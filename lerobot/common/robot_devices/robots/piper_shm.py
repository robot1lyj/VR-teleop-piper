"""Piper 双进程录制适配器（SharedMemory 只读）。

目标：把“实时控制链路(≈90Hz)”与“录制/写盘链路(≈30Hz)”彻底硬隔离，避免录制导致控制卡顿。

设计要点（非常重要）：
1) 本进程 **不启动** WebRTC/IK/硬件总线；只读 SharedMemory + 相机。
2) 相机采集线程内记录 ``time.monotonic_ns()``，作为跨进程一致的时间基准。
3) 与 LeRobot 默认采集行为一致：录制 loop 按数据集 fps（如 30Hz）运行；每帧读取“各相机最新帧”，
   允许重复旧帧（相机掉帧/低 fps 时仍尽量保持 30Hz 录制）；cmd/meas 取“当前时刻”ring 中最新样本。
   多相机时间戳偏差（cam_skew）与 cmd/meas 滞后超阈值只做诊断告警，不作为丢帧条件。
4) 录制触发沿用旧逻辑：任一手握持(grip) 即 active；按 active 上/下沿触发 start/stop。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.configs import (
    CameraConfig,
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.common.robot_devices.robots.utils import Robot

from vr_runtime.shm_ring import ShmRingReader, ShmStatusReader

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[4]

_CAMERA_TYPE_TO_CFG = {
    "intelrealsense": IntelRealSenseCameraConfig,
    "opencv": OpenCVCameraConfig,
}

_CAMERA_TYPE_ALIASES = {
    "intelrealsense": "intelrealsense",
    "realsense": "intelrealsense",
    "opencv": "opencv",
    "opencvcamera": "opencv",
}


def _coerce_camera_config(cfg: Any) -> CameraConfig:
    if isinstance(cfg, CameraConfig):
        return cfg
    if isinstance(cfg, dict):
        raw_type = cfg.get("type")
        cam_type_key = str(raw_type).lower() if raw_type is not None else ""
        cam_type = _CAMERA_TYPE_ALIASES.get(cam_type_key, cam_type_key)
        if cam_type not in _CAMERA_TYPE_TO_CFG:
            raise ValueError(f"Unsupported camera type '{raw_type}' in PiperShmRobotConfig.cameras")
        params = {k: v for k, v in cfg.items() if k != "type"}
        return _CAMERA_TYPE_TO_CFG[cam_type](**params)
    raise TypeError(f"Camera config must be CameraConfig or dict, got {type(cfg)}")


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _load_config_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件 {path} 必须是 JSON 对象")
    return data


@RobotConfig.register_subclass("piper_shm")
@dataclass
class PiperShmRobotConfig(RobotConfig):
    """Piper 双进程录制配置（录制端只读 SharedMemory）。"""

    teleop_config: str = "configs/piper_recording.json"
    shm_name: str = "piper_vr"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        self.cameras = {name: _coerce_camera_config(cfg) for name, cfg in self.cameras.items()}


class PiperShmRobot(Robot):
    """录制侧机器人：只读 cmd/meas ring + 相机，并输出 Lerobot 格式 observation/action。"""

    def __init__(self, config: PiperShmRobotConfig):
        self.config = config
        self.robot_type = "piper_shm"
        self.is_connected = False

        self.logs: Dict[str, Any] = {}
        self.cameras: Dict[str, Any] = {}

        self._status: ShmStatusReader | None = None
        self._cmd_rings: dict[str, ShmRingReader] = {}
        self._meas_rings: dict[str, ShmRingReader] = {}

        self._recording_active = False
        self._pending_recording_start = False
        self._pending_recording_stop = False

        self._last_action: Optional[np.ndarray] = None
        self._last_observation: Optional[np.ndarray] = None
        # 与 LeRobot 采集行为对齐：若某路相机掉帧/报错，则复用上一帧图像（允许重复帧）。
        self._last_images: dict[str, np.ndarray] = {}
        self._last_image_t_ns: dict[str, int] = {}

        self._hands: tuple[str, ...] = ()
        self._config_hands: tuple[str, ...] = ()

        self._ensure_config_cameras()
        if not self.cameras and self.config.cameras:
            # dataset create 阶段需要 camera keys/shape：占位用 config，connect 后替换为相机对象
            self.cameras = dict(self.config.cameras)

    # ------------------------------------------------------------------
    # 配置解析：hands/cameras
    # ------------------------------------------------------------------
    def _hand_order(self) -> list[str]:
        """固定 hand 顺序，确保平铺向量一致（右手优先，其次左手，其余按字典序）。"""

        hands: tuple[str, ...] = self._hands or self._config_hands
        if not hands:
            return []
        right_first = [h for h in hands if h == "right"]
        left_next = [h for h in hands if h == "left"]
        others = sorted([h for h in hands if h not in {"left", "right"}])
        return right_first + left_next + others

    def _infer_hands_from_config(self) -> tuple[str, ...]:
        if self._config_hands:
            return self._config_hands
        try:
            teleop_path = _resolve_path(self.config.teleop_config)
            cfg = _load_config_dict(teleop_path)
            raw = str(cfg.get("hands", "right")).lower()
            if raw == "both":
                self._config_hands = ("right", "left")
            else:
                self._config_hands = (raw,)
        except Exception:  # pylint: disable=broad-except
            self._config_hands = ("right",)
        return self._config_hands

    def _ensure_config_cameras(self) -> None:
        """在未 connect 前预先加载 teleop 配置里的相机，避免 dataset schema 缺失。"""

        if self.config.cameras:
            return
        try:
            teleop_path = _resolve_path(self.config.teleop_config)
            cfg = _load_config_dict(teleop_path)
            cam_cfgs = cfg.get("cameras")
            if cam_cfgs:
                self.config.cameras = {name: _coerce_camera_config(cam) for name, cam in cam_cfgs.items()}
                LOGGER.info("已从 %s 预加载相机配置：%s", teleop_path, list(self.config.cameras))
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("预加载相机配置失败，将在 connect 时再尝试: %s", exc)

    # ------------------------------------------------------------------
    # 数据集特征
    # ------------------------------------------------------------------
    @property
    def camera_features(self) -> Dict[str, Any]:
        self._ensure_config_cameras()
        camera_ft: Dict[str, Dict[str, Any]] = {}
        for cam_key in self.config.cameras:
            camera = self.cameras.get(cam_key)
            cfg = self.config.cameras[cam_key]
            height = getattr(camera, "height", None) or getattr(cfg, "height", 0) or 0
            width = getattr(camera, "width", None) or getattr(cfg, "width", 0) or 0
            channels = getattr(camera, "channels", None) or getattr(cfg, "channels", 3) or 3
            camera_ft[f"observation.images.{cam_key}"] = {
                "shape": (height, width, channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return camera_ft

    @property
    def motor_features(self) -> Dict[str, Any]:  # pragma: no cover - 兼容 LeRobotDataset.create
        """兼容 LeRobot 的 get_features_from_robot()：提供电机相关特征。"""

        return {
            "observation.state": self.features["observation.state"],
            "action": self.features["action"],
        }

    @property
    def features(self) -> Dict[str, Dict[str, Any]]:
        self._ensure_config_cameras()
        hands_list = self._hand_order()
        if not hands_list:
            hands_list = list(self._infer_hands_from_config())
        hands = hands_list or ["right"]

        joint_names: list[str] = []
        for hand in hands:
            prefix = f"{hand}_"
            joint_names.extend([f"{prefix}joint_{i}" for i in range(1, 7)])
            joint_names.append(f"{prefix}gripper")

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(joint_names),),
                "names": joint_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(joint_names),),
                "names": joint_names,
            },
        }
        for cam_key, cam_ft in self.camera_features.items():
            features[cam_key] = cam_ft
        return features

    # ------------------------------------------------------------------
    # 录制状态管理（沿用旧逻辑）
    # ------------------------------------------------------------------
    def reset_recording_state(self) -> None:
        self._recording_active = False
        self._pending_recording_start = False
        self._pending_recording_stop = False

    def consume_recording_events(self) -> tuple[bool, bool, bool]:
        started = self._pending_recording_start
        stopped = self._pending_recording_stop
        self._pending_recording_start = False
        self._pending_recording_stop = False
        return started, stopped, self._recording_active

    def _update_recording_state(self) -> None:
        """根据 shm status 的 grip_mask 更新 start/stop/active。"""

        status = self._status
        if status is None:
            return

        try:
            _t_ns, grip_mask = status.read()
        except Exception:
            return

        self.logs["grip_mask"] = int(grip_mask)

        enabled_mask = 0
        hands = self._hands or self._infer_hands_from_config()
        if "right" in hands:
            enabled_mask |= 0b01
        if "left" in hands:
            enabled_mask |= 0b10

        pressed = bool(int(grip_mask) & int(enabled_mask))
        if pressed and not self._recording_active:
            self._pending_recording_start = True
        if not pressed and self._recording_active:
            self._pending_recording_stop = True
        self._recording_active = pressed

    # ------------------------------------------------------------------
    # Robot 接口
    # ------------------------------------------------------------------
    def connect(self):
        if self.is_connected:
            return

        hands = tuple(self._infer_hands_from_config())
        if not hands:
            hands = ("right",)
        self._hands = hands

        base = str(self.config.shm_name)
        try:
            self._status = ShmStatusReader.attach(f"{base}_status")
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"未找到 SharedMemory '{base}_status'，请先启动遥操作控制进程并启用 --publish-shm（本机同一用户）。"
            ) from exc

        self._cmd_rings = {}
        self._meas_rings = {}
        for hand in self._hand_order():
            try:
                self._cmd_rings[hand] = ShmRingReader.attach(f"{base}_cmd_{hand}")
                self._meas_rings[hand] = ShmRingReader.attach(f"{base}_meas_{hand}")
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"未找到 SharedMemory ring：hand={hand}，请确认控制进程已创建 {base}_cmd_{hand}/{base}_meas_{hand}。"
                ) from exc

        self._ensure_config_cameras()
        self.cameras = make_cameras_from_configs(self.config.cameras) if self.config.cameras else {}
        for camera in self.cameras.values():
            try:
                camera.connect()
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("连接相机失败: %s", exc)

        self.reset_recording_state()
        self.is_connected = True

    def run_calibration(self):
        # 录制端只读，不负责回零/校准
        return

    def teleop_step(self, record_data: bool = False):
        if not self.is_connected:
            raise RuntimeError("PiperShmRobot 尚未 connect")

        self._update_recording_state()

        obs_dict: Dict[str, Any] = {}
        cam_ts: list[int] = []
        cam_ts_by_name: dict[str, int] = {}
        if record_data and self.cameras:
            # 与 LeRobot 默认采集方式保持一致：每帧读取“各相机最新帧”，允许重复旧帧。
            # 这能保证录制 loop 尽量稳定在 30Hz，而不是因为等待某路相机更新而把 loop 拉低。
            now_ns = time.monotonic_ns()
            for name in list(self.config.cameras.keys()):
                camera = self.cameras.get(name)
                img_np: np.ndarray | None = None
                t_ns_i: int | None = None
                status = "ok"
                try:
                    if camera is None:
                        status = "camera_missing"
                    elif hasattr(camera, "async_read_with_timestamp"):
                        image, t_ns = camera.async_read_with_timestamp()  # type: ignore[attr-defined]
                        img_np = np.asarray(image)
                        t_ns_i = int(t_ns)
                        self._last_images[name] = img_np
                        self._last_image_t_ns[name] = t_ns_i
                    else:
                        image = camera.async_read()
                        img_np = np.asarray(image)
                        t_ns_i = int(time.monotonic_ns())
                        self._last_images[name] = img_np
                        self._last_image_t_ns[name] = t_ns_i
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.debug("读取相机 %s 失败: %s", name, exc)
                    status = "read_error"

                if img_np is None:
                    if name in self._last_images:
                        img_np = self._last_images[name]
                        t_ns_i = int(self._last_image_t_ns.get(name, now_ns))
                        status = "reuse_last"
                    else:
                        cfg = self.config.cameras.get(name)
                        h = int(getattr(cfg, "height", 0) or 0)
                        w = int(getattr(cfg, "width", 0) or 0)
                        c = int(getattr(cfg, "channels", 3) or 3)
                        if h > 0 and w > 0 and c > 0:
                            img_np = np.zeros((h, w, c), dtype=np.uint8)
                        else:
                            img_np = np.zeros((1, 1, 3), dtype=np.uint8)
                        t_ns_i = int(now_ns)
                        status = "zeros"

                obs_dict[f"observation.images.{name}"] = torch.from_numpy(np.asarray(img_np))
                cam_ts.append(int(t_ns_i))
                cam_ts_by_name[name] = int(t_ns_i)
                self.logs[f"cam_status_{name}"] = status

            # 记录每路相机“本帧拿到的 latest 帧”新鲜度，便于定位哪一路在掉帧/卡住
            now_ns = time.monotonic_ns()
            for name, t_ns_i in cam_ts_by_name.items():
                self.logs[f"cam_t_ns_{name}"] = int(t_ns_i)
                self.logs[f"cam_age_ms_{name}"] = (int(now_ns) - int(t_ns_i)) / 1e6

        # ------------------------------
        # 多相机时间戳偏差（诊断用，不作为丢帧条件，与 LeRobot 行为一致）
        # ------------------------------
        if cam_ts:
            t_min = int(min(cam_ts))
            t_max = int(max(cam_ts))
            skew_ns = t_max - t_min
            skew_ms = skew_ns / 1e6
            self.logs["cam_skew_ms"] = float(skew_ms)

            min_name = None
            max_name = None
            if cam_ts_by_name:
                try:
                    min_name = min(cam_ts_by_name, key=lambda k: cam_ts_by_name[k])
                    max_name = max(cam_ts_by_name, key=lambda k: cam_ts_by_name[k])
                except Exception:
                    min_name = None
                    max_name = None
            self.logs["cam_skew_min_cam"] = str(min_name or "")
            self.logs["cam_skew_max_cam"] = str(max_name or "")
        else:
            self.logs["cam_skew_ms"] = 0.0
            self.logs["cam_skew_min_cam"] = ""
            self.logs["cam_skew_max_cam"] = ""

        # 与 LeRobot 一致：用“当前时刻”作为采样锚点（仅用于诊断 lag，不作为丢帧条件）
        anchor_ns = time.monotonic_ns()
        self.logs["anchor_ns"] = int(anchor_ns)

        hands = self._hand_order() or list(self._infer_hands_from_config())
        if not hands:
            hands = ["right"]
        expected = len(hands) * 7

        last_obs = self._last_observation if self._last_observation is not None else None
        last_act = self._last_action if self._last_action is not None else None
        if last_obs is not None and last_obs.size != expected:
            last_obs = None
        if last_act is not None and last_act.size != expected:
            last_act = None

        frame_valid = True
        invalid_reasons: list[str] = []

        def _add_reason(reason: str) -> None:
            if reason and reason not in invalid_reasons:
                invalid_reasons.append(reason)

        # ------------------------------
        # 从 ring buffer 取 cmd/meas：每手 best-effort 取 latest；缺失则复用上一条/填 0，保证 shape 恒定
        # ------------------------------
        cmd_chunks: list[np.ndarray] = []
        meas_chunks: list[np.ndarray] = []
        for idx, hand in enumerate(hands):
            start = idx * 7
            fallback_obs = last_obs[start : start + 7] if last_obs is not None else None
            fallback_act = last_act[start : start + 7] if last_act is not None else fallback_obs

            cmd_q: np.ndarray | None = None
            meas_q: np.ndarray | None = None

            cmd_ring = self._cmd_rings.get(hand)
            if cmd_ring is None:
                frame_valid = False
                _add_reason(f"missing_ring:{hand}")
            else:
                cmd_item = cmd_ring.get_latest()
                if cmd_item is None:
                    frame_valid = False
                    _add_reason(f"cmd_timeout:{hand}")
                else:
                    _seq_c, t_cmd_ns, meta_c, q_cmd = cmd_item
                    if (int(meta_c) & 0b01) == 0:
                        frame_valid = False
                        _add_reason(f"meta_invalid:{hand}")
                    else:
                        self.logs[f"cmd_lag_ms_{hand}"] = (int(anchor_ns) - int(t_cmd_ns)) / 1e6
                        cmd_q = np.asarray(q_cmd, dtype=np.float32).reshape(7)

            meas_ring = self._meas_rings.get(hand)
            if meas_ring is None:
                frame_valid = False
                _add_reason(f"missing_ring:{hand}")
            else:
                meas_item = meas_ring.get_latest()
                if meas_item is None:
                    frame_valid = False
                    _add_reason(f"meas_timeout:{hand}")
                else:
                    _seq_m, t_meas_ns, meta_m, q_meas = meas_item
                    if (int(meta_m) & 0b01) == 0:
                        frame_valid = False
                        _add_reason(f"meta_invalid:{hand}")
                    else:
                        self.logs[f"meas_lag_ms_{hand}"] = (int(anchor_ns) - int(t_meas_ns)) / 1e6
                        meas_q = np.asarray(q_meas, dtype=np.float32).reshape(7)

            if cmd_q is None:
                cmd_q = (
                    np.asarray(fallback_act, dtype=np.float32).reshape(7)
                    if fallback_act is not None
                    else np.zeros(7, dtype=np.float32)
                )
            if meas_q is None:
                meas_q = (
                    np.asarray(fallback_obs, dtype=np.float32).reshape(7)
                    if fallback_obs is not None
                    else np.zeros(7, dtype=np.float32)
                )

            cmd_chunks.append(cmd_q)
            meas_chunks.append(meas_q)

        action = np.concatenate(cmd_chunks, axis=0).astype(np.float32) if cmd_chunks else np.zeros(expected, dtype=np.float32)
        observation = (
            np.concatenate(meas_chunks, axis=0).astype(np.float32) if meas_chunks else np.zeros(expected, dtype=np.float32)
        )

        self.logs["frame_valid"] = bool(frame_valid)
        self.logs["frame_invalid_reason"] = ";".join(invalid_reasons) if not frame_valid else ""

        obs_tensor = torch.from_numpy(observation.astype(np.float32))
        act_tensor = torch.from_numpy(action.astype(np.float32))

        self._last_observation = observation
        self._last_action = action

        obs_dict["observation.state"] = obs_tensor
        action_dict = {"action": act_tensor}
        return obs_dict, action_dict

    def capture_observation(self):
        if not self.is_connected:
            raise RuntimeError("PiperShmRobot 尚未 connect")
        # 不读相机，只返回 best-effort 的最新 meas（缺失则复用上一条/填 0，保证 shape 恒定）
        hands = self._hand_order() or list(self._infer_hands_from_config())
        if not hands:
            hands = ["right"]
        expected = len(hands) * 7

        last_obs = self._last_observation if self._last_observation is not None else None
        if last_obs is not None and last_obs.size != expected:
            last_obs = None

        chunks: list[np.ndarray] = []
        for idx, hand in enumerate(hands):
            start = idx * 7
            fallback = last_obs[start : start + 7] if last_obs is not None else None
            meas_ring = self._meas_rings.get(hand)
            item = meas_ring.get_latest() if meas_ring is not None else None
            if item is None:
                chunk = fallback if fallback is not None else np.zeros(7, dtype=np.float32)
                chunks.append(chunk.astype(np.float32))
                continue
            _seq, _t_ns, meta, q = item
            if (int(meta) & 0b01) == 0:
                chunk = fallback if fallback is not None else np.zeros(7, dtype=np.float32)
                chunks.append(chunk.astype(np.float32))
                continue
            chunks.append(np.asarray(q, dtype=np.float32).reshape(7))

        observation = np.concatenate(chunks, axis=0).astype(np.float32) if chunks else np.zeros(expected, dtype=np.float32)
        return {"observation.state": torch.from_numpy(observation)}

    def send_action(self, action):
        # 录制端不下发动作，直接回传
        return action

    def disconnect(self):
        if not self.is_connected:
            return

        for ring in list(self._cmd_rings.values()):
            try:
                ring.close()
            except Exception:
                pass
        for ring in list(self._meas_rings.values()):
            try:
                ring.close()
            except Exception:
                pass
        self._cmd_rings = {}
        self._meas_rings = {}

        if self._status is not None:
            try:
                self._status.close()
            except Exception:
                pass
            self._status = None

        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except Exception:
                pass
        self.cameras = {}
        self.reset_recording_state()
        self.is_connected = False
