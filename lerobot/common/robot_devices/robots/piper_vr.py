"""LeRobot 与 Piper VR 遥操作管线的适配器。"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
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

from robot.real.piper import PiperMotorsBus
from scripts.teleop_common import build_session as build_teleop_session
from scripts.run_vr_piper import (
    PiperTeleopPipeline,
    build_parser as build_piper_parser,
    build_piper_pipeline,
    extract_piper_hand_configs,
    sync_reference_with_robot,
)
from vr_runtime.webrtc_endpoint import VRWebRTCServer


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
            raise ValueError(f"Unsupported camera type '{raw_type}' in PiperVRRobotConfig.cameras")
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


@RobotConfig.register_subclass("piper_vr")
@dataclass
class PiperVRRobotConfig(RobotConfig):
    """Piper VR 遥操作录制配置。"""

    teleop_config: str = "configs/piper_recording.json"
    hardware_config: str | None = None
    dry_run: bool = False
    telemetry_file: str | None = None
    telemetry_sample_measured: bool = False
    skip_home: bool = False
    power_off_on_disconnect: bool = False
    host: str | None = None
    port: int | None = None
    channel: str | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        self.cameras = {
            name: _coerce_camera_config(cfg)
            for name, cfg in self.cameras.items()
        }


class PiperVRRobot(Robot):
    """面向 LeRobot 录制流程的 Piper VR 遥操作设备封装。"""

    def __init__(self, config: PiperVRRobotConfig):
        self.config = config
        self.robot_type = "piper_vr"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._server: VRWebRTCServer | None = None
        self.pipeline: PiperTeleopPipeline | None = None
        self.bus: PiperMotorsBus | None = None
        self.bus_map: Dict[str, PiperMotorsBus | None] = {}
        self.session = None
        self.logs: Dict[str, float] = {}
        self.is_connected = False
        self._last_action: Optional[np.ndarray] = None
        self._last_observation: Optional[np.ndarray] = None
        self.cameras: Dict[str, Any] = {}
        self.calibration_dir = REPO_ROOT / ".cache" / "piper"
        # 兼容控制模块的日志/校验逻辑：暴露 leader/follower 字段
        self.leader_arms: Dict[str, Any] = {}
        self.follower_arms: Dict[str, Any] = {
            "piper": {
                "motors": (
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                    "gripper",
                )
            }
        }
        self._recording_active: bool = False
        self._pending_recording_start: bool = False
        self._pending_recording_stop: bool = False
        self._hands: tuple[str, ...] = ()
        self._primary_hand: str | None = None
        self._config_hands: tuple[str, ...] = ()
        # 提前加载相机配置，便于未 connect 时创建数据集包含图像特征
        self._ensure_config_cameras()
        if not self.cameras and self.config.cameras:
            # 占位使用 config，后续 connect() 会替换为实际相机对象
            self.cameras = dict(self.config.cameras)

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
                self.config.cameras = {
                    name: _coerce_camera_config(cam) for name, cam in cam_cfgs.items()
                }
                LOGGER.info("已从 %s 预加载相机配置：%s", teleop_path, list(self.config.cameras))
        except Exception as exc:  # pylint: disable-broad-except
            LOGGER.warning("预加载相机配置失败，将在 connect 时再尝试: %s", exc)

            LOGGER.warning("预加载相机配置失败，将在 connect 时再尝试: %s", exc)


    # ---------------------------------------------------------------------
    # Robot 接口
    # ---------------------------------------------------------------------
    def connect(self):
        if self.is_connected:
            return

        teleop_path = _resolve_path(self.config.teleop_config)
        hardware_path = _resolve_path(self.config.hardware_config or self.config.teleop_config)

        parser = build_piper_parser()
        teleop_overrides = _load_config_dict(teleop_path)
        camera_overrides = teleop_overrides.get("cameras")
        hw_overrides, hand_overrides = extract_piper_hand_configs(teleop_overrides)
        known_flags = {action.dest for action in parser._actions}
        overrides = {key: value for key, value in teleop_overrides.items() if key in known_flags}
        parser.set_defaults(**overrides)

        args = parser.parse_args([])
        args.config = str(teleop_path)
        args.piper_config = str(hardware_path)
        args.dry_run = bool(self.config.dry_run)
        if self.config.telemetry_file is not None:
            args.telemetry_file = self.config.telemetry_file
        if self.config.telemetry_sample_measured:
            args.telemetry_sample_measured = True
        if self.config.host is not None:
            args.host = self.config.host
        if self.config.port is not None:
            args.port = self.config.port
        if self.config.channel is not None:
            args.channel = self.config.channel

        self.session, teleop_pipeline = build_teleop_session(args)

        self.bus_map = {}
        target_hands = (
            sorted(teleop_pipeline.allowed_hands) if teleop_pipeline.allowed_hands else ["piper"]
        )
        if not self.config.dry_run:
            for hand in target_hands:
                overrides = hw_overrides.get(hand)
                bus = PiperMotorsBus(hardware_path, overrides)
                LOGGER.info("[%s] 初始化 Piper 机械臂，总线配置: %s", hand, hardware_path)
                if not bus.connect(True):
                    raise RuntimeError(f"[{hand}] Piper 机械臂使能失败，请检查硬件连接与急停状态")
                if not self.config.skip_home:
                    try:
                        LOGGER.info("[%s] 连接完成：移动至初始姿态（apply_calibration）", hand)
                        bus.apply_calibration()
                    except Exception as exc:  # pylint: disable=broad-except
                        LOGGER.warning("[%s] 初始回零失败: %s", hand, exc)
                self.bus_map[hand] = bus
        else:
            self.bus_map = {hand: None for hand in target_hands}

        self._primary_hand = target_hands[0] if target_hands else None
        self.bus = self.bus_map.get(self._primary_hand) if self._primary_hand else None

        if not self.config.cameras and camera_overrides:
            self.config.cameras = {
                name: _coerce_camera_config(cfg)
                for name, cfg in camera_overrides.items()
            }
            camera_summary = {
                name: {
                    "type": cam_cfg.type,
                    **{
                        key: getattr(cam_cfg, key)
                        for key in ("camera_index", "serial_number", "fps", "width", "height")
                        if hasattr(cam_cfg, key)
                    },
                }
                for name, cam_cfg in self.config.cameras.items()
            }
            LOGGER.info(f"加载相机配置：{camera_summary}")

        self.pipeline = build_piper_pipeline(
            args=args,
            session=self.session,
            teleop_pipeline=teleop_pipeline,
            bus_map=self.bus_map,
            hand_pipeline_overrides=hand_overrides,
        )

        self._hands = tuple(teleop_pipeline.allowed_hands)
        self.cameras = make_cameras_from_configs(self.config.cameras) if self.config.cameras else {}
        for camera in self.cameras.values():
            try:
                camera.connect()
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("连接相机失败: %s", exc)
        self.reset_recording_state()

        if self.bus_map and not self.config.dry_run:
            for hand, bus in self.bus_map.items():
                if bus is None:
                    continue
                try:
                    current = bus.read()
                    joints_rad = np.array(
                        [
                            current.get("joint_1", 0.0),
                            current.get("joint_2", 0.0),
                            current.get("joint_3", 0.0),
                            current.get("joint_4", 0.0),
                            current.get("joint_5", 0.0),
                            current.get("joint_6", 0.0),
                        ],
                        dtype=float,
                    )
                    joint_info = ", ".join(f"{np.degrees(v):.2f}" for v in joints_rad)
                    LOGGER.info("[%s] 同步实机关节到 IK，起始角度(度)：%s", hand, joint_info)
                    sync_reference_with_robot(self.session, self.pipeline, joints_rad, hand)
                except Exception as exc:  # pylint: disable-broad-except
                    LOGGER.warning("[%s] 同步实机关节失败：%s", hand, exc)

        stun_servers = [] if args.no_stun else list(args.stun)
        self._server = VRWebRTCServer(
            host=args.host,
            port=args.port,
            pipeline=self.pipeline,  # type: ignore[arg-type]
            channel_name=args.channel,
            stun_servers=stun_servers,
        )

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="piper-vr-webrtc",
            daemon=True,
        )
        self._loop_thread.start()
        try:
            asyncio.run_coroutine_threadsafe(self._server.start(), self._loop).result(timeout=5)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error("启动 VR WebRTC 服务失败: %s", exc)
            self.disconnect()
            raise

        self.is_connected = True

    def run_calibration(self):
        if not self.bus_map:
            LOGGER.debug("dry-run 模式下跳过 apply_calibration。")
            return
        try:
            if self.pipeline is not None:
                self.pipeline.reset()
                try:
                    self.pipeline._clear_pending_command()  # type: ignore[attr-defined]
                except AttributeError:
                    pass
            primary_bus = self.bus_map.get(self._primary_hand) or next(iter(self.bus_map.values()), None)
            for hand, bus in self.bus_map.items():
                if bus is None or self.config.dry_run:
                    continue
                try:
                    bus.apply_calibration()
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.warning("[%s] apply_calibration 失败: %s", hand, exc)
            # 等待主臂回零并同步参考
            if primary_bus is not None and not self.config.dry_run:
                target = np.asarray(primary_bus.init_joint_position[:6], dtype=float)
                joints_rad = self._wait_for_pose(target)
                if self.pipeline is not None and self.session is not None:
                    sync_reference_with_robot(self.session, self.pipeline, joints_rad, self._primary_hand)
                    joint_text = ", ".join(f"{np.degrees(v):.2f}" for v in joints_rad)
                    LOGGER.info(f"校准完成：同步 IK 参考，当前角度(度)：{joint_text}")
            # 为所有手同步实机参考，避免仅主臂更新
            if self.pipeline is not None and self.session is not None and not self.config.dry_run:
                for hand, bus in self.bus_map.items():
                    if bus is None:
                        continue
                    try:
                        state = bus.read()
                        joints_rad = np.array(
                            [
                                state["joint_1"],
                                state["joint_2"],
                                state["joint_3"],
                                state["joint_4"],
                                state["joint_5"],
                                state["joint_6"],
                            ],
                            dtype=float,
                        )
                        sync_reference_with_robot(self.session, self.pipeline, joints_rad, hand)
                        joint_text = ", ".join(f"{np.degrees(v):.2f}" for v in joints_rad)
                        LOGGER.info(f"[{hand}] 同步实机关节到 IK，起始角度(度)：{joint_text}")
                    except Exception as exc:  # pylint: disable=broad-except
                        LOGGER.warning("[%s] 校准后同步参考失败: %s", hand, exc)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("apply_calibration 失败: %s", exc)

    def teleop_step(self, record_data: bool = False):
        if not self.is_connected or self.pipeline is None:
            raise RuntimeError("PiperVRRobot 尚未连接或管线未初始化")

        observation = self._read_observation()
        action = self._get_latest_action()
        self._update_recording_state()

        obs_tensor = torch.from_numpy(observation.astype(np.float32))
        act_tensor = torch.from_numpy(action.astype(np.float32))

        self._last_observation = observation
        self._last_action = action

        obs_dict = {"observation.state": obs_tensor}
        action_dict = {"action": act_tensor}
        if record_data and self.cameras:
            for name, camera in self.cameras.items():
                before = time.perf_counter()
                try:
                    image = camera.async_read()
                    obs_dict[f"observation.images.{name}"] = torch.from_numpy(image)
                    self.logs[f"read_camera_{name}_dt_s"] = time.perf_counter() - before
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.debug("读取相机 %s 失败: %s", name, exc)
        return obs_dict, action_dict

    def capture_observation(self):
        observation = self._read_observation()
        self._last_observation = observation
        return {"observation.state": torch.from_numpy(observation.astype(np.float32))}

    def resync_reference(self) -> None:
        if self.bus is None or self.pipeline is None or self.session is None:
            return

        try:
            measured = self._read_observation()
            joints_rad = np.asarray(measured[:6], dtype=float)
            self.pipeline.reset()
            try:
                self.pipeline._clear_pending_command()  # type: ignore[attr-defined]
            except AttributeError:
                pass
            sync_reference_with_robot(self.session, self.pipeline, joints_rad)
            LOGGER.info(
                "同步当前姿态到 IK 参考，关节角(度)：%s",
                ", ".join(f"{np.degrees(v):.2f}" for v in joints_rad),
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("握持开始前同步参考失败: %s", exc)

    def _wait_for_pose(
        self,
        target_joints: np.ndarray,
        timeout: float = 8.0,
        tolerance_deg: float = 1.0,
    ) -> np.ndarray:
        tol_rad = math.radians(tolerance_deg)
        deadline = time.perf_counter() + timeout
        last = target_joints.copy()

        while time.perf_counter() < deadline:
            observation = self._read_observation()
            if observation is not None:
                current = np.asarray(observation[:6], dtype=float)
                last = current
                if np.max(np.abs(current - target_joints)) <= tol_rad:
                    return current
            time.sleep(0.05)

        max_err_deg = math.degrees(float(np.max(np.abs(last - target_joints)))) if last is not None else float("nan")
        LOGGER.warning("等待回初始姿态超时，最大偏差 %.2f 度", max_err_deg)
        return last

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.bus is None:
            return action.clone()

        values = action.detach().cpu().numpy().astype(float).reshape(-1)
        if values.size < 6:
            raise ValueError("send_action 需要至少 6 个关节目标值")

        if values.size == 6:
            gripper = 0.0
            state = None
            if self.pipeline is not None and hasattr(self.pipeline, "_get_state"):
                target_hand = self._primary_hand or (self._hands[0] if self._hands else None)
                if target_hand is not None:
                    state = self.pipeline._get_state(target_hand)  # type: ignore[attr-defined]
            if state is not None:
                try:
                    gripper = float(state.gripper_open)
                except Exception:
                    gripper = 0.0
            values = np.concatenate([values, [gripper]])

        try:
            self.bus.write(values.tolist())
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("直接写入 Piper 指令失败: %s", exc)
        return torch.from_numpy(values.astype(np.float32))

    def disconnect(self):
        if not self.is_connected:
            return

        if self._server is not None and self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._server.stop(), self._loop).result(timeout=5)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("关闭 WebRTC 服务失败: %s", exc)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=5)
            self._loop.close()
        self._loop = None
        self._loop_thread = None
        self._server = None

        if self.pipeline is not None:
            try:
                self.pipeline.shutdown()
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("关闭 Piper 管线失败: %s", exc)

        if self.bus_map and not self.config.dry_run:
            for hand, bus in self.bus_map.items():
                if bus is None:
                    continue
                try:
                    LOGGER.info("[%s] 退出：回到初始姿态", hand)
                    bus.apply_calibration()
                except Exception as exc:  # pylint: disable-broad-except
                    LOGGER.warning("[%s] 回初始姿态失败: %s", hand, exc)
                if self.config.power_off_on_disconnect:
                    try:
                        bus.connect(False)
                    except Exception:  # pylint: disable-broad-except
                        pass

        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except Exception:  # pylint: disable=broad-except
                pass
        self.pipeline = None
        self.bus = None
        self.bus_map = {}
        self.session = None
        self.reset_recording_state()
        self.is_connected = False

    # ------------------------------------------------------------------
    # 数据集接口辅助
    # ------------------------------------------------------------------
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

    @property
    def available_arms(self) -> list[str]:  # pragma: no cover - 兼容控制工具
        return list(self.follower_arms.keys())

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
    def motor_features(self) -> Dict[str, Any]:  # pragma: no cover - 兼容接口
        return {
            "observation.state": self.features["observation.state"],
            "action": self.features["action"],
        }

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _read_observation(self) -> np.ndarray:
        hands = self._hand_order() or list(self._infer_hands_from_config())
        if not hands:
            return np.zeros(7, dtype=float)

        last_obs = self._last_observation if self._last_observation is not None else None
        expected_len = len(hands) * 7
        if last_obs is not None and last_obs.size != expected_len:
            last_obs = None

        chunks: list[np.ndarray] = []
        for idx, hand in enumerate(hands):
            fallback = None
            if last_obs is not None:
                start = idx * 7
                fallback = last_obs[start : start + 7]

            bus = self.bus_map.get(hand) if hasattr(self, "bus_map") else None
            if bus is None:
                chunk = fallback if fallback is not None else np.zeros(7, dtype=float)
                chunks.append(chunk.astype(float))
                continue

            try:
                start_t = time.perf_counter()
                state = bus.read()
                self.logs[f"read_{hand}_pos_dt_s"] = time.perf_counter() - start_t
                chunk = np.array(
                    [
                        state.get("joint_1", 0.0),
                        state.get("joint_2", 0.0),
                        state.get("joint_3", 0.0),
                        state.get("joint_4", 0.0),
                        state.get("joint_5", 0.0),
                        state.get("joint_6", 0.0),
                        state.get("gripper", 0.0),
                    ],
                    dtype=float,
                )
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.debug("读取 Piper 状态失败 [%s]: %s", hand, exc)
                if fallback is not None:
                    chunk = fallback
                elif self._last_action is not None and self._last_action.size == expected_len:
                    start = idx * 7
                    chunk = self._last_action[start : start + 7]
                else:
                    chunk = np.zeros(7, dtype=float)
            chunks.append(chunk.astype(float))

        return np.concatenate(chunks) if chunks else np.zeros(7, dtype=float)

    def _get_latest_action(self) -> np.ndarray:
        hands = self._hand_order() or list(self._infer_hands_from_config())
        if not hands:
            return np.zeros(7, dtype=float)

        if self.pipeline is None:
            if self._last_action is not None and self._last_action.size == len(hands) * 7:
                return self._last_action.copy()
            return self._read_observation()

        last_action = self._last_action if self._last_action is not None and self._last_action.size == len(hands) * 7 else None
        last_obs = self._last_observation if self._last_observation is not None and self._last_observation.size == len(hands) * 7 else None

        chunks: list[np.ndarray] = []
        for idx, hand in enumerate(hands):
            joints, gripper = self.pipeline.get_latest_command(hand)

            state = None
            if hasattr(self.pipeline, "_get_state"):
                try:
                    state = self.pipeline._get_state(hand)  # type: ignore[attr-defined]
                except Exception:
                    state = None

            if joints is None:
                if last_action is not None:
                    start = idx * 7
                    chunk = last_action[start : start + 7]
                    chunks.append(chunk.copy())
                    continue
                if last_obs is not None:
                    start = idx * 7
                    chunk = last_obs[start : start + 7]
                    chunks.append(chunk.copy())
                    continue
                chunks.append(np.zeros(7, dtype=float))
                continue

            if gripper is None:
                if last_action is not None:
                    gripper = float(last_action[idx * 7 + 6])
                elif state is not None:
                    try:
                        gripper = float(state.gripper_open)
                    except Exception:
                        gripper = 0.0
                else:
                    gripper = 0.0

            chunk = np.concatenate([joints.reshape(-1), [float(gripper)]])
            chunks.append(chunk.astype(float))

        if not chunks:
            return np.zeros(7, dtype=float)
        return np.concatenate(chunks)

    # ------------------------------------------------------------------
    # 录制状态管理
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
        pressed = self._is_any_grip_active()
        grip_flags = self._grip_state_per_hand()
        if pressed and not self._recording_active:
            self._pending_recording_start = True
            self._log_recording_event("start", True, grip_flags)
        if not pressed and self._recording_active:
            self._pending_recording_stop = True
            self._log_recording_event("stop", False, grip_flags)
        self._recording_active = pressed

    def _grip_state_per_hand(self) -> Dict[str, bool]:
        mapper = getattr(self.session, "mapper", None)
        flags: Dict[str, bool] = {}
        if mapper is None or not hasattr(mapper, "controllers"):
            return flags
        for hand in self._hands:
            try:
                ctrl = mapper.controllers.get(hand)
                flags[hand] = bool(ctrl.grip_active) if ctrl is not None else False
            except Exception:
                continue
        return flags

    def _log_recording_event(self, event: str, active: bool, grip_flags: Dict[str, bool]) -> None:
        if self.pipeline is None:
            return
        try:
            arms = getattr(self.pipeline, "_arms", {})  # type: ignore[attr-defined]
            for state in arms.values():
                state.telemetry.log_recording_event(event, active, grip_flags)
        except Exception:
            pass

    def _is_any_grip_active(self) -> bool:
        mapper = getattr(self.session, "mapper", None)
        if mapper is None:
            return False
        for hand in self._hands:
            controller = mapper.controllers.get(hand)
            if controller and controller.grip_active:
                return True
        return False
