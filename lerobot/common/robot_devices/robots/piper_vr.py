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
        target_hands = list(teleop_pipeline.allowed_hands) if teleop_pipeline.allowed_hands else ["piper"]
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

        if self.bus is not None and not self.config.dry_run:
            try:
                current = self.bus.read()
                joints_rad = np.array(
                    [
                        current["joint_1"],
                        current["joint_2"],
                        current["joint_3"],
                        current["joint_4"],
                        current["joint_5"],
                        current["joint_6"],
                    ],
                    dtype=float,
                )
                joint_info = ", ".join(f"{np.degrees(v):.2f}" for v in joints_rad)
                LOGGER.info(f"同步实机关节到 IK，起始角度(度)：{joint_info}")
                sync_reference_with_robot(self.session, self.pipeline, joints_rad)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("同步实机关节失败：%s", exc)

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
            if primary_bus is not None and not self.config.dry_run:
                target = np.asarray(primary_bus.init_joint_position[:6], dtype=float)
                joints_rad = self._wait_for_pose(target)
                if self.pipeline is not None and self.session is not None:
                    sync_reference_with_robot(self.session, self.pipeline, joints_rad, self._primary_hand)
                    joint_text = ", ".join(f"{np.degrees(v):.2f}" for v in joints_rad)
                    LOGGER.info(f"校准完成：同步 IK 参考，当前角度(度)：{joint_text}")
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

        if self.config.power_off_on_disconnect and not self.config.dry_run and self.bus_map:
            for bus in set(self.bus_map.values()):
                if bus is None:
                    continue
                try:
                    bus.connect(False)
                except Exception:  # pylint: disable=broad-except
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
        joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper",
        ]
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
        if self.bus is None:
            if self._last_action is not None:
                return self._last_action.copy()
            return np.zeros(7, dtype=float)

        try:
            start = time.perf_counter()
            state = self.bus.read()
            self.logs["read_follower_piper_pos_dt_s"] = time.perf_counter() - start
            values = np.array(
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
            return values
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.debug("读取 Piper 状态失败: %s", exc)
            if self._last_observation is not None:
                return self._last_observation.copy()
            if self._last_action is not None:
                return self._last_action.copy()
            return np.zeros(7, dtype=float)

    def _get_latest_action(self) -> np.ndarray:
        if self.pipeline is None:
            if self._last_action is not None:
                return self._last_action.copy()
            return self._read_observation()

        joints, gripper = self.pipeline.get_latest_command(self._primary_hand)
        state = None
        if hasattr(self.pipeline, "_get_state"):
            target_hand = self._primary_hand or (self._hands[0] if self._hands else None)
            if target_hand is not None:
                state = self.pipeline._get_state(target_hand)  # type: ignore[attr-defined]

        if joints is None:
            if self._last_action is not None:
                return self._last_action.copy()
            snapshot = self._read_observation()
            self._last_action = snapshot.copy()
            return snapshot

        if gripper is None:
            if self._last_action is not None:
                gripper = float(self._last_action[-1])
            elif state is not None:
                try:
                    gripper = float(state.gripper_open)
                except Exception:
                    gripper = 0.0
            else:
                gripper = 0.0

        action = np.concatenate([joints.reshape(-1), [float(gripper)]])
        return action

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
        if pressed and not self._recording_active:
            self._pending_recording_start = True
        if not pressed and self._recording_active:
            self._pending_recording_stop = True
        self._recording_active = pressed

    def _is_any_grip_active(self) -> bool:
        mapper = getattr(self.session, "mapper", None)
        if mapper is None:
            return False
        for hand in self._hands:
            controller = mapper.controllers.get(hand)
            if controller and controller.grip_active:
                return True
        return False
