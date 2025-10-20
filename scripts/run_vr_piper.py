"""VR 手柄 -> Piper 实机遥操作入口。

继承 Meshcat 版的配置/IK 管线，额外对接 `PiperMotorsBus`，
在真实机械臂上复用同款的运动学求解参数。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import pathlib
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pinocchio as pin

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PIPER_CONFIG = ROOT_DIR / "configs" / "piper.json"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from robot.real.piper import PiperMotorsBus
from scripts.run_vr_meshcat import (  # type: ignore[import]
    TeleopPipeline,
    _create_config_parser,
    _load_config_file,
    build_arg_parser as build_meshcat_arg_parser,
    build_session as build_meshcat_session,
)
from vr_runtime.webrtc_endpoint import VRWebRTCServer


class PiperTeleopPipeline(TeleopPipeline):
    """在 Meshcat 版基础上加入 Piper 机械臂命令下发。"""

    def __init__(
        self,
        session,
        allowed_hands: Iterable[str],
        reference_translation: np.ndarray,
        reference_rotation: np.ndarray,
        *,
        bus: Optional[PiperMotorsBus],
        command_interval: float,
        gripper_open: float,
        gripper_closed: float,
        dry_run: bool,
        effort_samples: int,
        effort_interval: float,
        effort_mode: str,
        fine_scale: Optional[float] = None,
        fine_filter_alpha: float = 0.3,
        fine_deadband: float = 0.002,
    ) -> None:
        super().__init__(session, allowed_hands, reference_translation, reference_rotation)
        self.bus = bus
        self.command_interval = max(0.0, float(command_interval))
        self.gripper_open = float(gripper_open)
        self.gripper_closed = float(gripper_closed)
        self.dry_run = bool(dry_run)
        self._last_command_ts = 0.0
        self._effort_samples = max(0, int(effort_samples))
        self._effort_interval = max(0.0, float(effort_interval))
        self._effort_mode = effort_mode.lower()
        self._fine_scale = fine_scale
        self._fine_filter_alpha = fine_filter_alpha
        self._fine_deadband = fine_deadband
        self._pending_command: Optional[Tuple[List[float], float]] = None
        self._command_lock = threading.Lock()
        self._command_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_write_duration_ms: Optional[float] = None
        self._last_queue_delay_ms: Optional[float] = None
        self._last_gripper_effort: Optional[float] = None
        self._metrics_lock = threading.Lock()

        if self.bus is not None and not self.dry_run:
            self._worker = threading.Thread(
                target=self._command_worker,
                name="piper-write-worker",
                daemon=True,
            )
            self._worker.start()

        mapper = getattr(self.session, "mapper", None)
        if mapper is not None:
            if hasattr(mapper, "fine_filter_alpha"):
                mapper.fine_filter_alpha = float(np.clip(self._fine_filter_alpha, 0.0, 1.0))
            if hasattr(mapper, "fine_deadband"):
                mapper.fine_deadband = float(max(self._fine_deadband, 0.0))
            if hasattr(mapper, "fine_scale"):
                if self._fine_scale is not None:
                    mapper.fine_scale = float(self._fine_scale)
                    if hasattr(mapper, "_fine_scale_custom"):
                        mapper._fine_scale_custom = True
                elif hasattr(mapper, "_fine_scale_custom") and not mapper._fine_scale_custom:
                    mapper.fine_scale = float(min(getattr(mapper, "scale", 1.0), 1.0))

    def _get_gripper_target(self, closed: bool) -> float:
        return self.gripper_closed if closed else self.gripper_open

    def _can_command(self) -> bool:
        return self.bus is not None and not self.dry_run

    def _queue_command(self, joints: List[float], gripper_value: float) -> bool:
        if self.bus is None or self.dry_run:
            return False
        command = list(joints) + [gripper_value]
        queued_ts = time.monotonic()
        with self._command_lock:
            self._pending_command = (command, queued_ts)
            self._command_event.set()
        return True

    def _command_worker(self) -> None:
        assert self.bus is not None
        logger = logging.getLogger(__name__)
        next_send = time.monotonic()

        while not self._stop_event.is_set():
            self._command_event.wait(0.1)
            if self._stop_event.is_set():
                break

            with self._command_lock:
                entry = self._pending_command
                if entry is None:
                    self._command_event.clear()
                    continue
                self._pending_command = None
                self._command_event.clear()

            command, queued_ts = entry

            if self.command_interval > 0.0:
                now = time.monotonic()
                delay = next_send - now
                if delay > 0.0:
                    if self._stop_event.wait(delay):
                        break

            start = time.perf_counter()
            queue_delay_ms = (time.monotonic() - queued_ts) * 1000.0
            try:
                self.bus.write(command)
                duration_ms = (time.perf_counter() - start) * 1000.0
                with self._metrics_lock:
                    self._last_write_duration_ms = duration_ms
                    self._last_queue_delay_ms = queue_delay_ms
                logger.debug(
                    "Piper 指令耗时 %.2f ms (排队 %.2f ms)",
                    duration_ms,
                    queue_delay_ms,
                )

                if self._effort_samples > 0:
                    effort = self.bus.read_gripper_effort(
                        samples=self._effort_samples,
                        interval=self._effort_interval,
                        mode=self._effort_mode,
                    )
                    with self._metrics_lock:
                        self._last_gripper_effort = float(effort)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("写入 Piper 指令失败: %s", exc)
            finally:
                now = time.monotonic()
                self._last_command_ts = now
                if self.command_interval > 0.0:
                    next_send = now + self.command_interval

        logger.debug("Piper 指令线程已停止")

    def _clear_pending_command(self) -> None:
        with self._command_lock:
            self._pending_command = None
            self._command_event.clear()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._command_event.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None

    def reset(self) -> None:  # type: ignore[override]
        super().reset()
        self._clear_pending_command()

    def process_message(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:  # type: ignore[override]
        results = self.session.handle_vr_payload(payload)
        summaries: List[Dict[str, Any]] = []
        for item in results:
            summary: Dict[str, Any] = {
                "hand": item.hand,
                "success": item.success,
                "info": item.info,
                "gripper_closed": item.gripper_closed,
            }

            if not item.success or item.joints is None:
                summary["commanded"] = False
                summaries.append(summary)
                continue

            joints = np.asarray(item.joints, dtype=float).reshape(-1)
            summary["target_joints_rad"] = joints.tolist()
            summary["target_joints_deg"] = [math.degrees(val) for val in joints]
            gripper_target = self._get_gripper_target(item.gripper_closed)
            summary["gripper_target"] = gripper_target

            if self._can_command():
                queued = self._queue_command(list(joints), gripper_target)
                summary["commanded"] = queued
                summary["queued"] = queued
            else:
                summary["commanded"] = False
                summary["queued"] = False

            with self._metrics_lock:
                if self._last_write_duration_ms is not None:
                    summary["last_write_ms"] = self._last_write_duration_ms
                if self._last_queue_delay_ms is not None:
                    summary["last_queue_delay_ms"] = self._last_queue_delay_ms
                if self._last_gripper_effort is not None:
                    summary["gripper_effort"] = self._last_gripper_effort

            summaries.append(summary)

        return summaries


def _sync_reference_with_robot(
    session: ArmTeleopSession,
    pipeline: PiperTeleopPipeline,
    joints_rad: np.ndarray,
) -> None:
    """使用真实机械臂当前关节角刷新 IK 种子与手柄参考位姿。"""

    ik = session.ik_solver
    model = ik.reduced_robot.model
    data = ik.reduced_robot.data

    q = np.asarray(joints_rad, dtype=float).reshape(model.nq)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    ee_id = model.getFrameId("ee")
    if ee_id < 0 or ee_id >= len(data.oMf):
        raise RuntimeError("无法找到末端 Frame 'ee'，请检查 URDF 配置")

    ee_pose = data.oMf[ee_id]
    ik.set_seed(q)
    ik.q_last = q.copy()

    for hand in pipeline.allowed_hands:
        session.set_reference_pose(hand, ee_pose.translation, ee_pose.rotation)
    pipeline.reference_translation = ee_pose.translation
    pipeline.reference_rotation = ee_pose.rotation
    pipeline._apply_reference()


def build_parser(parent: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    parser = build_meshcat_arg_parser(parent)
    parser.description = "VR 遥操作 Piper 实机（复用 Meshcat IK 配置）"
    parser.add_argument(
        "--piper-config",
        default=str(DEFAULT_PIPER_CONFIG),
        help="Piper 机械臂硬件配置 JSON",
    )
    parser.add_argument(
        "--command-interval",
        type=float,
        default=0.02,
        help="连续关节指令之间的最小时间间隔（秒），0 表示不限制",
    )
    parser.add_argument(
        "--gripper-open",
        type=float,
        default=0.04,
        help="遥操作张开时的夹爪目标值（米或 SDK 要求单位）",
    )
    parser.add_argument(
        "--gripper-closed",
        type=float,
        default=0.0,
        help="遥操作闭合时的夹爪目标值",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印关节目标，不触发真实写入/连接",
    )
    parser.add_argument(
        "--skip-home",
        action="store_true",
        help="启动时跳过 apply_calibration()，避免自动回到初始位姿",
    )
    parser.add_argument(
        "--effort-samples",
        type=int,
        default=0,
        help="发送指令后采样夹爪扭矩的次数，0 表示不读取",
    )
    parser.add_argument(
        "--effort-interval",
        type=float,
        default=0.02,
        help="夹爪扭矩连续采样间隔（秒）",
    )
    parser.add_argument(
        "--effort-mode",
        choices=["mean", "median", "max", "min", "last"],
        default="mean",
        help="夹爪扭矩采样聚合方式",
    )
    parser.add_argument(
        "--fine-scale",
        type=float,
        default=None,
        help="按住细调键时的缩放倍率（缺省为 min(scale, 1.0)）",
    )
    parser.add_argument(
        "--fine-filter-alpha",
        type=float,
        default=0.3,
        help="细调模式下一阶滤波系数，越小越平滑",
    )
    parser.add_argument(
        "--fine-deadband",
        type=float,
        default=0.002,
        help="细调模式下的死区阈值（米）",
    )
    return parser


async def run_live(args: argparse.Namespace, pipeline: PiperTeleopPipeline) -> None:
    stun_servers = [] if args.no_stun else list(args.stun)
    server = VRWebRTCServer(
        host=args.host,
        port=args.port,
        pipeline=pipeline,  # type: ignore[arg-type]
        channel_name=args.channel,
        stun_servers=stun_servers,
    )

    await server.start()
    logging.info("Piper 遥操作服务已启动，等待 VR 手柄接入……")
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await server.stop()


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_parent = _create_config_parser()
    config_parent.set_defaults(config=str(DEFAULT_PIPER_CONFIG))
    config_args, remaining = config_parent.parse_known_args(argv)

    config_path = pathlib.Path(config_args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (ROOT_DIR / config_path).resolve()

    config_data: Dict[str, Any] = {}
    if config_path.exists():
        config_data = _load_config_file(config_path)

    parser = build_parser(config_parent)
    if config_data:
        known = {action.dest for action in parser._actions}
        overrides: Dict[str, Any] = {}
        for key, value in config_data.items():
            if key in known:
                overrides[key] = value
        parser.set_defaults(**overrides)

    args = parser.parse_args(remaining)
    args.config = str(config_path)
    if args.piper_config == str(DEFAULT_PIPER_CONFIG) and args.config != str(DEFAULT_PIPER_CONFIG):
        args.piper_config = args.config

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)
    logger.info("使用配置文件: %s", args.config)

    session, meshcat_pipeline = build_meshcat_session(args)

    bus: Optional[PiperMotorsBus] = None
    if not args.dry_run:
        logger.info("初始化 Piper 机械臂，总线: %s", args.piper_config)
        bus = PiperMotorsBus(args.piper_config)
        if not bus.connect(True):
            raise SystemExit("Piper 机械臂使能失败，请检查硬件连接与急停状态")
        if not args.skip_home:
            logger.info("移动至初始姿态（apply_calibration）")
            bus.apply_calibration()

    pipeline = PiperTeleopPipeline(
        session=session,
        allowed_hands=meshcat_pipeline.allowed_hands,
        reference_translation=meshcat_pipeline.reference_translation,
        reference_rotation=meshcat_pipeline.reference_rotation,
        bus=bus,
        command_interval=args.command_interval,
        gripper_open=args.gripper_open,
        gripper_closed=args.gripper_closed,
        dry_run=args.dry_run,
        effort_samples=args.effort_samples,
        effort_interval=args.effort_interval,
        effort_mode=args.effort_mode,
        fine_scale=args.fine_scale,
        fine_filter_alpha=args.fine_filter_alpha,
        fine_deadband=args.fine_deadband,
    )

    if bus is not None and not args.dry_run:
        try:
            current = bus.read()
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
            logger.info("同步实机关节到 IK，起始角度(度)：%s", ", ".join(f"{math.degrees(v):.2f}" for v in joints_rad))
            _sync_reference_with_robot(session, pipeline, joints_rad)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("同步实机关节失败：%s", exc)

    try:
        asyncio.run(run_live(args, pipeline))
    finally:
        try:
            pipeline.shutdown()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("关闭指令线程失败: %s", exc)

        if bus is not None:
            try:
                logger.info("退出遥操作：回到初始姿态待命")
                bus.apply_calibration()
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("回初始姿态失败: %s", exc)


if __name__ == "__main__":
    main()
