"""VR 手柄 -> Piper 实机遥操作入口。

继承 Meshcat 版的配置/IK 管线，额外对接 `PiperMotorsBus`，
在真实机械臂上复用同款的运动学求解参数。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import pathlib
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

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


class JointCommandFilter:
    """二阶临界阻尼滤波 + 速度/加速度限幅。"""

    def __init__(
        self,
        dof: int,
        max_speed_deg: Sequence[float],
        max_acc_deg: Sequence[float],
        kp: float = 16,
        kd: float = 8,
        error_deadband_deg: float = 0.12,
    ) -> None:
        self.dof = dof
        self._kp = float(kp)
        self._kd = float(kd)
        self._max_speed = np.deg2rad(np.asarray(max_speed_deg, dtype=float).reshape(dof))
        self._max_acc = np.deg2rad(np.asarray(max_acc_deg, dtype=float).reshape(dof))
        self._error_deadband = math.radians(abs(error_deadband_deg)) if error_deadband_deg > 0 else 0.0
        self._pos = np.zeros(dof, dtype=float)
        self._vel = np.zeros(dof, dtype=float)
        self._initialized = False
        self._last_ts: Optional[float] = None

    def reset(self) -> None:
        self._pos[:] = 0.0
        self._vel[:] = 0.0
        self._initialized = False
        self._last_ts = None

    def prime(self, joints: Sequence[float]) -> None:
        self._pos = np.asarray(joints, dtype=float).reshape(self.dof).copy()
        self._vel[:] = 0.0
        self._initialized = True
        self._last_ts = time.monotonic()

    def step(self, joints: Sequence[float]) -> np.ndarray:
        target = np.asarray(joints, dtype=float).reshape(self.dof)
        now = time.monotonic()
        if not self._initialized:
            self.prime(target)
            return target.copy()

        dt = now - self._last_ts if self._last_ts is not None else 0.0
        dt = max(dt, 1e-3)

        error = target - self._pos
        if self._error_deadband > 0.0:
            mask = np.abs(error) <= self._error_deadband
            if np.any(mask):
                error = error.copy()
                error[mask] = 0.0

        acc_cmd = self._kp * error - self._kd * self._vel
        acc_clamped = np.clip(acc_cmd, -self._max_acc, self._max_acc)

        vel = self._vel + acc_clamped * dt
        vel = np.clip(vel, -self._max_speed, self._max_speed)

        pos = self._pos + vel * dt

        self._pos = pos
        self._vel = vel
        self._last_ts = now
        return pos.copy()


class TelemetryLogger:
    """记录 IK、命令与实测关节数据，用于离线分析。"""

    def __init__(
        self,
        file_path: Optional[str],
        sample_measured: bool,
        dof: int,
    ) -> None:
        self._enabled = bool(file_path)
        self._sample_measured = bool(sample_measured) and self._enabled
        self._dof = dof
        self._lock = threading.Lock()
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._seq = 0
        self._file: Optional[TextIO] = None
        if self._enabled and file_path:
            path = pathlib.Path(file_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def should_sample_measured(self) -> bool:
        return self._sample_measured

    def log_command(self, q_ik: np.ndarray, q_cmd: np.ndarray) -> Optional[int]:
        if not self._enabled:
            return None
        with self._lock:
            seq = self._seq
            self._seq += 1
            record = {
                "seq": seq,
                "ts_ik": time.time(),
                "q_ik": np.asarray(q_ik, dtype=float).reshape(self._dof).tolist(),
                "q_cmd": np.asarray(q_cmd, dtype=float).reshape(self._dof).tolist(),
            }
            self._pending[seq] = record
            return seq

    def mark_enqueued(self, seq: Optional[int], queued_monotonic: float) -> None:
        if not self._enabled or seq is None:
            return
        with self._lock:
            record = self._pending.get(seq)
            if record is not None:
                record["ts_enqueued_mono"] = queued_monotonic

    def discard(self, seq: Optional[int]) -> None:
        if not self._enabled or seq is None:
            return
        with self._lock:
            self._pending.pop(seq, None)

    def reset_pending(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._pending.clear()

    def log_sent(
        self,
        seq: Optional[int],
        dt_send: float,
        write_ms: float,
        queue_delay_ms: float,
        q_meas: Optional[np.ndarray],
    ) -> None:
        if not self._enabled or seq is None:
            return
        with self._lock:
            record = self._pending.pop(seq, None)
            if record is None:
                return
            record.update(
                {
                    "ts_sent": time.time(),
                    "dt_send": dt_send,
                    "write_ms": write_ms,
                    "queue_delay_ms": queue_delay_ms,
                }
            )
            if q_meas is not None:
                record["q_meas"] = np.asarray(q_meas, dtype=float).reshape(self._dof).tolist()
            self._write_locked(record)

    def close(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def _write_locked(self, record: Dict[str, Any]) -> None:
        if not self._enabled or self._file is None:
            return
        try:
            json.dump(record, self._file, ensure_ascii=False)
            self._file.write("\n")
            self._file.flush()
        except Exception as exc:  # pylint: disable=broad-except
            logging.getLogger(__name__).warning("Telemetry write failed: %s", exc)



def _coerce_limits(values: Sequence[float], dof: int, name: str) -> List[float]:
    try:
        result = [float(v) for v in values]
    except TypeError as exc:  # values 不是可迭代
        raise ValueError(f"{name} 需提供 {dof} 个浮点数") from exc
    if len(result) != dof:
        raise ValueError(f"{name} 需要提供 {dof} 个数值，实际 {len(result)}")
    return result


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
        joint_speed_limits_deg: Sequence[float],
        joint_acc_limits_deg: Sequence[float],
        joint_error_deadband_deg: float,
        telemetry_file: Optional[str],
        telemetry_sample_measured: bool,
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
        self._pending_command: Optional[Tuple[List[float], float, Optional[int]]] = None
        self._command_lock = threading.Lock()
        self._command_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_write_duration_ms: Optional[float] = None
        self._last_queue_delay_ms: Optional[float] = None
        self._last_gripper_effort: Optional[float] = None
        self._metrics_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._last_filtered_joints: Optional[np.ndarray] = None
        self._last_gripper_target: Optional[float] = None

        dof = getattr(getattr(session.ik_solver, "reduced_robot", None), "model", None)
        if dof is not None:
            dof_value = getattr(dof, "nq", None)
        else:
            dof_value = None
        if not isinstance(dof_value, int) or dof_value <= 0:
            dof_value = len(getattr(session.ik_solver, "q_seed", []))
        if dof_value <= 0:
            dof_value = 6

        speed_limits = _coerce_limits(joint_speed_limits_deg, dof_value, "joint_speed_limits_deg")
        accel_limits = _coerce_limits(joint_acc_limits_deg, dof_value, "joint_acc_limits_deg")
        self._joint_filter = JointCommandFilter(
            dof=dof_value,
            max_speed_deg=speed_limits,
            max_acc_deg=accel_limits,
            error_deadband_deg=joint_error_deadband_deg,
        )
        self._telemetry = TelemetryLogger(
            telemetry_file,
            telemetry_sample_measured,
            dof=dof_value,
        )

        if self.bus is not None and not self.dry_run:
            self._worker = threading.Thread(
                target=self._command_worker,
                name="piper-write-worker",
                daemon=True,
            )
            self._worker.start()

    def _get_gripper_target(self, closed: bool) -> float:
        return self.gripper_closed if closed else self.gripper_open

    def _can_command(self) -> bool:
        return self.bus is not None and not self.dry_run

    def _queue_command(
        self,
        joints: Sequence[float],
        gripper_value: float,
        telemetry_seq: Optional[int],
    ) -> bool:
        if self.bus is None or self.dry_run:
            return False
        command = list(joints) + [gripper_value]
        queued_ts = time.monotonic()
        with self._command_lock:
            self._pending_command = (command, queued_ts, telemetry_seq)
            self._command_event.set()
        self._telemetry.mark_enqueued(telemetry_seq, queued_ts)
        with self._state_lock:
            self._last_filtered_joints = np.asarray(joints, dtype=float).copy()
            self._last_gripper_target = float(gripper_value)
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

            command, queued_ts, telemetry_seq = entry

            if self.command_interval > 0.0:
                now = time.monotonic()
                delay = next_send - now
                if delay > 0.0:
                    if self._stop_event.wait(delay):
                        break

            start = time.perf_counter()
            queue_delay_ms = (time.monotonic() - queued_ts) * 1000.0
            duration_ms = 0.0
            q_meas = None
            try:
                self.bus.write(command)
                duration_ms = (time.perf_counter() - start) * 1000.0
                if self._telemetry.should_sample_measured:
                    q_meas = self._read_measured_joints()
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
                dt_send = 0.0 if self._last_command_ts == 0.0 else now - self._last_command_ts
                self._last_command_ts = now
                self._telemetry.log_sent(
                    telemetry_seq,
                    dt_send,
                    duration_ms,
                    queue_delay_ms,
                    q_meas,
                )
                if self.command_interval > 0.0:
                    next_send = now + self.command_interval

        logger.debug("Piper 指令线程已停止")

    def _clear_pending_command(self) -> None:
        with self._command_lock:
            entry = self._pending_command
            self._pending_command = None
            self._command_event.clear()
        if entry is not None:
            _, _, telemetry_seq = entry
            self._telemetry.discard(telemetry_seq)

    def _read_measured_joints(self) -> Optional[np.ndarray]:
        if self.bus is None:
            return None
        try:
            state = self.bus.read()
        except Exception as exc:  # pylint: disable=broad-except
            logging.getLogger(__name__).debug("读取实机关节失败: %s", exc)
            return None
        joints = [
            state.get("joint_1"),
            state.get("joint_2"),
            state.get("joint_3"),
            state.get("joint_4"),
            state.get("joint_5"),
            state.get("joint_6"),
        ]
        if any(val is None for val in joints):
            return None
        return np.asarray(joints, dtype=float).reshape(-1)

    def prime_joint_filter(self, joints: Sequence[float]) -> None:
        try:
            self._joint_filter.prime(joints)
            with self._state_lock:
                self._last_filtered_joints = (
                    np.asarray(joints, dtype=float).reshape(self._joint_filter.dof)
                )
                self._last_gripper_target = None
        except Exception:
            self._joint_filter.reset()
        self._telemetry.reset_pending()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._command_event.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None
        self._joint_filter.reset()
        with self._state_lock:
            self._last_filtered_joints = None
            self._last_gripper_target = None
        self._telemetry.reset_pending()
        self._telemetry.close()

    def reset(self) -> None:  # type: ignore[override]
        super().reset()
        self._clear_pending_command()
        self._joint_filter.reset()
        with self._state_lock:
            self._last_filtered_joints = None
            self._last_gripper_target = None
        self._telemetry.reset_pending()

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
            summary["ik_joints_rad"] = joints.tolist()
            summary["ik_joints_deg"] = [math.degrees(val) for val in joints]
            summary["target_joints_rad"] = summary["ik_joints_rad"]
            summary["target_joints_deg"] = summary["ik_joints_deg"]
            gripper_target = self._get_gripper_target(item.gripper_closed)
            summary["gripper_target"] = gripper_target

            filtered = self._joint_filter.step(joints)
            summary["command_joints_rad"] = filtered.tolist()
            summary["command_joints_deg"] = [math.degrees(val) for val in filtered]
            with self._state_lock:
                self._last_filtered_joints = filtered.copy()
                self._last_gripper_target = float(gripper_target)

            telemetry_seq = self._telemetry.log_command(joints, filtered)

            if self._can_command():
                queued = self._queue_command(filtered, gripper_target, telemetry_seq)
                summary["commanded"] = queued
                summary["queued"] = queued
                if not queued:
                    self._telemetry.discard(telemetry_seq)
            else:
                summary["commanded"] = False
                summary["queued"] = False
                self._telemetry.discard(telemetry_seq)

            with self._metrics_lock:
                if self._last_write_duration_ms is not None:
                    summary["last_write_ms"] = self._last_write_duration_ms
                if self._last_queue_delay_ms is not None:
                    summary["last_queue_delay_ms"] = self._last_queue_delay_ms
                if self._last_gripper_effort is not None:
                    summary["gripper_effort"] = self._last_gripper_effort

            summaries.append(summary)

        self._handle_grip_release_events()
        return summaries

    def _handle_grip_release_events(self) -> None:
        mapper = getattr(self.session, "mapper", None)
        if mapper is None or not hasattr(mapper, "consume_grip_release"):
            return
        for hand in getattr(self, "allowed_hands", []):
            try:
                released = mapper.consume_grip_release(hand)
            except Exception:  # pragma: no cover - defensive
                released = False
            if released:
                self._resync_after_grip_release(hand)

    def _resync_after_grip_release(self, hand: str) -> None:
        logger = logging.getLogger(__name__)
        joints: Optional[np.ndarray] = None
        with self._state_lock:
            if self._last_filtered_joints is not None:
                joints = self._last_filtered_joints.copy()
        if joints is None:
            joints = self._read_measured_joints()

        if joints is None:
            logger.warning("握持释放后无法获取当前关节，跳过参考同步")
            return

        try:
            _sync_reference_with_robot(self.session, self, joints)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("握持释放后同步参考失败: %s", exc)
            return

        self._clear_pending_command()
        logger.info("[%s] 握持释放 -> 重置增量基准", hand)

    def get_latest_command(self) -> tuple[Optional[np.ndarray], Optional[float]]:
        """返回最近一次滤波后的关节目标与夹爪值。"""

        with self._state_lock:
            joints = None if self._last_filtered_joints is None else self._last_filtered_joints.copy()
            gripper = self._last_gripper_target
        return joints, gripper


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
    pipeline.prime_joint_filter(joints_rad)


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
        "--joint-speed-limits-deg",
        type=float,
        nargs="+",
        default=[180.0, 195.0, 180.0, 225.0, 225.0, 225.0],
        help="关节最大速度限制(度/秒)，按 joint1..joint6 顺序提供",
    )
    parser.add_argument(
        "--joint-acc-limits-deg",
        type=float,
        nargs="+",
        default=[400.0, 400.0, 400.0, 450.0, 450.0, 450.0],
        help="关节最大加速度限制(度/秒^2)，按 joint1..joint6 顺序提供",
    )
    parser.add_argument(
        "--joint-error-deadband-deg",
        type=float,
        default=0.12,
        help="关节误差死区(度)，误差低于该值时视为零",
    )
    parser.add_argument(
        "--telemetry-file",
        default="",
        help="Telemetry JSONL 输出文件路径，留空表示关闭",
    )
    parser.add_argument(
        "--telemetry-sample-measured",
        action="store_true",
        help="启用后在发送指令后同步读取实机关节用于日志",
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
        joint_speed_limits_deg=args.joint_speed_limits_deg,
        joint_acc_limits_deg=args.joint_acc_limits_deg,
        joint_error_deadband_deg=args.joint_error_deadband_deg,
        telemetry_file=args.telemetry_file or None,
        telemetry_sample_measured=args.telemetry_sample_measured,
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
