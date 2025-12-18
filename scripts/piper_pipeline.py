"""Piper 遥操作管线相关的通用组件，拆分自 run_vr_piper.py。"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

import numpy as np
import pinocchio as pin

from robot.real.piper import PiperMotorsBus
from robot.teleop import ArmTeleopSession
from scripts.teleop_common import TeleopPipeline

__all__ = [
    "JointCommandFilter",
    "TelemetryLogger",
    "PiperTeleopPipeline",
    "extract_piper_hand_configs",
    "select_telemetry_file",
    "sync_reference_with_robot",
]


class JointCommandFilter:
    """关节指令二阶滤波 + 限速/限加/限位。

    使用临界阻尼的二阶系统平滑 IK 目标：
        acc = ω^2 * (target - pos) - 2ζω * vel
    然后对加速度/速度/位置施加限幅，必要时再裁剪信赖域。
    """

    def __init__(
        self,
        dof: int,
        max_speed_deg: Sequence[float],
        max_acc_deg: Sequence[float],
        error_deadband_deg: float = 0.12,
        trust_region_rad: Optional[Sequence[float]] = None,
        joint_lower: Optional[Sequence[float]] = None,
        joint_upper: Optional[Sequence[float]] = None,
        dt: float = 0.02,
        natural_freq: float = 8.0,
        damping: float = 1.0,
    ) -> None:
        self.dof = dof
        self._max_speed = np.deg2rad(np.asarray(max_speed_deg, dtype=float).reshape(dof))
        self._max_acc = np.deg2rad(np.asarray(max_acc_deg, dtype=float).reshape(dof))
        self._error_deadband = math.radians(abs(error_deadband_deg)) if error_deadband_deg > 0 else 0.0
        self._epsilon = 1e-6
        self._trust_region = (
            np.asarray(trust_region_rad, dtype=float).reshape(dof)
            if trust_region_rad is not None
            else None
        )
        self._joint_lower = (
            np.asarray(joint_lower, dtype=float).reshape(dof)
            if joint_lower is not None
            else np.full(dof, -np.inf, dtype=float)
        )
        self._joint_upper = (
            np.asarray(joint_upper, dtype=float).reshape(dof)
            if joint_upper is not None
            else np.full(dof, np.inf, dtype=float)
        )
        self._dt = float(dt)
        self._wn = max(1e-3, float(natural_freq))
        self._zeta = max(0.0, float(damping))
        self._pos = np.zeros(dof, dtype=float)
        self._vel = np.zeros(dof, dtype=float)
        self._initialized = False
        self._last_ts: Optional[float] = None

    def reset(self) -> None:
        """清空内部状态，在重新同步实机关节或关闭线程前调用。"""

        self._pos[:] = 0.0
        self._vel[:] = 0.0
        self._initialized = False
        self._last_ts = None

    def prime(self, joints: Sequence[float]) -> None:
        """以实机关节角初始化滤波器，避免第一帧产生突跳。"""

        self._pos = np.asarray(joints, dtype=float).reshape(self.dof).copy()
        self._vel[:] = 0.0
        self._initialized = True
        self._last_ts = time.monotonic()

    def step(self, joints: Sequence[float], velocity_hint: Optional[Sequence[float]] = None) -> np.ndarray:
        """根据目标和速度前馈输出平滑后的指令，自动应用速度/加速度限制。"""

        target = np.asarray(joints, dtype=float).reshape(self.dof)
        now = time.monotonic()
        if not self._initialized:
            self.prime(target)
            return target.copy()

        dt = now - self._last_ts if self._last_ts is not None else self._dt
        dt = max(dt, 1e-3)

        pos0 = self._pos
        vel0 = self._vel
        if self._error_deadband > 0.0:
            err0 = target - pos0
            mask = np.abs(err0) <= self._error_deadband
            if np.any(mask):
                target = target.copy()
                target[mask] = pos0[mask]

        # 二阶系统加速度（临界阻尼/可调阻尼）
        wn = self._wn
        zeta = self._zeta
        acc_cmd = (wn ** 2) * (target - pos0) - 2.0 * zeta * wn * vel0

        # 加速度限幅
        acc_cmd = np.clip(acc_cmd, -self._max_acc, self._max_acc)
        vel_new = vel0 + acc_cmd * dt
        # 速度限幅
        vel_new = np.clip(vel_new, -self._max_speed, self._max_speed)

        pos_new = pos0 + vel_new * dt

        # 信赖域裁剪（如果提供）
        if self._trust_region is not None:
            delta = pos_new - pos0
            delta = np.clip(delta, -self._trust_region, self._trust_region)
            pos_new = pos0 + delta

        # 位置限幅
        pos_new = np.minimum(np.maximum(pos_new, self._joint_lower + 1e-6), self._joint_upper - 1e-6)

        self._pos = pos_new
        self._vel = vel_new
        self._last_ts = now
        return pos_new.copy()


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

    def log_command(self, q_ik: np.ndarray, q_cmd: np.ndarray, extra: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """记录一次 IK 输出/滤波输出，返回序号供后续补充队列/实测信息。"""

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
            if extra:
                record.update(extra)
            self._pending[seq] = record
            return seq

    def mark_enqueued(self, seq: Optional[int], queued_monotonic: float) -> None:
        """在指令被发送线程接管时记录排队时间戳，便于分析延迟。"""

        if not self._enabled or seq is None:
            return
        with self._lock:
            record = self._pending.get(seq)
            if record is not None:
                record["ts_enqueued_mono"] = queued_monotonic

    def discard(self, seq: Optional[int]) -> None:
        """如果指令被取消（如 VR 断开）则丢弃队列中的临时记录。"""

        if not self._enabled or seq is None:
            return
        with self._lock:
            self._pending.pop(seq, None)

    def reset_pending(self) -> None:
        """清空尚未补全的记录，常在复位或 prime 之后调用。"""

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
        client_ts: Optional[float] = None,
        client_dt: Optional[float] = None,
    ) -> None:
        """写入最终耗时与测量值，在成功发送一次指令后调用。"""

        if not self._enabled:
            return
        with self._lock:
            record = None if seq is None else self._pending.pop(seq, None)
            if record is None:
                record = {"seq": seq if seq is not None else -1}
            record.update(
                {
                    "ts_sent": time.time(),
                    "dt_send": dt_send,
                    "write_ms": write_ms,
                    "queue_delay_ms": queue_delay_ms,
                }
            )
            if client_ts is not None:
                record["client_ts"] = client_ts
            if client_dt is not None:
                record["client_dt"] = client_dt
            if q_meas is not None:
                record["q_meas"] = np.asarray(q_meas, dtype=float).reshape(self._dof).tolist()
            self._write_locked(record)

    def log_recording_event(self, event: str, active: bool, grip_flags: Optional[Dict[str, bool]]) -> None:
        """记录握持事件（start/stop/active heartbeat），便于排查掉线/抖动。"""

        if not self._enabled:
            return
        record = {
            "type": "recording_event",
            "event": event,
            "active": bool(active),
            "ts": time.time(),
        }
        if grip_flags:
            record["grip_flags"] = {k: bool(v) for k, v in grip_flags.items()}
        self._write_locked(record)

    def close(self) -> None:
        """关闭日志文件句柄，确保缓存刷新到磁盘。"""

        if not self._enabled:
            return
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def _write_locked(self, record: Dict[str, Any]) -> None:
        """在持有互斥锁的情况下将 JSON 记录写入文件。"""

        if not self._enabled or self._file is None:
            return
        try:
            json.dump(record, self._file, ensure_ascii=False)
            self._file.write("\n")
            self._file.flush()
        except Exception as exc:  # pylint: disable=broad-except
            logging.getLogger(__name__).warning("Telemetry write failed: %s", exc)


@dataclass
class _ArmState:
    """管理单条手臂的滤波、写入线程与测量缓存。"""

    hand: str
    bus: Optional[PiperMotorsBus]
    command_interval: float
    gripper_open: float
    gripper_closed: float
    dry_run: bool
    joint_filter: JointCommandFilter
    telemetry: TelemetryLogger
    velocity_window: int
    stop_event: threading.Event
    command_lock: threading.Lock = field(default_factory=threading.Lock)
    command_event: threading.Event = field(default_factory=threading.Event)
    worker: Optional[threading.Thread] = None
    pending_command: Optional[Tuple[List[float], float, Optional[int]]] = None
    last_command_ts: float = 0.0
    metrics_lock: threading.Lock = field(default_factory=threading.Lock)
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    last_filtered_joints: Optional[np.ndarray] = None
    last_gripper_target: Optional[float] = None
    last_write_duration_ms: Optional[float] = None
    last_queue_delay_ms: Optional[float] = None
    velocity_history: Deque[np.ndarray] = field(default_factory=deque)
    last_ik_state: Optional[Tuple[float, np.ndarray]] = None
    last_client_ts: Optional[float] = None
    last_client_dt: Optional[float] = None
    pitch_mode_active: bool = False
    pitch_base_joints: Optional[np.ndarray] = None
    pitch_base_angle: float = 0.0
    grip_synced: bool = False
    init_gripper: float = 0.0
    init_joint_position: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))

    def __post_init__(self) -> None:
        self.velocity_history = deque(maxlen=self.velocity_window or 1)
        if self.bus is not None and hasattr(self.bus, "init_joint_position"):
            try:
                self.init_joint_position = np.asarray(self.bus.init_joint_position[:6], dtype=float)
            except Exception:
                self.init_joint_position = np.zeros(6, dtype=float)
        self.init_gripper = 0.0

    def start_worker(self) -> None:
        if self.bus is None or self.dry_run:
            return
        self.worker = threading.Thread(
            target=self._command_worker,
            name=f"piper-write-{self.hand}",
            daemon=True,
        )
        self.worker.start()

    def can_command(self) -> bool:
        return self.bus is not None and not self.dry_run

    def queue_command(
        self,
        joints: Sequence[float],
        gripper_value: float,
        telemetry_seq: Optional[int],
    ) -> bool:
        if not self.can_command():
            return False
        command = list(joints) + [gripper_value]
        queued_ts = time.monotonic()
        with self.command_lock:
            self.pending_command = (command, queued_ts, telemetry_seq)
            self.command_event.set()
        self.telemetry.mark_enqueued(telemetry_seq, queued_ts)
        with self.state_lock:
            self.last_filtered_joints = np.asarray(joints, dtype=float).copy()
            self.last_gripper_target = float(gripper_value)
        return True

    def _command_worker(self) -> None:
        assert self.bus is not None
        logger = logging.getLogger(__name__)
        next_send = time.monotonic()

        while not self.stop_event.is_set():
            self.command_event.wait(0.1)
            if self.stop_event.is_set():
                break

            with self.command_lock:
                entry = self.pending_command
                if entry is None:
                    self.command_event.clear()
                else:
                    self.pending_command = None
                    self.command_event.clear()

            # 合并等待期间的新命令，始终仅发送最新的一条
            while True:
                with self.command_lock:
                    newer = self.pending_command
                    if newer is not None:
                        entry = newer
                        self.pending_command = None
                        self.command_event.clear()
                if newer is None:
                    break

            if entry is None:
                continue

            command, queued_ts, telemetry_seq = entry

            if self.command_interval > 0.0:
                now = time.monotonic()
                delay = next_send - now
                if delay > 0.0:
                    if self.stop_event.wait(delay):
                        break

            start = time.perf_counter()
            queue_delay_ms = (time.monotonic() - queued_ts) * 1000.0
            duration_ms = 0.0
            q_meas = None
            try:
                self.bus.write(command)
                duration_ms = (time.perf_counter() - start) * 1000.0
                if self.telemetry.should_sample_measured:
                    q_meas = self.read_measured_joints()
                client_ts = self.last_client_ts
                client_dt = self.last_client_dt
                with self.metrics_lock:
                    self.last_write_duration_ms = duration_ms
                    self.last_queue_delay_ms = queue_delay_ms
                logger.debug(
                    "[%s] Piper 指令耗时 %.2f ms (排队 %.2f ms)",
                    self.hand,
                    duration_ms,
                    queue_delay_ms,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("[%s] 写入 Piper 指令失败: %s", self.hand, exc)
            finally:
                now = time.monotonic()
                dt_send = 0.0 if self.last_command_ts == 0.0 else now - self.last_command_ts
                self.last_command_ts = now
                self.telemetry.log_sent(
                    telemetry_seq,
                    dt_send,
                    duration_ms,
                    queue_delay_ms,
                    q_meas,
                    client_ts,
                    client_dt,
                )
                if self.command_interval > 0.0:
                    next_send = now + self.command_interval

        logger.debug("[%s] Piper 指令线程已停止", self.hand)

    def clear_pending_command(self) -> None:
        with self.command_lock:
            entry = self.pending_command
            self.pending_command = None
            self.command_event.clear()
        if entry is not None:
            _, _, telemetry_seq = entry
            self.telemetry.discard(telemetry_seq)

    def read_measured_joints(self) -> Optional[np.ndarray]:
        if self.bus is None:
            return None
        try:
            state = self.bus.read()
        except Exception as exc:  # pylint: disable=broad-except
            logging.getLogger(__name__).debug("[%s] 读取实机关节失败: %s", self.hand, exc)
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

    def reset_velocity_history(self) -> None:
        self.last_ik_state = None
        self.velocity_history.clear()

    def prime_joint_filter(self, joints: Sequence[float]) -> None:
        try:
            self.joint_filter.prime(joints)
            with self.state_lock:
                self.last_filtered_joints = (
                    np.asarray(joints, dtype=float).reshape(self.joint_filter.dof)
                )
                self.last_gripper_target = None
        except Exception:
            self.joint_filter.reset()
        self.telemetry.reset_pending()
        self.reset_velocity_history()

    def shutdown(self) -> None:
        self.command_event.set()
        if self.worker is not None:
            self.worker.join(timeout=1.0)
            self.worker = None
        self.joint_filter.reset()
        with self.state_lock:
            self.last_filtered_joints = None
            self.last_gripper_target = None
        self.telemetry.reset_pending()
        self.reset_velocity_history()

    def reset_state(self) -> None:
        self.clear_pending_command()
        self.joint_filter.reset()
        with self.state_lock:
            self.last_filtered_joints = None
            self.last_gripper_target = None
        self.telemetry.reset_pending()
        self.reset_velocity_history()
        self.last_queue_delay_ms = None
        self.last_write_duration_ms = None
        self.pitch_mode_active = False
        self.pitch_base_joints = None
        self.pitch_base_angle = 0.0
        self.grip_synced = False

    def wait_for_pose(
        self,
        target: np.ndarray,
        timeout: float = 10.0,
        tolerance_deg: float = 2.0,
    ) -> bool:
        """等待实机关节接近目标，返回是否在超时前到位。"""
        if self.bus is None:
            return True
        tol = math.radians(tolerance_deg)
        deadline = time.perf_counter() + timeout
        last = None
        while time.perf_counter() < deadline:
            current = self.read_measured_joints()
            if current is not None:
                last = current
                if np.max(np.abs(current - target)) <= tol:
                    return True
            time.sleep(0.05)
        if last is not None:
            max_err = math.degrees(float(np.max(np.abs(last - target))))
            logging.getLogger(__name__).warning(
                "[%s] 回位等待超时，最大偏差 %.2f 度", self.hand, max_err
            )
        return False


def _coerce_limits(values: Sequence[float], dof: int, name: str) -> List[float]:
    """确保速度/加速度限制包含 ``dof`` 个浮点数，并在报错信息中包含配置名。"""

    try:
        result = [float(v) for v in values]
    except TypeError as exc:  # values 不是可迭代
        raise ValueError(f"{name} 需提供 {dof} 个浮点数") from exc
    if len(result) != dof:
        raise ValueError(f"{name} 需要提供 {dof} 个数值，实际 {len(result)}")
    return result


class PiperTeleopPipeline(TeleopPipeline):
    """在通用 VR/IK 管线基础上加入 Piper 机械臂命令下发，并支持多手柄。"""

    def __init__(
        self,
        session,
        allowed_hands: Iterable[str],
        reference_translation: np.ndarray,
        reference_rotation: np.ndarray,
        *,
        buses: Dict[str, Optional[PiperMotorsBus]],
        command_interval: float,
        gripper_open: float,
        gripper_closed: float,
        dry_run: bool,
        joint_speed_limits_deg: Sequence[float],
        joint_acc_limits_deg: Sequence[float],
        joint_error_deadband_deg: float,
        telemetry_file: Optional[str],
        telemetry_sample_measured: bool,
        velocity_filter_window: int,
        hand_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        shm_status_writer: Optional[Any] = None,
    ) -> None:
        super().__init__(session, allowed_hands, reference_translation, reference_rotation)
        self._stop_event = threading.Event()
        self._hand_overrides = hand_overrides or {}
        self._arms: Dict[str, _ArmState] = {}
        self._hand_reference: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._pitch_joint_index = 4  # joint5 (0-based)
        self._joint_lower: Optional[np.ndarray] = None
        self._joint_upper: Optional[np.ndarray] = None
        # SharedMemory：发布握持状态给录制进程（bit0=right, bit1=left）
        self._shm_status_writer = shm_status_writer

        per_hand_refs = getattr(self, "per_hand_reference", {})
        for hand, pose in sorted(per_hand_refs.items()):
            try:
                self._hand_reference[hand] = (
                    np.asarray(pose["translation"], dtype=float).reshape(3),
                    np.asarray(pose["rotation"], dtype=float).reshape(3, 3),
                )
            except Exception:
                continue

        ik_for_limits = session.get_ik() if hasattr(session, "get_ik") else getattr(session, "ik_solver", None)
        reduced = getattr(ik_for_limits, "reduced_robot", None)
        model = getattr(reduced, "model", None)
        if model is not None:
            self._joint_lower = np.asarray(model.lowerPositionLimit, dtype=float).copy()
            self._joint_upper = np.asarray(model.upperPositionLimit, dtype=float).copy()
            dof_value = int(getattr(model, "nq", 0))
        else:
            dof_value = 0
        if dof_value <= 0:
            dof_value = len(getattr(ik_for_limits, "q_seed", []))
        if dof_value <= 0:
            dof_value = 6
        if self._joint_lower is None or self._joint_upper is None:
            try:
                robot_model = ik_for_limits.reduced_robot.model  # type: ignore[union-attr]
                self._joint_lower = np.asarray(robot_model.lowerPositionLimit, dtype=float).copy()
                self._joint_upper = np.asarray(robot_model.upperPositionLimit, dtype=float).copy()
            except Exception:
                self._joint_lower = np.full(dof_value, -np.inf, dtype=float)
                self._joint_upper = np.full(dof_value, np.inf, dtype=float)

        speed_limits = _coerce_limits(joint_speed_limits_deg, dof_value, "joint_speed_limits_deg")
        accel_limits = _coerce_limits(joint_acc_limits_deg, dof_value, "joint_acc_limits_deg")
        trust_region = getattr(ik_for_limits, "trust_region", None)
        command_dt = float(command_interval) if command_interval > 0 else 0.02

        hand_count = max(1, len(list(self.allowed_hands)))
        for hand in self.allowed_hands:
            overrides = dict(self._hand_overrides.get(hand, {}))
            state_velocity_window = max(0, int(overrides.get("velocity_filter_window", velocity_filter_window)))
            telemetry_path = select_telemetry_file(
                overrides.get("telemetry_file"),
                telemetry_file,
                hand,
                hand_count,
            )
            hand_dt = float(overrides.get("command_interval", command_interval))
            hand_dt = hand_dt if hand_dt > 0 else command_dt
            state = _ArmState(
                hand=hand,
                bus=buses.get(hand),
                command_interval=float(overrides.get("command_interval", command_interval)),
                gripper_open=float(overrides.get("gripper_open", gripper_open)),
                gripper_closed=float(overrides.get("gripper_closed", gripper_closed)),
                dry_run=bool(dry_run),
                joint_filter=JointCommandFilter(
                    dof=dof_value,
                    max_speed_deg=speed_limits,
                    max_acc_deg=accel_limits,
                    error_deadband_deg=joint_error_deadband_deg,
                    trust_region_rad=trust_region,
                    joint_lower=self._joint_lower,
                    joint_upper=self._joint_upper,
                    dt=hand_dt,
                    natural_freq=6.0,
                    damping=1.3,
                ),
                telemetry=TelemetryLogger(
                    telemetry_path,
                    telemetry_sample_measured,
                    dof=dof_value,
                ),
                velocity_window=state_velocity_window,
                stop_event=self._stop_event,
            )
            state.start_worker()
            self._arms[hand] = state
            self._hand_reference[hand] = (
                self.reference_translation.copy(),
                self.reference_rotation.copy(),
            )
        self._last_solve_mono: Dict[str, float] = {hand: 0.0 for hand in self.allowed_hands}

    def _get_state(self, hand: str) -> Optional[_ArmState]:
        return self._arms.get(hand)

    def _get_gripper_target(self, state: _ArmState, closed: bool) -> float:
        return state.gripper_closed if closed else state.gripper_open

    def _compute_velocity_hint(self, state: _ArmState, joints: np.ndarray) -> Optional[np.ndarray]:
        if state.velocity_window <= 0:
            state.last_ik_state = (time.monotonic(), joints.copy())
            return None
        now = time.monotonic()
        last_entry = state.last_ik_state
        velocity_hint = None
        if last_entry is not None:
            last_time, last_joints = last_entry
            dt = now - last_time
            if dt > 1e-3:
                try:
                    vel = (joints - last_joints) / dt
                except Exception:
                    vel = None
                if vel is not None:
                    state.velocity_history.append(vel)
                    if state.velocity_history:
                        stacked = np.stack(list(state.velocity_history), axis=0)
                        velocity_hint = np.mean(stacked, axis=0)
        state.last_ik_state = (now, joints.copy())
        return velocity_hint

    def prime_joint_filter(self, hand: str, joints: Sequence[float]) -> None:
        state = self._get_state(hand)
        if state is None:
            return
        state.prime_joint_filter(joints)

    def shutdown(self) -> None:
        self._stop_event.set()
        for hand in sorted(self._arms.keys()):
            state = self._arms[hand]
            state.shutdown()
            state.telemetry.close()
        self._publish_grip_mask(force_zero=True)

    def reset(self) -> None:  # type: ignore[override]
        super().reset()
        for state in self._arms.values():
            state.reset_state()
        for hand in list(self._last_solve_mono):
            self._last_solve_mono[hand] = 0.0
        self._apply_reference_overrides()
        # 连接断开时 pipeline.reset 会被调用：此时应清零握持状态，避免录制端误触发。
        self._publish_grip_mask(force_zero=True)

    def _apply_reference_overrides(self) -> None:
        for hand, pose in self._hand_reference.items():
            self.session.set_reference_pose(hand, pose[0], pose[1])

    def _set_hand_reference(self, hand: str, translation: np.ndarray, rotation: np.ndarray) -> None:
        pose_t = np.asarray(translation, dtype=float).reshape(3)
        pose_r = np.asarray(rotation, dtype=float).reshape(3, 3)
        self._hand_reference[hand] = (pose_t, pose_r)
        self.session.set_reference_pose(hand, pose_t, pose_r)

    def _publish_grip_mask(self, *, force_zero: bool = False) -> None:
        """将当前握持状态发布到 SharedMemory（录制进程只读）。

        约定：bit0=right, bit1=left。
        """

        writer = self._shm_status_writer
        if writer is None:
            return

        mask = 0
        if not force_zero:
            mapper = getattr(self.session, "mapper", None)
            controllers = getattr(mapper, "controllers", None)
            if controllers:
                if "right" in self.allowed_hands:
                    ctrl = controllers.get("right")
                    if ctrl is not None and bool(getattr(ctrl, "grip_active", False)):
                        mask |= 0b01
                if "left" in self.allowed_hands:
                    ctrl = controllers.get("left")
                    if ctrl is not None and bool(getattr(ctrl, "grip_active", False)):
                        mask |= 0b10
        try:
            writer.update(mask)
        except Exception:
            # shm 发布失败不应影响实时控制
            pass

    def process_message(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:  # type: ignore[override]
        goals = self.session.process_vr_payload(payload)
        # 注意：握持状态在 process_vr_payload 内更新，即便本帧没有 goal，也需要发布给录制端。
        self._publish_grip_mask()
        summaries: List[Dict[str, Any]] = []
        for goal in goals:
            now_mono = time.monotonic()
            summary: Dict[str, Any] = {
                "hand": goal.hand,
                "success": False,
                "info": "pending",
                "gripper_closed": bool(goal.gripper_closed),
            }
            state = self._get_state(goal.hand)
            if state is None:
                summary["commanded"] = False
                summaries.append(summary)
                continue

            client_ts = payload.get("client_ts")
            client_dt = payload.get("client_dt")
            if client_ts is not None:
                try:
                    state.last_client_ts = float(client_ts) / 1000.0  # Date.now() ms -> s
                except Exception:
                    state.last_client_ts = None
            if client_dt is not None:
                try:
                    state.last_client_dt = float(client_dt) / 1000.0
                except Exception:
                    state.last_client_dt = None
            # 一并写入遥测，便于链路频率定位
            if client_ts is not None or client_dt is not None:
                try:
                    state.telemetry.log_sent(
                        seq=None,
                        dt_send=0.0,
                        write_ms=0.0,
                        queue_delay_ms=0.0,
                        q_meas=None,
                        client_ts=state.last_client_ts,
                        client_dt=state.last_client_dt,
                    )
                except Exception:
                    pass

            # 握持开始时同步参考/prime（仅在未同步的本次握持内触发），首帧跳过下发
            if not state.grip_synced:
                mapper = getattr(self.session, "mapper", None)
                if mapper is not None and hasattr(mapper, "reset_hand_state"):
                    try:
                        mapper.reset_hand_state(goal.hand)
                    except Exception:
                        pass
                measured = state.init_joint_position.copy()
                try:
                    self.prime_joint_filter(goal.hand, measured)
                    sync_reference_with_robot(self.session, self, measured, goal.hand)
                    with state.state_lock:
                        state.grip_synced = True
                    summary["commanded"] = False
                    summary["info"] = "synced_reference"
                    summary["success"] = True
                    summaries.append(summary)
                    continue
                except Exception as exc:  # pragma: no cover - 防御
                    logging.getLogger(__name__).warning("[%s] 握持开始同步参考失败: %s", goal.hand, exc)
                    summary["commanded"] = False
                    summary["info"] = "sync_failed"
                    summaries.append(summary)
                    continue

            interval = float(state.command_interval) if state.command_interval > 0 else 0.0
            last_solve = self._last_solve_mono.get(goal.hand, 0.0)
            if interval > 0.0 and last_solve > 0.0 and (now_mono - last_solve) < interval:
                summary["commanded"] = False
                summary["success"] = True
                summary["info"] = "throttled"
                summaries.append(summary)
                continue

            self._last_solve_mono[goal.hand] = now_mono
            item = self.session.solve_goal(goal)

            summary["success"] = item.success
            summary["info"] = item.info
            summary["gripper_closed"] = item.gripper_closed

            if item.pitch_mode:
                summary["pitch_mode"] = True
                joints = self._build_pitch_joints(state, item.pitch_angle)
                if joints is None:
                    summary["commanded"] = False
                    summaries.append(summary)
                    continue
            else:
                state.pitch_mode_active = False
                state.pitch_base_joints = None
                state.pitch_base_angle = 0.0

                if not item.success or item.joints is None:
                    summary["commanded"] = False
                    summaries.append(summary)
                    continue

                joints = np.asarray(item.joints, dtype=float).reshape(-1)

            summary["ik_joints_rad"] = joints.tolist()
            summary["ik_joints_deg"] = [math.degrees(val) for val in joints]
            summary["target_joints_rad"] = summary["ik_joints_rad"]
            summary["target_joints_deg"] = summary["ik_joints_deg"]
            gripper_target = self._get_gripper_target(state, item.gripper_closed)
            summary["gripper_target"] = gripper_target

            velocity_hint = self._compute_velocity_hint(state, joints)
            filtered = state.joint_filter.step(joints, velocity_hint)
            summary["command_joints_rad"] = filtered.tolist()
            summary["command_joints_deg"] = [math.degrees(val) for val in filtered]
            with state.state_lock:
                state.last_filtered_joints = filtered.copy()
                state.last_gripper_target = float(gripper_target)

            telemetry_seq = state.telemetry.log_command(
                joints,
                filtered,
                extra={
                    "client_ts": state.last_client_ts,
                    "client_dt": state.last_client_dt,
                },
            )

            if state.can_command():
                queued = state.queue_command(filtered, gripper_target, telemetry_seq)
                summary["commanded"] = queued
                summary["queued"] = queued
                if not queued:
                    state.telemetry.discard(telemetry_seq)
            else:
                summary["commanded"] = False
                summary["queued"] = False
                # dry-run 或无硬件时也写入遥测，便于离线分析
                state.telemetry.log_sent(
                    telemetry_seq,
                    dt_send=0.0,
                    write_ms=0.0,
                    queue_delay_ms=0.0,
                    q_meas=None,
                    client_ts=state.last_client_ts,
                    client_dt=state.last_client_dt,
                )

            with state.metrics_lock:
                if state.last_write_duration_ms is not None:
                    summary["last_write_ms"] = state.last_write_duration_ms
                if state.last_queue_delay_ms is not None:
                    summary["last_queue_delay_ms"] = state.last_queue_delay_ms

            summaries.append(summary)

        self._handle_grip_release_events()
        return summaries

    def _build_pitch_joints(self, state: _ArmState, pitch_angle: Optional[float]) -> Optional[np.ndarray]:
        if pitch_angle is None:
            pitch_angle = 0.0
        if not state.pitch_mode_active:
            base = self._get_pitch_base(state)
            if base is None:
                return None
            state.pitch_base_joints = base.copy()
            state.pitch_base_angle = base[self._pitch_joint_index]
            state.pitch_mode_active = True

        if state.pitch_base_joints is None:
            return None

        lower = -np.inf
        upper = np.inf
        if self._joint_lower is not None and self._joint_upper is not None:
            lower = float(self._joint_lower[self._pitch_joint_index])
            upper = float(self._joint_upper[self._pitch_joint_index])

        target = state.pitch_base_joints.copy()
        target_angle = np.clip(state.pitch_base_angle + pitch_angle, lower, upper)
        target[self._pitch_joint_index] = target_angle
        return target

    def _get_pitch_base(self, state: _ArmState) -> Optional[np.ndarray]:
        with state.state_lock:
            if state.last_filtered_joints is not None:
                return state.last_filtered_joints.copy()
        return state.read_measured_joints()

    def _handle_grip_release_events(self) -> None:
        mapper = getattr(self.session, "mapper", None)
        if mapper is None or not hasattr(mapper, "consume_grip_release"):
            return
        for hand in sorted(getattr(self, "allowed_hands", [])):  # 固定顺序，先右后左
            state = self._get_state(hand)
            if state is None:
                continue
            try:
                released = mapper.consume_grip_release(hand)
            except Exception:  # pragma: no cover - defensive
                released = False
            if released:
                # 清手柄缓存，避免用旧的 controller frame
                if mapper is not None and hasattr(mapper, "reset_hand_state"):
                    try:
                        mapper.reset_hand_state(hand)
                    except Exception:
                        pass
                self._resync_after_grip_release(hand, state)
                self._return_home_after_release(hand, state)

    def _resync_after_grip_release(self, hand: str, state: _ArmState) -> None:
        logger = logging.getLogger(__name__)
        # 释放后同步参考统一用初始姿态，避免用上一次过滤结果造成下一次握持跳变
        joints = state.init_joint_position.copy()
        try:
            sync_reference_with_robot(self.session, self, joints, hand)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[%s] 握持释放后同步参考失败: %s", hand, exc)
            return

        mapper = getattr(self.session, "mapper", None)
        if mapper is not None and hasattr(mapper, "reset_hand_state"):
            try:
                mapper.reset_hand_state(hand)
            except Exception as exc:  # pragma: no cover - 防御
                logger.debug("[%s] 重置手柄状态失败: %s", hand, exc)

        # 释放后清空 IK/滤波/命令缓存，保持 init 基准，并标记为需首帧同步
        state.clear_pending_command()
        state.joint_filter.reset()
        state.reset_velocity_history()
        with state.state_lock:
            state.last_filtered_joints = None
            state.last_gripper_target = None
            state.last_ik_state = None
            state.pitch_mode_active = False
            state.pitch_base_joints = None
            state.pitch_base_angle = 0.0
            state.grip_synced = False
        logger.info("[%s] 握持释放 -> 重置增量基准", hand)

    def _return_home_after_release(self, hand: str, state: _ArmState) -> None:
        """握持松开后，以限速限加插补回初始姿态。"""
        logger = logging.getLogger(__name__)
        with state.state_lock:
            target = np.asarray(state.init_joint_position[:6], dtype=float)
            gripper_target = float(state.init_gripper)
        try:
            queued = state.queue_command(target, gripper_target, telemetry_seq=-1)
            if queued:
                # 不等待到位，避免阻塞信令/VR 会话；由后台下发线程异步执行
                logger.info("[%s] 松开后回初始姿态（异步）", hand)
            else:
                logger.warning("[%s] 松开后回位未入队", hand)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[%s] 松开后回初始姿态失败: %s", hand, exc)

        # 额外短暂保持：以 20ms 间隔重复初始位 0.5s，防止 MIT 模式掉落
        def _pulse_hold() -> None:
            if state.bus is None or not state.can_command():
                return
            hold_duration = 0.5
            interval = 0.02
            end_time = time.monotonic() + hold_duration
            while time.monotonic() < end_time and not state.stop_event.is_set():
                ok = state.queue_command(target, gripper_target, telemetry_seq=-1)
                if not ok:
                    break
                if state.stop_event.wait(interval):
                    break

        threading.Thread(target=_pulse_hold, daemon=True).start()

    def get_latest_command(self, hand: Optional[str] = None) -> tuple[Optional[np.ndarray], Optional[float]]:
        target_hand = hand
        if target_hand is None:
            if len(self._arms) == 1:
                target_hand = next(iter(self._arms))
            else:
                return None, None
        state = self._get_state(target_hand)
        if state is None:
            return None, None
        with state.state_lock:
            joints = None if state.last_filtered_joints is None else state.last_filtered_joints.copy()
            gripper = state.last_gripper_target
        return joints, gripper


def select_telemetry_file(
    override_path: Optional[str],
    base_path: Optional[str],
    hand: str,
    hand_count: int,
) -> Optional[str]:
    """根据全局/手柄覆盖生成遥测路径，用于区分多臂日志。"""

    if override_path:
        return str(pathlib.Path(override_path).expanduser())
    if not base_path:
        return None
    path = pathlib.Path(base_path).expanduser()
    if hand_count <= 1:
        return str(path)
    stem = path.stem or path.name
    suffix = path.suffix
    if suffix:
        return str(path.with_name(f"{stem}_{hand}{suffix}"))
    return str(path.with_name(f"{path.name}_{hand}"))


def extract_piper_hand_configs(config: Dict[str, Any]) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """解析 JSON 中的 piper_arms 字段，拆分硬件参数与管线覆盖。"""

    arms = config.get("piper_arms", {})
    if not isinstance(arms, dict):
        return {}, {}

    hw_overrides: Dict[str, Dict[str, Any]] = {}
    pipeline_overrides: Dict[str, Dict[str, Any]] = {}
    hw_keys = {"can_name", "init_joint_position", "safe_disable_position", "joint_factor"}
    pipeline_keys = {
        "gripper_open",
        "gripper_closed",
        "command_interval",
        "velocity_filter_window",
        "telemetry_file",
    }

    for hand, payload in arms.items():
        if hand not in {"left", "right"}:
            continue
        if not isinstance(payload, dict):
            continue
        hw_entry = {key: payload[key] for key in hw_keys if key in payload}
        if hw_entry:
            hw_overrides[hand] = hw_entry
        pipe_entry = {key: payload[key] for key in pipeline_keys if key in payload}
        if pipe_entry:
            pipeline_overrides[hand] = pipe_entry

    return hw_overrides, pipeline_overrides


def sync_reference_with_robot(
    session: ArmTeleopSession,
    pipeline: PiperTeleopPipeline,
    joints_rad: np.ndarray,
    hand: Optional[str] = None,
) -> None:
    """使用真实机械臂当前关节角刷新 IK 种子与手柄参考位姿。"""

    target_hands = [hand] if hand is not None else list(pipeline.allowed_hands)
    if not target_hands:
        target_hands = list(pipeline.allowed_hands)

    for target in target_hands:
        ik = session.get_ik(target) if hasattr(session, "get_ik") else session.ik_solver
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

        pipeline._set_hand_reference(target, ee_pose.translation, ee_pose.rotation)
        pipeline.prime_joint_filter(target, joints_rad)
