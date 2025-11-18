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
from scripts.run_vr_meshcat import TeleopPipeline

__all__ = [
    "JointCommandFilter",
    "TelemetryLogger",
    "PiperTeleopPipeline",
    "extract_piper_hand_configs",
    "select_telemetry_file",
    "sync_reference_with_robot",
]


class JointCommandFilter:
    """二阶临界阻尼滤波 + 速度/加速度限幅，用于硬件侧的关节平滑。"""

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

        dt = now - self._last_ts if self._last_ts is not None else 0.0
        dt = max(dt, 1e-3)

        error = target - self._pos
        if self._error_deadband > 0.0:
            mask = np.abs(error) <= self._error_deadband
            if np.any(mask):
                error = error.copy()
                error[mask] = 0.0

        vel_target = None
        if velocity_hint is not None:
            try:
                vel_target = np.asarray(velocity_hint, dtype=float).reshape(self.dof)
            except Exception:
                vel_target = None
        if vel_target is None:
            vel_target = np.zeros(self.dof, dtype=float)

        acc_cmd = self._kp * error - self._kd * (self._vel - vel_target)
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
    ) -> None:
        """写入最终耗时与测量值，在成功发送一次指令后调用。"""

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
    effort_samples: int
    effort_interval: float
    effort_mode: str
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
    last_gripper_effort: Optional[float] = None
    velocity_history: Deque[np.ndarray] = field(default_factory=deque)
    last_ik_state: Optional[Tuple[float, np.ndarray]] = None
    pitch_mode_active: bool = False
    pitch_base_joints: Optional[np.ndarray] = None
    pitch_base_angle: float = 0.0

    def __post_init__(self) -> None:
        self.velocity_history = deque(maxlen=self.velocity_window or 1)

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
                    continue
                self.pending_command = None
                self.command_event.clear()

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
                with self.metrics_lock:
                    self.last_write_duration_ms = duration_ms
                    self.last_queue_delay_ms = queue_delay_ms
                logger.debug(
                    "[%s] Piper 指令耗时 %.2f ms (排队 %.2f ms)",
                    self.hand,
                    duration_ms,
                    queue_delay_ms,
                )

                if self.effort_samples > 0:
                    effort = self.bus.read_gripper_effort(
                        samples=self.effort_samples,
                        interval=self.effort_interval,
                        mode=self.effort_mode,
                    )
                    with self.metrics_lock:
                        self.last_gripper_effort = float(effort)
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
        self.last_gripper_effort = None
        self.last_queue_delay_ms = None
        self.last_write_duration_ms = None
        self.pitch_mode_active = False
        self.pitch_base_joints = None
        self.pitch_base_angle = 0.0


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
    """在 Meshcat 版基础上加入 Piper 机械臂命令下发，并支持多手柄。"""

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
        effort_samples: int,
        effort_interval: float,
        effort_mode: str,
        joint_speed_limits_deg: Sequence[float],
        joint_acc_limits_deg: Sequence[float],
        joint_error_deadband_deg: float,
        telemetry_file: Optional[str],
        telemetry_sample_measured: bool,
        velocity_filter_window: int,
        hand_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(session, allowed_hands, reference_translation, reference_rotation)
        self._stop_event = threading.Event()
        self._hand_overrides = hand_overrides or {}
        self._arms: Dict[str, _ArmState] = {}
        self._hand_reference: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._pitch_joint_index = 4  # joint5 (0-based)
        self._joint_lower: Optional[np.ndarray] = None
        self._joint_upper: Optional[np.ndarray] = None

        reduced = getattr(session.ik_solver, "reduced_robot", None)
        model = getattr(reduced, "model", None)
        if model is not None:
            self._joint_lower = np.asarray(model.lowerPositionLimit, dtype=float).copy()
            self._joint_upper = np.asarray(model.upperPositionLimit, dtype=float).copy()
            dof_value = int(getattr(model, "nq", 0))
        else:
            dof_value = 0
        if dof_value <= 0:
            dof_value = len(getattr(session.ik_solver, "q_seed", []))
        if dof_value <= 0:
            dof_value = 6
        if self._joint_lower is None or self._joint_upper is None:
            try:
                robot_model = session.ik_solver.reduced_robot.model
                self._joint_lower = np.asarray(robot_model.lowerPositionLimit, dtype=float).copy()
                self._joint_upper = np.asarray(robot_model.upperPositionLimit, dtype=float).copy()
            except Exception:
                self._joint_lower = np.full(dof_value, -np.inf, dtype=float)
                self._joint_upper = np.full(dof_value, np.inf, dtype=float)

        speed_limits = _coerce_limits(joint_speed_limits_deg, dof_value, "joint_speed_limits_deg")
        accel_limits = _coerce_limits(joint_acc_limits_deg, dof_value, "joint_acc_limits_deg")

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
                ),
                telemetry=TelemetryLogger(
                    telemetry_path,
                    telemetry_sample_measured,
                    dof=dof_value,
                ),
                velocity_window=state_velocity_window,
                stop_event=self._stop_event,
                effort_samples=int(overrides.get("effort_samples", effort_samples)),
                effort_interval=float(overrides.get("effort_interval", effort_interval)),
                effort_mode=str(overrides.get("effort_mode", effort_mode)).lower(),
            )
            state.start_worker()
            self._arms[hand] = state
            self._hand_reference[hand] = (
                self.reference_translation.copy(),
                self.reference_rotation.copy(),
            )

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
        for state in self._arms.values():
            state.shutdown()
            state.telemetry.close()

    def reset(self) -> None:  # type: ignore[override]
        super().reset()
        for state in self._arms.values():
            state.reset_state()
        self._apply_reference_overrides()

    def _apply_reference_overrides(self) -> None:
        for hand, pose in self._hand_reference.items():
            self.session.set_reference_pose(hand, pose[0], pose[1])

    def _set_hand_reference(self, hand: str, translation: np.ndarray, rotation: np.ndarray) -> None:
        pose_t = np.asarray(translation, dtype=float).reshape(3)
        pose_r = np.asarray(rotation, dtype=float).reshape(3, 3)
        self._hand_reference[hand] = (pose_t, pose_r)
        self.session.set_reference_pose(hand, pose_t, pose_r)

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
            state = self._get_state(item.hand)
            if state is None:
                summary["commanded"] = False
                summaries.append(summary)
                continue
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

            telemetry_seq = state.telemetry.log_command(joints, filtered)

            if state.can_command():
                queued = state.queue_command(filtered, gripper_target, telemetry_seq)
                summary["commanded"] = queued
                summary["queued"] = queued
                if not queued:
                    state.telemetry.discard(telemetry_seq)
            else:
                summary["commanded"] = False
                summary["queued"] = False
                state.telemetry.discard(telemetry_seq)

            with state.metrics_lock:
                if state.last_write_duration_ms is not None:
                    summary["last_write_ms"] = state.last_write_duration_ms
                if state.last_queue_delay_ms is not None:
                    summary["last_queue_delay_ms"] = state.last_queue_delay_ms
                if state.last_gripper_effort is not None:
                    summary["gripper_effort"] = state.last_gripper_effort

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
        for hand in getattr(self, "allowed_hands", []):
            state = self._get_state(hand)
            if state is None:
                continue
            try:
                released = mapper.consume_grip_release(hand)
            except Exception:  # pragma: no cover - defensive
                released = False
            if released:
                self._resync_after_grip_release(hand, state)

    def _resync_after_grip_release(self, hand: str, state: _ArmState) -> None:
        logger = logging.getLogger(__name__)
        joints: Optional[np.ndarray] = None
        with state.state_lock:
            if state.last_filtered_joints is not None:
                joints = state.last_filtered_joints.copy()
        if joints is None:
            joints = state.read_measured_joints()

        if joints is None:
            logger.warning("[%s] 握持释放后无法获取当前关节，跳过参考同步", hand)
            return

        try:
            sync_reference_with_robot(self.session, self, joints, hand)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[%s] 握持释放后同步参考失败: %s", hand, exc)
            return

        state.clear_pending_command()
        logger.info("[%s] 握持释放 -> 重置增量基准", hand)

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
        "effort_samples",
        "effort_interval",
        "effort_mode",
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

    targets = [hand] if hand is not None else list(pipeline.allowed_hands)
    if not targets:
        targets = list(pipeline.allowed_hands)
    for target in targets:
        pipeline._set_hand_reference(target, ee_pose.translation, ee_pose.rotation)
        pipeline.prime_joint_filter(target, joints_rad)
