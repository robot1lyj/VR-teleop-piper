"""Piper 实机 CAN 控制封装，负责桥接 JSON 配置与 Piper SDK 调用。

与录制解耦相关的约束：
- Piper SDK 不是线程安全的：所有 SDK 调用必须串行化（否则容易出现长尾阻塞/抖动）。
- “实际下发时刻”的时间戳只能在硬件写线程里拿到，因此 cmd/meas 的 SharedMemory 发布也应在此处完成。
- 录制进程只读 SharedMemory + 相机，不再触碰 Piper SDK，避免录制写盘/相机线程抢占控制线程导致卡顿。
"""

import json
import logging
import math
import time
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple


from piper_sdk import *


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "piper_teleop.json"


def _clamp(value: float, lower: float, upper: float) -> float:
    return lower if value < lower else upper if value > upper else value


def _resolve_config_path(config_path: str | Path | None) -> Path:
    """将传入路径解析为仓库根目录下的绝对路径。"""

    if config_path is None:
        return DEFAULT_CONFIG_PATH

    resolved = Path(config_path)
    if not resolved.is_absolute():
        resolved = (ROOT_DIR / resolved).resolve()
    return resolved


def _ensure_float_list(values: Any, key: str) -> List[float]:
    """校验并转换关节角列表为浮点数列表。"""

    if values is None:
        raise ValueError(f"配置项 '{key}' 缺失或为空")

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"配置项 '{key}' 必须是长度为 7 的序列")

    items = list(values)
    if len(items) != 7:
        raise ValueError(f"配置项 '{key}' 需要提供 7 个数值 (6 轴 + 夹爪)")

    try:
        return [float(item) for item in items]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"配置项 '{key}' 中存在无法转换的数值") from exc


def _degrees_to_internal(values: List[float]) -> List[float]:
    """将配置中的关节角（度）转换为内部使用的弧度，夹爪值保持原样。"""

    converted = [math.radians(val) for val in values[:6]]
    if len(values) >= 7:
        converted.append(values[6])
    return converted


def _load_piper_config(
    config_path: str | Path | None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """读取 Piper 配置文件，补全缺省值并校验关键字段。"""

    defaults: Dict[str, Any] = {
        "can_name": "can0",
        "init_joint_position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "safe_disable_position": None,
        "joint_factor": 57296.0,
    }

    path = _resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到 Piper 配置文件: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"解析 Piper 配置文件失败: {path}\n{exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(f"Piper 配置文件必须是 JSON 对象: {path}")

    config: Dict[str, Any] = defaults | loaded
    if overrides:
        for key, value in overrides.items():
            config[key] = value

    can_name = config.get("can_name")
    if not isinstance(can_name, str) or not can_name.strip():
        raise ValueError("配置项 'can_name' 必须为非空字符串")
    config["can_name"] = can_name.strip()

    init_joint_deg = _ensure_float_list(config["init_joint_position"], "init_joint_position")
    config["init_joint_position"] = _degrees_to_internal(init_joint_deg)

    safe_disable = config.get("safe_disable_position")
    if safe_disable is None:
        safe_disable_deg = list(init_joint_deg)
    else:
        safe_disable_deg = _ensure_float_list(safe_disable, "safe_disable_position")
    config["safe_disable_position"] = _degrees_to_internal(safe_disable_deg)

    try:
        config["joint_factor"] = float(config["joint_factor"])
    except (TypeError, ValueError) as exc:
        raise ValueError("配置项 'joint_factor' 无法转换为浮点数") from exc

    config["config_path"] = str(path)
    return config


class PiperMotorsBus:
    """对 Piper SDK 的二次封装，参数可通过 JSON 配置覆盖。"""

    def __init__(
        self,
        config_path: str | Path | None = None,
        overrides: Dict[str, Any] | None = None,
    ):
        config = _load_piper_config(config_path, overrides)

        self.config = config
        self._logger = logging.getLogger(__name__)
        self.config_path = config["config_path"]
        self.can_name = config["can_name"]
        self.init_joint_position = config["init_joint_position"]
        self.safe_disable_position = config["safe_disable_position"]
        self.joint_factor = config["joint_factor"]  # 用于兼容旧配置的缩放因子（默认 rad -> mdeg）
        self.rad_to_mdeg = float(config.get("joint_factor", 180.0 / math.pi * 1000.0))
        self.mdeg_to_rad = math.pi / 180.0 / 1000.0
        # MIT 控制参数使用官方默认值，不对外暴露
        self._mit_kp = 10.0
        self._mit_kd = 0.8
        self._mit_torque_ref = 0.0

        # Piper SDK 并非线程安全：所有 SDK 调用必须串行化（写线程为主）。
        self._sdk_lock = threading.RLock()

        self.piper = C_PiperInterface_V2(self.can_name)
        self.piper.ConnectPort()

        self._effort_history: Deque[float] = deque(maxlen=50)
        self._joint_mode_configured = False
        self._last_cmd_rad: List[float] | None = None
        self._last_cmd_ts = 0.0

        # 异步发送：只保留最新命令，独立线程写硬件，避免主线程阻塞。
        self._send_lock = threading.Lock()
        self._send_event = threading.Event()
        # (joints_rad[6], vels[6], gripper_range_int, gripper_value_float)
        self._pending_cmd: Tuple[List[float], List[float], int, float] | None = None
        self._send_stop = threading.Event()

        # SharedMemory 发布（可选）：用于“控制进程 -> 录制进程”严格对齐。
        # 只发布小数据（q7 + t_ns），不发布图像。
        self._shm_cmd_writer: Any | None = None
        self._shm_meas_writer: Any | None = None
        self._shm_meas_period_ns = 0
        self._shm_last_meas_ns = 0
        self._last_measured_q7: Optional[List[float]] = None

        self._send_thread = threading.Thread(target=self._send_worker, daemon=True)
        self._send_thread.start()

    def enable_shm_publish(self, *, cmd_writer: Any = None, meas_writer: Any = None, meas_hz: float = 0.0) -> None:
        """启用 SharedMemory 发布（仅控制进程调用）。

        设计目标：录制进程不再 import/调用 Piper SDK，只通过 SharedMemory 读取 q_cmd/q_meas，
        并以 ``time.monotonic_ns()`` 对齐到相机采集时间戳，避免录制写盘导致控制抖动。
        """

        self._shm_cmd_writer = cmd_writer
        self._shm_meas_writer = meas_writer
        self._shm_last_meas_ns = 0
        if meas_hz and meas_hz > 0:
            self._shm_meas_period_ns = max(1, int(1e9 / float(meas_hz)))
        else:
            self._shm_meas_period_ns = 0

    def _read_q7_locked(self) -> List[float]:
        """读取 q7（6 轴弧度 + 夹爪），要求外部已持有 ``_sdk_lock``。"""

        joint_msg = self.piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state

        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state

        return [
            joint_state.joint_1 * self.mdeg_to_rad,
            joint_state.joint_2 * self.mdeg_to_rad,
            joint_state.joint_3 * self.mdeg_to_rad,
            joint_state.joint_4 * self.mdeg_to_rad,
            joint_state.joint_5 * self.mdeg_to_rad,
            joint_state.joint_6 * self.mdeg_to_rad,
            gripper_state.grippers_angle / 1000,
        ]




    def _read_enable_flags(self) -> List[bool]:
        """读取 6 个关节驱动的上电状态，供上电流程轮询。"""

        info = self.piper.GetArmLowSpdInfoMsgs()
        return [
            bool(info.motor_1.foc_status.driver_enable_status),
            bool(info.motor_2.foc_status.driver_enable_status),
            bool(info.motor_3.foc_status.driver_enable_status),
            bool(info.motor_4.foc_status.driver_enable_status),
            bool(info.motor_5.foc_status.driver_enable_status),
            bool(info.motor_6.foc_status.driver_enable_status),
        ]

    def connect(self, enable: bool = True) -> bool:
        """上电或断电 Piper。

        Args:
            enable: ``True`` 表示依次上电六个关节并启动夹爪，``False`` 则执行软断电。

        Returns:
            bool: 在超时前完成目标状态返回 True，否则 False。
        """

        timeout = 5.0
        interval = 0.2
        deadline = time.time() + timeout

        while True:
            if enable:
                initial_flags = self._read_enable_flags()
                if all(initial_flags):
                    print("所有关节已上电，跳过重复使能")
                    return True

                if not self.piper.EnablePiper():
                    time.sleep(0.01)
                    continue
                self.piper.EnableArm(7)
            else:
                self.piper.DisableArm(7)

            flags = self._read_enable_flags()
            enabled_count = sum(flags)
            print("--------------------")
            print(f"当前上电关节数量: {enabled_count}/6")
            print("--------------------")

            if enable and all(flags):
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
                self._joint_mode_configured = False
                return True
            if not enable and not any(flags):
                self.piper.GripperCtrl(0, 1000, 0x02, 0)
                self._joint_mode_configured = False
                return True

            if time.time() >= deadline:
                print("使能超时，停止尝试")
                return False

            time.sleep(interval)



    def set_calibration(self):
        """保留的校准接口，预留给未来流程。"""

        return

    def revert_calibration(self):
        """恢复校准的占位函数，当前硬件版本无需实现。"""

        return

    def apply_calibration(self):
        """将机械臂移动到配置中指定的初始关节角，常用于回零或安全待命。"""

        self.write(target_joint=self.init_joint_position)

    def write(self, target_joint: List[float]) -> None:
        """写入 6 轴+夹爪目标（弧度/米），下发 MIT 控制。"""

        if len(target_joint) != 7:
            raise ValueError("target_joint 长度必须为 7 (6 轴 + 夹爪)")

        now = time.monotonic()
        joints_rad = [_clamp(float(val), -12.5, 12.5) for val in target_joint[:6]]
        velocities = [0.0] * 6
        if self._last_cmd_rad is not None:
            dt = now - self._last_cmd_ts
            if dt > 1e-4:
                velocities = [
                    _clamp((pos - prev) / dt, -45.0, 45.0)
                    for pos, prev in zip(joints_rad, self._last_cmd_rad)
                ]
        self._last_cmd_rad = joints_rad
        self._last_cmd_ts = now

        gripper_value = float(target_joint[6])
        gripper_range = round(target_joint[6] * 1000.0 * 100.0)
        if gripper_range < 0:
            gripper_range = 0

        with self._send_lock:
            self._pending_cmd = (joints_rad, velocities, gripper_range, gripper_value)
            self._send_event.set()

    def _send_worker(self) -> None:
        while not self._send_stop.is_set():
            self._send_event.wait(0.05)
            if self._send_stop.is_set():
                break
            with self._send_lock:
                entry = self._pending_cmd
                self._pending_cmd = None
                self._send_event.clear()
            if entry is None:
                continue
            joints_rad, joint_vels, gripper_range, gripper_value = entry
            try:
                # 以“进入写线程准备下发”的时刻作为 cmd 时间戳，便于与相机的 monotonic_ns 对齐。
                t_cmd_ns = time.monotonic_ns()
                start = time.perf_counter()
                with self._sdk_lock:
                    if not self._joint_mode_configured:
                        # MIT 模式：CAN 指令、MOVE M、MIT 标志开启
                        self.piper.MotionCtrl_2(0x01, 0x04, 0, 0xAD)
                        self._joint_mode_configured = True
                    for idx, (pos, vel) in enumerate(zip(joints_rad, joint_vels), start=1):
                        self.piper.JointMitCtrl(
                            idx, pos, vel, self._mit_kp, self._mit_kd, self._mit_torque_ref
                        )
                    self.piper.GripperCtrl(abs(gripper_range), 500, 0x01, 0)  # 单位 0.001°

                    # measured 采样：与写线程同线程读，避免并发读写 SDK 引发阻塞/抖动。
                    if self._shm_meas_writer is not None and self._shm_meas_period_ns > 0:
                        now_ns = time.monotonic_ns()
                        if now_ns - int(self._shm_last_meas_ns) >= int(self._shm_meas_period_ns):
                            q7 = self._read_q7_locked()
                            t_meas_ns = time.monotonic_ns()
                            self._shm_last_meas_ns = int(t_meas_ns)
                            self._last_measured_q7 = list(q7)
                            try:
                                self._shm_meas_writer.append(int(t_meas_ns), q7, meta=1)
                            except Exception:
                                pass
                duration_ms = (time.perf_counter() - start) * 1000.0

                # cmd 发布：写线程完成本次下发后写入 ring（时间戳仍使用 t_cmd_ns）。
                if self._shm_cmd_writer is not None:
                    try:
                        self._shm_cmd_writer.append(
                            int(t_cmd_ns),
                            [*joints_rad, float(gripper_value)],
                            meta=1,
                        )
                    except Exception:
                        pass

                # 若单次写入耗时过长，跳过后续等待，尽快处理下一条
                if duration_ms > 20.0:
                    continue
            except Exception:
                # 发送异常时忽略，等待下一帧
                continue
    

    def read(self) -> Dict:
        """读取当前关节状态（弧度）和夹爪开度。"""
        with self._sdk_lock:
            joint_msg = self.piper.GetArmJointMsgs()
            joint_state = joint_msg.joint_state

            gripper_msg = self.piper.GetArmGripperMsgs()
            gripper_state = gripper_msg.gripper_state
        
        return {
            "joint_1": joint_state.joint_1 * self.mdeg_to_rad,
            "joint_2": joint_state.joint_2 * self.mdeg_to_rad,
            "joint_3": joint_state.joint_3 * self.mdeg_to_rad,
            "joint_4": joint_state.joint_4 * self.mdeg_to_rad,
            "joint_5": joint_state.joint_5 * self.mdeg_to_rad,
            "joint_6": joint_state.joint_6 * self.mdeg_to_rad,
            "gripper": gripper_state.grippers_angle / 1000
        }
    
    def read_gripper_effort(
        self,
        samples: int = 3,
        interval: float = 0.02,
        mode: str = "mean",
        return_samples: bool = False,
    ) -> float | Tuple[float, List[float]]:
        """采样夹爪扭矩，支持多次采样与简单滤波。

        Args:
            samples: 采样次数，大于等于 1。
            interval: 连续采样之间的等待时间（秒）。
            mode: 对采样值进行聚合的方式，可选 ``mean``、``median``、``max``、``min``、``last``。
            return_samples: 若为 True，则同时返回原始采样列表。

        Returns:
            聚合后的扭矩数值；当 ``return_samples`` 为 True 时返回 ``(数值, 原始列表)``。
        """

        if samples <= 0:
            raise ValueError("参数 'samples' 需大于 0")

        efforts: List[float] = []
        for idx in range(samples):
            msg = self.piper.GetArmGripperMsgs()
            effort = float(msg.gripper_state.grippers_effort)
            efforts.append(effort)
            if idx < samples - 1 and interval > 0:
                time.sleep(interval)

        match mode.lower():
            case "mean":
                aggregated = sum(efforts) / len(efforts)
            case "median":
                sorted_efforts = sorted(efforts)
                mid = len(sorted_efforts) // 2
                if len(sorted_efforts) % 2 == 1:
                    aggregated = sorted_efforts[mid]
                else:
                    aggregated = (sorted_efforts[mid - 1] + sorted_efforts[mid]) / 2
            case "max":
                aggregated = max(efforts)
            case "min":
                aggregated = min(efforts)
            case "last":
                aggregated = efforts[-1]
            case _:
                raise ValueError("参数 'mode' 仅支持 mean/median/max/min/last")

        self._effort_history.append(aggregated)

        if return_samples:
            return aggregated, efforts
        return aggregated

    def get_gripper_effort_history(self) -> List[float]:
        """返回近期夹爪扭矩的缓存列表（旧值在前）。"""

        return list(self._effort_history)
    
    def safe_disconnect(self):
        """在断电/急停前移动到更安全的姿态，避免姿态突变。"""

        self.write(target_joint=self.safe_disable_position)
