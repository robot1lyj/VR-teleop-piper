import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Sequence, Tuple


from piper_sdk import *


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "piper.json"


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


def _load_piper_config(config_path: str | Path | None) -> Dict[str, Any]:
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

    def __init__(self, config_path: str | Path | None = None):
        config = _load_piper_config(config_path)

        self.config = config
        self.config_path = config["config_path"]
        self.can_name = config["can_name"]
        self.init_joint_position = config["init_joint_position"]
        self.safe_disable_position = config["safe_disable_position"]
        self.joint_factor = config["joint_factor"]  # 用于兼容旧配置的缩放因子（默认 rad -> mdeg）
        self.rad_to_mdeg = float(config.get("joint_factor", 180.0 / math.pi * 1000.0))
        self.mdeg_to_rad = math.pi / 180.0 / 1000.0

        self.piper = C_PiperInterface_V2(self.can_name)
        self.piper.ConnectPort()

        self._effort_history: Deque[float] = deque(maxlen=50)
        self._joint_mode_configured = False




    def _read_enable_flags(self) -> List[bool]:
        info = self.piper.GetArmLowSpdInfoMsgs()
        return [
            bool(info.motor_1.foc_status.driver_enable_status),
            bool(info.motor_2.foc_status.driver_enable_status),
            bool(info.motor_3.foc_status.driver_enable_status),
            bool(info.motor_4.foc_status.driver_enable_status),
            bool(info.motor_5.foc_status.driver_enable_status),
            bool(info.motor_6.foc_status.driver_enable_status),
        ]

    def connect(self, enable:bool=True) -> bool:
        '''
            使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        '''
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
        return
    
    def revert_calibration(self):
        return

    def apply_calibration(self):
        """
            移动到初始位置
        """
        self.write(target_joint=self.init_joint_position)

    def write(self, target_joint: List[float]) -> None:
        """Joint control（输入单位为弧度，内部自动转换成 0.001°）。"""

        if len(target_joint) != 7:
            raise ValueError("target_joint 长度必须为 7 (6 轴 + 夹爪)")

        joint_cmds = [round(val * self.rad_to_mdeg) for val in target_joint[:6]]
        gripper_range = round(target_joint[6] * 1000.0 * 100.0)

        if joint_cmds[4] < -70000:
            joint_cmds[4] = -70000

        if joint_cmds[3] < -178000:
            joint_cmds[3] = -173000

        if gripper_range < 0:
            gripper_range = 0

        if not self._joint_mode_configured:
            self.piper.MotionCtrl_2(0x01, 0x01, 80, 0x00)  # joint control
            self._joint_mode_configured = True

        self.piper.JointCtrl(*joint_cmds)
        self.piper.GripperCtrl(abs(gripper_range), 1000, 0x01, 0)  # 单位 0.001°
    

    def read(self) -> Dict:
        """读取当前关节状态（弧度）和夹爪开度。"""
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
        """ 
            Move to safe disconnect position
        """
        self.write(target_joint=self.safe_disable_position)
