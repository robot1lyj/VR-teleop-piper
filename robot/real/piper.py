import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


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


def _load_piper_config(config_path: str | Path | None) -> Dict[str, Any]:
    """读取 Piper 配置文件，补全缺省值并校验关键字段。"""

    defaults: Dict[str, Any] = {
        "can_name": "can0",
        "init_joint_position": [-5, 80, -161, 26.2, -8.3, -4.1, 0.0],
        "safe_disable_position": None,
        "pose_factor": 1000,
        "joint_factor": 1000,
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

    config["init_joint_position"] = _ensure_float_list(config["init_joint_position"], "init_joint_position")

    safe_disable = config.get("safe_disable_position")
    if safe_disable is None:
        safe_disable = list(config["init_joint_position"])
    else:
        safe_disable = _ensure_float_list(safe_disable, "safe_disable_position")
    config["safe_disable_position"] = safe_disable

    try:
        config["pose_factor"] = float(config["pose_factor"])
    except (TypeError, ValueError) as exc:
        raise ValueError("配置项 'pose_factor' 无法转换为浮点数") from exc

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
        self.pose_factor = config["pose_factor"]  # 单位 0.001 mm
        self.joint_factor = config["joint_factor"]  # 1000*180/pi， rad -> 度（单位 0.001 deg）

        self.piper = C_PiperInterface_V2(self.can_name)
        self.piper.ConnectPort()




    def connect(self, enable:bool=True) -> bool:
        '''
            使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        '''
        enable_flag = False
        loop_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        while not (loop_flag):
            elapsed_time = time.time() - start_time
            print("--------------------")
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            if(enable):
                enable_flag = all(enable_list)
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0,1000,0x01, 0)
            else:
                # move to safe disconnect position
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0,1000,0x02, 0)
            print(f"使能状态: {enable_flag}")
            print("--------------------")
            if(enable_flag == enable):
                loop_flag = True
                enable_flag = True
            else: 
                loop_flag = False
                enable_flag = False
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print("超时....")
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        resp = enable_flag
        print(f"Returning response: {resp}")
        return resp
    


    def set_calibration(self):
        return
    
    def revert_calibration(self):
        return

    def apply_calibration(self):
        """
            移动到初始位置
        """
        self.write(target_joint=self.init_joint_position)

    def write(self, target_joint:list):
        """
            Joint control
            - target joint: in radians
                joint_1 (float): 关节1角度 (-92000~92000) / 57324.840764
                joint_2 (float): 关节2角度 -1300 ~ 90000 / 57324.840764
                joint_3 (float): 关节3角度 2400 ~ -80000 / 57324.840764
                joint_4 (float): 关节4角度 -90000~90000 / 57324.840764
                joint_5 (float): 关节5角度 19000~-77000 / 57324.840764
                joint_6 (float): 关节6角度 -90000~90000 / 57324.840764
                gripper_range: 夹爪角度 0~0.08
        """
        joint_0 = round(target_joint[0]*self.joint_factor)
        joint_1 = round(target_joint[1]*self.joint_factor)
        joint_2 = round(target_joint[2]*self.joint_factor)
        joint_3 = round(target_joint[3]*self.joint_factor)
        joint_4 = round(target_joint[4]*self.joint_factor)
        joint_5 = round(target_joint[5]*self.joint_factor)
        gripper_range = round(target_joint[6]*1000*100)

        if joint_4 < -70000:
            joint_4 = -70000
        if joint_3 <-178:
            joint_3 = -173
        if gripper_range < 1000000:
            gripper_range = 0

        self.piper.MotionCtrl_2(0x01, 0x01, 80, 0xAD) # joint control
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(abs(gripper_range), 1000, 0x01, 0) # 单位 0.001°
    

    def read(self) -> Dict:
        """
            - 机械臂关节消息,单位0.001度
            - 机械臂夹爪消息
        """
        joint_msg = self.piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state

        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state
        
        return {
            "joint_1": joint_state.joint_1 / 1000,
            "joint_2": joint_state.joint_2 / 1000,
            "joint_3": joint_state.joint_3 / 1000,
            "joint_4": joint_state.joint_4 / 1000,
            "joint_5": joint_state.joint_5 / 1000,
            "joint_6": joint_state.joint_6 / 1000,
            "gripper": gripper_state.grippers_angle / 1000
        }
    
    def safe_disconnect(self):
        """ 
            Move to safe disconnect position
        """
        self.write(target_joint=self.safe_disable_position)
