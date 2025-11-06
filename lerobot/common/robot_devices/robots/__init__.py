"""机器人设备适配器集合，确保 ChoiceRegistry 正确注册子类。"""

from .configs import RobotConfig
from .piper_vr import PiperVRRobot, PiperVRRobotConfig

__all__ = [
    "RobotConfig",
    "PiperVRRobot",
    "PiperVRRobotConfig",
]
