"""轻量级的 VR 手柄状态模型，记录握持、触发器和姿态信息。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ControllerState:
    """记录单个 VR 手柄的状态。

    hand
        手柄所在的手，取值为 ``"left"`` 或 ``"right"``。
    grip_active
        当前握持键是否按下。握持键按下后才开始计算位移增量。
    trigger_active
        扳机键是否按下，用于控制夹爪开闭。
    origin_position
        握持键首次按下时记录的世界坐标（numpy 向量）。
    origin_quaternion
        握持键首次按下时记录的方向（四元数，x/y/z/w）。
    accumulated_quaternion
        手柄当前姿态的四元数，用于与 origin 比较得到相对旋转。
    z_axis_rotation
        根据相对四元数计算得到的 Z 轴旋转角（度），用于手腕 roll。
    x_axis_rotation
        根据相对四元数计算得到的 X 轴旋转角（度），用于手腕 pitch。
    """

    hand: str
    grip_active: bool = False
    trigger_active: bool = False

    origin_position: Optional[np.ndarray] = None
    origin_quaternion: Optional[np.ndarray] = None
    accumulated_quaternion: Optional[np.ndarray] = None

    z_axis_rotation: float = 0.0
    x_axis_rotation: float = 0.0
    menu_active: bool = False

    def reset_grip(self) -> None:
        """重置握持相关状态，在松开握持键或断开连接时调用。"""

        self.grip_active = False
        self.origin_position = None
        self.origin_quaternion = None
        self.accumulated_quaternion = None
        self.z_axis_rotation = 0.0
        self.x_axis_rotation = 0.0
        self.menu_active = False


# 两个全局实例分别跟踪左右手柄的状态，供服务器循环复用。
LEFT_CONTROLLER = ControllerState("left")
RIGHT_CONTROLLER = ControllerState("right")
