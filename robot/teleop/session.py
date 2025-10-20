"""封装 VR 增量控制与 IK 求解的会话逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pinocchio as pin

from robot.ik import BaseArmIK

from .incremental_mapper import IncrementalPoseMapper, TeleopGoal


@dataclass
class TeleopResult:
    """单次 IK 求解的结果封装。"""

    hand: str
    success: bool
    info: str
    joints: Optional[np.ndarray]
    target: pin.SE3
    gripper_closed: bool


class ArmTeleopSession:
    """管理 VR 报文、增量映射与 IK 求解的完整链路。"""

    def __init__(
        self,
        ik_solver: BaseArmIK,
        mapper: Optional[IncrementalPoseMapper] = None,
        check_collision: bool = True,
    ) -> None:
        self.ik_solver = ik_solver
        self.mapper = mapper or IncrementalPoseMapper()
        self.check_collision = bool(check_collision)

    def set_reference_pose(self, hand: str, position: np.ndarray, rotation: np.ndarray) -> None:
        """更新某只手柄对应的末端基准位姿。"""

        self.mapper.set_reference_pose(hand, position, rotation)

    def set_reference_pose_se3(self, hand: str, pose: pin.SE3) -> None:
        """使用 Pinocchio 的 SE3 直接设置参考位姿。"""

        self.mapper.set_reference_pose(hand, pose.translation, pose.rotation)

    def clear_reference_pose(self, hand: str) -> None:
        """释放参考位姿并重置手柄状态。"""

        self.mapper.clear_reference_pose(hand)

    def set_scale(self, scale: float) -> None:
        """调整位置增量缩放比例。"""

        if hasattr(self.mapper, "set_scale") and callable(getattr(self.mapper, "set_scale")):
            self.mapper.set_scale(float(scale))
        else:
            self.mapper.scale = float(scale)

    def set_check_collision(self, enabled: bool) -> None:
        """切换是否在 IK 结束后进行自碰撞检测。"""

        self.check_collision = bool(enabled)

    def handle_vr_payload(self, payload: Dict[str, Any]) -> List[TeleopResult]:
        """处理一帧 VR 报文，返回 IK 结果列表。"""

        goals = self.mapper.process(payload)
        results: List[TeleopResult] = []

        for goal in goals:
            self.ik_solver.set_gripper_state(goal.gripper_closed)
            target = pin.SE3(goal.rotation, goal.position)
            joints, success, info = self.ik_solver.solve(target, check_collision=self.check_collision)
            if isinstance(joints, np.ndarray):
                joints_array: Optional[np.ndarray] = joints.copy()
            elif joints is None:
                joints_array = None
            else:
                joints_array = np.asarray(joints, dtype=float).reshape(-1)

            results.append(
                TeleopResult(
                    hand=goal.hand,
                    success=bool(success),
                    info=str(info),
                    joints=joints_array,
                    target=target,
                    gripper_closed=goal.gripper_closed,
                )
            )

        return results
