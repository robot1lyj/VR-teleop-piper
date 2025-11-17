"""封装 VR 增量控制与 IK 求解的会话逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import math

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
    pitch_mode: bool = False
    pitch_angle: Optional[float] = None


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
        model = self.ik_solver.reduced_robot.model
        self._joint_lower = model.lowerPositionLimit.copy()
        self._joint_upper = model.upperPositionLimit.copy()
        constraint_manager = getattr(self.ik_solver, "constraint_manager", None)
        if constraint_manager is not None:
            hard_lower = getattr(constraint_manager, "_hard_lower", None)
            hard_upper = getattr(constraint_manager, "_hard_upper", None)
            if isinstance(hard_lower, np.ndarray):
                self._joint_lower = np.maximum(self._joint_lower, hard_lower)
            if isinstance(hard_upper, np.ndarray):
                finite = np.isfinite(hard_upper)
                upper = self._joint_upper.copy()
                upper[finite] = np.minimum(upper[finite], hard_upper[finite])
                self._joint_upper = upper

        self._wrist_indices = self._resolve_wrist_indices(model)
        self._last_wrist: Dict[str, np.ndarray] = {}

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

            if goal.pitch_mode:
                results.append(
                    TeleopResult(
                        hand=goal.hand,
                        success=True,
                        info="pitch-mode",
                        joints=None,
                        target=target,
                        gripper_closed=goal.gripper_closed,
                        pitch_mode=True,
                        pitch_angle=goal.pitch_angle,
                    )
                )
                continue

            joints, success, info = self.ik_solver.solve(target, check_collision=self.check_collision)
            if isinstance(joints, np.ndarray):
                joints_array: Optional[np.ndarray] = joints.copy()
            elif joints is None:
                joints_array = None
            else:
                joints_array = np.asarray(joints, dtype=float).reshape(-1)

            if success and joints_array is not None:
                joints_array = self._stabilize_wrist_branch(goal.hand, joints_array)

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

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    def _resolve_wrist_indices(self, model: pin.Model) -> List[int]:
        indices: List[int] = []
        for name in ("joint4", "joint5", "joint6"):
            joint_id = model.getJointId(name)
            if joint_id <= 0:
                continue
            joint = model.joints[joint_id]
            if joint.nq != 1:
                continue
            indices.append(joint.idx_q)
        return indices

    def _stabilize_wrist_branch(self, hand: str, joints: np.ndarray) -> np.ndarray:
        if len(self._wrist_indices) != 3:
            return joints
        wrist = joints[self._wrist_indices].copy()
        candidates = [wrist]

        flipped = wrist.copy()
        flipped[0] = self._wrap_angle(flipped[0] + math.pi)
        flipped[1] = -flipped[1]
        flipped[2] = self._wrap_angle(flipped[2] + math.pi)
        candidates.append(flipped)

        lower = self._joint_lower[self._wrist_indices]
        upper = self._joint_upper[self._wrist_indices]

        prev = self._last_wrist.get(hand)
        best = None
        best_cost = float("inf")
        for cand in candidates:
            if np.any(cand < lower) or np.any(cand > upper):
                continue
            if prev is None:
                best = cand
                break
            cost = np.linalg.norm(cand - prev)
            if cost < best_cost:
                best_cost = cost
                best = cand

        if best is None:
            best = wrist

        for offset, idx in enumerate(self._wrist_indices):
            joints[idx] = best[offset]

        self._last_wrist[hand] = best.copy()
        return joints
