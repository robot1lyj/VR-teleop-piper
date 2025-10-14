"""VR 增量姿态映射逻辑，负责将手柄位姿转换为机械臂基座系目标。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from vr_runtime.controller_state import ControllerState

# VR 世界坐标系 -> 机械臂基座系的固定旋转矩阵
R_BV_DEFAULT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


@dataclass
class TeleopGoal:
    """单帧增量目标。"""

    hand: str
    position: np.ndarray  # 机械臂基座系下的末端位置
    rotation: np.ndarray  # 机械臂基座系下的末端旋转矩阵
    gripper_closed: bool


def _quaternion_from_payload(payload: Dict[str, float]) -> np.ndarray:
    """将手柄报文中的四元数字段转为 numpy 向量。"""

    return np.array(
        [
            float(payload.get("x", 0.0)),
            float(payload.get("y", 0.0)),
            float(payload.get("z", 0.0)),
            float(payload.get("w", 1.0)),
        ],
        dtype=float,
    )


def _normalize_quaternion(quat: Optional[np.ndarray]) -> np.ndarray:
    """归一化四元数，避免求逆时的数值问题。"""

    if quat is None:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return quat
    return quat / norm


def _quaternion_conjugate(quat: np.ndarray) -> np.ndarray:
    """求共轭四元数。"""

    x, y, z, w = quat
    return np.array([-x, -y, -z, w], dtype=float)


def _quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """按数学定义相乘两个四元数。"""

    x1, y1, z1, w1 = lhs
    x2, y2, z2, w2 = rhs
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


def _quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """将四元数转换为 3x3 旋转矩阵。"""

    quat = _normalize_quaternion(quat)
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


class IncrementalPoseMapper:
    """将 VR 手柄增量映射到机械臂基座系。"""

    def __init__(
        self,
        scale: float = 1.0,
        allowed_hands: Optional[Iterable[str]] = None,
        rotation_vr_to_base: np.ndarray = R_BV_DEFAULT,
    ) -> None:
        allowed_set = set(allowed_hands or {"left", "right"})
        invalid = allowed_set - {"left", "right"}
        if invalid:
            raise ValueError(f"Unsupported hand identifiers: {sorted(invalid)}")

        self.scale = float(scale)
        self.allowed_hands = allowed_set
        self.rotation_vr_to_base = np.asarray(rotation_vr_to_base, dtype=float).reshape(3, 3)
        self.rotation_base_to_vr = self.rotation_vr_to_base.T

        self.controllers: Dict[str, ControllerState] = {
            "left": ControllerState("left"),
            "right": ControllerState("right"),
        }
        self.reference_poses: Dict[str, Dict[str, np.ndarray]] = {}
        # 每只手柄握持时缓存一次局部坐标映射，避免重复计算。
        self.controller_frames: Dict[str, Dict[str, np.ndarray]] = {}

    def set_reference_pose(self, hand: str, position: np.ndarray, rotation: np.ndarray) -> None:
        """记录机械臂当前末端位姿，作为增量累加的基准。"""

        if hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        if hand not in self.allowed_hands:
            return

        pos = np.asarray(position, dtype=float).reshape(3)
        rot = np.asarray(rotation, dtype=float).reshape(3, 3)
        self.reference_poses[hand] = {"position": pos, "rotation": rot}
        self.controller_frames.pop(hand, None)

    def clear_reference_pose(self, hand: str) -> None:
        """当控制权移交或重置时移除参考位姿。"""

        self.reference_poses.pop(hand, None)
        self.controller_frames.pop(hand, None)
        controller = self.controllers.get(hand)
        if controller:
            controller.reset_grip()

    def process(self, payload: Dict[str, Any]) -> List[TeleopGoal]:
        """处理来自 WebRTC 通道的手柄报文。"""

        goals: List[TeleopGoal] = []

        if "leftController" in payload or "rightController" in payload:
            if "leftController" in payload and "left" in self.allowed_hands:
                goal = self._handle_single("left", payload["leftController"])
                if goal:
                    goals.append(goal)
            if "rightController" in payload and "right" in self.allowed_hands:
                goal = self._handle_single("right", payload["rightController"])
                if goal:
                    goals.append(goal)
            return goals

        hand = payload.get("hand")
        if hand in self.allowed_hands:
            goal = self._handle_single(hand, payload)
            if goal:
                goals.append(goal)

        return goals

    def _cache_controller_frame(
        self,
        hand: str,
        reference: Dict[str, np.ndarray],
        origin_quaternion: Optional[np.ndarray],
    ) -> None:
        """基于当前参考姿态与手柄零位四元数缓存局部映射。"""

        reference_rotation = reference["rotation"].copy()
        vr_origin_matrix = np.eye(3)
        if origin_quaternion is not None:
            vr_origin_matrix = _quaternion_to_matrix(origin_quaternion)

        ctrl_to_base = self.rotation_vr_to_base @ vr_origin_matrix
        ctrl_to_ee = reference_rotation.T @ ctrl_to_base
        self.controller_frames[hand] = {
            "reference_rotation": reference_rotation,
            "vr_origin_matrix": vr_origin_matrix,
            "vr_origin_matrix_T": vr_origin_matrix.T,
            "ctrl_to_ee": ctrl_to_ee,
            "ctrl_to_ee_T": ctrl_to_ee.T,
        }

    def _handle_single(self, hand: str, payload: Dict[str, Any]) -> Optional[TeleopGoal]:
        """处理单个手柄的增量计算。"""

        controller = self.controllers[hand]
        reference = self.reference_poses.get(hand)
        if reference is None:
            return None

        position = payload.get("position")
        quaternion_payload = payload.get("quaternion")
        grip_active = bool(payload.get("gripActive", False))
        trigger_value = float(payload.get("trigger", 0.0))
        trigger_active = trigger_value > 0.5
        controller.trigger_active = trigger_active

        if position is None:
            return None

        if not grip_active:
            if controller.grip_active:
                controller.reset_grip()
                self.controller_frames.pop(hand, None)
            return None

        position_vec = np.array(
            [
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
                float(position.get("z", 0.0)),
            ],
            dtype=float,
        )

        if not controller.grip_active:
            controller.grip_active = True
            controller.origin_position = position_vec
            if quaternion_payload:
                controller.origin_quaternion = _quaternion_from_payload(quaternion_payload)
                controller.accumulated_quaternion = controller.origin_quaternion
            else:
                controller.origin_quaternion = None
                controller.accumulated_quaternion = None
            self._cache_controller_frame(hand, reference, controller.origin_quaternion)
            return None

        if quaternion_payload:
            quaternion_now = _quaternion_from_payload(quaternion_payload)
            controller.accumulated_quaternion = quaternion_now
            if controller.origin_quaternion is None:
                controller.origin_quaternion = quaternion_now
                self._cache_controller_frame(hand, reference, controller.origin_quaternion)
            elif hand not in self.controller_frames:
                self._cache_controller_frame(hand, reference, controller.origin_quaternion)

        origin_position = controller.origin_position
        if origin_position is None:
            origin_position = np.zeros(3)
        delta_vr = position_vec - origin_position
        delta_base = self.rotation_vr_to_base @ delta_vr
        goal_position = reference["position"] + self.scale * delta_base

        goal_rotation = reference["rotation"]
        frame_cache = self.controller_frames.get(hand)
        if (
            frame_cache
            and controller.accumulated_quaternion is not None
            and controller.origin_quaternion is not None
            and "vr_origin_matrix" in frame_cache
        ):
            vr_now = _quaternion_to_matrix(controller.accumulated_quaternion)
            vr_origin_matrix_T = frame_cache["vr_origin_matrix_T"]
            ctrl_rel_local = vr_origin_matrix_T @ vr_now
            ctrl_to_ee = frame_cache["ctrl_to_ee"]
            ctrl_to_ee_T = frame_cache["ctrl_to_ee_T"]
            rel_in_ee = ctrl_to_ee @ ctrl_rel_local @ ctrl_to_ee_T
            reference_rotation = frame_cache["reference_rotation"]
            goal_rotation = reference_rotation @ rel_in_ee
        elif controller.accumulated_quaternion is not None and controller.origin_quaternion is not None:
            q_now = _normalize_quaternion(controller.accumulated_quaternion)
            q_origin = _normalize_quaternion(controller.origin_quaternion)
            q_rel = _quaternion_multiply(q_now, _quaternion_conjugate(q_origin))
            rot_rel_vr = _quaternion_to_matrix(q_rel)
            rot_rel_base = self.rotation_vr_to_base @ rot_rel_vr @ self.rotation_base_to_vr
            goal_rotation = reference["rotation"] @ rot_rel_base

        return TeleopGoal(
            hand=hand,
            position=goal_position,
            rotation=goal_rotation,
            gripper_closed=not controller.trigger_active,
        )
