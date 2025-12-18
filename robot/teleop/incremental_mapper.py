"""VR 增量姿态映射逻辑，负责将手柄位姿转换为机械臂基座系目标。"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pinocchio as pin

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


class PoseHistoryFilter:
    """基于时间窗的多项式拟合，用历史帧平滑位置/姿态。"""

    def __init__(
        self,
        window_sec: float,
        degree: int,
        min_samples: int,
        lookahead_sec: float,
    ) -> None:
        self.enabled = window_sec > 0.0
        self.window_sec = max(0.0, float(window_sec))
        self.degree = max(1, int(degree))
        required = max(self.degree + 1, int(min_samples))
        self.min_samples = max(required, 3)
        self.lookahead_sec = max(0.0, float(lookahead_sec))
        self._samples: Deque[Tuple[float, np.ndarray, Optional[np.ndarray]]] = deque()

    def filter_sample(
        self,
        timestamp: float,
        position: np.ndarray,
        quaternion: Optional[np.ndarray],
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.enabled:
            return position, quaternion

        pos = np.asarray(position, dtype=float).reshape(3)
        quat = None if quaternion is None else np.asarray(quaternion, dtype=float).reshape(4)
        self._samples.append((timestamp, pos, quat))
        self._trim(timestamp)

        filtered_pos = self._smooth_positions()
        filtered_rot = self._smooth_orientation()
        return filtered_pos if filtered_pos is not None else pos, filtered_rot if filtered_rot is not None else quat

    def reset(self) -> None:
        """清空历史窗口，握持重新开始时避免用旧样本参与平滑。"""

        self._samples.clear()

    def _trim(self, latest_ts: float) -> None:
        if not self.enabled or self.window_sec <= 0.0:
            return
        while self._samples and latest_ts - self._samples[0][0] > self.window_sec:
            self._samples.popleft()

    def _relative_times(self) -> np.ndarray:
        times = np.array([sample[0] for sample in self._samples], dtype=float)
        if times.size == 0:
            return times
        times -= times[-1]
        return times

    def _fit_series(
        self,
        times: np.ndarray,
        values: np.ndarray,
        lookahead: float,
    ) -> Optional[np.ndarray]:
        if times.size == 0 or values.shape[0] == 0:
            return None
        degree = min(self.degree, values.shape[0] - 1)
        if degree <= 0:
            return values[-1]
        try:
            coeffs = np.polyfit(times, values, deg=degree)
            result = np.polyval(coeffs, lookahead)
            return result
        except Exception:
            return None

    def _smooth_positions(self) -> Optional[np.ndarray]:
        if len(self._samples) < self.min_samples:
            return None
        rel_times = self._relative_times()
        values = np.stack([sample[1] for sample in self._samples], axis=0)
        result = np.empty(3, dtype=float)
        for axis in range(3):
            fitted = self._fit_series(rel_times, values[:, axis], self.lookahead_sec)
            if fitted is None:
                return None
            result[axis] = fitted
        return result

    def _smooth_orientation(self) -> Optional[np.ndarray]:
        rotation_vectors: List[np.ndarray] = []
        times: List[float] = []
        rel_times = self._relative_times()
        for idx, sample in enumerate(self._samples):
            quat = sample[2]
            if quat is None:
                continue
            try:
                rot_vec = pin.log3(_quaternion_to_matrix(quat))
            except Exception:
                continue
            rotation_vectors.append(rot_vec)
            times.append(rel_times[idx])
        if len(rotation_vectors) < max(self.min_samples, self.degree + 1):
            return None
        values = np.stack(rotation_vectors, axis=0)
        times_arr = np.array(times, dtype=float)
        fitted = np.empty(3, dtype=float)
        for axis in range(3):
            axis_fit = self._fit_series(times_arr, values[:, axis], self.lookahead_sec)
            if axis_fit is None:
                return None
            fitted[axis] = axis_fit
        try:
            rot_matrix = pin.exp3(fitted)
        except Exception:
            return None
        return _matrix_to_quaternion(rot_matrix)


@dataclass
class TeleopGoal:
    """单帧增量目标。"""

    hand: str
    position: np.ndarray  # 机械臂基座系下的末端位置
    rotation: np.ndarray  # 机械臂基座系下的末端旋转矩阵
    gripper_closed: bool
    pitch_mode: bool = False
    pitch_angle: float = 0.0  # 仅在 pitch_mode 时生效（弧度）


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


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(mat))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (mat[2, 1] - mat[1, 2]) / s
        y = (mat[0, 2] - mat[2, 0]) / s
        z = (mat[1, 0] - mat[0, 1]) / s
    else:
        if mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = math.sqrt(max(0.0, 1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])) * 2.0
            x = 0.25 * s
            y = (mat[0, 1] + mat[1, 0]) / s
            z = (mat[0, 2] + mat[2, 0]) / s
            w = (mat[2, 1] - mat[1, 2]) / s
        elif mat[1, 1] > mat[2, 2]:
            s = math.sqrt(max(0.0, 1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])) * 2.0
            x = (mat[0, 1] + mat[1, 0]) / s
            y = 0.25 * s
            z = (mat[1, 2] + mat[2, 1]) / s
            w = (mat[0, 2] - mat[2, 0]) / s
        else:
            s = math.sqrt(max(0.0, 1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])) * 2.0
            x = (mat[0, 2] + mat[2, 0]) / s
            y = (mat[1, 2] + mat[2, 1]) / s
            z = 0.25 * s
            w = (mat[1, 0] - mat[0, 1]) / s
    quat = np.array([x, y, z, w], dtype=float)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm


class IncrementalPoseMapper:
    """将 VR 手柄增量映射到机械臂基座系。"""

    def __init__(
        self,
        scale: float = 1.0,
        allowed_hands: Optional[Iterable[str]] = None,
        rotation_vr_to_base: np.ndarray | Dict[str, np.ndarray] = R_BV_DEFAULT,
        pose_filter_window_sec: float = 0.0,
        pose_filter_degree: int = 2,
        pose_filter_min_samples: int = 15,
        pose_filter_lookahead_sec: float = 0.02,
    ) -> None:
        allowed_set = set(allowed_hands or {"left", "right"})
        invalid = allowed_set - {"left", "right"}
        if invalid:
            raise ValueError(f"Unsupported hand identifiers: {sorted(invalid)}")

        self.scale = float(scale)
        self.allowed_hands = allowed_set
        self._default_rotation = np.asarray(
            rotation_vr_to_base if not isinstance(rotation_vr_to_base, dict) else R_BV_DEFAULT,
            dtype=float,
        ).reshape(3, 3)

        self.rotation_vr_to_base_map: Dict[str, np.ndarray] = {}
        if isinstance(rotation_vr_to_base, dict):
            for hand, mat in rotation_vr_to_base.items():
                if hand not in allowed_set:
                    continue
                self.rotation_vr_to_base_map[hand] = np.asarray(mat, dtype=float).reshape(3, 3)
        if not self.rotation_vr_to_base_map:
            for hand in allowed_set:
                self.rotation_vr_to_base_map[hand] = self._default_rotation
        self.rotation_base_to_vr_map: Dict[str, np.ndarray] = {
            hand: rot.T for hand, rot in self.rotation_vr_to_base_map.items()
        }

        self.controllers: Dict[str, ControllerState] = {
            "left": ControllerState("left"),
            "right": ControllerState("right"),
        }
        self.reference_poses: Dict[str, Dict[str, np.ndarray]] = {}
        # 每只手柄握持时缓存一次局部坐标映射，避免重复计算。
        self.controller_frames: Dict[str, Dict[str, np.ndarray]] = {}
        # 握持姿态补偿：对齐手柄零位与末端零位，修正后续相对旋转
        self._grip_rot_comp: Dict[str, Dict[str, np.ndarray]] = {}
        self._pose_filters: Dict[str, PoseHistoryFilter] = {}
        if pose_filter_window_sec > 0.0:
            for hand in self.allowed_hands:
                self._pose_filters[hand] = PoseHistoryFilter(
                    window_sec=pose_filter_window_sec,
                    degree=pose_filter_degree,
                    min_samples=pose_filter_min_samples,
                    lookahead_sec=pose_filter_lookahead_sec,
                )
        self._grip_released: Dict[str, bool] = {
            "left": False,
            "right": False,
        }
        self._pitch_states: Dict[str, Dict[str, Any]] = {
            "left": {"active": False},
            "right": {"active": False},
        }
        self._pitch_angle_limit = np.deg2rad(80.0)

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
        self._grip_released[hand] = False

    def set_scale(self, scale: float) -> None:
        self.scale = float(scale)

    def clear_reference_pose(self, hand: str) -> None:
        """当控制权移交或重置时移除参考位姿。"""

        self.reference_poses.pop(hand, None)
        self.controller_frames.pop(hand, None)
        self._grip_rot_comp.pop(hand, None)
        controller = self.controllers.get(hand)
        if controller:
            controller.reset_grip()
        self._grip_released[hand] = False

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

        rot_bv = self.rotation_vr_to_base_map.get(hand, self._default_rotation)
        reference_rotation = reference["rotation"].copy()
        vr_origin_matrix = np.eye(3)
        if origin_quaternion is not None:
            vr_origin_matrix = _quaternion_to_matrix(origin_quaternion)

        # 握持时对齐手柄零位与末端零位，补偿仅作用于后续相对旋转
        try:
            ctrl0_base = rot_bv @ vr_origin_matrix
            comp = reference_rotation @ ctrl0_base.T
            self._grip_rot_comp[hand] = {
                "comp": comp,
                "ctrl0_base": ctrl0_base,
                "reference_rotation": reference_rotation,
                "vr_origin_matrix": vr_origin_matrix,
            }
        except Exception:
            self._grip_rot_comp.pop(hand, None)

        ctrl_to_base = rot_bv @ vr_origin_matrix
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
        quaternion_array = _quaternion_from_payload(quaternion_payload) if quaternion_payload else None
        grip_active = bool(payload.get("gripActive", False))
        trigger_value = float(payload.get("trigger", 0.0))
        trigger_active = trigger_value > 0.5
        controller.trigger_active = trigger_active
        controller.menu_active = bool(payload.get("menuPressed", False))
        if position is None:
            return None

        if not grip_active:
            if controller.grip_active:
                self._grip_released[hand] = True
                controller.reset_grip()
                self.controller_frames.pop(hand, None)
                self._pitch_states[hand] = {"active": False}
                pose_filter = self._pose_filters.get(hand)
                if pose_filter is not None:
                    pose_filter.reset()
            return None

        position_vec = np.array(
            [
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
                float(position.get("z", 0.0)),
            ],
            dtype=float,
        )

        pose_filter = self._pose_filters.get(hand)
        if pose_filter is not None:
            timestamp = time.monotonic()
            position_vec, quaternion_array = pose_filter.filter_sample(timestamp, position_vec, quaternion_array)

        if not controller.grip_active:
            controller.grip_active = True
            controller.origin_position = position_vec
            if quaternion_array is not None:
                controller.origin_quaternion = quaternion_array
                controller.accumulated_quaternion = controller.origin_quaternion
            else:
                controller.origin_quaternion = None
                controller.accumulated_quaternion = None
            self._cache_controller_frame(hand, reference, controller.origin_quaternion)
            return None

        if quaternion_array is not None:
            controller.accumulated_quaternion = quaternion_array
            if controller.origin_quaternion is None:
                controller.origin_quaternion = quaternion_array
                self._cache_controller_frame(hand, reference, controller.origin_quaternion)
            elif hand not in self.controller_frames:
                self._cache_controller_frame(hand, reference, controller.origin_quaternion)

        origin_position = controller.origin_position
        if origin_position is None:
            origin_position = np.zeros(3)
        delta_vr = position_vec - origin_position
        rot_bv = self.rotation_vr_to_base_map.get(hand, self._default_rotation)
        rot_vb = self.rotation_base_to_vr_map.get(hand, rot_bv.T)
        delta_base = rot_bv @ delta_vr
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
            rot_rel_base = rot_bv @ rot_rel_vr @ rot_vb
            comp_entry = self._grip_rot_comp.get(hand)
            if comp_entry is not None:
                vr_origin_matrix = comp_entry.get("vr_origin_matrix", np.eye(3))
                try:
                    rot_rel_zero = vr_origin_matrix.T @ rot_rel_vr @ vr_origin_matrix
                    goal_rotation = reference["rotation"] @ rot_rel_zero
                except Exception:
                    goal_rotation = reference["rotation"] @ rot_rel_base
            else:
                goal_rotation = reference["rotation"] @ rot_rel_base

        goal_position, goal_rotation, pitch_active, pitch_angle = self._apply_pitch_mode(
            hand,
            controller,
            goal_position,
            goal_rotation,
            controller.accumulated_quaternion is not None,
        )

        return TeleopGoal(
            hand=hand,
            position=goal_position,
            rotation=goal_rotation,
            gripper_closed=not controller.trigger_active,
            pitch_mode=pitch_active,
            pitch_angle=pitch_angle,
        )

    def reset_hand_state(self, hand: str) -> None:
        """清空单手柄的缓存（不移除参考位姿），用于重握后彻底重置。"""

        controller = self.controllers.get(hand)
        if controller is not None:
            controller.reset_grip()
        self.controller_frames.pop(hand, None)
        self._grip_rot_comp.pop(hand, None)
        self._grip_released[hand] = False
        self._pitch_states[hand] = {"active": False}
        pose_filter = self._pose_filters.get(hand)
        if pose_filter is not None:
            pose_filter.reset()

    def consume_grip_release(self, hand: str) -> bool:
        if hand not in {"left", "right"}:
            return False
        released = self._grip_released.get(hand, False)
        if released:
            self._grip_released[hand] = False
        return released

    def _apply_pitch_mode(
        self,
        hand: str,
        controller: ControllerState,
        goal_position: np.ndarray,
        goal_rotation: np.ndarray,
        has_orientation: bool,
    ) -> tuple[np.ndarray, np.ndarray, bool, float]:
        state = self._pitch_states[hand]
        active = bool(state.get("active"))
        if controller.menu_active:
            if not active:
                state["active"] = True
                state["ref_position"] = goal_position.copy()
                state["ref_rotation"] = goal_rotation.copy()
                state["pitch_angle"] = 0.0
                if has_orientation and controller.accumulated_quaternion is not None:
                    state["controller_origin"] = _quaternion_to_matrix(controller.accumulated_quaternion)
                else:
                    state["controller_origin"] = None
                return state["ref_position"], state["ref_rotation"], True, 0.0

            ref_pos = state.get("ref_position")
            ref_rot = state.get("ref_rotation")
            if ref_pos is None or ref_rot is None:
                return goal_position, goal_rotation, False, 0.0
            controller_origin = state.get("controller_origin")
            if controller_origin is None or controller.accumulated_quaternion is None:
                return ref_pos, ref_rot, True, 0.0
            vr_now = _quaternion_to_matrix(controller.accumulated_quaternion)
            rel = controller_origin.T @ vr_now
            try:
                omega = pin.log3(rel)
            except Exception:  # pragma: no cover
                omega = np.zeros(3)
            pitch_angle = float(np.clip(omega[0], -self._pitch_angle_limit, self._pitch_angle_limit))
            axis_tool = ref_rot[:, 1]
            rot_vec = axis_tool * pitch_angle
            try:
                rot_pitch = pin.exp3(rot_vec)
            except Exception:  # pragma: no cover
                rot_pitch = np.eye(3)
            new_rot = ref_rot @ rot_pitch
            state["pitch_angle"] = pitch_angle
            return ref_pos, new_rot, True, pitch_angle

        if active:
            state.clear()
            state["active"] = False
        return goal_position, goal_rotation, False, 0.0
