"""封装 Meshcat 可视化的 IK 求解器派生类。"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple, Dict, Any
import numpy as np
import pinocchio as pin

from .base_solver import BaseArmIK

try:  # Meshcat 非强依赖，导入失败时自动降级
    import meshcat.geometry as mg
    from pinocchio.visualize import MeshcatVisualizer
    _HAS_MESHCAT = True
except Exception:  # pylint: disable=broad-except
    _HAS_MESHCAT = False
    MeshcatVisualizer = None  # type: ignore[misc,assignment]


class MeshcatArmIK(BaseArmIK):
    """在基础 IK 上叠加 Meshcat 可视化能力。"""

    def __init__(
        self,
        urdf_path: str,
        joints_to_lock: Optional[list[str]] = None,
        add_ee_on_joint: str = "joint6",
        add_ee_translation: Tuple[float, float, float] = (0.0, 0.0, 0.13),
        add_ee_rpy: Tuple[float, float, float] = (0.0, -np.pi / 2.0, 0.0),
        position_weight: float = 20.0,
        orientation_weight: float = 20.0,
        reg_weight: float = 0.01,
        smooth_weight: Optional[float] = None,
        joint_reg_weights: Optional[Iterable[float]] = None,
        joint_smooth_weights: Optional[Iterable[float]] = None,
        swivel_limit: Optional[float] = None,
        trust_region: Optional[Iterable[float]] = None,
        joint_constraints: Optional[Dict[str, Any]] = None,
        solver_max_iter: int = 50,
        solver_tol: float = 1e-4,
        enable_viewer: bool = True,
        open_viewer: bool = True,
    ) -> None:
        super().__init__(
            urdf_path=urdf_path,
            joints_to_lock=joints_to_lock,
            add_ee_on_joint=add_ee_on_joint,
            add_ee_translation=add_ee_translation,
            add_ee_rpy=add_ee_rpy,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            reg_weight=reg_weight,
            smooth_weight=smooth_weight,
            joint_reg_weights=joint_reg_weights,
            joint_smooth_weights=joint_smooth_weights,
            swivel_limit=swivel_limit,
            trust_region=trust_region,
            joint_constraints=joint_constraints,
            solver_max_iter=solver_max_iter,
            solver_tol=solver_tol,
        )

        self.use_meshcat = bool(enable_viewer and _HAS_MESHCAT)
        self.vis: Optional[MeshcatVisualizer] = None
        self._meshcat_base_transform = np.eye(4)
        self._full_config_template = pin.neutral(self.robot.model)

        gripper_indices: list[int] = []
        gripper_lower: list[float] = []
        gripper_upper: list[float] = []
        for joint_name in ("joint7", "joint8"):
            joint_id = self.robot.model.getJointId(joint_name)
            if joint_id <= 0:
                raise ValueError(f"未在模型中找到 {joint_name}")
            joint = self.robot.model.joints[joint_id]
            if joint.nq != 1:
                raise ValueError(f"手爪关节 {joint_name} 的自由度数量异常: {joint.nq}")
            idx_q = joint.idx_q
            gripper_indices.append(idx_q)
            gripper_lower.append(self.robot.model.lowerPositionLimit[idx_q])
            gripper_upper.append(self.robot.model.upperPositionLimit[idx_q])

        self._gripper_joint_indices = np.array(gripper_indices, dtype=int)
        self._gripper_lower = np.array(gripper_lower, dtype=float)
        self._gripper_upper = np.array(gripper_upper, dtype=float)

        open_targets = []
        for lower, upper in zip(self._gripper_lower, self._gripper_upper):
            if abs(upper) >= abs(lower):
                open_targets.append(0.9 * upper)
            else:
                open_targets.append(0.9 * lower)
        self._gripper_closed_config = np.zeros_like(self._gripper_lower)
        self._gripper_open_config = np.clip(np.array(open_targets, dtype=float), self._gripper_lower, self._gripper_upper)
        self._current_gripper_q = self._gripper_closed_config.copy()
        self._ee_frame_id_visual = self.robot.model.getFrameId("ee")

        if self.use_meshcat:
            assert MeshcatVisualizer is not None
            try:
                self.vis = MeshcatVisualizer(
                    self.robot.model,
                    self.robot.collision_model,
                    self.robot.visual_model,
                )
                self.vis.initViewer(open=open_viewer)
                self.vis.loadViewerModel("pinocchio")

                if self._ee_frame_id_visual >= 0:
                    self.vis.displayFrames(True, frame_ids=[self._ee_frame_id_visual], axis_length=0.12, axis_width=5)
                self.vis.display(self._compose_full_q(self.q_seed))

                frame_viz_name = "ee_target"
                axis_positions = np.array(
                    [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]],
                    dtype=np.float32,
                ).T
                axis_colors = np.array(
                    [[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]],
                    dtype=np.float32,
                ).T
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(position=0.1 * axis_positions, color=axis_colors),
                        mg.LineBasicMaterial(linewidth=10, vertexColors=True),
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                logging.getLogger(__name__).warning("Meshcat 初始化失败：%s", exc)
                self.vis = None
                self.use_meshcat = False
        else:
            if enable_viewer and not _HAS_MESHCAT:
                logging.getLogger(__name__).warning("未检测到 Meshcat 依赖，自动降级为纯求解模式")

    # -----------------------------
    # 对基类钩子的实现
    # -----------------------------
    def _before_solve(self, target: np.ndarray) -> None:
        if not self.use_meshcat or self.vis is None:
            return
        try:
            display_target = self._meshcat_base_transform @ target
            self.vis.viewer["ee_target"].set_transform(display_target)
        except Exception:  # pylint: disable=broad-except
            pass

    def _after_solve_success(self, q: np.ndarray, target: np.ndarray) -> None:
        if not self.use_meshcat or self.vis is None:
            return
        try:
            self.vis.display(self._compose_full_q(q))
        except Exception:  # pylint: disable=broad-except
            pass

    def _after_solve_failure(self, error: Exception, target: np.ndarray) -> None:  # noqa: D401
        if not self.use_meshcat or self.vis is None:
            return
        logging.getLogger(__name__).debug("Meshcat IK 求解失败：%s", error)

    # -----------------------------
    # 额外的 Meshcat 接口
    # -----------------------------
    def set_visual_base_transform(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        """设置 Meshcat 中的基座姿态，rotation 为 3x3，translation 为长度 3。"""

        self._meshcat_base_transform = np.eye(4)
        self._meshcat_base_transform[:3, :3] = rotation
        self._meshcat_base_transform[:3, 3] = translation.reshape(3)

        if not self.use_meshcat or self.vis is None:
            return
        try:
            self.vis.viewer["pinocchio"].set_transform(self._meshcat_base_transform)
        except Exception:  # pylint: disable=broad-except
            logging.getLogger(__name__).warning("Meshcat 基座变换设置失败")

    def refresh_visual(self, q: Optional[np.ndarray] = None) -> None:
        """强制刷新 Meshcat 中当前关节配置，便于外部在设置初始位姿后同步。"""

        if not self.use_meshcat or self.vis is None:
            return
        pose = self._compose_full_q(self.q_last if q is None else np.asarray(q, dtype=float).reshape(-1))
        try:
            self.vis.display(pose)
        except Exception:  # pylint: disable=broad-except
            logging.getLogger(__name__).warning("Meshcat 刷新失败")

    def set_gripper_state(self, closed: bool) -> None:
        super().set_gripper_state(closed)
        target = self._gripper_closed_config if closed else self._gripper_open_config
        if np.allclose(self._current_gripper_q, target, atol=1e-4):
            return
        self._current_gripper_q = target.copy()
        if not self.use_meshcat or self.vis is None:
            return
        try:
            self.vis.display(self._compose_full_q(self.q_last))
        except Exception:  # pylint: disable=broad-except
            logging.getLogger(__name__).debug("Meshcat 手爪刷新失败", exc_info=False)

    def _compose_full_q(
        self,
        q: Optional[np.ndarray] = None,
        gripper: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        full_q = self._full_config_template.copy()
        q_source = self.q_last if q is None else np.asarray(q, dtype=float).reshape(-1)
        full_q[: self.reduced_robot.model.nq] = q_source
        gripper_cfg = self._current_gripper_q if gripper is None else np.asarray(gripper, dtype=float).reshape(-1)
        full_q[self._gripper_joint_indices] = np.clip(gripper_cfg, self._gripper_lower, self._gripper_upper)
        return full_q


__all__ = ["MeshcatArmIK"]
