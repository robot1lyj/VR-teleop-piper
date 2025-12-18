"""基础的 IK 求解器，不涉及 Meshcat 可视化逻辑。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
import os
import numpy as np
import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper

from robot.constraints import JointConstraintManager


# -----------------------------
# 工具函数
# -----------------------------

def rpy_to_quat(roll: float, pitch: float, yaw: float) -> pin.Quaternion:
    """将 XYZ 欧拉角转换为 Pinocchio 使用的四元数。"""

    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    w = cr * cp * cy - sr * sp * sy
    x = sr * cp * cy + cr * sp * sy
    y = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy
    return pin.Quaternion(w, x, y, z)


def xyzrpy_to_SE3(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> pin.SE3:
    """快速构造 SE3 位姿，便于测试。"""

    quat = rpy_to_quat(roll, pitch, yaw)
    return pin.SE3(quat, np.array([x, y, z], dtype=float))


class BaseArmIK:
    """负责构建 Pinocchio + CasADi 的 IK 求解问题，子类可扩展可视化等功能。"""

    def __init__(
        self,
        urdf_path: str,
        joints_to_lock: Optional[List[str]] = None,
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
    ) -> None:
        resolved_urdf = os.path.abspath(urdf_path)
        if not os.path.isfile(resolved_urdf):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        urdf_dir = os.path.dirname(resolved_urdf)
        repo_root = os.path.dirname(os.path.dirname(urdf_dir))  # 假设 URDF 位于 piper_description/urdf
        package_dirs = []
        for path in (urdf_dir, repo_root, os.getcwd()):
            if path and path not in package_dirs:
                package_dirs.append(path)

        self.robot: RobotWrapper = RobotWrapper.BuildFromURDF(resolved_urdf, package_dirs=package_dirs)
        if joints_to_lock is None:
            joints_to_lock = ["joint7", "joint8"]

        # 构建锁定手爪后的约化模型
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=joints_to_lock,
            reference_configuration=np.zeros(self.robot.model.nq),
        )
        self.nq = self.reduced_robot.model.nq
        self._joint_lower = self.reduced_robot.model.lowerPositionLimit.copy()
        self._joint_upper = self.reduced_robot.model.upperPositionLimit.copy()

        self.joint_reg_weights = self._sanitize_joint_weights(joint_reg_weights, self.nq, "joint_reg_weights")
        if smooth_weight is not None:
            self.joint_smooth_weights = self._sanitize_joint_weights(
                joint_smooth_weights, self.nq, "joint_smooth_weights"
            )
            smooth_sq = np.square(self.joint_smooth_weights)
            self._smooth_weights_sq_dm = casadi.DM(smooth_sq).reshape((self.nq, 1))
        else:
            self.joint_smooth_weights = None
            self._smooth_weights_sq_dm = None
        reg_sq = np.square(self.joint_reg_weights)
        self._reg_weights_sq_dm = casadi.DM(reg_sq).reshape((self.nq, 1))

        # 约束与信赖域配置
        self.swivel_limit = self._sanitize_swivel_limit(swivel_limit)
        self._swivel_eps = 1e-9
        self.trust_region = self._sanitize_trust_region(trust_region, self.nq)
        self.constraint_manager = JointConstraintManager.from_config(
            self.reduced_robot.model, joint_constraints or {}
        )

        # 记录肘部相关关节，供 swivel 约束使用
        self.shoulder_joint_id = self.reduced_robot.model.getJointId("joint2")
        self.elbow_joint_id = self.reduced_robot.model.getJointId("joint4")
        self.wrist_joint_id = self.reduced_robot.model.getJointId("joint6")
        for name, jid in {
            "joint2": self.shoulder_joint_id,
            "joint4": self.elbow_joint_id,
            "joint6": self.wrist_joint_id,
        }.items():
            if jid <= 0:
                raise ValueError(f"未在模型中找到 {name}，请检查 URDF")

        # 追加便捷的末端 Frame
        ee_quat = rpy_to_quat(*add_ee_rpy)
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "ee",
                self.reduced_robot.model.getJointId(add_ee_on_joint),
                pin.SE3(ee_quat, np.array(add_ee_translation, dtype=float)),
                pin.FrameType.OP_FRAME,
            )
        )
        if not self.robot.model.existFrame("ee"):
            self.robot.model.addFrame(
                pin.Frame(
                    "ee",
                    self.robot.model.getJointId(add_ee_on_joint),
                    pin.SE3(ee_quat, np.array(add_ee_translation, dtype=float)),
                    pin.FrameType.OP_FRAME,
                )
            )
        self.robot.data = self.robot.model.createData()
        self.reduced_robot.data = self.reduced_robot.model.createData()

        # 碰撞模型（子类可选择忽略）
        self.geom_model = pin.buildGeomFromUrdf(
            self.robot.model, resolved_urdf, pin.GeometryType.COLLISION, package_dirs=package_dirs
        )
        self.geometry_data = pin.GeometryData(self.geom_model)
        try:
            for i in range(4, 9):
                for j in range(0, 3):
                    self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        except Exception:
            pass

        # 初始化中性姿态，用于求解种子与 swivel 参考
        q_neutral = pin.neutral(self.reduced_robot.model)
        self.q_seed = q_neutral.copy()
        self.q_last = q_neutral.copy()
        self._gripper_closed_state: bool = True

        self._ref_elbow_normal: Optional[np.ndarray] = None
        self._ref_elbow_axis: Optional[np.ndarray] = None
        if self.swivel_limit is not None:
            ref_normal = self._compute_elbow_normal(q_neutral)
            ref_axis = self._compute_elbow_axis(q_neutral)
            if ref_normal is not None and ref_axis is not None:
                self._ref_elbow_normal = ref_normal
                self._ref_elbow_axis = ref_axis
            else:
                self.swivel_limit = None

        # 构建 CasADi 符号模型
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        self.ee_fid = self.reduced_robot.model.getFrameId("ee")

        log_vec = cpin.log6(self.cdata.oMf[self.ee_fid].inverse() * cpin.SE3(self.cTf)).vector
        weights = casadi.diag(
            casadi.vcat([
                position_weight,
                position_weight,
                position_weight,
                orientation_weight,
                orientation_weight,
                orientation_weight,
            ])
        )
        weighted = weights @ log_vec
        self.error_fun = casadi.Function("error", [self.cq, self.cTf], [weighted])

        # Opti 变量与参数
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.nq)
        self.param_tf = self.opti.parameter(4, 4)
        self.param_q_last = self.opti.parameter(self.nq) if smooth_weight else None

        tracking_cost = casadi.sumsqr(self.error_fun(self.var_q, self.param_tf))
        reg_cost = reg_weight * casadi.dot(self._reg_weights_sq_dm, casadi.power(self.var_q, 2))
        total_cost = tracking_cost + reg_cost
        if smooth_weight is not None:
            smooth_residual = self.var_q - self.param_q_last
            if self._smooth_weights_sq_dm is not None:
                total_cost += smooth_weight * casadi.dot(
                    self._smooth_weights_sq_dm, casadi.power(smooth_residual, 2)
                )
            else:
                total_cost += smooth_weight * casadi.sumsqr(smooth_residual)

        self.param_swivel_limit = None
        self._elbow_normal_fun: Optional[casadi.Function] = None
        if (
            self.swivel_limit is not None
            and self._ref_elbow_normal is not None
            and self._ref_elbow_axis is not None
        ):
            shoulder_pos = self.cdata.oMi[self.shoulder_joint_id].translation
            elbow_pos = self.cdata.oMi[self.elbow_joint_id].translation
            wrist_pos = self.cdata.oMi[self.wrist_joint_id].translation

            def _casadi_cross(a: casadi.SX, b: casadi.SX) -> casadi.SX:
                return casadi.vertcat(
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                )

            upper_vec = elbow_pos - shoulder_pos
            forearm_vec = wrist_pos - elbow_pos
            normal_raw = _casadi_cross(upper_vec, forearm_vec)
            normal_unit = normal_raw / casadi.sqrt(casadi.sumsqr(normal_raw) + self._swivel_eps)
            self._elbow_normal_fun = casadi.Function("elbow_normal", [self.cq], [normal_unit])

            normal_ref_vec = casadi.MX(casadi.DM(self._ref_elbow_normal))
            axis_ref_vec = casadi.MX(casadi.DM(self._ref_elbow_axis))
            normal_unit_expr = self._elbow_normal_fun(self.var_q)
            angle_num = casadi.dot(axis_ref_vec, casadi.cross(normal_ref_vec, normal_unit_expr))
            angle_den = casadi.dot(normal_ref_vec, normal_unit_expr)
            swivel_delta = casadi.atan2(angle_num, angle_den)

            self.param_swivel_limit = self.opti.parameter()
            self.opti.subject_to(swivel_delta <= self.param_swivel_limit)
            self.opti.subject_to(-self.param_swivel_limit <= swivel_delta)

        lower_bound = self.reduced_robot.model.lowerPositionLimit
        upper_bound = self.reduced_robot.model.upperPositionLimit
        self.opti.subject_to(self.opti.bounded(lower_bound, self.var_q, upper_bound))

        self.param_step_lower = self.opti.parameter(self.nq)
        self.param_step_upper = self.opti.parameter(self.nq)
        self.opti.subject_to(self.param_step_lower <= self.var_q)
        self.opti.subject_to(self.var_q <= self.param_step_upper)
        self.opti.minimize(total_cost)

        opts = {
            "ipopt": {
                "print_level": 0,
                "max_iter": int(solver_max_iter),
                "tol": float(solver_tol),
            },
            "print_time": False,
        }
        self.opti.solver("ipopt", opts)

    # -----------------------------
    # 钩子接口：子类可覆盖
    # -----------------------------
    def _before_solve(self, target: np.ndarray) -> None:
        """求解前回调，子类可用于更新可视化。"""

    def _after_solve_success(self, q: np.ndarray, target: np.ndarray) -> None:
        """求解成功后的回调。"""

    def _after_solve_failure(self, error: Exception, target: np.ndarray) -> None:
        """求解失败后的回调。"""

    # -----------------------------
    # 公共接口
    # -----------------------------
    @staticmethod
    def _sanitize_joint_weights(
        weights: Optional[Iterable[float] | Dict[int | str, float]],
        size: int,
        name: str,
    ) -> np.ndarray:
        if weights is None:
            return np.ones(size, dtype=float)

        if isinstance(weights, dict):
            arr = np.ones(size, dtype=float)
            for key, value in weights.items():
                if isinstance(key, str):
                    key_lower = key.lower().replace("joint", "")
                    if key_lower.startswith("j"):
                        key_lower = key_lower[1:]
                    idx = int(key_lower) - 1
                else:
                    idx = int(key)
                if not 0 <= idx < size:
                    raise ValueError(f"{name} index {key} out of range for nq={size}")
                arr[idx] = float(value)
            if np.any(arr <= 0):
                raise ValueError(f"{name} must contain positive weights")
            return arr

        arr = np.asarray(list(weights), dtype=float).reshape(-1)
        if arr.shape[0] != size:
            raise ValueError(f"{name} length mismatch: expect {size}, got {arr.shape[0]}")
        if np.any(arr <= 0):
            raise ValueError(f"{name} must contain positive weights")
        return arr

    @staticmethod
    def _sanitize_trust_region(
        region: Optional[Iterable[float] | float],
        size: int,
    ) -> Optional[np.ndarray]:
        if region is None:
            return None

        if np.isscalar(region):
            value = float(region)
            if value <= 0:
                raise ValueError("trust_region 必须为正数或 None")
            return np.full(size, value, dtype=float)

        arr = np.asarray(list(region), dtype=float).reshape(-1)
        if arr.size == 1:
            value = float(arr[0])
            if value <= 0:
                raise ValueError("trust_region 必须为正数或 None")
            return np.full(size, value, dtype=float)
        if arr.size != size:
            raise ValueError(f"trust_region 大小不匹配: 期望 {size}, 实际 {arr.size}")
        if np.any(arr <= 0):
            raise ValueError("trust_region 中必须全部为正数")
        return arr

    @staticmethod
    def _sanitize_swivel_limit(limit: Optional[float]) -> Optional[float]:
        if limit is None:
            return None
        value = float(limit)
        if value <= 0:
            return None
        if value > np.pi:
            value = np.pi
        return value

    def set_seed(self, q_seed: Iterable[float]) -> None:
        q_arr = np.asarray(list(q_seed), dtype=float).reshape(-1)
        if q_arr.shape[0] != self.reduced_robot.model.nq:
            raise ValueError("seed dim mismatch")
        self.q_seed = q_arr.copy()

    def set_gripper_state(self, closed: bool) -> None:
        """基类仅记录闭合状态，具体行为由派生类实现。"""

        self._gripper_closed_state = bool(closed)

    def solve(self, target: pin.SE3 | np.ndarray, check_collision: bool = True) -> Tuple[Optional[np.ndarray], bool, str]:
        if isinstance(target, pin.SE3):
            T = target.homogeneous
        else:
            T = np.asarray(target, dtype=float)
            if T.shape != (4, 4):
                raise ValueError("target must be SE3 or (4,4) array")

        self.opti.set_initial(self.var_q, self.q_seed)
        self.opti.set_value(self.param_tf, T)
        if self.param_q_last is not None:
            self.opti.set_value(self.param_q_last, self.q_last)

        if self.param_step_lower is not None and self.param_step_upper is not None:
            if self.trust_region is not None:
                lower = np.maximum(self._joint_lower, self.q_last - self.trust_region)
                upper = np.minimum(self._joint_upper, self.q_last + self.trust_region)
            else:
                lower = self._joint_lower.copy()
                upper = self._joint_upper.copy()
            if self.constraint_manager is not None:
                lower, upper = self.constraint_manager.adjust_bounds(self.q_last, lower, upper)
            self.opti.set_value(self.param_step_lower, lower)
            self.opti.set_value(self.param_step_upper, upper)

        if self.param_swivel_limit is not None and self.swivel_limit is not None:
            self.opti.set_value(self.param_swivel_limit, self.swivel_limit)

        self._before_solve(T)

        try:
            self.opti.solve_limited()
            q = np.asarray(self.opti.value(self.var_q)).reshape(-1)
            self.q_last = q.copy()
            self.q_seed = q.copy()
            if self.constraint_manager is not None:
                self.constraint_manager.update_after_solve(q)

            success = True
            info = "ok"
            if check_collision and self.is_self_collision(q):
                success = False
                info = "self-collision detected"

            self._after_solve_success(q, T)
            return q, success, info

        except Exception as exc:  # pylint: disable=broad-except
            self._after_solve_failure(exc, T)
            return None, False, f"solve failed: {exc}"

    # -----------------------------
    # 几何工具函数
    # -----------------------------
    def _compute_elbow_normal(self, q: np.ndarray) -> Optional[np.ndarray]:
        data = self.reduced_robot.data
        pin.forwardKinematics(self.reduced_robot.model, data, q)
        shoulder = data.oMi[self.shoulder_joint_id].translation
        elbow = data.oMi[self.elbow_joint_id].translation
        wrist = data.oMi[self.wrist_joint_id].translation

        upper = elbow - shoulder
        forearm = wrist - elbow
        normal = np.cross(upper, forearm)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            return None
        return normal / norm

    def _compute_elbow_axis(self, q: np.ndarray) -> Optional[np.ndarray]:
        data = self.reduced_robot.data
        pin.forwardKinematics(self.reduced_robot.model, data, q)
        elbow = data.oMi[self.elbow_joint_id].translation
        wrist = data.oMi[self.wrist_joint_id].translation
        axis = wrist - elbow
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            return None
        return axis / norm

    def is_self_collision(self, q: np.ndarray, gripper: Optional[np.ndarray] = None) -> bool:
        if gripper is None:
            gripper = np.zeros(2)
        q_full = np.concatenate([q.reshape(-1), gripper.reshape(-1)], axis=0)
        pin.forwardKinematics(self.robot.model, self.robot.data, q_full)
        pin.updateGeometryPlacements(
            self.robot.model, self.robot.data, self.geom_model, self.geometry_data
        )
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        return bool(collision)


__all__ = ["BaseArmIK", "rpy_to_quat", "xyzrpy_to_SE3"]
