"""
ArmIK: a ROS‑free inverse kinematics helper built on Pinocchio + CasADi.

• Loads Piper URDF, locks finger joints (joint7, joint8) by default, adds a convenience 'ee' frame on joint6.
• Solves IK with an Opti (Ipopt) problem on SE(3) log residuals.
• Optional Meshcat visualization (off by default).
• No ROS, no piper_msgs; you call ArmIK.solve(...) directly and handle the result.

Usage (example):

    from arm_ik import ArmIK, xyzrpy_to_SE3

    ik = ArmIK(urdf_path="/path/to/piper_description.urdf", use_meshcat=True)
    T = xyzrpy_to_SE3(0.3, 0.0, 0.2, 0.0, -1.57, 0.0)
    q, ok, info = ik.solve(T)
    if ok:
        print("q (rad):", q)
    else:
        print("IK failed:", info)

"""
from __future__ import annotations
import os
import numpy as np
import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper

try:
    # Meshcat is optional; keep import local and guarded
    import meshcat.geometry as mg
    from pinocchio.visualize import MeshcatVisualizer
    _HAS_MESHCAT = True
except Exception:
    _HAS_MESHCAT = False

# -----------------------------
# Utilities
# -----------------------------

def rpy_to_quat(roll: float, pitch: float, yaw: float) -> pin.Quaternion:
    """XYZ (sxyz) Euler to Pinocchio Quaternion (w, x, y, z).
    We avoid tf.transformations and keep deps light.
    """
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    # ZYX? We want sxyz Roll->Pitch->Yaw intrinsic; equivalent to XYZ extrinsic
    # The original code used tf.transformations.quaternion_from_euler(roll, pitch, yaw, axes='sxyz')
    # We'll reproduce sxyz directly:
    w = cr*cp*cy - sr*sp*sy
    x = sr*cp*cy + cr*sp*sy
    y = cr*sp*cy - sr*cp*sy
    z = cr*cp*sy + sr*sp*cy
    return pin.Quaternion(w, x, y, z)


def xyzrpy_to_SE3(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> pin.SE3:
    q = rpy_to_quat(roll, pitch, yaw)
    return pin.SE3(q, np.array([x, y, z], dtype=float))


# -----------------------------
# IK Class (no ROS)
# -----------------------------

class ArmIK:
    def __init__(
        self,
        urdf_path: str,
        joints_to_lock: list[str] | None = None,
        add_ee_on_joint: str = "joint6",
        add_ee_translation = (0.0, 0.0, 0.0),
        add_ee_rpy = (0.0, 0.0, 0.0),
        position_weight: float = 20.0,
        orientation_weight: float = 20.0,
        reg_weight: float = 0.01,
        smooth_weight: float | None = None,  # if set, adds ||q - q_last||^2
        joint_reg_weights: np.ndarray | list[float] | None = None,
        joint_smooth_weights: np.ndarray | list[float] | None = None,
        solver_max_iter: int = 50,
        solver_tol: float = 1e-4,
        use_meshcat: bool = False,
    ) -> None:
        """
        Build reduced robot, SE(3) log-residual cost, and an Ipopt Opti.

        position_weight/orientation_weight scale the 6D log residual as [w_p, w_p, w_p, w_o, w_o, w_o].
        """
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self.robot: RobotWrapper = RobotWrapper.BuildFromURDF(urdf_path)
        if joints_to_lock is None:
            joints_to_lock = ["joint7", "joint8"]  # lock gripper sliders by default

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=joints_to_lock,
            reference_configuration=np.zeros(self.robot.model.nq),
        )
        self.nq = self.reduced_robot.model.nq

        # 逐关节正则/平滑权重；默认 1.0，后续可针对特定关节提高惩罚力度抑制抖动
        self.joint_reg_weights = self._sanitize_joint_weights(joint_reg_weights, self.nq, "joint_reg_weights")
        if smooth_weight is not None:
            self.joint_smooth_weights = self._sanitize_joint_weights(
                joint_smooth_weights, self.nq, "joint_smooth_weights"
            )
            self._smooth_weights_sq_dm = casadi.DM(np.square(self.joint_smooth_weights)).reshape((self.nq, 1))
        else:
            self.joint_smooth_weights = None
            self._smooth_weights_sq_dm = None
        self._reg_weights_sq_dm = casadi.DM(np.square(self.joint_reg_weights)).reshape((self.nq, 1))

        # Add a convenience end-effector frame on the wrist joint (default joint6)
        q_ee = rpy_to_quat(*add_ee_rpy)
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "ee",
                self.reduced_robot.model.getJointId(add_ee_on_joint),
                pin.SE3(q_ee, np.array(add_ee_translation, dtype=float)),
                pin.FrameType.OP_FRAME,
            )
        )
        # 新增 frame 后需重建 data，确保 oMf 长度与 frame 数一致
        self.reduced_robot.data = self.reduced_robot.model.createData()

        # Build collision model from URDF (optional pairs — you can extend)
        self.geom_model = pin.buildGeomFromUrdf(self.robot.model, urdf_path, pin.GeometryType.COLLISION)
        self.geometry_data = pin.GeometryData(self.geom_model)
        # A few example pairs to test; adapt to your robot indices as needed
        try:
            for i in range(4, 9):
                for j in range(0, 3):
                    self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        except Exception:
            pass  # ignore if indices invalid for the specific URDF

        # Meshcat (optional)
        self.use_meshcat = bool(use_meshcat and _HAS_MESHCAT)
        if self.use_meshcat:
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            # show base & ee frames if exist
            ee_id = self.reduced_robot.model.getFrameId("ee")
            self.vis.displayFrames(True, frame_ids=[ee_id], axis_length=0.12, axis_width=5)
            self.vis.display(pin.neutral(self.reduced_robot.model))
            # add an ee_target axis object
            frame_viz_name = "ee_target"
            try:
                import meshcat.geometry as mg
                FRAME_AXIS_POSITIONS = (
                    np.array([[0,0,0],[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1]], dtype=np.float32).T
                )
                FRAME_AXIS_COLORS = (
                    np.array([[1,0,0],[1,0.6,0],[0,1,0],[0.6,1,0],[0,0,1],[0,0.6,1]], dtype=np.float32).T
                )
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(position=0.1 * FRAME_AXIS_POSITIONS, color=FRAME_AXIS_COLORS),
                        mg.LineBasicMaterial(linewidth=10, vertexColors=True),
                    )
                )
            except Exception:
                pass

        # Seed & history
        self.q_seed = np.zeros(self.nq)
        self.q_last = self.q_seed.copy()

        # --- CasADi symbolic model ---
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        self.ee_fid = self.reduced_robot.model.getFrameId("ee")

        # 6D log residual with separate weights for position/orientation
        # log6 returns [vx, vy, vz, wx, wy, wz]
        log_vec = cpin.log6(self.cdata.oMf[self.ee_fid].inverse() * cpin.SE3(self.cTf)).vector
        W = casadi.diag(casadi.vcat([
            position_weight, position_weight, position_weight,
            orientation_weight, orientation_weight, orientation_weight,
        ]))
        weighted = W @ log_vec
        self.error_fun = casadi.Function("error", [self.cq, self.cTf], [weighted])

        # --- Build Opti ---
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.nq)
        self.param_tf = self.opti.parameter(4, 4)
        self.param_q_last = self.opti.parameter(self.nq) if smooth_weight else None

        cost_tracking = casadi.sumsqr(self.error_fun(self.var_q, self.param_tf))
        cost_reg = reg_weight * casadi.dot(self._reg_weights_sq_dm, casadi.power(self.var_q, 2))
        total_cost = cost_tracking + cost_reg
        if smooth_weight is not None:
            smooth_residual = self.var_q - self.param_q_last
            if self._smooth_weights_sq_dm is not None:
                total_cost = total_cost + smooth_weight * casadi.dot(
                    self._smooth_weights_sq_dm, casadi.power(smooth_residual, 2)
                )
            else:
                total_cost = total_cost + smooth_weight * casadi.sumsqr(smooth_residual)

        # Joint limits from reduced model
        lb = self.reduced_robot.model.lowerPositionLimit
        ub = self.reduced_robot.model.upperPositionLimit
        self.opti.subject_to(self.opti.bounded(lb, self.var_q, ub))
        self.opti.minimize(total_cost)

        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': int(solver_max_iter),
                'tol': float(solver_tol)
            },
            'print_time': False
        }
        self.opti.solver("ipopt", opts)

    # -----------------------------
    # Public API
    # -----------------------------
    @staticmethod
    def _sanitize_joint_weights(
        weights: np.ndarray | list[float] | dict[int | str, float] | None,
        size: int,
        name: str,
    ) -> np.ndarray:
        """标准化逐关节权重，便于统一处理输入格式。"""

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

        arr = np.asarray(weights, dtype=float).reshape(-1)
        if arr.shape[0] != size:
            raise ValueError(f"{name} length mismatch: expect {size}, got {arr.shape[0]}")
        if np.any(arr <= 0):
            raise ValueError(f"{name} must contain positive weights")
        return arr

    def set_seed(self, q_seed: np.ndarray | list[float]) -> None:
        q_seed = np.asarray(q_seed, dtype=float).reshape(-1)
        if q_seed.shape[0] != self.reduced_robot.model.nq:
            raise ValueError("seed dim mismatch")
        self.q_seed = q_seed.copy()

    def solve(self, target: pin.SE3 | np.ndarray, check_collision: bool = True) -> tuple[np.ndarray | None, bool, str]:
        """
        target: pin.SE3 or 4x4 homogeneous np.array
        Returns: (q, success, info)
        """
        if isinstance(target, pin.SE3):
            T = target.homogeneous
        else:
            T = np.asarray(target, dtype=float)
            if T.shape != (4,4):
                raise ValueError("target must be SE3 or (4,4) array")

        # initial guess
        self.opti.set_initial(self.var_q, self.q_seed)
        self.opti.set_value(self.param_tf, T)
        if self.param_q_last is not None:
            self.opti.set_value(self.param_q_last, self.q_last)

        if self.use_meshcat:
            try:
                self.vis.viewer['ee_target'].set_transform(T)
            except Exception:
                pass

        try:
            sol = self.opti.solve_limited()
            q = np.asarray(self.opti.value(self.var_q)).reshape(-1)
            self.q_last = q.copy()
            self.q_seed = q.copy()  # warm-start next call

            ok = True
            info = "ok"

            if check_collision:
                if self.is_self_collision(q):
                    ok = False
                    info = "self-collision detected"

            if self.use_meshcat:
                try:
                    self.vis.display(q)
                except Exception:
                    pass

            return q, ok, info

        except Exception as e:
            # On failure, keep last seed; user may swap seed and retry.
            return None, False, f"solve failed: {e}"

    def is_self_collision(self, q: np.ndarray, gripper: np.ndarray | None = None) -> bool:
        """Very simple pair-based collision check using full model geom.
        If your URDF indexes differ, adapt the pairs in __init__.
        """
        if gripper is None:
            # if you locked joint7/8, assume zeros for them in full model concatenation
            gripper = np.zeros(2)
        q_full = np.concatenate([q.reshape(-1), gripper.reshape(-1)], axis=0)
        pin.forwardKinematics(self.robot.model, self.robot.data, q_full)
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        return bool(collision)


# -----------------------------
# Minimal CLI test (optional)
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ROS‑free IK test for Piper using Pinocchio+CasADi")
    parser.add_argument("--urdf", required=True, help="Path to piper_description.urdf")
    parser.add_argument("--xyzrpy", nargs=6, type=float, default=[0.3, 0.0, 0.2, 0.0, -1.57, 0.0],
                        help="Target pose: x y z roll pitch yaw (rad)")
    parser.add_argument("--meshcat", action="store_true", help="Enable Meshcat viewer")
    args = parser.parse_args()

    x,y,z,rr,pp,yy = args.xyzrpy
    T = xyzrpy_to_SE3(x,y,z,rr,pp,yy)

    ik = ArmIK(
        urdf_path=args.urdf,
        use_meshcat=args.meshcat,
        position_weight=20.0,
        orientation_weight=20.0,
        reg_weight=0.01,
        smooth_weight=0.1,
    )

    q, ok, info = ik.solve(T)
    if ok:
        print("IK q (rad):", np.array2string(q, precision=5))
    else:
        print("IK failed:", info)
