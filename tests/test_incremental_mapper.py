import numpy as np

from robot.teleop.incremental_mapper import (
    IncrementalPoseMapper,
    R_BV_DEFAULT,
    _quaternion_to_matrix,
)


def test_incremental_mapper_position_and_rotation():
    mapper = IncrementalPoseMapper(scale=1.0)
    mapper.set_reference_pose("left", np.array([0.4, 0.0, 0.2]), np.eye(3))

    # 首帧锁定原点
    mapper.process(
        {
            "leftController": {
                "position": {"x": 0.1, "y": 0.2, "z": -0.3},
                "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "gripActive": True,
                "trigger": 0.0,
            }
        }
    )

    # 第二帧：向 VR 的 -Z 方向移动 0.1 m，并绕 X 轴旋转 90°
    half_pi = np.pi / 2.0
    mapper_output = mapper.process(
        {
            "leftController": {
                "position": {"x": 0.1, "y": 0.2, "z": -0.4},
                "quaternion": {
                    "x": np.sin(half_pi / 2.0),
                    "y": 0.0,
                    "z": 0.0,
                    "w": np.cos(half_pi / 2.0),
                },
                "gripActive": True,
                "trigger": 0.0,
            }
        }
    )

    assert len(mapper_output) == 1
    goal = mapper_output[0]

    # 位置检查：VR -Z -> 基座 +X
    np.testing.assert_allclose(goal.position, np.array([0.5, 0.0, 0.2]), atol=1e-6)

    # 旋转检查：使用同样的映射公式验证
    rot_rel_vr = _quaternion_to_matrix(
        np.array([
            np.sin(half_pi / 2.0),
            0.0,
            0.0,
            np.cos(half_pi / 2.0),
        ])
    )
    expected_rot = R_BV_DEFAULT @ rot_rel_vr @ R_BV_DEFAULT.T
    np.testing.assert_allclose(goal.rotation, expected_rot, atol=1e-6)

    assert goal.gripper_closed is True
