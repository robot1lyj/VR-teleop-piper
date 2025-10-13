# 控制链路函数与命名说明

## robot/constraints.py
- `JointConstraintManager.from_config(model, config)`：解析 JSON/字典形式的关节约束配置，生成统一管理器；常配合 `joint_constraints` 字段使用。
- `JointConstraintManager.adjust_bounds(q_ref, lower, upper)`：在每次 IK 求解前调用，融合 URDF 上/下限、信赖域与步长/硬区间约束，返回最新界限。
- `JointConstraintManager.update_after_solve(q_solution)`：求解成功后刷新内部滤波状态，使步长限制能够跟踪最新姿态。
- `JointConstraintManager._build_joint_index_map(model)`：根据 Pinocchio 模型生成关节索引映射，支持 `joint4` 这类名称引用。

## robot/ik/base_solver.py
- `BaseArmIK.__init__(..., swivel_limit, trust_region, joint_constraints, ...)`：构建 Pinocchio + CasADi 的核心 IK 求解器；`swivel_limit` 为肘部 ±范围（弧度），`trust_region` 控制单步最大改变量，`joint_constraints` 接受与配置文件一致的字典。默认的末端 Frame `ee` 先绕 Y 轴 -90°，再沿旋转后 `+X` 偏移 0.13 m（距离 130 mm）。
- `BaseArmIK.solve(target, check_collision)`：给定 `pin.SE3` 目标求解六关节角度，内部调用 `JointConstraintManager` 调整步长/范围，并在成功时更新 `q_last`、`q_seed`。
- `BaseArmIK._sanitize_trust_region(region, size)`：把标量/列表转成长度为关节数的数组，单位弧度。
- `BaseArmIK._sanitize_swivel_limit(limit)`：将肘部范围裁剪到 `(0, π]`；`None` 表示关闭。

## robot/ik/meshcat_solver.py
- `MeshcatArmIK.__init__(..., enable_viewer)`：在 `BaseArmIK` 的基础上带上 Meshcat 可视化；如果 Meshcat 依赖缺失会自动降级。
- `MeshcatArmIK.set_visual_base_transform(rotation, translation)`：设置 Meshcat 中的基座姿态（旋转矩阵 + 平移向量），用于模拟不同安装方式。
- `MeshcatArmIK.refresh_visual(q)`：手动刷新关节角到 Meshcat，常用于更新初始姿态或加载回放数据。

## scripts/run_vr_meshcat.py
- `build_session(args)`：解析配置后构建遥操作会话；负责把 `swivel_range_deg`→弧度，并传入 `joint_constraints`。
- `_parse_joint_constraints_arg(value)`：支持从 JSON 字符串或配置字典读取约束；格式例如 `{"step_limits_deg":{"joint4":15}}`。
- `_replay_trajectory_sync(session, trajectory, speed, loop_playback)`：离线回放工具，便于在 Meshcat/离线模式下验证 IK 与约束行为。
