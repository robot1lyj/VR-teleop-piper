# 真实机器人适配教程

本指南基于仓库默认的 VR 手柄 → IK 遥操作“基线”实现，逐步说明如何将管线对接到真实机械臂。本教程假定你已经能在本地成功运行 `scripts/run_vr_meshcat.py` 并确认 Meshcat 中的模型能随手柄运动；接下来我们将把同样的轨迹送到实体机器人。快速接入可直接使用 `scripts/run_vr_piper.py`（默认读取 `configs/piper.json`，内含 VR + Piper 硬件的统一参数），该脚本已将 VR 管线与 Piper SDK 串联，可按需参考下文细节继续自定义。

## 1. 整体数据链路回顾

```
VR 头显/浏览器 ── WebRTC DataChannel ──▶ vr_runtime/controller_pipeline.py
                                                  │
                                                  ▼
                                   robot/teleop/ArmTeleopSession
                                   （增量映射 + BaseArmIK 求解）
                                                  │
                                                  ▼
                                 自定义 Robot Driver（关节/笛卡尔指令）
                                                  │
                                                  ▼
                                           实体机械臂 + 手爪
```

核心模块：

- `robot/teleop/incremental_mapper.py`：将 VR 手柄增量映射为机械臂基座系下的位置/姿态增量。
- `robot/teleop/session.py`：把增量结果喂给 `BaseArmIK`，得到 6 轴关节角及 `gripper_closed` 状态。
- `robot/ik/base_solver.py`：基于 Pinocchio + CasADi 的迭代 IK，默认末端 Frame 名称为 `ee`，具备用于 Piper 机械臂的工具偏置。
- `robot/ik/meshcat_solver.py`：在基类基础上附加 Meshcat 可视化与手爪开闭的显示能力（真实机器人时可继续复用读取关节结果）。

> **提示**：对接真实机器人时，Meshcat 仍是宝贵的可视化验证工具。建议在推送实体指令前，优先确保 Meshcat 显示的姿态与期待一致。

## 2. 准备工作

1. **依赖环境**：
   ```bash
   pip install aiortc websockets numpy pin python-casadi meshcat
   ```
   同时准备供应商 SDK / ROS 驱动等发送关节指令所需的库。

2. **URDF/几何**：
   - 将真实机械臂的 URDF（含正确的惯量与关节限制）放入仓库可访问的位置，例如 `piper_description/urdf/real_arm.urdf`。
   - 若末端工具与默认的“夹爪朝向 + 0.13 m”不同，请确认 `BaseArmIK` 初始化参数：
     ```python
     MeshcatArmIK(
         urdf_path="...",
         add_ee_on_joint="joint6",
         add_ee_translation=(tx, ty, tz),
         add_ee_rpy=(roll, pitch, yaw),
     )
     ```
     其中 `(tx, ty, tz)` / `(roll, pitch, yaw)` 使用米 / 弧度。真实工具长度或角度变化只需修改此处，无需改 URDF。

3. **网络与安全**：
   - 建议全链路在可信局域网运行，沿用 `--no-stun`；远程部署需额外加固 TLS、认证与开机限位。
   - 机械臂应启用硬件急停、软件安全区、速度/力矩限制等保护措施，首次联调请在低速、关节空间测试模式下进行。

## 3. 标定机械臂基座姿态

真实机械臂的安装姿态可能与仿真不同，需要确保 VR → 机械臂基座的旋转映射正确。

1. **采集当前姿态**：保持机械臂在“出厂零位”或易于辨识的固定姿态，记录末端实际朝向与位置。
2. **配置 `run_vr_meshcat.py`**：
   - 在 `configs/run_vr_meshcat.json` 新增或修改：
     ```json
     {
       "urdf": "piper_description/urdf/real_arm.urdf",
       "mount_rpy_deg": [rx_deg, ry_deg, rz_deg],
       "mount_offset": [ox, oy, oz]
     }
     ```
   - `mount_rpy_deg` 代表机械臂基座绕 XYZ 的安装角（度），`mount_offset` 是基座在 Meshcat 中的视觉平移（米）。
   - 调整参数直至 Meshcat 模型与实体朝向一致；同样的旋转矩阵会传给增量映射，保证 VR 坐标与实体基座对齐。

> **快速校准技巧**：使用激光/水平仪对齐机械臂前向，与 Meshcat 中的 `+X`/`+Y` 轴比对；必要时在控制器上微动并观察实体末端移动方向是否与 VR 中一致。

## 4. 末端工具偏置与手爪开闭

- `BaseArmIK` 默认追加的末端 Frame 已实现：先绕局部 Y 轴 -90°，再沿新 X 方向平移 0.13 m（已在项目中更新）。若夹爪实际长度或朝向不同，修改 `add_ee_translation` / `add_ee_rpy` 即可。
- 现在扳机状态会随 `TeleopGoal.gripper_closed` 传递到 IK：
  - `True` → `joint7/joint8` 收回（默认 0）
  - `False` → 打开到各自极限的 90%
- 若真实手爪由外部控制器驱动，可在下游读取 `TeleopResult.gripper_closed` 并转成开合命令；Meshcat 的可视化变化可用来确认触发逻辑无误。

## 5. 编写真实机器人驱动脚本

以下示例展示如何复用现有管线，把 IK 结果发送到实体机器人。假设你提供了两个函数：

- `robot_iface.get_joint_positions()`：读取当前 6 轴关节角。
- `robot_iface.command_joint_positions(target_q, duration=0.2)`：在给定时长内插补到目标角度。
- `robot_iface.set_gripper(closed: bool)`：控制手爪开闭。

```python
from robot.teleop import ArmTeleopSession, IncrementalPoseMapper
from robot.ik import BaseArmIK
from vr_runtime.webrtc_endpoint import VRWebRTCServer

ik = BaseArmIK(
    urdf_path="piper_description/urdf/real_arm.urdf",
    add_ee_translation=(0.0, 0.0, 0.145),  # 示例：工具长度 145 mm
    add_ee_rpy=(0.0, -1.5708, 0.0),
    joint_reg_weights=[5, 1, 1, 5, 1, 1],
    smooth_weight=0.05,
)

# 使用读取到的当前姿态作为 IK 种子 & 参考位姿
q_home = robot_iface.get_joint_positions()
ik.set_seed(q_home)

mapper = IncrementalPoseMapper(rotation_vr_to_base=your_rotation_matrix)
session = ArmTeleopSession(ik_solver=ik, mapper=mapper)
session.set_reference_pose("right", position=ref_pos, rotation=ref_rot)

# 组装到 TeleopPipeline/VRWebRTCServer 中
...

def handle_results(results):
    for item in results:
        if item.success and item.joints is not None:
            robot_iface.command_joint_positions(item.joints)
            robot_iface.set_gripper(item.gripper_closed)
```

关键要点：

- **初始参考位姿**：将真实机器人当前末端姿态写入 `session.set_reference_pose`，使增量映射以实体的“零点”作为基准。可以在上电后先读取一次关节角，利用 Pinocchio 计算末端位姿。
- **单位统一**：`scripts/run_vr_piper.py` 与 `PiperMotorsBus` 以弧度（rad）接收关节目标，在写入前自动转换成控制器要求的 0.001°。配置文件 `configs/piper.json` 中的 6 个关节角建议用角度（deg）填写，驱动加载时会统一转成弧度；`gripper_open` / `gripper_closed` 仍按 SDK 约定维持线性位移（米）。
- **平滑/信赖域**：真实机械臂对突变指令更敏感，建议在 `configs/run_vr_meshcat.json` 或脚本中开启：
  - `smooth_weight`（默认 0.05）+ `joint_smooth_weights`，抑制帧间大幅摆动。
  - `trust_region`，限制每帧关节步长，例如 `0.1`（弧度）或 `[0.1, 0.1, ...]` 列表。
  - `joint_constraints` 中的 `step_limits_deg` 可以单独控制肘、腕的瞬时改变量。
- **命令频率**：VR 手柄默认 50 Hz；对真实机器人可根据控制接口降采样，例如每隔两帧发送一次，以减小网络抖动影响。

## 6. 运行步骤建议

1. **离线验证**
   - 启动 `scripts/run_vr_meshcat.py`，调节 `mount_rpy_deg`、工具偏置等直至视觉一致。
   - 录制一段 VR 轨迹（`scripts/record_vr_trajectory.py`），再用 `--replay` 回放确认 IK 求解稳定。

2. **联调前静态检查**
   - 将真实机械臂置于安全空间并上电，使之保持在参考姿态。
   - 运行真实 robot driver 脚本但暂不发送指令，只打印 `TeleopResult` 验证触发逻辑。

3. **低速模式试运行**
   - 降低 `command_joint_positions` 的目标速度或拉大插补时长，例如 0.5 s。
   - 在安全员监控下进行小范围操作，逐步扩大运动范围。

4. **常规操作**
   - 每次操作前重新调用 `session.set_reference_pose` 锁定新的基准。
   - 保持 Meshcat 界面开启，随时对照模型与真实动作。

## 7. 常见问题排查

| 现象 | 排查方向 |
| ---- | -------- |
| 扳机开闭与手爪动作相反 | 检查 `robot_iface.set_gripper` 中闭合逻辑，必要时取反 `item.gripper_closed`。 |
| 末端运动方向与预期不符 | 再次确认 `rotation_vr_to_base`、`mount_rpy_deg`；必要时在控制器上做三个轴的独立位移，观察实体响应。 |
| IK 失败或抖动 | 调高 `reg_weight` / `smooth_weight`，或缩小 `trust_region`；确认初始参考姿态不处于奇异位。 |
| 实体姿态与 Meshcat 模型常有差值 | 确保 `command_joint_positions` 执行到位后再刷新 `session.set_reference_pose`；也可定期从真实关节角反算位姿并重置参考。 |
| 手爪几何与 Meshcat 不一致 | 修改 IK 初始化的 `add_ee_translation`/`add_ee_rpy`，或为 URDF 添加更贴合的末端模型。 |

## 8. 后续扩展建议

- **双手协作**：允许运行两个 `ArmTeleopSession` 实例分别控制左右机械臂，或在单体机器人内映射不同手柄功能。
- **力/触觉反馈**：在 `TeleopResult.info` 中扩展额外字段（如碰撞检测结果），再通过 WebRTC 反向发送给前端。
- **ROS/工业现场集成**：可将 `TeleopResult` 发布为 ROS 话题，或映射成 Modbus/工业以太网指令，与现有 PLC/机器人控制器对接。

> **总结**：真实机器人适配的关键是“对齐参考 + 限制速度 + 逐步验证”。把 VR 管线当成笛卡尔指令生成器，先在 Meshcat 中确保姿态正确，再小心地把同样的 IK 结果发给实体即可。若严格按照上面的步骤逐项验证，便能将仓库的基线方案安全地迁移到真实机械臂上。
