# 遥操作运动学与坐标系说明

## 坐标系与符号
- **$W_{vr}$（VR 世界）**：由 WebXR/A-Frame 提供的舞台坐标系，`x` 指向右手侧，`y` 向上，`z` 指向观察者。浏览器端上报的控制器位置 `position` 与四元数 `quaternion` 均位于该系。
- **$G$（握持局部系）**：用户按下握持键时锁定的控制器瞬时位姿，作为增量的零点。其位置为 $p_G^{W_{vr}}$，朝向为 $R_G^{W_{vr}}$。
- **$B$（机器人基座）**：Pinocchio 模型的根坐标系，即机械臂基座。IK 求解器 `BaseArmIK` 在该系中接收目标位姿。
- **$\text{EE}$（末端工具系）**：在 `BaseArmIK.__init__` 中通过 `addFrame("ee", joint6, …)` 固定在第 6 轴末端的操作框架，IK 目标直接对应此 Frame。默认工具偏置满足 $T_{joint6}^{ee} = R_y(-90^\circ) \cdot \mathrm{Trans}(0.13, 0, 0)$，即末端先绕 Y 轴 -90°，再沿旋转后 `+X` 方向平移 130 mm。

符号使用：$R_X^Y$ 表示从坐标系 $X$ 旋转到 $Y$ 的方向余弦矩阵，$p_X^Y$ 表示点在 $Y$ 系下的坐标。

## 数据链路概览
1. **浏览器采集**（`web-ui/vr_app.js`）
   - 每帧读取左右手控制器的 `object3D.position` / `quaternion`，连同握持、扳机状态通过 WebRTC DataChannel 发送。
2. **遥操作增量映射**（`robot/teleop/incremental_mapper.py`）
   - `IncrementalPoseMapper` 为每只手维持一个 `ControllerState` 与可选的参考位姿 `reference_poses`（由机器人当前末端姿态写入）。
   - 当握持键初次按下时锁定 $G$，随后使用增量映射将 VR 位姿转换到机械臂基座下的目标位姿。
3. **IK 目标求解**（`robot/teleop/session.py`）
   - `ArmTeleopSession.handle_vr_payload` 调用 mapper 获得 `TeleopGoal`，并将 `goal.rotation` / `goal.position` 直接封装为 `pin.SE3` 传入 `BaseArmIK.solve`。
4. **关节角输出**
   - `BaseArmIK.solve` 求解得到的关节角 `q` 回写给上层，用于驱动机器人或仿真。

## 增量坐标变换细节
`IncrementalPoseMapper` 的核心步骤如下：

1. **握持锁定**
   ```text
   p_G^{W_{vr}} = position_vec
   q_G = quaternion_payload
   ```
   `ControllerState` 记录握持时的原点与朝向。

2. **增量计算**
   ```text
   \Delta p^{W_{vr}} = p_{now}^{W_{vr}} - p_G^{W_{vr}}
   \Delta R^{W_{vr}} = R(q_{now} q_G^{-1})
   ```
   其中 `R()` 将四元数转换成旋转矩阵。

3. **坐标变换：VR → 基座**
   默认常数矩阵 `R_{B}^{W_{vr}} =` `R_BV_DEFAULT`
   ```text
   \Delta p^{B} = R_{B}^{W_{vr}} \; \Delta p^{W_{vr}}
   R_{rel}^{B} = R_{B}^{W_{vr}} \; \Delta R^{W_{vr}} \; (R_{B}^{W_{vr}})^T
   ```
   `R_BV_DEFAULT` 将 VR 的 `+z` 映射为机器人 `-x`，`+x` 映射为机器人 `-y`，`+y` 映射为机器人 `+z`，满足机械臂基座前进/左/上的右手系约定。

4. **拼接参考位姿**
   设握持开始前记录的末端参考位姿为 $\{p_{ref}^B, R_{ref}^B\}$，则目标位姿为：
   ```text
   p_{target}^B = p_{ref}^B + s \; \Delta p^{B}
   R_{target}^B = R_{ref}^B \; R_{rel}^{B}
   ```
   其中 `s = self.scale` 是增益系数（默认 1.0）。

5. **握持/扳机状态**
   - 扳机值通过 `controller.trigger_active` 映射为 `TeleopGoal.gripper_closed`，IK 求解器不直接处理，但上层可用于手爪命令。

## IK 目标帧来源
- `ArmTeleopSession` 在 `handle_vr_payload` 中将 `goal.position` / `goal.rotation` 打包为 `pin.SE3(goal.rotation, goal.position)`。
- 该目标位姿明确位于机器人基座系 $B$，并对应 `BaseArmIK` 内添加的 `ee` Frame。
- 因此 IK 求解的目标帧是“机械臂基座系下的末端工具框架”，与 VR 头显或基站无直接关系。

## 无需显式头显基站变换的原因
- WebXR 已经在浏览器端把手柄与头显 pose 统一到同一舞台坐标系 $W_{vr}$；A-Frame 的控制器实体 `object3D.position` 直接给出该世界坐标。
- 本项目采用**增量映射**：握持键按下时锁定 $G$，之后仅使用相对位移/旋转 $\Delta p^{W_{vr}}$、$\Delta R^{W_{vr}}$。任何整体平移（例如头显与基站的偏差）都会在差分中抵消。
- 真正需要与机械臂对齐的只有轴向约定，故用固定矩阵 `R_{B}^{W_{vr}}` 即可；若实际安装方向不同，只需修改/配置该矩阵，而无需引入头显基站的动态姿态。
- 相比其它项目的“控制器 → 基站 → 机器人”两段式转换，我们的设计将第一段省略，依赖 WebXR 的统一坐标与握持参考来消除误差。

## 检查清单
- 若发现位置方向反了，优先验证 `rotation_vr_to_base`（即 `R_BV_DEFAULT`）是否符合实际安装。
- 如需绝对标定，可在握持前通过额外流程更新 `reference_poses`，而无需修改 IK 求解器。
- 任何新数据流或坐标映射应记录在 `README.md` 的遥操作部分，保持与本文一致。
