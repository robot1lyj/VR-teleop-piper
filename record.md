# 改造记录（持续补充）

## 2025-11-26：双臂基础改造（去 Meshcat 后的第一轮重构）

### 架构概览
```
VR/WebRTC payload
      │
      ▼
IncrementalPoseMapper (per-hand rotation, filters)
      │
      ▼
ArmTeleopSession + HandRegistry (per-hand IK)
      │
      ▼
TeleopPipeline (per-hand参考基准)
      │
      ▼
PiperTeleopPipeline (队列/滤波/遥测/硬件下发)
```

### 改动内容
- **per-hand IK 管理**：新增 `HandRegistry`，`ArmTeleopSession` 支持为左右手维护独立 IK 状态，避免 `q_last/trust_region` 互相污染。
- **强类型配置**：新增 `TeleopConfig`，在 `scripts/teleop_common.py` 中集中解析 per-hand 安装角、关节约束、滤波参数，生成 per-hand 旋转矩阵与 IK。
- **映射增强**：`IncrementalPoseMapper` 支持按手柄提供 `rotation_vr_to_base`，握持重置时清空滤波历史。
- **管线组装**：`TeleopPipeline` 支持 per-hand 参考位姿；`PiperTeleopPipeline` 兼容 per-hand 参考与 IK。
- **配置样例**：`configs/piper_teleop_dual.json` 增加 `hand_mount_rpy_deg`（左 -75° / 右 +75°）与 `hand_joint_constraints`，左/右臂硬件参数拆分。

### 改动必要性
- 为双臂适配打基础：左右臂的安装角、硬约束、IK 状态需要隔离，单 IK/单矩阵会导致互相干扰与约束冲突。
- 提升可读性与可复现性：配置集中、类型明确，运行日志可打印 per-hand 生效参数，便于现场对齐与回归。
- 消除隐式共享状态：将 per-hand 旋转矩阵、q_last、信赖域/约束封装到 HandRegistry + TeleopConfig，减少跨文件散落的分支逻辑。

### 待续方向
- 把硬件适配（Piper bus/遥测）与控制平面进一步解耦，提供 mock 硬件以便 CI 回放。
- 增加配置校验/指纹打印及最小回放测试，确保每次修改可快速复现。

## 2025-11-27：配置精简与起始姿态对齐

### 配置加载链路
```
configs/piper_teleop*.json
      │
      ▼
teleop_config.py (TeleopConfig.from_args)  <- hand_mount_rpy_deg / hand_home_q_deg
      │
      ▼
teleop_common.build_session -> per-hand IK seeds +参考位姿
      │
      ▼
PiperTeleopPipeline (硬件总线覆盖 piper_arms)
```

### 改动内容
- 删除未使用的配置项：`no_meshcat`、`mount_offset`、`replay*`，避免误导与重复。
- 统一安装姿态写法：单臂配置改用 `hand_mount_rpy_deg` 明确手侧，去掉全局空 `mount_rpy_deg`。
- 起始姿态单一来源：将 `home_q_deg` 与硬件 `init_joint_position` 对齐，双臂配置使用 `hand_home_q_deg` 分别 warm-start。
- 暂时移除夹爪扭矩采样：去掉 `effort_samples/interval/mode` 相关 CLI、配置与写入线程逻辑，避免下发阻塞，后续需要再按需并行重引入。
- 关节滤波恢复二阶阻尼：`JointCommandFilter` 回到 PD（临界阻尼）+限速/限加方案，避免关节层摆动；如需降延迟可在未来改为可配置开关。
- 新增双臂采集配置：增加 `configs/piper_recording_dual.json`，与双臂 teleop 配置保持一致（per-hand 安装角/起始姿态/约束/速度限幅），并内置右腕相机示例配置。

### 改动必要性
- 减少噪声字段，降低“填但未用”的认知成本。
- 明确 per-hand 安装角/起始角，避免重握时跳回中性位或左右共用姿态导致漂移。
- 为后续配置校验与自动指纹打印打基础，便于上线前快速核对现场参数。
