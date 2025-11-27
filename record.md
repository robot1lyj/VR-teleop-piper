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
