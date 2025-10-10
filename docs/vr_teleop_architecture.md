# VR 增量遥操作架构

```mermaid
graph LR
    subgraph VR前端
        A[手柄传感器\nweb-ui/vr_app.js]
        B[WebRTC DataChannel]
    end
    subgraph Python服务
        C[VRWebRTCServer\ncontroller_stream.py]
        D[IncrementalPoseMapper\nrobot/teleop/incremental_mapper.py]
        E[ArmTeleopSession + ArmIK\nrobot/teleop/session.py]
        G[轨迹录制/回放脚本\nscripts/record_vr_trajectory.py & run_vr_meshcat.py]
    end
    F[机械臂执行栈\n(关节控制器/日志)]

    A --> B --> C
    C -->|VR 姿态 JSON| D
    D -->|R_bv 映射后的 SE(3) 目标| E
    E -->|关节角/夹爪指令| F
    C -->|离线写入| G
    G -->|轨迹回放| D
```

- VR 前端按 50 Hz 推送手柄位置/四元数，`controller_stream.py` 负责接收。
- `IncrementalPoseMapper` 读取手柄增量，套用固定矩阵 `R_bv` 将位置和旋转转换到机械臂基座系。
- `ArmTeleopSession` 将目标姿态送入 `ArmIK.solve`，获取 Piper 机械臂的关节解，并附带夹爪状态。
- 下游可以根据 `TeleopResult` 中的关节矢量选择推送至实时控制器或写入调试日志。 
- `record_vr_trajectory.py` 支持单独录制 DataChannel 帧，`run_vr_meshcat.py --replay` 可直接离线回放轨迹验证映射与 IK。 
