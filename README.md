# VR-New 模块概览（WebRTC 版）

```
├── vr_runtime/                    # VR 采集/信令层（与机器人逻辑解耦）
│   ├── __init__.py                # 导出 ControllerPipeline / VRWebRTCServer
│   ├── controller_pipeline.py     # WebRTC 信令 CLI + ControllerPipeline 实现
│   ├── controller_state.py        # 单手柄状态结构，记录握持/扳机/姿态等信息
│   ├── webrtc_endpoint.py         # WebSocket 信令 + aiortc DataChannel 服务端
│   └── common/
│       └── math.py                # （占位）通用四元数 / 姿态工具
├── robot/                         # VR 目标 → 机械臂运动学
│   ├── teleop/                    # 增量映射 + Teleop 会话
│   └── ik/                        # IK 求解器
└── web-ui/                        # 浏览器侧 A-Frame 客户端
    ├── index.html                 # 信令地址输入、状态显示、A-Frame 场景
    ├── interface.js               # UI 交互与日志面板
    ├── styles.css                 # 深色主题样式
    └── vr_app.js                  # WebRTCBridge 与手柄数据采集
```

## 数据流

```
  浏览器 / 头显 (web-ui/vr_app.js)
        │ 信令：WebSocket (offer/answer/ICE)
        │ 数据：WebRTC DataChannel(JSON，姿态 50 Hz)
        ▼
  vr_runtime/controller_pipeline.py + vr_runtime/webrtc_endpoint.py
        │ ControllerPipeline 解析 -> 更新 ControllerState
        ▼
      标准输出 / 后续机器人控制模块
```

- `ControllerPipeline` 将左/右手柄的位移、四元数转换为目标字典（位置、腕部角度、夹爪状态），方便单元测试复用。
- `VRWebRTCServer` 单客户端模式：保留现有 WebSocket 端口，仅承担信令交换；姿态数据通过名为 `controller` 的 DataChannel 传输。
- `web-ui/vr_app.js` 中的 `WebRTCBridge` 负责创建 `RTCPeerConnection`、管理 DataChannel 并以 20 ms（≈50 Hz）节奏发送手柄姿态。

## 局域网快速上手

1. **安装依赖**（仅需一次）：
   ```bash
   pip install aiortc websockets numpy
   ```
   若使用 Conda 环境，请先激活目标环境。

2. **启动信令 + DataChannel 服务**（局域网可关闭 STUN）：
   ```bash
   PYTHONPATH=. python -m vr_runtime.controller_pipeline \
     --host 0.0.0.0 \
     --port 8442 \
     --no-stun \
     --log-level info
   ```
   - 默认同时追踪双手柄，可用 `--hands left` 或 `--hands right` 仅启用单侧。
   - 如遇到 NAT 导致协商失败，再追加 `--stun stun:stun.l.google.com:19302`。

3. **启动前端界面**（本地文件即可）：
   ```bash
   python -m http.server 8080 --directory web-ui
   ```

4. **在浏览器 / 头显中操作**：
   - 访问 `http://<服务器IP>:8080`。
   - 在输入框填写 `ws://<服务器IP>:8442`，点击「连接」。
   - 成功建立 DataChannel 后点击「开启手柄追踪」，允许浏览器进入 VR/AR 模式。
   - 手柄长按侧键或页面按钮可随时停止追踪；终端会实时打印目标字典。



![image-20250928145800282](https://raw.githubusercontent.com/robot1lyj/image_typora/main/image-20250928145800282.png)

## CLI 选项速查

| 参数 | 说明 |
| ---- | ---- |
| `--host` / `--port` | WebSocket 信令监听地址与端口（默认 `0.0.0.0:8442`）。 |
| `--hands` | `both` / `left` / `right`，限制 ControllerPipeline 处理的手柄。 |
| `--scale` | 位移缩放系数，影响 `target_position`。 |
| `--channel-name` | DataChannel 名称，需与前端 `WebRTCBridge` 中一致。 |
| `--no-stun` | 只使用局域网 host-candidate（默认行为）。 |
| `--stun URL` | 追加可选 STUN 服务器；可重复指定多个 URL。 |
| `--log-level` | Python 日志等级，如 `debug` / `info`。 |

## 桌面多视角采集 UI（Qt）
- 入口：`python scripts/qt_recorder.py --repo-id local/piper_vr_demo --teleop-config configs/piper_recording.json --hardware-config configs/piper_recording.json --fps 30 --video --resume`
- 多路相机同时预览，名称与数据集键一致（`observation.images.<name>`），按 `left_wrist`/`right_wrist`/`laptop` 优先占位，其余相机顺序排布；每个预览下方都会显示视角名称，接入几个相机就展示几个。
- 预览布局靠上，减少顶部留白；握持触发录制，按钮控制开始/放弃/下一集，右侧日志提示保存/压缩耗时。
- Episode 保存与视频编码在后台线程执行，录制结束时会提示“保存中…”，待提示完成再复位场景开始下一集，避免阻塞采集。

## 调试建议（局域网）

- **观察终端输出**：每条 DataChannel 消息触发的目标字典会打印到标准输出，可直接验证姿态解算是否正确。
- **浏览器日志**：`web-ui` 页面左侧日志实时显示信令协商、DataChannel 状态和错误信息。
- **断线重连**：若连接状态变为 `disconnected`/`failed`，前端会自动重新协商；必要时刷新页面即可恢复。
- **纯脚本测试**：可以通过 `python - <<'PY'` 直接构造 `ControllerPipeline` 并注入样例 payload，便于单元级验证。

> 当前方案专为局域网场景设计，默认不启用 TLS/证书，也不提供多客户端抢占逻辑。如需互联网部署或并发接入，可在此基础上扩展。

### 遥操作链路遥测

- Piper 实机运行时可通过 `--telemetry-file` 记录 IK 输出、滤波后命令与实测关节角，例如：
  ```bash
  python scripts/run_vr_piper.py \
    --config configs/piper_teleop.json \
    --telemetry-file output/telemetry.jsonl \
    --telemetry-sample-measured
  ```
- 日志为 JSONL，每帧包含 `q_ik`、`q_cmd`、`q_meas`、`dt_send` 等字段。搭配可视化脚本快速排查链路抖动：
  ```bash
  python scripts/plot_telemetry.py output/telemetry.jsonl --save out/telemetry
  ```
- 会生成 `out/telemetry_joints.png`（IK / Command / Measured 曲线）与 `out/telemetry_dt.png`（指令间隔），便于定位抖动来源。

## 真实机器人接入（Piper）

### 依赖与准备

- 安装 Piper 官方 SDK（示例：`pip3 install piper_sdk`，具体包名以厂家发布为准），并确保已激活对应的 Python 环境。
- 根据 `robot/real/piper/can_config.sh` 中的要求准备系统工具：`sudo apt install ethtool can-utils`，并确认内核已加载 `gs_usb` 驱动。

### CAN 接口命名与激活

- `robot/real/piper/can_config.sh` 支持单/双 CAN 模块自动重命名与设定波特率，修改脚本顶部的 `EXPECTED_CAN_COUNT`、`USB_PORTS` 后执行：
  ```bash
  sudo bash robot/real/piper/can_config.sh
  ```
- 若需要在插拔时自动寻找指定 USB 口，可改用 `robot/real/piper/can_find_and_config.sh`：
  ```bash
  sudo bash robot/real/piper/can_find_and_config.sh can_right 1000000 1-10.2:1.0
  ```

### 控制封装入口

- `robot/real/piper.py` 提供 `PiperMotorsBus` 类，对 `C_PiperInterface_V2` 进行二次封装，核心方法：
  - `connect(enable=True)`：上电/下电六轴与夹爪。
  - `apply_calibration()`：将关节移动到 `init_joint_position` 设定的初始角。
  - `write(target_joint)`：发送 6 轴 + 夹爪目标（弧度 / 线性开度），内部自动转换成 0.001° 并限制 4 号关节与夹爪安全范围。
  - `read()`：读取当前关节（弧度）与夹爪状态。
  - `safe_disconnect()`：在下电前回到 `safe_disable_position`。
- 使用前请在 `configs/piper_teleop*.json` 中将 `can_name`、`init_joint_position` 等参数改成现场配置；关节角以角度填写，加载后自动转换。

### VR 管线入口

- `scripts/run_vr_piper.py` 直接将 VR -> IK -> Piper 硬件串联：
  ```bash
  # 正装默认参数 + 实机（默认读取 configs/piper_teleop.json）
  python scripts/run_vr_piper.py

  # 仅观察指令，不下发硬件
  python scripts/run_vr_piper.py --dry-run

  # 双臂示例（左右臂独立基座姿态/硬件配置）
  python scripts/run_vr_piper.py --config configs/piper_teleop_dual.json --piper-config configs/piper_teleop_dual.json
  ```
- `piper_teleop.json` / `piper_teleop_left.json` / `piper_teleop_dual.json` 中的 `mount_rpy_deg` 控制 VR → 基座旋转映射（注意该角度会在代码中取转置后生效）；`piper_arms` 字段可为左右臂提供各自的 CAN 名称与初始/禁用姿态。
- 遥操作链路默认启用了两段滤波：`pose_filter_*`（VR 位姿时间窗平滑 + 前视预测）与 `velocity_filter_window`（关节速度前馈），可在配置或命令行调整。

### 配置文件说明

- `configs/piper_teleop*.json` 作为 VR→IK 的主配置入口，缺失字段会使用脚本默认值。
- 支持直接修改布尔值（如 `"no_stun": true`）、数值，以及长度为 6 的关节权重列表，例如 `"joint_reg_weights": [5,1,1,5,1,1]`、`"joint_smooth_weights": [8,1,1,8,1,1]`。
- `swivel_range_deg`（肘部旋转硬约束范围，度）与 `trust_region`（逐关节单步上限，弧度）可在 JSON 中调整；`joint_constraints` 支持 `hard_limits` / `step_limits`（角度或弧度）。
- `home_q_deg` 以度为单位描述初始关节姿态（按 joint1→jointN 排序）；填写后会用于 warm-start IK，并作为增量参考。
- 命令行参数始终优先生效，便于快速对比不同配置；可复制文件后通过 `--config path/to/file.json` 切换不同安装/硬件参数。


## 真实机器人接入（Piper）

### 依赖与准备

- 安装 Piper 官方 SDK（示例：`pip3 install piper_sdk`，具体包名以厂家发布为准），并确保已激活对应的 Python 环境。
- 根据 `robot/real/piper/can_config.sh` 中的要求准备系统工具：`sudo apt install ethtool can-utils`，并确认内核已加载 `gs_usb` 驱动。
- VR 端保持与仿真一致的 WebRTC/遥操作流程；本节仅描述额外的硬件 bring-up 步骤。

### CAN 接口命名与激活

- `robot/real/piper/can_config.sh` 支持单/双 CAN 模块自动重命名与设定波特率，修改脚本顶部的 `EXPECTED_CAN_COUNT`、`USB_PORTS` 后执行：
  ```bash
  sudo bash robot/real/piper/can_config.sh
  ```
- 若需要在插拔时自动寻找指定 USB 口，可改用 `robot/real/piper/can_find_and_config.sh`：
  ```bash
  sudo bash robot/real/piper/can_find_and_config.sh can_right 1000000 1-10.2:1.0
  ```
  其中第三个参数为 `ethtool -i canX` 获得的 `bus-info` 字段。
- 常见辅助脚本：`can_activate.sh`（单路激活）、`can_muti_activate.sh`（批量启用）、`find_all_can_port.sh`（列出当前可见的 USB 地址）。

### 控制封装入口

- `robot/real/piper.py` 提供 `PiperMotorsBus` 类，对 `C_PiperInterface_V2` 进行二次封装，核心方法如下：
  - `connect(enable: bool = True)`：上电或下电六轴与夹爪，内部自检 5 s 超时并记录返回状态。
  - `apply_calibration()`：将关节移动到 `init_joint_position` 设定的初始角。
  - `write(target_joint: list)`：发送 6 轴 + 夹爪目标（弧度 / 线性开度），内部自动转换成 0.001° 并限制 4 号关节与夹爪安全范围。
  - `read()`：读取当前关节（弧度）与夹爪状态。
  - `safe_disconnect()`：在下电前回到 `safe_disable_position`。
- 使用前请在 `configs/piper_teleop.json` 中将 `can_name`、`init_joint_position` 等参数改成现场的实际配置；配置文件中的 6 个关节角以角度（deg）填写，加载后会自动转换为弧度并在发送前转成 Piper 控制器要求的 0.001°。
  `gripper_open` / `gripper_closed` 仍沿用 SDK 默认的线性位移单位（米），如需改用角度可在驱动层统一变换。
  建议在业务逻辑中包装 try/finally，确保异常时调用 `safe_disconnect()` 与 `connect(False)`。

### 上位机自检脚本

- 位置归零、点动示例：`robot/real/piper/piper_ctrl_go_zero.py`、`robot/real/piper/piper_ctrl_right.py`。
- 状态读取：`robot/real/piper/piper_read_joint_state.py`、`robot/real/piper/piper_read_status.py`、`robot/real/piper/piper_read_fk.py`。
- 夹爪工具：`robot/real/piper/piper_set_gripper_zero.py`、`robot/real/piper/piper_read_gripper_status.py`。
- 这些脚本均假设先执行 CAN 配置，并在运行前安装 Piper SDK。可在 `python -m robot.real.piper` 环境中逐一验证硬件响应。

### 与 VR 管线集成

- `scripts/run_vr_piper.py` 负责将 VR 遥操作链路直接对接 Piper 实机：
  ```bash
  # 正装默认参数 + 实机（默认读取 configs/piper_teleop.json）
  python scripts/run_vr_piper.py

  # 仅观察指令，不下发硬件
  python scripts/run_vr_piper.py --dry-run

  # 若需自定义另一套配置，可共享同一个 JSON
  python scripts/run_vr_piper.py --config configs/piper_teleop_left.json

  # 双臂示例：镜像安装、can_left/can_right（会自动启用 --hands both）
  python scripts/run_vr_piper.py \
      --config configs/piper_teleop_dual.json \
      --piper-config configs/piper_teleop_dual.json
  ```
  该脚本会复用 `scripts/teleop_common.py` 的 IK/映射设置，并按需调用 `PiperMotorsBus.connect()`、`apply_calibration()`，再由 `scripts/piper_pipeline.py` 中的多臂 `PiperTeleopPipeline` 将关节结果写入实机。通过配置文件或 `--command-interval/--gripper-open/--effort-samples` 等参数即可调整指令频率、手爪目标和扭矩采样策略；当配置文件内包含 `piper_arms.left/right` 字段时，脚本会自动为每只手臂构造独立的 CAN 总线、遥测与滤波。若未显式指定 `--piper-config`，依旧会默认复用 `--config` 所指向的 JSON。
- 遥操作链路默认启用了两段滤波：`pose_filter_*` 参数（窗口 0.8 s、二阶拟合）在 `IncrementalPoseMapper` 中对 VR 位姿做时间窗平滑+前视预测，`velocity_filter_window`（默认 5 帧）则用于在关节空间平均速度并作为 `JointCommandFilter` 的前馈，进一步抑制加速度冲击。需要更灵敏或更平滑时，可在命令行传 `--pose-filter-window-sec / --velocity-filter-window` 或直接修改 `configs/piper_teleop.json` / `configs/piper_recording.json`，同时 `joint_speed_limits_deg` / `joint_acc_limits_deg` 仍由配置控制，确保机械臂速度限制可一致管理。
- 若需要手动对接或进一步定制，可继续参考 `ArmTeleopSession` / `IncrementalPoseMapper` 的组合：
  1. `PiperMotorsBus.connect()` 使能 → `apply_calibration()` 对齐零位。
  2. 在 Teleop 循环中调用 `bus.write(target_joint=q.tolist())`；若遇异常立即调用 `safe_disconnect()` 并下电。
  3. 收工前执行 `safe_disconnect()` 和 `connect(False)`，同时记录 `bus.read()` 的最终状态以便回放。
