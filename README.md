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
- 多路相机同时预览，名称与数据集键一致（`observation.images.<name>`）；握持触发录制，按钮控制开始/放弃/下一集，右侧日志提示保存/压缩耗时。
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

## VR 增量遥操作完整流程

### 依赖准备

```bash
pip install aiortc websockets numpy pin python-casadi meshcat
```

确保浏览器端可以访问本机 8442（信令）和 8080（http.server）端口，Meshcat 页面默认会在脚本启动后自动打开或输出可访问的 URL。

### 实时遥操作（Meshcat 演示）

1. 启动 Meshcat 演示脚本：
   ```bash
   python scripts/run_vr_meshcat.py \
     --urdf piper_description/urdf/piper_description.urdf \
     --hands right \
     --scale 1.0 \
     --log-level info
   ```
   - 首次运行会加载 Piper 中立关节姿态作为参考点。
   - Meshcat 页面的 URL 会出现在终端中，复制到浏览器即可查看机器人模型。
   - 默认参数集中在 `configs/run_vr_meshcat.json`，修改该文件即可调整端口、缩放、关节权重等设置；如需临时覆盖，可继续使用命令行标志，例如 `--joint-reg-weights joint1=5,joint4=8`。
2. 在另一终端启动 Web UI：
   ```bash
   python -m http.server 8080 --directory web-ui
   ```
3. 在头显 / 浏览器访问 `http://<电脑IP>:8080` → 填写 `ws://<电脑IP>:8442` 建立连接 → 点击「开启手柄追踪」。
4. 长按握持键进入增量控制：
   - 位移由固定矩阵 `R_bv` 映射到机械臂基座坐标系，再与参考位姿相加。
   - 四元数先转换为相对旋转，再通过 `R_bv` 映射到基座系，更新末端姿态。
   - 终端会输出 IK 结果（成功 / 失败原因）以及当前目标平移位置。
5. 松开握持键即可复位增量；扳机值大于 0.5 时会在日志中标记夹爪闭合。

### 轨迹录制（可选）

1. 运行录制脚本（示例开启自动开始/停止）：
   ```bash
   python scripts/record_vr_trajectory.py output.jsonl \
     --hands right \
     --channel controller \
     --no-stun \
     --auto-start \
     --auto-stop
   ```
   - 自动开始：任意参与录制的手柄握持键持续按下 ≈3 帧（可用 `--start-grip-threshold` 调整）后写入。
2. 按照实时遥操作同样的方式连接手柄，若未开启自动停止，可按 `Ctrl+C` 手动结束，JSONL 文件会保存每帧的原始报文和归一化结果。
3. 可为不同测试场景建立多个轨迹文件，以便复用。

### 离线回放（无需硬件）

```bash
python scripts/run_vr_meshcat.py \
  --replay output.jsonl \
  --replay-speed 1.0 \
  --hands right \
  --urdf piper_description/urdf/piper_description.urdf \
  [--no-meshcat] [--no-collision]
```

若需验证 30° 安装配置，可直接：

```bash
python scripts/run_vr_meshcat.py --config configs/run_vr_meshcat_side30.json --replay output.jsonl --replay-speed 1.0
```

- `--replay-speed` 用于加速 / 减速回放（例如 0.5 表示慢放，2.0 表示倍速）。
- 配合 `--replay-loop` 可以循环播放，便于长时间观察 Meshcat 中的末端路径。
- 回放期间同样会输出 IK 结果和目标位姿，验证计算链路是否稳定。

### 主要参数说明

| 参数 | 脚本 | 说明 |
| ---- | ---- | ---- |
| `--scale` | `run_vr_meshcat.py` | 控制 VR 位移映射后的缩放，调大可放大手柄位移。 |
| `--hands` | 两个脚本 | 指定使用哪只手柄（`right`/`left`/`both`），需与 DataChannel 推送的键一致。 |
| `--channel` | 两个脚本 | DataChannel 名称，需与 `web-ui/vr_app.js` 中配置保持一致。 |
| `--no-stun` / `--stun` | 两个脚本 | 是否禁用 STUN（局域网推荐 `--no-stun`）。 |
| `--log-level` | 两个脚本 | 调整日志详情，调试时可改为 `debug`。 |
| `--auto-start` / `--auto-stop` | `record_vr_trajectory.py` | 握持触发录制、菜单键长按自动完结。 |
| `--start-grip-threshold` | `record_vr_trajectory.py` | 自动开始所需的连续握持帧数（默认 3）。 |
| `--replay-speed` / `--replay-loop` | `run_vr_meshcat.py` | 离线回放速度倍率与是否循环。 |
| `--no-meshcat` | `run_vr_meshcat.py` | 仅回放/求解 IK，不启动 Meshcat（服务器冲突时使用）。 |
| `--no-collision` | `run_vr_meshcat.py` | 禁用自碰撞检测，规避几何配置异常导致的崩溃。 |
| `--urdf` | `run_vr_meshcat.py` | 机械臂 URDF 路径，可替换为自定义模型。 |
| `--config` | `run_vr_meshcat.py` | 指定 JSON 配置文件，默认 `configs/run_vr_meshcat.json`。 |
| `--joint-reg-weights` | `run_vr_meshcat.py` | 逐关节正则权重，支持 `joint1=6,joint4=6` 或 `6,1,1,6,1,1`，输入 `none` 关闭自定义。 |
| `--joint-smooth-weights` | `run_vr_meshcat.py` | 逐关节平滑权重，格式同上；未设置时沿用默认抑制 1/4 号关节方案。 |
| `--swivel-range-deg` | `run_vr_meshcat.py` | 肘部 swivel 角硬约束（度），以中立姿态为零点，40 表示允许 ±40°，0 关闭。 |
| `--trust-region` | `run_vr_meshcat.py` | 单步信赖域（弧度），限制 `q` 偏离上一帧的幅度，可填标量或 6 元列表。 |
| `--joint-constraints` | `run_vr_meshcat.py` | 以 JSON 描述额外的关节硬约束/步长限制，如 `{"step_limits_deg":{"joint4":15}}`，命令行优先生效。 |
| `--mount-rpy-deg` | `run_vr_meshcat.py` | 基座安装姿态（roll、pitch、yaw，单位度），如 `0,30,0`；用于兼容侧装等朝向变化。 |
| `--mount-offset` | `run_vr_meshcat.py` | 基座在 Meshcat 中的平移偏置（米），如 `0,0,0.15`，用于虚拟抬升/平移模型。 |
| `--home-q-deg` | `run_vr_meshcat.py` | 自定义初始关节角（度），按 joint1→jointN 顺序提供；Meshcat 与 IK 将以此姿态为基准。 |

### 配置文件说明

- `configs/run_vr_meshcat.json` 会在脚本启动时自动加载，缺失字段则使用脚本默认值。
- 支持直接修改布尔值（如 `"no_stun": true`）、数值，以及长度为 6 的关节权重列表，例如 `"joint_reg_weights": [5,1,1,5,1,1]`、`"joint_smooth_weights": [8,1,1,8,1,1]`。
- 新增 `swivel_range_deg`（肘部旋转硬约束范围，单位度，以中立姿态为零点）与 `trust_region`（逐关节单步上限，单位弧度）字段，可直接在 JSON 中调整，脚本会自动加载。
- `joint_constraints` 支持 `hard_limits` / `hard_limits_deg`（收窄物理范围）与 `step_limits` / `step_limits_deg`（限制单步改变量），当前默认只对 `joint4` 生效，可按需扩展到其他关节；如需滤波步长中心，可设置 `filter_alpha`（0-1 之间）。
- `mount_rpy_deg` 使用 Roll-Pitch-Yaw（XYZ 顺序，单位度）描述机械臂基座相对于默认水平姿态的安装角；脚本会自动将 VR 坐标系、IK 基座与 Meshcat 同步旋转，无需修改 URDF。
- `mount_offset` 提供基座位置平移（米），常用于 Meshcat 中模拟台面高度差或相机视角调整，不影响 IK 参考坐标系。
- `home_q_deg` 以度为单位描述初始关节姿态（按 joint1→jointN 排序）。填写后脚本会：① 用该姿态 warm-start IK；② 以其末端位姿作为增量参考；③ Meshcat 初始显示真实姿态。若角度超出 URDF 限制会给出警告，可结合 `trust_region` / `joint_constraints` 放宽可行区间。
- 命令行参数始终优先生效，便于快速对比不同配置；若想切换成另一套完整配置，可复制该文件并通过 `--config path/to/file.json` 指定。

仓库额外提供 `configs/run_vr_meshcat_side30.json` 作为完整字段示例：在默认参数基础上覆写安装姿态、平移偏置与真实关节初始角，便于直接测试 30°（或自行调整）安装，不需要手动补齐缺失字段。

示例：水平正装使用默认配置，若需要测试绕 X 轴 30° 的侧装，可直接：

```bash
# 水平安装（默认）
python scripts/run_vr_meshcat.py

# 侧装 30° 配置
python scripts/run_vr_meshcat.py --config configs/run_vr_meshcat_side30.json
```

如需自定义姿态，可修改 JSON 中的 `mount_rpy_deg` / `mount_offset` / `home_q_deg`（推荐复制 `configs/run_vr_meshcat_side30.json` 再修改），或在命令行追加 `--mount-rpy-deg 0,45,0 --mount-offset 0,0,0.2 --home-q-deg -88,157,-150,-100,12,-172` 等覆盖参数。

### 安装姿态快速预览

使用 `scripts/preview_mount_pose.py` 可以在 Meshcat 中直接查看当前配置对应的基座朝向：

```bash
# 读取默认配置
python scripts/preview_mount_pose.py

# 指定侧装配置
python scripts/preview_mount_pose.py --config configs/run_vr_meshcat_side30.json

# 直接覆盖安装角/平移/初始关节角
python scripts/preview_mount_pose.py --mount-rpy-deg 0,30,0 --mount-offset 0,0,0.2 --home-q-deg -88,157,-150,-100,12,-172
```

脚本会加载配置中的 URDF、`mount_rpy_deg`、`mount_offset` 与 `home_q_deg`，并在 Meshcat 中同步更新基座姿态和平移以及初始关节角，便于在不改 URDF 的情况下核对不同安装方式或零位。

### 常见排查

- **Meshcat 页面空白**：重启脚本并确保浏览器访问的是最新的 URL；如仍失败，可单独启动 `meshcat-server` 并设置环境变量以复用已有服务端口。
- **IK 频繁失败**：降低 `--scale` 或调整参考姿态，确保目标位姿落在 Piper 的可达空间；必要时暂时关闭 `check_collision`。
- **轨迹文件过大**：录制脚本默认逐帧 flush，可改为运行结束后手动压缩，或在后续处理管线中按时间段切片。


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

- 新增脚本 `scripts/run_vr_piper.py` 已将 Meshcat 遥操作链路直接对接 Piper 实机：
  ```bash
  # 正装默认参数 + 实机（默认读取 configs/piper_teleop.json）
  python scripts/run_vr_piper.py

  # 仅观察指令，不下发硬件
  python scripts/run_vr_piper.py --dry-run

  # 若需自定义另一套配置，可共享同一个 JSON
  python scripts/run_vr_piper.py --config configs/run_vr_meshcat_side30.json

  # 双臂示例：镜像安装、can_left/can_right（会自动启用 --hands both）
  python scripts/run_vr_piper.py \
      --config configs/piper_teleop_dual.json \
      --piper-config configs/piper_teleop_dual.json
  ```
  该脚本会复用 `run_vr_meshcat.py` 的 IK/映射设置，并按需调用 `PiperMotorsBus.connect()`、`apply_calibration()`，再由 `scripts/piper_pipeline.py` 中的多臂 `PiperTeleopPipeline` 将关节结果写入实机。通过配置文件或 `--command-interval/--gripper-open/--effort-samples` 等参数即可调整指令频率、手爪目标和扭矩采样策略；当配置文件内包含 `piper_arms.left/right` 字段时，脚本会自动为每只手臂构造独立的 CAN 总线、遥测与滤波。若未显式指定 `--piper-config`，依旧会默认复用 `--config` 所指向的 JSON。
- 遥操作链路默认启用了两段滤波：`pose_filter_*` 参数（窗口 0.8 s、二阶拟合）在 `IncrementalPoseMapper` 中对 VR 位姿做时间窗平滑+前视预测，`velocity_filter_window`（默认 5 帧）则用于在关节空间平均速度并作为 `JointCommandFilter` 的前馈，进一步抑制加速度冲击。需要更灵敏或更平滑时，可在命令行传 `--pose-filter-window-sec / --velocity-filter-window` 或直接修改 `configs/piper_teleop.json` / `configs/piper_recording.json`，同时 `joint_speed_limits_deg` / `joint_acc_limits_deg` 仍由配置控制，确保机械臂速度限制可一致管理。
- 若需要手动对接或进一步定制，可继续参考 `ArmTeleopSession` / `IncrementalPoseMapper` 的组合：
  1. `PiperMotorsBus.connect()` 使能 → `apply_calibration()` 对齐零位。
  2. 启动 `scripts/run_vr_meshcat.py`，确认 Meshcat 中的末端姿态与实机一致。
  3. 在 Teleop 循环中调用 `bus.write(target_joint=q.tolist())`；若遇异常立即调用 `safe_disconnect()` 并下电。
  4. 收工前执行 `safe_disconnect()` 和 `connect(False)`，同时记录 `bus.read()` 的最终状态以便回放。
