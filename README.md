# VR-New 模块概览（WebRTC 版）

```
├── __init__.py                # 暴露 run_vr_controller_stream 入口
├── controller_state.py        # 单手柄状态结构，记录握持/扳机/姿态等信息
├── controller_stream.py       # WebRTC 信令入口，封装姿态处理管线与 CLI
├── webrtc_endpoint.py         # WebSocket 信令 + aiortc DataChannel 服务端
└── web-ui/                    # 浏览器侧 A-Frame 客户端
    ├── index.html             # 信令地址输入、状态显示、A-Frame 场景
    ├── interface.js           # UI 交互与日志面板
    ├── styles.css             # 深色主题样式
    └── vr_app.js              # WebRTCBridge 与手柄数据采集
```

## 数据流

```
  浏览器 / 头显 (web-ui/vr_app.js)
        │ 信令：WebSocket (offer/answer/ICE)
        │ 数据：WebRTC DataChannel(JSON，姿态 50 Hz)
        ▼
  controller_stream.py + webrtc_endpoint.py
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
   PYTHONPATH=. python -m controller_stream \
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

## 调试建议（局域网）

- **观察终端输出**：每条 DataChannel 消息触发的目标字典会打印到标准输出，可直接验证姿态解算是否正确。
- **浏览器日志**：`web-ui` 页面左侧日志实时显示信令协商、DataChannel 状态和错误信息。
- **断线重连**：若连接状态变为 `disconnected`/`failed`，前端会自动重新协商；必要时刷新页面即可恢复。
- **纯脚本测试**：可以通过 `python - <<'PY'` 直接构造 `ControllerPipeline` 并注入样例 payload，便于单元级验证。

> 当前方案专为局域网场景设计，默认不启用 TLS/证书，也不提供多客户端抢占逻辑。如需互联网部署或并发接入，可在此基础上扩展。 

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
   - 自动停止：手柄长按 A/B（或 X/Y）约 0.8 秒触发 `menuPressed`，脚本自动收尾并退出。
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

### 配置文件说明

- `configs/run_vr_meshcat.json` 会在脚本启动时自动加载，缺失字段则使用脚本默认值。
- 支持直接修改布尔值（如 `"no_stun": true`）、数值或关节权重字典（键可写成 `joint1` 或数字编号）。
- 命令行参数始终优先生效，便于快速对比不同配置；若想切换成另一套完整配置，可复制该文件并通过 `--config path/to/file.json` 指定。

### 常见排查

- **Meshcat 页面空白**：重启脚本并确保浏览器访问的是最新的 URL；如仍失败，可单独启动 `meshcat-server` 并设置环境变量以复用已有服务端口。
- **IK 频繁失败**：降低 `--scale` 或调整参考姿态，确保目标位姿落在 Piper 的可达空间；必要时暂时关闭 `check_collision`。
- **轨迹文件过大**：录制脚本默认逐帧 flush，可改为运行结束后手动压缩，或在后续处理管线中按时间段切片。
