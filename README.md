# VR-Teleop-Piper 快速指南

本仓库聚焦两件事：
- **VR 遥操作（控制进程）**：`scripts/run_vr_piper.py` 内置 WebRTC 信令 + IK + Piper 下发（建议 90 Hz）。
- **数据采集（录制进程）**：`scripts/qt_recorder.py` **只读 SharedMemory + 相机** 并写入数据集（30 Hz），与控制进程硬隔离，避免“录制写盘导致机械臂卡顿”。

> 重要：录制必须本机 local（SharedMemory 不能跨机器）；并且先启动控制进程 `--publish-shm`，再启动 Qt。

## 环境
- Python 3.11.7（推荐 Conda）：`conda create -n teleop python=3.11.7 && conda activate teleop`
- 安装：`pip install -r requirements.txt && pip install -e .`

## 仓库结构（精简）
- `web-ui/`：VR 页面（A-Frame），浏览器/头显访问 `http://<host>:8080`
- `scripts/run_vr_piper.py`：控制进程（WebRTC + IK + 硬件下发 + 可选 shm 发布）
- `scripts/qt_recorder.py`：录制进程（相机 + shm 采样 + 数据集写盘）
- `vr_runtime/shm_ring.py`：SharedMemory ring 协议
- `configs/`：遥操作/录制配置

## 启动网页
```bash
python -m http.server 8080 --directory web-ui
```
头显/浏览器访问 `http://<host>:8080`，连接地址填 `ws://<host>:8442`。

## 遥操作（控制进程）
右臂示例（建议 90Hz，并为录制发布 shm）：
```bash
python scripts/run_vr_piper.py --config configs/piper_teleop.json \
  --publish-shm --shm-name piper_vr
```
- 默认 90Hz：已在 `configs/piper_teleop*.json` 中将 `command_interval` 统一为 `0.0111`（≈90Hz），因此一般不需要再在终端传参；如需临时覆盖可用 `--command-interval`。
- 干跑：加 `--dry-run`（不连接硬件）。
- 左臂：`--config configs/piper_teleop_left.json`
- 双臂：`--config configs/piper_teleop_dual.json --piper-config configs/piper_teleop_dual.json`

SharedMemory 段名约定：
- 状态：`<shm-name>_status`（握持掩码 bit0=right, bit1=left）
- 指令：`<shm-name>_cmd_<hand>`，测量：`<shm-name>_meas_<hand>`（hand 为 `right/left`）

可调参数（控制进程）：
- `--shm-meas-hz`：measured 采样频率（默认 60Hz）
- `--shm-cmd-capacity`、`--shm-meas-capacity`：ring buffer 深度

## 数据采集（Qt，录制进程，30Hz）
录制链路需要启动的进程（建议按顺序）：
- 进程 1（可选，但头显/浏览器访问 VR 页面需要）：网页静态服务器 `python -m http.server 8080 --directory web-ui`
- 进程 2（必须，本机）：遥操作控制进程（WebRTC + IK + 硬件下发 + shm 发布）
- 进程 3（必须，本机）：Qt 录制进程（只读 shm + 相机 + 写入数据集）

注意：
- SharedMemory 不能跨机器；控制进程与 Qt 必须在同一台机器、同一用户下运行。
- 必须先启动控制进程（带 `--publish-shm --shm-name ...`），再启动 Qt。
- 若使用 `--resume`：请保持 `--fps/--video` 与既有数据集一致；不一致会自动备份并重建新数据集，避免“集数/时长统计错误”。
- 保存过程中窗口会禁止退出；请等待界面提示“保存完成”后再关闭。

双臂录制示例（控制进程 + Qt 录制进程）：
```bash
# 控制进程（双臂 + shm 发布）
python scripts/run_vr_piper.py --config configs/piper_teleop_dual.json \
  --piper-config configs/piper_teleop_dual.json \
  --publish-shm --shm-name piper_vr

# 录制进程（双臂，30Hz）
python scripts/qt_recorder.py --repo-id local/piper_vr_demo \
  --teleop-config configs/piper_recording_dual.json \
  --shm-name piper_vr \
  --fps 30 --preview-fps 10 --resume \
  --single-task "teleop"
```

采集策略（与 LeRobot 一致，best-effort）：
- 录制 loop 按数据集 `--fps` 运行（如 30Hz）；每帧读取“各相机最新帧”，允许重复旧帧（相机掉帧/低 fps 时仍尽量保持 30Hz）。
- 动作/状态从 shm ring 取“当前时刻”最新 cmd/meas；若控制进程短暂断流/采样缺失，则复用上一条并在 Qt 日志提示（不丢帧）。
- 多相机不同步不会丢帧：Qt 会记录 `cam_skew_ms/cam_age_ms_*` 供排查；若三相机需要严格同步必须做硬件同步/触发，否则出现 1–2 帧偏差是常态。

写盘并发（录制进程，影响“是否会卡顿/掉帧”的关键）：
- 推荐：`--num-image-writer-processes 0`（不开子进程），`--num-image-writer-threads` **不填**（自动=4×相机数，更稳）
- `--num-image-writer-threads N` 语义是“总线程数”（不会再按相机数自动放大）；N 太大可能抢 CPU 影响控制丝滑，太小可能写盘跟不上导致保存变慢/内存升高
- 录制进程默认会把自身 `nice` 调低（`+10`）并降低 PNG 压缩等级，以“更大磁盘占用”换“更稳的 90Hz 控制”；如需极限压缩可自行改回
- 默认 **不把 PNG bytes 内嵌进 parquet**（parquet 仅保存图片路径，图片仍在 `images/` 目录），可显著降低“保存阶段”的 CPU/IO 压力并避免巨型 parquet；如需单文件可移植/冗余可加 `--embed-images-in-parquet`（代价很大）
- `--video` 会在每集保存阶段额外编码 mp4（吃 CPU/IO）；建议先不加，录制完用 `python scripts/postprocess_encode_videos.py ...` 后处理

## 相机索引探测（OpenCV）
```bash
python lerobot/common/robot_devices/cameras/opencv.py --images-dir output/cam_probe
```

## VR 报文轨迹录制（可选，仅调试）
仅保存 VR 手柄报文为 JSONL（不含相机/数据集），用于离线回放/排查：
```bash
python scripts/record_vr_trajectory.py output/trajectory.jsonl --hands both --auto-start
```

## 遥测与排查（控制进程）
- 开启：`--telemetry-file output/telemetry.jsonl [--telemetry-sample-measured]`
- 可视化：`python scripts/plot_telemetry.py output/telemetry.jsonl --save output/telemetry`
- 关注字段：`queue_delay_ms`（发送排队）、`write_ms`（硬件写耗时）、`dt_send`（指令间隔）。

## 常用命令速查
- 启动网页：`python -m http.server 8080 --directory web-ui`
- 遥操作（右臂，90Hz+录制发布）：`python scripts/run_vr_piper.py --config configs/piper_teleop.json --publish-shm --shm-name piper_vr`
- 遥操作（双臂，90Hz+录制发布）：`python scripts/run_vr_piper.py --config configs/piper_teleop_dual.json --piper-config configs/piper_teleop_dual.json --publish-shm --shm-name piper_vr`
- 遥操作干跑：`python scripts/run_vr_piper.py --config configs/piper_teleop.json --dry-run`
- Qt 录制（右臂，数据集）：`python scripts/qt_recorder.py --repo-id local/piper_vr_demo --teleop-config configs/piper_recording.json --shm-name piper_vr --fps 30`
- Qt 录制（双臂，数据集）：`python scripts/qt_recorder.py --repo-id local/piper_vr_demo --teleop-config configs/piper_recording_dual.json --shm-name piper_vr --fps 30`
- 相机索引探测：`python lerobot/common/robot_devices/cameras/opencv.py --images-dir output/cam_probe`
- 录制后视频编码：`python scripts/postprocess_encode_videos.py --root ~/data/local/pen3 --repo-id local/pen3 --overwrite`

## 提示
- 默认信令内置在控制脚本里，无需单独运行 `vr_runtime.controller_pipeline`。
- 调整配置后记得在 `record.md` 留痕，说明改动原因与必要性。
