# Piper VR 数据采集指南

本指南面向运行 VR 遥操作 + Piper 机械臂的数据采集场景。按照下面步骤，就能在本地 `/home/lyj/data` 下持续保存 30 Hz 的关节数据和相机画面。

---

## 1. 采集前的准备
- Python 依赖：建议在虚拟环境中安装 `aiortc、websockets、numpy、pin` 等包。
- Piper SDK 与 CAN：确认 `piper_sdk` 正常工作，并完成 `can` 口配置（默认 `can_right`）。常用串口脚本：
  ```bash
  # 列出可用 CAN 设备
  python robot/real/piper/find_all_can_port.py

  # 初始化指定 CAN 口（按需设定脚本参数）
  sudo bash robot/real/piper/can_config.sh can_right
  ```
- 相机接口确认：
  ```bash
  # RealSense 查看所有设备及序列号
  rs-enumerate-devices

  # OpenCV 摄像头：按持续时间抓取测试图（此例约 6 帧，保存到指定目录）
  python lerobot/common/robot_devices/cameras/opencv.py \
    --images-dir outputs/opencv_probe \
    --fps 30 \
    --record-time-s 0.2
  ```
- VR WebRTC 服务：`vr_runtime/controller_pipeline.py` 能正常接受手柄数据；遥操作时握持键、扳机键都应有响应。
- 配置文件：`configs/piper_recording.json` 已预留相机字段，可按实际设备修改：
  ```json
  "cameras": {
    "front_rgb": {
      "type": "opencv",
      "camera_index": 0,
      "fps": 30,
      "width": 640,
      "height": 480
    }
  }
  ```
  同一份配置还包含 `pose_filter_*`（VR 位姿历史滤波）与 `velocity_filter_window`（关节速度滑窗）等遥操作参数，默认值较 `piper_teleop` 更平滑，必要时可根据采集任务调整；关节限速/限加继续通过 `joint_speed_limits_deg`、`joint_acc_limits_deg` 控制，确保机械臂始终在安全范围内运行。

---

## 2. 一次采集怎么操作
1. 运行命令（举例）：
   ```bash
   python lerobot/scripts/control_robot.py \
     --robot.type=piper_vr \
     --robot.teleop_config=configs/piper_recording.json \
     --robot.hardware_config=configs/piper_recording.json \
     --control.type=record \
     --control.repo_id=local/piper_vr_demo \
     --control.single_task="test" \
     --control.fps=30 \
     --control.num_episodes=10 \
     --control.video=true
   ```
   - 根目录默认 `/home/lyj/data`，无需额外指定。
   - `push_to_hub` 已关闭，所有数据仅保存在本地。
2. 连接完成后，机械臂会立即执行 `apply_calibration()`，回到 `configs/piper_recording.json` 中的 `init_joint_position`。如需跳过，可在命令行追加 `--robot.skip_home=true`。
3. 程序接着进入 warmup（默认 10 秒），此时你可以用 VR 手柄微调确认对齐。
4. **按住 VR 手柄的侧握键（grip）即可开始录制**。系统会在检测到握持后才启动本轮计时；从这一帧起，`observation.state` 和 `action` 以 30 Hz 写入数据。
5. **松开握持键，当前 Episode 立即结束并保存**。系统会调用 `apply_calibration()` 让 Piper 回到初始位姿，并在终端提示“请复位”。等待回位完成后，再次握持即可继续录制下一条数据。
6. 当你结束程序（或按 `Esc` 停止录制）时，脚本同样会在断开连接前自动执行 `apply_calibration()`，确保机械臂回到初始姿态等待下一次操作。
7. 键盘辅助仍可使用：
   - `→` 跳过当前复位阶段。
   - `←` 删除上一条数据并重新录制。
   - `Esc` 终止整个录制流程。
8. 如果误触导致帧数太少（甚至为 0），系统会自动跳过保存——直接重新握持即可。

---

## 3. 多轮采集技巧
- 想追加数据到同一数据集，命令中加 `--control.resume=true` 再跑一次。
- 如果要按场景拆分数据，换一个新的 `repo_id`（例如 `local/piper_vr_day2`），后续训练再合并。
- `single_task` 会写入每一帧，可以用来区分不同任务。

---

## 4. 相机与 WebRTC 配置
- 最推荐的方式是在 `configs/piper_recording.json` 内修改 `"cameras"`，包括类型、序列号、分辨率等。
- 也可以在命令行临时覆盖：`--robot.cameras='{"front_rgb": {"type": "intelrealsense", ...}}'`。
- 需要重定向 WebRTC 服务时，可以传 `--robot.host / --robot.port / --robot.channel`。
- 若暂时不采图像，将 `"cameras"` 字段删掉即可。

---

## 5. 数据保存格式
- 所有结果写到 `/home/lyj/data/<repo_id>/`，典型结构如下：
  ```
  /home/lyj/data/local/piper_vr_demo/
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── episode_000001.parquet
    ├── videos/
    │   └── chunk-000/
    │       └── front_rgb/
    │           ├── episode_000000.mp4
    │           └── episode_000001.mp4
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl
    │   └── episodes_stats.jsonl
    └── images/  # 编码 mp4 前的临时 PNG，保存后会自动删除
  ```
- 每帧数据包含：
  - `observation.state`: `[joint1, …, joint6, gripper]`（弧度 + 夹爪线性值）。
  - `action`: 同样 7 维，为当帧送给 Piper 的目标。
  - 若启用相机且 `--control.video=true`，会额外保存 `observation.images.<camera>`（RGB 数组）并在 `videos/` 中打包为 mp4。
- 读取示例（含相机数据）：
  ```python
  import pandas as pd
  df = pd.read_parquet("/home/lyj/data/local/piper_vr_demo/data/chunk-000/episode_000000.parquet")
  print(df.columns)
  # -> ['observation.state', 'action', 'observation.images.front_rgb', ...]
  state = df.loc[0, 'observation.state']
  action = df.loc[0, 'action']
  rgb = df.loc[0, 'observation.images.front_rgb']
  print(state.shape, action.shape, rgb.shape)
  # -> (7,), (7,), (480, 640, 3)  # 尺寸取决于配置
  ```

---

## 6. 可视化与复查
- 使用 `lerobot/scripts/visualize_dataset.py` 快速查看录制结果：
  ```bash
  python lerobot/scripts/visualize_dataset.py \
    --repo-id local/piper_vr_demo \
    --episode-index 0 \
    --keys observation.state action
  ```
  如果要看视频，可添加 `--keys observation.images.front_rgb` 或直接播放 `videos/` 下的 mp4。
- 若仅想检查实时相机画面，可在录制时加入 `--control.display_data=true`，终端会弹出 Rerun 视窗。
- 终端输出中如果看到 “Episode 保存完毕…正在回到初始姿态”，说明松手流程已结束；若没有这条信息，说明握持不足或数据未写入。

---

## 7. 常见问题
- **仅测试流程？** 加 `--robot.dry_run=true`，不会写实机也不会回零。
- **相机暂时不用？** 把配置清空，或在命令行传空字典。
- **要加新传感器？** 在 `PiperVRRobot.features` 里拓展字段，并保证 `teleop_step()` 返回对应数据即可。

祝采集顺利！
