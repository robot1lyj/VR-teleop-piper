"""后台线程：采集、握持事件、相机预览推送。

录制进程职责（与控制进程硬隔离）：
- 只负责：读相机 + 读 SharedMemory(控制进程发布) + 写入 LeRobotDataset。
- 不负责：WebRTC/IK/硬件控制（这些都在控制进程中）。

关键目标：与 LeRobot 一致的 best-effort 采集 —— 不因多相机不同步/缺帧丢帧；必要时复用上一帧/黑图填充，
并把 cam_skew/cam_age/shm 采样异常写入日志，方便离线诊断/过滤。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np
from PyQt5 import QtCore

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import busy_wait
from lerobot.common.robot_devices.robots.utils import Robot


class RecorderWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(str, np.ndarray)
    status_changed = QtCore.pyqtSignal(str)
    episode_saved = QtCore.pyqtSignal(int, int)
    error = QtCore.pyqtSignal(str)
    counters = QtCore.pyqtSignal(int, int)
    log_event = QtCore.pyqtSignal(str)
    saving_state = QtCore.pyqtSignal(bool)
    metrics_update = QtCore.pyqtSignal(dict)

    def __init__(
        self,
        robot: Robot,
        dataset: LeRobotDataset,
        single_task: str,
        fps: int,
        min_frames: int,
        preview_fps: int,
    ):
        super().__init__()
        self.robot = robot
        self.dataset = dataset
        self.single_task = single_task
        self.fps = int(fps)
        self.min_frames = int(min_frames)
        self.preview_fps = int(preview_fps)

        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._allow_recording = False
        self._recording = False
        self._saving = False
        self._frame_count = 0
        self._inactive_since: float | None = None

        self._discard_last_requested = False
        self._next_requested = False

        self._preview_ts: dict[str, float] = {}
        self._cam_fps: dict[str, float] = {}
        self._next_preview_fetch = 0.0

        self._loop_dt_ema = 0.0
        self._last_metrics_emit = 0.0

        self._save_thread: Optional[threading.Thread] = None
        self._record_start_perf: float | None = None

        # LeRobot 风格：best-effort 录制，不因多相机 skew 丢帧；但仍提示对齐/缺图警告（节流打印，避免刷屏）
        self._warn_count = 0
        self._warn_last_reason = ""
        self._warn_last_emit = 0.0

        # 若某路相机缺帧：复用上一帧（LeRobot 允许重复帧）；若无上一帧则填充黑图
        self._last_images: dict[str, np.ndarray] = {}
        self._blank_images: dict[str, np.ndarray] = {}

    def stop(self):
        self._stop.set()

    def set_allow_recording(self, allowed: bool):
        with self._lock:
            self._allow_recording = bool(allowed)

    def request_discard_last(self):
        with self._lock:
            self._discard_last_requested = True

    def request_next(self):
        with self._lock:
            self._next_requested = True

    def _should_discard_last(self) -> bool:
        with self._lock:
            flag = self._discard_last_requested
            self._discard_last_requested = False
            return flag

    def _should_next(self) -> bool:
        with self._lock:
            flag = self._next_requested
            self._next_requested = False
            return flag

    def _is_allowed(self) -> bool:
        with self._lock:
            return self._allow_recording

    def _emit_preview(self, observation: dict[str, Any]) -> None:
        """把所有相机帧推到 UI，按 preview_fps 节流。"""

        if self.preview_fps <= 0:
            return

        now = time.perf_counter()
        for key, image in observation.items():
            if not key.startswith("observation.images."):
                continue
            last_t = self._preview_ts.get(key, 0.0)
            if now - last_t < 1 / max(1, self.preview_fps):
                continue
            if last_t > 0:
                fps = 1.0 / max(1e-6, now - last_t)
                prev = self._cam_fps.get(key, fps)
                self._cam_fps[key] = 0.8 * prev + 0.2 * fps
            np_img = image.numpy() if hasattr(image, "numpy") else image
            if not isinstance(np_img, np.ndarray):
                continue
            self.frame_ready.emit(key, np_img)
            self._preview_ts[key] = now

    def _start_async_save(self, episode_buffer: dict, frames: int, record_elapsed_s: float | None = None) -> None:
        """后台保存 Episode，避免阻塞采集循环。"""

        def _job():
            try:
                self.dataset.save_episode(episode_buffer)
                ep_idx_raw = episode_buffer.get("episode_index", 0)
                try:
                    saved_idx = int(np.asarray(ep_idx_raw).reshape(-1)[0])
                except Exception:
                    saved_idx = int(ep_idx_raw) if not isinstance(ep_idx_raw, (list, tuple)) else int(ep_idx_raw[0])
                saved_idx = int(np.asarray(saved_idx).reshape(-1)[0])
                frames_int = int(np.asarray(frames).reshape(-1)[0])
                try:
                    approx_record_s = float(record_elapsed_s) if record_elapsed_s is not None else frames_int / max(1, self.fps)
                except Exception:
                    approx_record_s = frames_int / max(1, self.fps)
                self.episode_saved.emit(saved_idx, frames_int)
                save_s = time.perf_counter() - t0
                total_eps = getattr(self.dataset.meta, "total_episodes", None)
                total_frames = getattr(self.dataset.meta, "total_frames", None)
                tail = ""
                if total_eps is not None and total_frames is not None:
                    tail = f"；累计 {int(total_eps)} 集 / {int(total_frames)} 帧"
                self.log_event.emit(
                    f"Episode {saved_idx:04d} 已保存：{frames_int} 帧（约 {approx_record_s:.1f}s）"
                    f"，保存耗时 {save_s:.2f}s{tail}"
                )
            except Exception as exc:  # pylint: disable=broad-except
                import traceback

                tb = traceback.format_exc()
                self.error.emit(f"{exc}\n{tb}")
                self.log_event.emit(f"保存失败：{exc}")
            finally:
                self._saving = False
                self.saving_state.emit(False)
                self.dataset.episode_buffer = self.dataset.create_episode_buffer()

        self._saving = True
        t0 = time.perf_counter()
        self.log_event.emit("开始后台保存，请等待完成再开始下一集")
        self.saving_state.emit(True)
        # 注意：保存线程不可设为 daemon，否则窗口关闭/进程退出时会被强制杀死，
        # 造成 meta/info.json 只写了一半（表现为“集数/时长统计不对”）。
        self._save_thread = threading.Thread(target=_job, name="episode-save", daemon=False)
        self._save_thread.start()

    def _log_warn_throttled(self, reason: str) -> None:
        """对齐/缺图警告：仍然保存帧，但做节流提示避免刷屏。"""

        self._warn_count += 1
        self._warn_last_reason = reason
        now = time.perf_counter()
        if now - self._warn_last_emit < 1.0:
            return
        self._warn_last_emit = now
        self.log_event.emit(f"采集警告（仍保存帧）：{self._warn_count}（最近原因：{self._warn_last_reason}）")
        self._warn_count = 0

    def run(self):
        try:
            if not getattr(self.robot, "is_connected", False):
                self.robot.connect()

            self.status_changed.emit("就绪：点亮“录制允许”后握持开始")

            while not self._stop.is_set():
                loop_start = time.perf_counter()

                # 录制中必须抓图；未录制时按 preview_fps 节流抓图（降低相机/拷贝负载）
                need_images = self._recording
                if not need_images and self.preview_fps > 0 and loop_start >= self._next_preview_fetch:
                    need_images = True
                    self._next_preview_fetch = loop_start + 1.0 / max(1, self.preview_fps)

                observation, action = self.robot.teleop_step(record_data=need_images)
                started, stopped, active = self.robot.consume_recording_events()  # type: ignore[attr-defined]

                buffer_size = self.dataset.episode_buffer["size"] if self.dataset.episode_buffer is not None else 0
                allow_record = self._is_allowed()

                # --- UI 请求：下一集 / 放弃上一集 ---
                if self._should_next() and not self._recording and not self._saving:
                    self.dataset.clear_episode_buffer()
                    self._frame_count = 0
                    self._inactive_since = None
                    self.status_changed.emit("已清空缓冲，等待下一集握持")
                    self.log_event.emit("下一集：已清空缓冲")

                if self._should_discard_last():
                    if self._recording or self._saving:
                        self.log_event.emit("录制/保存中，暂不可放弃上一集")
                    elif self.dataset.meta.total_episodes <= 0:
                        self.log_event.emit("暂无可放弃的集")
                    else:
                        try:
                            ep_idx, ep_len = self.dataset.discard_last_episode()
                            self.status_changed.emit(f"已放弃上一集 {ep_idx:04d}（{ep_len} 帧）")
                            self.log_event.emit(f"上一集已放弃：Episode {ep_idx:04d}（{ep_len} 帧）")
                            self.counters.emit(0, self.dataset.meta.total_episodes)
                        except Exception as exc:  # pylint: disable=broad-except
                            self.error.emit(str(exc))
                            self.log_event.emit(f"放弃上一集失败：{exc}")

                # --- 触发录制 ---
                if allow_record and not self._recording and not self._saving and (started or (active and buffer_size == 0)):
                    self.dataset.clear_episode_buffer()
                    self._recording = True
                    self._frame_count = 0
                    self._inactive_since = None
                    self._record_start_perf = time.perf_counter()
                    self.status_changed.emit("录制中（松开握持结束）")
                    self.log_event.emit("开始录制")

                # --- 写入帧（与 LeRobot 一致：best-effort，不因多相机 skew 丢帧） ---
                if self._recording:
                    # 录制启动的同一轮 loop 里，可能因为 need_images=False 而没抓到图像键。
                    # 这会触发大量“missing_images”告警并导致首帧黑图/复用旧帧。
                    # 处理：若本集首帧缺少任何相机键，则跳过该帧，等待下一轮拿到完整图像再写入。
                    if self._frame_count == 0 and self.dataset.meta.camera_keys:
                        missing_first = [k for k in self.dataset.meta.camera_keys if k not in observation]
                        if missing_first:
                            self._log_warn_throttled(
                                "start_warmup:missing_images:" + ",".join([k.split(".")[-1] for k in missing_first])
                            )
                            # 不写入 frame，不更新计数；保持录制状态，下一轮会强制 need_images=True
                            # （因为 self._recording=True），从而拿到完整图像键。
                            pass
                        else:
                            missing_first = []
                    else:
                        missing_first = []

                    if missing_first:
                        # 仍推预览/维持节拍，但不写入数据集
                        self._emit_preview(observation)
                        dt = time.perf_counter() - loop_start
                        self._loop_dt_ema = (
                            dt if self._loop_dt_ema == 0.0 else 0.9 * self._loop_dt_ema + 0.1 * dt
                        )
                        now_perc = time.perf_counter()
                        if now_perc - self._last_metrics_emit > 1.0:
                            loop_fps = 1.0 / self._loop_dt_ema if self._loop_dt_ema > 0 else 0.0
                            self.metrics_update.emit({"loop_fps": loop_fps, "cam_fps": dict(self._cam_fps)})
                            self._last_metrics_emit = now_perc
                        busy_wait(1 / max(1, self.fps) - dt)
                        continue

                    # 1) 更新“上一帧图像”缓存（用于缺帧时复用）
                    if self.dataset.meta.camera_keys:
                        for key in self.dataset.meta.camera_keys:
                            if key in observation:
                                img = observation[key]
                                np_img = img.numpy() if hasattr(img, "numpy") else img
                                if isinstance(np_img, np.ndarray):
                                    self._last_images[key] = np_img

                    # 2) 若某路相机缺帧：复用上一帧；若无上一帧则填充黑图（保证 schema 完整、录制 fps 稳定）
                    if self.dataset.meta.camera_keys:
                        for key in self.dataset.meta.camera_keys:
                            if key in observation:
                                continue
                            if key in self._last_images:
                                observation[key] = self._last_images[key]
                                self._log_warn_throttled(f"missing_images:reuse_last:{key}")
                                continue
                            blank = self._blank_images.get(key)
                            if blank is None:
                                shape = (self.dataset.features.get(key) or {}).get("shape")
                                try:
                                    h, w, c = [int(x) for x in shape]
                                except Exception:
                                    h, w, c = 1, 1, 3
                                h = max(1, h)
                                w = max(1, w)
                                c = max(1, c)
                                blank = np.zeros((h, w, c), dtype=np.uint8)
                                self._blank_images[key] = blank
                            observation[key] = blank
                            self._log_warn_throttled(f"missing_images:zeros:{key}")

                    logs = getattr(self.robot, "logs", {}) or {}
                    frame_valid = bool(logs.get("frame_valid", True))
                    invalid_reason = str(logs.get("frame_invalid_reason", "")) if not frame_valid else ""

                    # 注意：即便 shm 读数缺失/异常，也不丢帧（与 LeRobot 一致）。
                    # 这类帧可能存在“图像-动作”时间偏差，可用于离线诊断/过滤（或按需剔除）。
                    if not frame_valid:
                        self._log_warn_throttled(f"shm_align:{invalid_reason}")

                    frame = {**observation, **action, "task": self.single_task}
                    self.dataset.add_frame(frame)
                    self._frame_count += 1
                    self.counters.emit(self._frame_count, self.dataset.meta.total_episodes)
                    if active:
                        self._inactive_since = None
                    elif self._inactive_since is None:
                        self._inactive_since = time.perf_counter()

                # --- 停止录制（握持松开/防抖） ---
                should_stop = False
                if self._recording:
                    if stopped:
                        should_stop = True
                    elif (
                        not active
                        and self._inactive_since is not None
                        and (time.perf_counter() - self._inactive_since) > 0.3
                        and self._frame_count >= self.min_frames
                    ):
                        should_stop = True

                if should_stop:
                    frames = self._frame_count
                    record_elapsed_s = (
                        (time.perf_counter() - self._record_start_perf) if self._record_start_perf is not None else None
                    )
                    if frames <= 0:
                        self.dataset.clear_episode_buffer()
                        self.log_event.emit("未记录到帧，已丢弃（握持过短或未采到有效数据）")
                        self._recording = False
                        self._frame_count = 0
                        self._inactive_since = None
                        self._record_start_perf = None
                    elif stopped and frames < self.min_frames:
                        # 明确 stop 事件但帧数过短：认为误触/掉线，直接丢弃
                        self.dataset.clear_episode_buffer()
                        self.log_event.emit(f"帧数过短（{frames}/{self.min_frames}），已丢弃本集（stop 事件）")
                        self.status_changed.emit("帧数过短，已丢弃")
                        self._recording = False
                        self._frame_count = 0
                        self._inactive_since = None
                        self._record_start_perf = None
                    else:
                        # 性能关键：episode_buffer 里包含大量图像张量，deepcopy 会产生巨额内存拷贝，
                        # 甚至在 stop 边沿造成“卡一下/丢控制”。这里直接把 buffer 交给保存线程，
                        # 主线程立刻切换到新的空 buffer，避免重复拷贝。
                        ep_buffer = self.dataset.episode_buffer
                        self.dataset.episode_buffer = self.dataset.create_episode_buffer()
                        self._start_async_save(ep_buffer, frames, record_elapsed_s=record_elapsed_s)
                        if frames < self.min_frames:
                            self.status_changed.emit("帧数偏少，正在保存（可按放弃上一集重录）")
                            self.log_event.emit(f"帧数偏少（{frames}/{self.min_frames}），已保存本集")
                        else:
                            self.status_changed.emit("保存中…等待完成")
                        self._recording = False
                        self._frame_count = 0
                        self._inactive_since = None
                        self._record_start_perf = None

                self._emit_preview(observation)

                # --- 指标与节拍 ---
                dt = time.perf_counter() - loop_start
                self._loop_dt_ema = dt if self._loop_dt_ema == 0.0 else 0.9 * self._loop_dt_ema + 0.1 * dt
                now_perc = time.perf_counter()
                if now_perc - self._last_metrics_emit > 1.0:
                    loop_fps = 1.0 / self._loop_dt_ema if self._loop_dt_ema > 0 else 0.0
                    self.metrics_update.emit({"loop_fps": loop_fps, "cam_fps": dict(self._cam_fps)})
                    self._last_metrics_emit = now_perc

                busy_wait(1 / max(1, self.fps) - dt)

        except Exception as exc:  # pylint: disable=broad-except
            self.error.emit(str(exc))
            self.log_event.emit(f"错误：{exc}")
        finally:
            # 退出前确保后台保存完成，否则 meta/episodes/parquet 可能处于“写了一半”的不一致状态。
            try:
                if self._save_thread is not None and self._save_thread.is_alive():
                    self.log_event.emit("退出中：等待后台保存完成…")
                    self._save_thread.join()
            except Exception:
                pass
            try:
                if getattr(self.dataset, "image_writer", None) is not None:
                    self.dataset.stop_image_writer()
            except Exception:
                pass
            try:
                self.robot.disconnect()
            except Exception:
                pass
