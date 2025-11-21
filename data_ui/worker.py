"""后台线程：采集、握持事件、相机预览推送。"""

from __future__ import annotations

import copy
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
    episode_discarded = QtCore.pyqtSignal()
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
        preview_key: Optional[str],
    ):
        super().__init__()
        self.robot = robot
        self.dataset = dataset
        self.single_task = single_task
        self.fps = fps
        self.min_frames = min_frames
        self.preview_fps = preview_fps
        self.preview_key = preview_key

        self._stop = threading.Event()
        self._recording = False
        self._frame_count = 0
        self._ready = False
        self._allow_recording = False
        self._discard_requested = False
        self._next_requested = False
        self._preview_ts: dict[str, float] = {}
        self._lock = threading.Lock()
        self._saving = False
        self._save_thread: Optional[threading.Thread] = None
        self._loop_dt_ema = 0.0
        self._cam_fps: dict[str, float] = {}
        self._last_metrics_emit = 0.0

    def stop(self):
        self._stop.set()

    def set_allow_recording(self, allowed: bool):
        with self._lock:
            self._allow_recording = allowed

    def request_discard(self):
        with self._lock:
            self._discard_requested = True

    def request_next(self):
        with self._lock:
            self._next_requested = True

    def _should_discard(self) -> bool:
        with self._lock:
            flag = self._discard_requested
            self._discard_requested = False
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
        """把所有相机帧推到 UI，按 camera key 节流。"""
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
            np_img = image
            if hasattr(image, "numpy"):
                np_img = image.numpy()
            if not isinstance(np_img, np.ndarray):
                continue
            self.frame_ready.emit(key, np_img)
            self._preview_ts[key] = now

    def _ensure_ready(self) -> None:
        """回零后进入 Ready 状态。"""
        self.status_changed.emit("回零中...")
        try:
            self.robot.run_calibration()
        except Exception as exc:  # pylint: disable=broad-except
            self.error.emit(f"回零失败: {exc}")
            return
        self._ready = True
        self.status_changed.emit("准备完毕：握持 + 开始按钮后录制")
        self.log_event.emit("回零完成，等待握持/开始按钮")

    def _start_async_save(self, episode_buffer: dict, frames: int) -> None:
        """在后台线程保存 Episode，避免阻塞采集循环。"""

        def _job():
            try:
                self.dataset.save_episode(episode_buffer)
                saved_idx = episode_buffer["episode_index"]
                self.episode_saved.emit(saved_idx, frames)
                self.log_event.emit(
                    f"Episode {saved_idx:04d} 已保存（{frames} 帧），耗时 {time.perf_counter() - t0:.2f}s（含视频/压缩）"
                )
            except Exception as exc:  # pylint: disable=broad-except
                self.error.emit(str(exc))
                self.log_event.emit(f"保存失败：{exc}")
            finally:
                self._saving = False
                self.saving_state.emit(False)
                self.dataset.episode_buffer = self.dataset.create_episode_buffer()

        self._saving = True
        t0 = time.perf_counter()
        self.log_event.emit("开始后台保存/编码，请等待完成再开始下一集")
        self.saving_state.emit(True)
        self._save_thread = threading.Thread(target=_job, name="episode-save", daemon=True)
        self._save_thread.start()

    def run(self):
        try:
            if not self.robot.is_connected:
                self.robot.connect()
            self._ensure_ready()
            while not self._stop.is_set():
                loop_start = time.perf_counter()
                observation, action = self.robot.teleop_step(record_data=True)
                started, stopped, active = self.robot.consume_recording_events()  # type: ignore[attr-defined]

                buffer_size = self.dataset.episode_buffer["size"] if self.dataset.episode_buffer is not None else 0

                if self._should_next() and not self._recording:
                    self.dataset.clear_episode_buffer()
                    self._ensure_ready()

                if self._should_discard():
                    if self._recording:
                        self.dataset.clear_episode_buffer()
                        self._recording = False
                        self._frame_count = 0
                        self.episode_discarded.emit()
                        self.status_changed.emit("已丢弃当前缓存")
                        self._ensure_ready()

                allow_record = self._is_allowed()
                if (
                    self._ready
                    and allow_record
                    and not self._recording
                    and not self._saving
                    and (started or (active and buffer_size == 0))
                ):
                    self.dataset.clear_episode_buffer()
                    self._recording = True
                    self._frame_count = 0
                    self._ready = False
                    self.status_changed.emit("录制中（握持松开结束）")
                    self.log_event.emit("开始录制，本集准备用于保存")

                if self._recording:
                    frame = {**observation, **action, "task": self.single_task}
                    self.dataset.add_frame(frame)
                    self._frame_count += 1
                    self.counters.emit(self._frame_count, self.dataset.meta.total_episodes)

                if self._recording and (stopped or not active):
                    if self._frame_count >= self.min_frames:
                        # 拷贝 buffer 后台保存，主循环不阻塞
                        ep_buffer = copy.deepcopy(self.dataset.episode_buffer)
                        self.dataset.clear_episode_buffer()
                        self._start_async_save(ep_buffer, self._frame_count)
                        self.status_changed.emit("保存中…等待完成")
                    else:
                        self.dataset.clear_episode_buffer()
                        self.episode_discarded.emit()
                        self.status_changed.emit("帧数不足，本集已丢弃，回零中...")
                        self.log_event.emit("本集帧数不足，已丢弃")
                    self._recording = False
                    self._frame_count = 0
                    self._ensure_ready()

                self._emit_preview(observation)

                dt = time.perf_counter() - loop_start
                # 指标更新
                if self._loop_dt_ema == 0.0:
                    self._loop_dt_ema = dt
                else:
                    self._loop_dt_ema = 0.9 * self._loop_dt_ema + 0.1 * dt
                now_perc = time.perf_counter()
                if now_perc - self._last_metrics_emit > 1.0:
                    loop_fps = 1.0 / self._loop_dt_ema if self._loop_dt_ema > 0 else 0.0
                    metrics = {"loop_fps": loop_fps, "cam_fps": dict(self._cam_fps)}
                    self.metrics_update.emit(metrics)
                    self._last_metrics_emit = now_perc

                busy_wait(1 / self.fps - dt)
        except Exception as exc:  # pylint: disable=broad-except
            self.error.emit(str(exc))
            self.log_event.emit(f"错误：{exc}")

        try:
            self.robot.disconnect()
        except Exception:  # pylint: disable=broad-except
            pass
