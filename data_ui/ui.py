"""Qt 前端界面：多路相机预览 + 控制按钮 + 日志。"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class MainWindow(QtWidgets.QWidget):
    def __init__(self, worker, dataset: LeRobotDataset):
        super().__init__()
        self.worker = worker
        self.dataset = dataset
        self.setWindowTitle("Piper VR 采集 / Neon Deck")
        self._accent = "#0A84FF"  # Apple 蓝
        self._warn = "#E8733F"
        self._ok = "#1F8F4A"
        self._saving = False

        self._apply_styles()

        self.status_label = QtWidgets.QLabel("初始化中…")
        self.status_label.setObjectName("statusText")
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.status_label.setMaximumHeight(22)

        self.status_pill = QtWidgets.QLabel("●")
        self.status_pill.setObjectName("statusPill")
        self.status_pill.setAlignment(QtCore.Qt.AlignCenter)
        self.status_pill.setMaximumHeight(20)

        self.episode_label = QtWidgets.QLabel(f"集数：{dataset.meta.total_episodes}")
        self.episode_label.setObjectName("metric")
        self.episode_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.episode_label.setMaximumHeight(20)
        self.frame_label = QtWidgets.QLabel("本集帧数：0")
        self.frame_label.setObjectName("metric")
        self.frame_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.frame_label.setMaximumHeight(20)

        # 多相机容器
        self.preview_grid = QtWidgets.QGridLayout()
        self.preview_labels: Dict[str, QtWidgets.QLabel] = {}
        self.preview_captions: Dict[str, QtWidgets.QLabel] = {}
        self.preview_cards: Dict[str, QtWidgets.QWidget] = {}
        self.camera_keys = list(dict.fromkeys(getattr(dataset.meta, "camera_keys", [])))
        self._init_previews(self.camera_keys)

        self.status_bar = QtWidgets.QProgressBar()
        self.status_bar.setRange(0, 0)
        self.status_bar.setVisible(False)
        self.status_bar.setTextVisible(True)
        self.status_bar.setFormat("保存中…（如启用视频会编码）")
        self.status_bar.setMinimumHeight(14)
        self.status_bar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.start_toggle = QtWidgets.QPushButton("开始录制")
        self.start_toggle.setObjectName("primaryButton")
        self.next_btn = QtWidgets.QPushButton("下一集（清缓存）")
        self.next_btn.setObjectName("ghostButton")
        self.discard_btn = QtWidgets.QPushButton("放弃上一集")
        self.discard_btn.setObjectName("dangerButton")
        self.quit_btn = QtWidgets.QPushButton("退出")
        self.quit_btn.setObjectName("ghostButton")

        header = QtWidgets.QHBoxLayout()
        header.addWidget(self.status_pill, 0, QtCore.Qt.AlignLeft)
        header.addWidget(self.status_label, 1)
        header.addStretch()
        header.addWidget(self.episode_label)
        header.addWidget(self.frame_label)
        header.setContentsMargins(0, 0, 0, 4)

        header2 = QtWidgets.QHBoxLayout()
        header2.addWidget(self.status_bar)
        header2.addStretch()

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.start_toggle)
        btn_row.addWidget(self.next_btn)
        btn_row.addWidget(self.discard_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.quit_btn)
        btn_row.setContentsMargins(0, 6, 0, 0)

        glass = QtWidgets.QFrame()
        glass.setObjectName("glassCard")
        glass_layout = QtWidgets.QVBoxLayout(glass)
        glass_layout.setContentsMargins(6, 4, 6, 6)
        glass_layout.addLayout(header)
        glass_layout.addLayout(header2)
        glass_layout.addLayout(self.preview_grid)
        glass_layout.addLayout(btn_row)
        glass_layout.setSpacing(8)
        glass_layout.setStretch(0, 0)
        glass_layout.setStretch(1, 0)
        glass_layout.setStretch(2, 1)
        glass_layout.setStretch(3, 0)

        info_card = QtWidgets.QFrame()
        info_card.setObjectName("glassCard")
        info_layout = QtWidgets.QVBoxLayout(info_card)
        info_title = QtWidgets.QLabel("数据/日志")
        info_title.setObjectName("statusText")
        info_layout.addWidget(info_title)

        info_grid = QtWidgets.QGridLayout()
        info_grid.addWidget(QtWidgets.QLabel("repo_id"), 0, 0)
        info_grid.addWidget(QtWidgets.QLabel(dataset.repo_id), 0, 1)
        info_grid.addWidget(QtWidgets.QLabel("root"), 1, 0)
        info_grid.addWidget(QtWidgets.QLabel(str(dataset.root)), 1, 1)
        info_grid.addWidget(QtWidgets.QLabel("fps"), 2, 0)
        info_grid.addWidget(QtWidgets.QLabel(str(dataset.fps)), 2, 1)
        info_grid.addWidget(QtWidgets.QLabel("min_frames（仅提示）"), 3, 0)
        info_grid.addWidget(QtWidgets.QLabel(str(worker.min_frames)), 3, 1)
        info_grid.addWidget(QtWidgets.QLabel("preview_fps"), 4, 0)
        info_grid.addWidget(QtWidgets.QLabel(str(worker.preview_fps)), 4, 1)
        info_grid.addWidget(QtWidgets.QLabel("loop_fps"), 5, 0)
        self.loop_fps_label = QtWidgets.QLabel("-")
        self.loop_fps_label.setObjectName("metric")
        info_grid.addWidget(self.loop_fps_label, 5, 1)
        info_grid.addWidget(QtWidgets.QLabel("cam_fps"), 6, 0)
        self.cam_fps_label = QtWidgets.QLabel("-")
        self.cam_fps_label.setObjectName("metric")
        info_grid.addWidget(self.cam_fps_label, 6, 1)
        info_layout.addLayout(info_grid)

        self.hint = QtWidgets.QLabel(
            "提示：握持开始录制，松开结束；出现“保存中…”时请等待完成再操作下一集（启用视频会顺带编码）；非录制/保存时可放弃上一集（硬删除上一集数据）。"
        )
        self.hint.setWordWrap(True)
        self.hint.setObjectName("metric")
        info_layout.addWidget(self.hint)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setObjectName("logBox")
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumWidth(320)
        info_layout.addWidget(self.log_box, 1)

        layout_root = QtWidgets.QHBoxLayout()
        layout_root.addWidget(glass, 2)
        layout_root.addWidget(info_card, 1)
        self.setLayout(layout_root)

        self.start_toggle.setCheckable(True)
        self.start_toggle.clicked.connect(self._toggle_start)
        self.next_btn.clicked.connect(self.worker.request_next)
        self.discard_btn.clicked.connect(self._discard_last)
        self.quit_btn.clicked.connect(self._quit)

        self.worker.frame_ready.connect(self._update_frame)
        self.worker.status_changed.connect(self._set_status)
        self.worker.episode_saved.connect(self._on_saved)
        self.worker.error.connect(self._on_error)
        self.worker.counters.connect(self._on_counters)
        self.worker.log_event.connect(self._append_log)
        self.worker.saving_state.connect(self._on_saving_state)
        self.worker.metrics_update.connect(self._on_metrics)
        # SharedMemory 录制模式不再做“软重置/回零”，控制进程负责机械臂状态管理

    # --- 回调 ---
    def _toggle_start(self, checked: bool):
        self._set_start_text(checked)
        self.worker.set_allow_recording(checked)

    def _set_start_text(self, allowed: bool):
        self.start_toggle.setText("录制允许（握持开始）" if allowed else "开始录制")

    def _on_saved(self, ep_index: int, frames: int):
        self.episode_label.setText(f"集数：{ep_index + 1}")
        self.frame_label.setText(f"本集帧数：{frames}")
        self._append_log(f"Episode {ep_index:04d} 已保存，帧数 {frames}；可开始下一集")

    def _on_counters(self, frames: int, ep_index: int):
        self.frame_label.setText(f"本集帧数：{frames}")
        self.episode_label.setText(f"集数：{ep_index}")

    def _set_status(self, text: str):
        self.status_label.setText(text)
        warn_words = ("失败", "不足", "放弃", "偏少")
        pill_color = self._warn if any(w in text for w in warn_words) else self._ok
        self.status_pill.setStyleSheet(f"color: {pill_color};")

    def _on_error(self, text: str):
        QtWidgets.QMessageBox.critical(self, "错误", text)
        self._append_log(f"错误：{text}")
        # 保留窗口，避免立即退出丢失日志

    def _on_saving_state(self, saving: bool):
        self._saving = saving
        self.status_bar.setVisible(saving)
        self._update_start_toggle_enabled()
        self.next_btn.setEnabled(not saving)
        self.discard_btn.setEnabled(not saving)
        if saving:
            self.status_bar.setRange(0, 0)
            self.status_label.setText("保存中，请稍候…（启用视频会顺带编码）")
            self._append_log("保存中，请等待完成后再开始下一集（启用视频会顺带编码）")
        else:
            self.status_bar.setRange(0, 1)
            self.status_bar.setValue(1)
            self.status_label.setText("保存完成，可开始下一集")

    def _on_metrics(self, metrics: dict):
        loop_fps_raw = metrics.get("loop_fps", 0.0)
        try:
            loop_fps = float(loop_fps_raw)
        except Exception:
            loop_fps = 0.0
        cam_fps_raw = metrics.get("cam_fps", {})
        self.loop_fps_label.setText(f"{loop_fps:.1f}")
        if cam_fps_raw:
            parts = []
            for k, v in cam_fps_raw.items():
                try:
                    parts.append(f"{k.split('.')[-1]}:{float(v):.1f}")
                except Exception:
                    continue
            self.cam_fps_label.setText(", ".join(parts))
        else:
            self.cam_fps_label.setText("-")

    def _update_start_toggle_enabled(self):
        self.start_toggle.setEnabled(not self._saving)

    def _discard_last(self):
        if self._saving:
            QtWidgets.QMessageBox.information(self, "提示", "保存中，暂不可放弃上一集。")
            return
        if self.worker.dataset.meta.total_episodes <= 0:
            self._append_log("暂无可放弃的集")
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "放弃上一集",
            "将硬删除上一集（数据/视频/统计），无法恢复，确认要放弃吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self.worker.request_discard_last()

    def _update_frame(self, key: str, img: np.ndarray):
        h, w = img.shape[:2]
        if img.ndim == 2:
            qformat = QtGui.QImage.Format_Grayscale8
        else:
            qformat = QtGui.QImage.Format_RGB888
            if img.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            img = img[..., :3]
        qimg = QtGui.QImage(img.data, w, h, img.strides[0], qformat)
        label = self._ensure_preview_label(key)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            label.width(),
            label.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        label.setPixmap(pix)
        label.setToolTip(key)

    def closeEvent(self, event):  # noqa: N802
        # 重要：保存过程中强制退出会导致 meta/info.json 与 parquet/图片不一致（表现为集数/时长统计错误）。
        if self._saving:
            QtWidgets.QMessageBox.information(self, "提示", "正在保存，请等待“保存完成”后再退出。")
            event.ignore()
            return

        # 若当前还有未保存的帧（录制中/缓冲未清），退出会直接丢弃本集，默认不允许误操作。
        try:
            ep_buf = getattr(self.worker.dataset, "episode_buffer", None)  # type: ignore[attr-defined]
            buffer_size = int(ep_buf.get("size", 0)) if isinstance(ep_buf, dict) else 0
        except Exception:
            buffer_size = 0
        if buffer_size > 0:
            reply = QtWidgets.QMessageBox.question(
                self,
                "退出确认",
                f"检测到当前还有未保存的帧（{buffer_size} 帧）。\n退出将丢弃本集，是否仍要退出？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return

        self._quit()
        event.accept()

    def _quit(self):
        self.worker.stop()
        # 等待采集线程彻底退出（避免后台保存/写盘被强制中断）
        self.worker.wait()
        try:
            self.dataset.stop_image_writer()
        except Exception:
            pass
        QtWidgets.QApplication.quit()

    # --- 样式与预览 ---
    def _apply_styles(self):
        self.setStyleSheet(
            f"""
            QWidget {{
                background: #eef1f5;
                color: #111111;
                font-family: "JetBrains Mono","Fira Code","Consolas",monospace;
                font-size: 14px;
            }}
            #glassCard {{
                background: rgba(255,255,255,0.75);
                border: 1px solid rgba(0,0,0,0.08);
                border-radius: 12px;
                padding: 14px;
            }}
            #preview {{
                background: #ffffff;
                border: 1px solid rgba(0,0,0,0.08);
                border-radius: 10px;
                color: #6b7280;
            }}
            #statusText {{
                font-size: 13px;
                font-weight: 600;
                color: {self._accent};
            }}
            #statusPill {{
                font-size: 14px;
                color: {self._accent};
            }}
            #metric {{
                color: #343844;
                padding-left: 8px;
                font-size: 12px;
            }}
            QPushButton {{
                padding: 9px 14px;
                border-radius: 10px;
                border: 1px solid rgba(0,0,0,0.08);
                background: #f9fafb;
                color: #111111;
            }}
            QPushButton#primaryButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0a84ff, stop:1 #5ba8ff);
                border: none;
                font-weight: 700;
                color: #ffffff;
            }}
            QPushButton#primaryButton:checked {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1f8f4a, stop:1 #3ecf8e);
                color: #ffffff;
            }}
            QPushButton#dangerButton {{
                border: 1px solid rgba(232, 115, 63, 0.4);
                color: #b53c00;
            }}
            QPushButton#ghostButton {{
                border: 1px solid rgba(0,0,0,0.08);
            }}
            QPushButton:hover {{
                border-color: {self._accent};
            }}
            QMessageBox {{
                background: #f8fafc;
            }}
            #logBox {{
                background: #ffffff;
                border: 1px solid rgba(0,0,0,0.08);
                border-radius: 10px;
                color: #111111;
            }}
            #previewCaption {{
                font-size: 12px;
                color: #4b5563;
            }}
            """
        )

    def _append_log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{ts}] {text}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def _init_previews(self, keys: list[str]) -> None:
        self.preview_grid.setSpacing(8)
        self.preview_grid.setContentsMargins(0, 0, 0, 0)
        self.preview_grid.setAlignment(QtCore.Qt.AlignTop)
        self._layout_previews(keys)

    def _camera_display_name(self, key: str) -> str:
        name = key.replace("observation.images.", "")
        for suffix in ("_rgb", "_depth"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name

    def _ordered_camera_keys(self, keys: list[str]) -> list[str]:
        preferred_names = ["left_wrist", "right_wrist", "laptop"]
        ordered: list[str] = []
        for pref in preferred_names:
            ordered.extend([k for k in keys if self._camera_display_name(k) == pref and k not in ordered])
        ordered.extend([k for k in keys if k not in ordered])
        return ordered

    def _make_preview_widget(self, key: str) -> tuple[QtWidgets.QWidget, QtWidgets.QLabel, QtWidgets.QLabel]:
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(2, 0, 2, 0)
        vbox.setSpacing(4)

        lbl = QtWidgets.QLabel()
        lbl.setObjectName("preview")
        lbl.setMinimumSize(360, 220)
        lbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setText(self._camera_display_name(key))

        caption = QtWidgets.QLabel(self._camera_display_name(key))
        caption.setObjectName("previewCaption")
        caption.setAlignment(QtCore.Qt.AlignCenter)

        vbox.addWidget(lbl)
        vbox.addWidget(caption)
        return container, lbl, caption

    def _get_or_create_preview_widget(
        self, key: str
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QLabel, QtWidgets.QLabel]:
        if key in self.preview_labels:
            return self.preview_cards[key], self.preview_labels[key], self.preview_captions[key]
        container, lbl, caption = self._make_preview_widget(key)
        self.preview_cards[key] = container
        self.preview_labels[key] = lbl
        self.preview_captions[key] = caption
        return container, lbl, caption

    def _pop_key_by_name(self, keys: list[str], used: set[str], display_name: str) -> str | None:
        for key in keys:
            if key in used:
                continue
            if self._camera_display_name(key) == display_name:
                return key
        return None

    def _layout_previews(self, keys: list[str]) -> None:
        while self.preview_grid.count():
            item = self.preview_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        ordered = self._ordered_camera_keys(keys)
        used: set[str] = set()

        def place(key: str, row: int, col: int, colspan: int = 1):
            container, label, caption = self._get_or_create_preview_widget(key)
            label.setText(self._camera_display_name(key))
            caption.setText(self._camera_display_name(key))
            label.setToolTip(key)
            self.preview_grid.addWidget(container, row, col, 1, colspan)
            used.add(key)

        has_wrist = False
        left_key = self._pop_key_by_name(ordered, used, "left_wrist")
        if left_key:
            place(left_key, 0, 0)
            has_wrist = True
        right_key = self._pop_key_by_name(ordered, used, "right_wrist")
        if right_key:
            place(right_key, 0, 1)
            has_wrist = True
        row_cursor = 1 if has_wrist else 0

        laptop_key = self._pop_key_by_name(ordered, used, "laptop")
        if laptop_key:
            place(laptop_key, row_cursor, 0, colspan=2)
            row_cursor += 1

        remaining = [k for k in ordered if k not in used]
        for idx, key in enumerate(remaining):
            row = row_cursor + idx // 2
            col = idx % 2
            place(key, row, col)

    def _ensure_preview_label(self, key: str) -> QtWidgets.QLabel:
        if key in self.preview_labels:
            return self.preview_labels[key]
        if key not in self.camera_keys:
            self.camera_keys.append(key)
        self._layout_previews(self.camera_keys)
        return self.preview_labels[key]
