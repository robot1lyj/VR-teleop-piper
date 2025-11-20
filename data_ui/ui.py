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
        self._init_previews(list(getattr(dataset.meta, "camera_keys", [])))

        self.start_toggle = QtWidgets.QPushButton("开始录制")
        self.start_toggle.setObjectName("primaryButton")
        self.next_btn = QtWidgets.QPushButton("下一集（回零）")
        self.next_btn.setObjectName("ghostButton")
        self.discard_btn = QtWidgets.QPushButton("放弃当前")
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
        glass_layout.addLayout(header)
        glass_layout.addLayout(self.preview_grid)
        glass_layout.addLayout(btn_row)
        glass_layout.setSpacing(6)
        glass_layout.setStretch(0, 0)
        glass_layout.setStretch(1, 12)
        glass_layout.setStretch(2, 1)

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
        info_grid.addWidget(QtWidgets.QLabel("min_frames"), 3, 0)
        info_grid.addWidget(QtWidgets.QLabel(str(worker.min_frames)), 3, 1)
        info_grid.addWidget(QtWidgets.QLabel("preview_fps"), 4, 0)
        info_grid.addWidget(QtWidgets.QLabel(str(worker.preview_fps)), 4, 1)
        info_layout.addLayout(info_grid)

        self.hint = QtWidgets.QLabel("提示：保存后等待视频压缩完成，再复位场景并开始下一集。")
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
        self.discard_btn.clicked.connect(self.worker.request_discard)
        self.quit_btn.clicked.connect(self._quit)

        self.worker.frame_ready.connect(self._update_frame)
        self.worker.status_changed.connect(self._set_status)
        self.worker.episode_saved.connect(self._on_saved)
        self.worker.episode_discarded.connect(self._on_discarded)
        self.worker.error.connect(self._on_error)
        self.worker.counters.connect(self._on_counters)
        self.worker.log_event.connect(self._append_log)

    # --- 回调 ---
    def _toggle_start(self, checked: bool):
        if checked:
            self.start_toggle.setText("录制允许（握持开始）")
            self.worker.set_allow_recording(True)
        else:
            self.start_toggle.setText("开始录制")
            self.worker.set_allow_recording(False)

    def _on_saved(self, ep_index: int, frames: int):
        self.episode_label.setText(f"集数：{ep_index + 1}")
        self.frame_label.setText(f"本集帧数：{frames}")
        self._append_log(f"Episode {ep_index:04d} 已保存，帧数 {frames}")

    def _on_discarded(self):
        self.frame_label.setText("本集帧数：0")
        self._append_log("当前缓存已丢弃")

    def _on_counters(self, frames: int, ep_index: int):
        self.frame_label.setText(f"本集帧数：{frames}")
        self.episode_label.setText(f"集数：{ep_index}")

    def _set_status(self, text: str):
        self.status_label.setText(text)
        pill_color = self._warn if "失败" in text or "不足" in text else self._ok
        self.status_pill.setStyleSheet(f"color: {pill_color};")

    def _on_error(self, text: str):
        QtWidgets.QMessageBox.critical(self, "错误", text)
        self._append_log(f"错误：{text}")
        self._quit()

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
        self._quit()
        event.accept()

    def _quit(self):
        self.worker.stop()
        self.worker.wait(1000)
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
            """
        )

    def _append_log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{ts}] {text}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def _init_previews(self, keys: list[str]) -> None:
        if not keys:
            keys = ["待接入相机"]
        self.preview_grid.setSpacing(6)
        cols = 2
        for idx, key in enumerate(keys):
            row = idx // cols
            col = idx % cols
            label = self._make_preview_label(key)
            self.preview_grid.addWidget(label, row, col)
            self.preview_labels[key] = label

    def _make_preview_label(self, key: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel()
        lbl.setObjectName("preview")
        lbl.setFixedSize(460, 300)
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setText(key)
        return lbl

    def _ensure_preview_label(self, key: str) -> QtWidgets.QLabel:
        if key in self.preview_labels:
            return self.preview_labels[key]
        label = self._make_preview_label(key)
        count = len(self.preview_labels)
        cols = 2
        row = count // cols
        col = count % cols
        self.preview_grid.addWidget(label, row, col)
        self.preview_labels[key] = label
        return label
