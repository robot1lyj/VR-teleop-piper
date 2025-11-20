"""Qt 采集界面入口，封装参数解析与 Worker/UI 拼装。"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

from PyQt5 import QtWidgets

# 确保仓库根目录在 sys.path 中，避免 scripts.* 模块导入失败
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_ui.worker import RecorderWorker
from data_ui.ui import MainWindow
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.robots.piper_vr import PiperVRRobotConfig
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config


def _load_or_create_dataset(
    repo_id: str,
    root: Path,
    robot: Robot,
    fps: int,
    use_videos: bool,
    resume: bool,
    num_image_writer_threads: int,
    num_image_writer_processes: int,
) -> LeRobotDataset:
    root.parent.mkdir(parents=True, exist_ok=True)
    meta_dir = root / "meta"
    meta_ready = (meta_dir / "info.json").exists() and (meta_dir / "tasks.jsonl").exists()
    if resume and meta_ready:
        dataset = LeRobotDataset(repo_id, root=root, download_videos=use_videos)
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=num_image_writer_processes,
                num_threads=num_image_writer_threads * len(robot.cameras),
            )
    else:
        if resume and not meta_ready:
            logging.warning("本地数据集缺少 meta 文件，转为重新创建（不会从 hub 拉取）。目录：%s", meta_dir)
        if root.exists():
            entries = list(root.iterdir())
            only_meta = len(entries) == 0 or (len(entries) == 1 and entries[0].name == "meta")
            meta_files = list(meta_dir.iterdir()) if meta_dir.exists() else []
            only_meta_info = only_meta and all(f.name == "info.json" for f in meta_files)
            if only_meta_info:
                shutil.rmtree(root)
            elif not only_meta:
                raise RuntimeError(f"数据集目录已存在但 meta 不完整，请备份后删除或补全 meta: {root}")
        dataset = LeRobotDataset.create(
            repo_id,
            fps,
            root=root,
            robot=robot,
            use_videos=use_videos,
            image_writer_processes=num_image_writer_processes,
            image_writer_threads=num_image_writer_threads * len(robot.cameras),
        )
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Piper VR Qt 采集器")
    parser.add_argument("--repo-id", required=True, help="数据集名称，如 local/piper_vr_demo")
    parser.add_argument("--single-task", default="teleop", help="任务描述，会写入每帧 task 字段")
    parser.add_argument("--root", type=Path, default=Path("/home/lyj/data"), help="数据集根目录")
    parser.add_argument("--teleop-config", default="configs/piper_recording.json")
    parser.add_argument("--hardware-config", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preview-fps", type=int, default=12)
    parser.add_argument("--min-frames", type=int, default=5, help="少于该帧数则丢弃本集")
    parser.add_argument("--resume", action="store_true", help="继续写入已有数据集")
    parser.add_argument("--video", action="store_true", help="保存视频（若配置了相机）")
    parser.add_argument("--dry-run", action="store_true", help="dry-run 不写实机")
    parser.add_argument("--num-image-writer-threads", type=int, default=4)
    parser.add_argument("--num-image-writer-processes", type=int, default=0)
    return parser.parse_args()


def build_robot(cfg: argparse.Namespace) -> Robot:
    robot_cfg = PiperVRRobotConfig(
        teleop_config=cfg.teleop_config,
        hardware_config=cfg.hardware_config,
        dry_run=cfg.dry_run,
    )
    return make_robot_from_config(robot_cfg)


def run_app(args: Optional[argparse.Namespace] = None) -> None:
    cfg = args or parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    robot = build_robot(cfg)
    dataset_root = (Path(cfg.root) / cfg.repo_id).expanduser()
    dataset = _load_or_create_dataset(
        repo_id=cfg.repo_id,
        root=dataset_root,
        robot=robot,
        fps=cfg.fps,
        use_videos=cfg.video,
        resume=cfg.resume,
        num_image_writer_threads=cfg.num_image_writer_threads,
        num_image_writer_processes=cfg.num_image_writer_processes,
    )

    preview_key = None
    if robot.config.cameras:
        first_camera = next(iter(robot.config.cameras.keys()))
        preview_key = f"observation.images.{first_camera}"

    app = QtWidgets.QApplication(sys.argv)
    worker = RecorderWorker(
        robot=robot,
        dataset=dataset,
        single_task=cfg.single_task,
        fps=cfg.fps,
        min_frames=cfg.min_frames,
        preview_fps=cfg.preview_fps,
        preview_key=preview_key,
    )
    window = MainWindow(worker, dataset)
    window.resize(1200, 720)
    window.show()
    worker.start()
    sys.exit(app.exec_())


def main():
    run_app()


if __name__ == "__main__":
    main()
