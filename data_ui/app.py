"""Qt 采集界面入口，封装参数解析与 Worker/UI 拼装。"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
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
from lerobot.common.robot_devices.robots.piper_shm import PiperShmRobotConfig
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config


def _load_or_create_dataset(
    repo_id: str,
    root: Path,
    robot: Robot,
    fps: int,
    use_videos: bool,
    resume: bool,
    num_image_writer_threads: int | None,
    num_image_writer_processes: int,
    embed_images_in_parquet: bool,
) -> LeRobotDataset:
    num_cameras = len(getattr(robot, "cameras", {}) or {})
    # AsyncImageWriter 的 num_threads 语义是“总线程数”，不是“每路相机线程数”。
    # 但实践经验（LeRobot 建议）是：每路相机约 4 个写盘线程更稳。
    # 这里做一个折中：未显式指定时自动 = 4 * 相机数；显式指定则严格按用户值（避免悄悄放大导致抢 CPU）。
    if num_cameras <= 0:
        writer_threads = 0
    else:
        if num_image_writer_threads is None or int(num_image_writer_threads) <= 0:
            writer_threads = 4 * num_cameras
        else:
            writer_threads = int(num_image_writer_threads)
    root.parent.mkdir(parents=True, exist_ok=True)
    meta_dir = root / "meta"
    meta_ready = (meta_dir / "info.json").exists() and (meta_dir / "tasks.jsonl").exists()
    if resume and meta_ready:
        dataset = LeRobotDataset(repo_id, root=root, download_videos=use_videos)
        dataset.embed_images_in_parquet = bool(embed_images_in_parquet)
        # 数据集元信息一旦创建就固定：fps / 是否使用视频（video dtype）都不应在 resume 时改变，
        # 否则会出现“时长显示不对 / 采样间隔不一致 / 统计混乱”等严重问题。
        incompat_reasons: list[str] = []
        if int(dataset.fps) != int(fps):
            incompat_reasons.append(f"fps 不一致（数据集={dataset.fps}，当前={fps}）")
        existing_use_videos = len(getattr(dataset.meta, "video_keys", []) or []) > 0
        if bool(existing_use_videos) != bool(use_videos):
            incompat_reasons.append(
                f"video 模式不一致（数据集={'video' if existing_use_videos else 'image'}，当前={'video' if use_videos else 'image'}）"
            )

        expected_keys = set(robot.features.keys())
        actual_keys = set(dataset.features.keys())
        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            logging.warning(
                "检测到数据集特征与当前机器人不一致，尝试重新创建。缺失: %s，多余: %s",
                sorted(missing),
                sorted(extra),
            )
        if incompat_reasons:
            logging.warning("检测到 resume 参数与现有数据集不兼容：%s，将重新创建数据集。", "；".join(incompat_reasons))
        if expected_keys != actual_keys or incompat_reasons:
            backup_root = root.with_name(f"{root.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}")
            logging.warning("将现有数据集备份到: %s", backup_root)
            shutil.move(str(root), str(backup_root))
            dataset = LeRobotDataset.create(
                repo_id,
                fps,
                root=root,
                robot=robot,
                use_videos=use_videos,
                image_writer_processes=num_image_writer_processes if writer_threads > 0 else 0,
                image_writer_threads=writer_threads,
                embed_images_in_parquet=bool(embed_images_in_parquet),
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
            image_writer_processes=num_image_writer_processes if writer_threads > 0 else 0,
            image_writer_threads=writer_threads,
            embed_images_in_parquet=bool(embed_images_in_parquet),
        )
    # 兼容 resume 跑出旧的异步 writer 配置：强制停掉 image_writer，避免残留，并按当前参数重启
    if dataset.image_writer is not None:
        dataset.stop_image_writer()
    if writer_threads > 0:
        dataset.start_image_writer(
            num_processes=num_image_writer_processes,
            num_threads=writer_threads,
        )
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Piper Qt 采集器（录制进程：SharedMemory + 相机）")
    parser.add_argument("--repo-id", required=True, help="数据集名称，如 local/piper_vr_demo")
    parser.add_argument("--single-task", default="teleop", help="任务描述，会写入每帧 task 字段")
    default_root = Path.home() / "data"
    parser.add_argument("--root", type=Path, default=default_root, help="数据集根目录")
    parser.add_argument("--teleop-config", default="configs/piper_recording.json", help="用于读取 hands/cameras 配置")
    parser.add_argument("--shm-name", default="piper_vr", help="SharedMemory 段名前缀（需与控制进程一致）")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preview-fps", type=int, default=30)
    parser.add_argument("--min-frames", type=int, default=5, help="少于该帧数会提示（不再自动丢弃）")
    parser.add_argument("--resume", action="store_true", help="继续写入已有数据集")
    parser.add_argument("--video", action="store_true", help="保存视频（若配置了相机）")
    parser.add_argument(
        "--embed-images-in-parquet",
        action="store_true",
        help=(
            "将每帧 PNG bytes 内嵌进 parquet（更可移植/冗余，但会显著增加 CPU/IO 并导致 parquet 巨大）。"
            "默认关闭（parquet 仅保存图片路径，图片仍保存在 images/ 目录）。"
        ),
    )
    parser.add_argument(
        "--num-image-writer-threads",
        type=int,
        default=None,
        help="异步写图线程总数；不填则自动=4*相机数（更稳），填了就严格按该值",
    )
    parser.add_argument("--num-image-writer-processes", type=int, default=0)
    return parser.parse_args()


def build_robot(cfg: argparse.Namespace) -> Robot:
    robot_cfg = PiperShmRobotConfig(
        teleop_config=cfg.teleop_config,
        shm_name=cfg.shm_name,
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
        embed_images_in_parquet=cfg.embed_images_in_parquet,
    )

    app = QtWidgets.QApplication(sys.argv)
    worker = RecorderWorker(
        robot=robot,
        dataset=dataset,
        single_task=cfg.single_task,
        fps=cfg.fps,
        min_frames=cfg.min_frames,
        preview_fps=cfg.preview_fps,
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
