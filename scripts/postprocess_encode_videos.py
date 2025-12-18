"""数据集后处理：将图片帧按 episode/相机编码为视频，写回 LeRobotDataset。"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata, CODEBASE_VERSION
from lerobot.common.datasets.utils import (
    DEFAULT_IMAGE_PATH,
    DEFAULT_VIDEO_PATH,
    load_episodes,
    write_info,
    write_jsonlines,
)
from lerobot.common.datasets.video_utils import encode_video_frames, get_video_info


def _extract_episode_images(meta, root: Path, ep_idx: int, cam_keys: list[str]) -> None:
    """若缺少图片目录，则从 parquet 中提取内嵌 PNG bytes 落盘。"""
    parquet_path = root / meta.get_data_file_path(ep_idx)
    if not parquet_path.exists():
        logging.warning("Episode %d parquet 不存在，无法提取：%s", ep_idx, parquet_path)
        return
    df = pd.read_parquet(parquet_path)
    for cam_key in cam_keys:
        if cam_key not in df.columns:
            continue
        img_dir = (root / Path(DEFAULT_IMAGE_PATH.format(
            image_key=cam_key, episode_index=ep_idx, frame_index=0
        ))).parent
        img_dir.mkdir(parents=True, exist_ok=True)
        for i, val in enumerate(df[cam_key]):
            data = None
            if isinstance(val, dict) and "bytes" in val:
                data = val["bytes"]
            elif isinstance(val, (bytes, bytearray, memoryview)):
                data = bytes(val)
            if data is None:
                continue
            (img_dir / f"frame_{i:06d}.png").write_bytes(data)


def _list_episode_indices(root: Path) -> list[int]:
    files = sorted(root.glob("data/chunk-*/*episode_*.parquet"))
    eps: list[int] = []
    for f in files:
        try:
            num = int(f.stem.split("_")[-1])
            eps.append(num)
        except Exception:
            continue
    return sorted(set(eps))


def encode_dataset_videos(meta, fps: int, cam_keys: list[str], root: Path, overwrite: bool) -> None:
    """遍历数据集的 episode，将图片编码为 mp4，并刷新 meta 为 video dtype。"""
    # 若未设置 video_path，补上默认模板
    if meta.video_path is None:
        meta.info["video_path"] = DEFAULT_VIDEO_PATH

    ep_indices = _list_episode_indices(root)
    if not ep_indices:
        logging.warning("未找到任何 parquet，退出")
        return
    logging.info("开始编码视频：%d 集，摄像头：%s", len(ep_indices), cam_keys)

    total_frames = 0
    episode_lengths: dict[int, int] = {}

    for ep_idx in tqdm(ep_indices, desc="Episodes"):
        for cam_key in cam_keys:
            img_dir = (root / Path(DEFAULT_IMAGE_PATH.format(
                image_key=cam_key, episode_index=ep_idx, frame_index=0
            ))).parent
            if not img_dir.exists():
                logging.warning("Episode %d 相机 %s 缺少图片目录，尝试从 parquet 提取", ep_idx, cam_key)
                _extract_episode_images(meta, root, ep_idx, cam_keys)
            if not img_dir.exists():
                logging.warning("Episode %d 相机 %s 仍缺少图片目录，跳过", ep_idx, cam_key)
                continue

            frames = sorted(img_dir.glob("frame_*.png"))
            if not frames:
                logging.warning("Episode %d 相机 %s 无帧，跳过", ep_idx, cam_key)
                continue

            video_rel = Path(meta.get_video_file_path(ep_index=ep_idx, vid_key=cam_key))
            video_path = root / video_rel
            if video_path.exists():
                if overwrite:
                    video_path.unlink()
                else:
                    logging.info("视频已存在，跳过：%s", video_path)
                    continue
            video_path.parent.mkdir(parents=True, exist_ok=True)

            fps_val = fps if fps else 30
            logging.info("编码 Episode %d 相机 %s -> %s (fps=%s)", ep_idx, cam_key, video_path, fps_val)
            try:
                encode_video_frames(
                    imgs_dir=img_dir,
                    video_path=video_path,
                    fps=int(fps_val),
                    vcodec="libsvtav1",
                    crf=30,
                    overwrite=True,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logging.warning("编码失败，跳过 Episode %d 相机 %s：%s", ep_idx, cam_key, exc)
                continue
        # 统计帧数
        try:
            pq = root / meta.get_data_file_path(ep_idx)
            ep_len = len(pd.read_parquet(pq))
            total_frames += ep_len
            episode_lengths[ep_idx] = ep_len
        except Exception:
            episode_lengths[ep_idx] = 0

    # 刷新 meta：将相机 dtype 标记为 video，写入 video_path 和视频 info
    info = meta.info
    info["video_path"] = info.get("video_path") or DEFAULT_VIDEO_PATH
    info["total_episodes"] = len(ep_indices)
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{len(ep_indices)}"}
    for key in cam_keys:
        if key not in info["features"]:
            logging.warning("特征中缺少相机键 %s，无法标记为 video", key)
            continue
        info["features"][key]["dtype"] = "video"
        # 写入视频 info（取第 0 集的视频）
        sample_video = None
        # 找到第一个存在的视频文件填充 info
        for ep_idx in _list_episode_indices(root):
            candidate = root / meta.get_video_file_path(ep_index=ep_idx, vid_key=key)
            if candidate.exists():
                sample_video = candidate
                break
        if sample_video and sample_video.exists():
            info["features"][key]["info"] = get_video_info(sample_video)
    info["total_videos"] = len(ep_indices) * len(cam_keys)
    # 回写 meta
    meta.info = info
    write_info(info, root)
    logging.info(
        "写回 meta：episodes=%d, frames=%d, videos=%d",
        info["total_episodes"],
        info["total_frames"],
        info["total_videos"],
    )

    # 回写 episodes（长度取 parquet），避免 meta/episodes 不一致
    try:
        existing_eps = load_episodes(root)
    except Exception:
        existing_eps = {}
    # 默认任务：已有 episodes 的任务或全部任务列表
    default_tasks = None
    if existing_eps:
        default_tasks = next(iter(existing_eps.values())).get("tasks", None)
    if not default_tasks and getattr(meta, "tasks", None):
        default_tasks = list(meta.tasks.values())

    episode_entries = []
    for ep_idx in ep_indices:
        tasks = existing_eps.get(ep_idx, {}).get("tasks", default_tasks) or []
        episode_entries.append(
            {"episode_index": ep_idx, "tasks": tasks, "length": episode_lengths.get(ep_idx, 0)}
        )
    if episode_entries:
        write_jsonlines(episode_entries, root / "meta" / "episodes.jsonl")

    # 将 parquet 中的图片列移除（只保留状态/动作/索引等），与视频存储对齐
    for ep_idx in ep_indices:
        pq = root / meta.get_data_file_path(ep_idx)
        try:
            df = pd.read_parquet(pq)
            drop_cols = [c for c in cam_keys if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)
                df.to_parquet(pq, index=False)
                logging.info("已从 %s 移除图片列：%s", pq, drop_cols)
        except Exception as exc:
            logging.warning("清理图片列失败 %s：%s", pq, exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="后处理：将图片帧编码为视频并写回数据集")
    parser.add_argument("--root", type=Path, required=True, help="数据集根目录，如 ~/data/local/pen")
    parser.add_argument("--repo-id", required=True, help="数据集 repo_id，如 local/pen3（必填）")
    parser.add_argument("--cam-keys", nargs="+", default=None, help="需要编码的相机键，留空则取 meta.camera_keys")
    parser.add_argument("--overwrite", action="store_true", help="存在视频文件时是否覆盖")
    parser.add_argument("--backup", action="store_true", help="编码前备份数据集目录（同级 _backup_时间戳）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser()

    def _infer_repo_id(r: Path) -> str:
        # 若父目录名为 local，则推断为 local/<name>，否则用目录名
        parts = r.parts
        if len(parts) >= 2 and parts[-2] == "local":
            return f"local/{parts[-1]}"
        return r.name

    if args.backup:
        backup_dir = root.with_name(root.name + "_backup")
        logging.info("备份数据集目录到 %s", backup_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(root, backup_dir)

    repo_id = args.repo_id or _infer_repo_id(root)
    # 优先尝试本地元数据，避免无网时访问 hub
    meta = LeRobotDatasetMetadata(
        repo_id=repo_id,
        root=root,
        revision=CODEBASE_VERSION,
        force_cache_sync=False,
    )
    fps = meta.info["fps"]
    cam_keys = args.cam_keys or list(meta.camera_keys)
    if not cam_keys:
        logging.warning("数据集 meta 未定义相机键，退出")
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    encode_dataset_videos(meta, fps, cam_keys, root, overwrite=args.overwrite)
    logging.info("编码完成")


if __name__ == "__main__":
    main()
