"""合并多个同构的本地 LeRobot 数据集到一个新目录。

使用示例：
    python scripts/merge_datasets.py \
        --sources ~/data/local/pen_a ~/data/local/pen_b \
        --target  ~/data/local/pen_merged --repo-id local/pen_merged

要求：
- 各源数据集的 fps/features/相机/video 配置一致。
- 仅合并最新版本结构（v2.1），不处理 Hub 远端。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PARQUET_PATH,
    DEFAULT_VIDEO_PATH,
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_tasks,
    serialize_dict,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
    write_jsonlines,
    write_stats,
)


def _dataset_paths(root: Path, info: dict, ep_index: int) -> Tuple[Path, List[Path]]:
    chunk_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    ep_chunk = ep_index // chunk_size
    data_tpl = info.get("data_path", DEFAULT_PARQUET_PATH)
    data_path = root / data_tpl.format(episode_chunk=ep_chunk, episode_index=ep_index)
    video_tpl = info.get("video_path", DEFAULT_VIDEO_PATH)
    vids = []
    for key in info.get("features", {}):
        ft = info["features"][key]
        if ft.get("dtype") == "video":
            vids.append(root / video_tpl.format(episode_chunk=ep_chunk, video_key=key, episode_index=ep_index))
    return data_path, vids


def _assert_compatible(info_base: dict, info_new: dict) -> None:
    keys = (
        "fps",
        "features",
        "video_path",
        "data_path",
        "chunks_size",
        "video_keys",
        "camera_keys",
    )
    for k in keys:
        if info_base.get(k) != info_new.get(k):
            raise ValueError(f"数据集不兼容，字段 {k} 不一致")


def _load_source(root: Path) -> dict:
    info = load_info(root)
    tasks, task_to_idx = load_tasks(root)
    episodes = load_episodes(root)
    ep_stats = load_episodes_stats(root) or {}
    return {
        "root": root,
        "info": info,
        "tasks": tasks,
        "task_to_idx": task_to_idx,
        "episodes": episodes,
        "ep_stats": ep_stats,
    }


def _init_target(info: dict, tasks: Dict[int, str]) -> dict:
    info_new = json.loads(json.dumps(info))
    info_new.update(
        {
            "total_episodes": 0,
            "total_frames": 0,
            "total_chunks": 0,
            "total_videos": 0,
            "splits": {"train": "0:0"},
            "total_tasks": len(tasks),
        }
    )
    return info_new


def _write_tasks(tasks: Dict[int, str], root: Path) -> None:
    task_items = [{"task_index": idx, "task": task} for idx, task in sorted(tasks.items())]
    write_jsonlines(task_items, root / TASKS_PATH)


def merge_datasets(sources: List[Path], target_root: Path, repo_id: str, overwrite: bool = False) -> None:
    if target_root.exists():
        if not overwrite:
            raise RuntimeError(f"目标目录已存在：{target_root}，请使用 --overwrite 或更换目录")
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / "meta").mkdir(exist_ok=True)

    loaded = [_load_source(Path(src).expanduser()) for src in sources]
    base = loaded[0]
    info_base = base["info"]
    tasks_map: Dict[int, str] = dict(base["tasks"])
    task_str_to_idx = {v: k for k, v in tasks_map.items()}

    # 校验兼容性
    for item in loaded[1:]:
        _assert_compatible(info_base, item["info"])
        for t in item["tasks"].values():
            if t not in task_str_to_idx:
                new_idx = len(task_str_to_idx)
                task_str_to_idx[t] = new_idx
                tasks_map[new_idx] = t

    info_target = _init_target(info_base, tasks_map)
    total_eps = 0
    total_frames = 0
    episode_entries = []
    episode_stats_entries = []

    for item in loaded:
        info_src = item["info"]
        eps = item["episodes"]
        stats_map = item["ep_stats"]
        for src_ep_idx in sorted(eps):
            ep_meta = eps[src_ep_idx]
            new_idx = total_eps
            length = int(ep_meta.get("length", 0) or 0)

            data_src, vids_src = _dataset_paths(item["root"], info_src, src_ep_idx)
            data_tgt, vids_tgt = _dataset_paths(target_root, info_target, new_idx)
            data_tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data_src, data_tgt)
            for vs, vt in zip(vids_src, vids_tgt):
                if vs.exists():
                    vt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(vs, vt)

            episode_entries.append(
                {
                    "episode_index": new_idx,
                    "tasks": ep_meta.get("tasks", []),
                    "length": length,
                }
            )
            if src_ep_idx in stats_map:
                episode_stats_entries.append(
                    {"episode_index": new_idx, "stats": serialize_dict(stats_map[src_ep_idx])}
                )

            total_eps += 1
            total_frames += length

    # 汇总统计
    stats_agg = {}
    if episode_stats_entries:
        stats_agg = aggregate_stats([entry["stats"] for entry in episode_stats_entries])

    info_target["total_episodes"] = total_eps
    info_target["total_frames"] = total_frames
    info_target["total_videos"] = total_eps * len(info_target.get("video_keys", []))
    info_target["total_chunks"] = (total_eps + info_target.get("chunks_size", DEFAULT_CHUNK_SIZE) - 1) // info_target.get(
        "chunks_size", DEFAULT_CHUNK_SIZE
    )
    info_target["splits"] = {"train": f"0:{total_eps}"}
    info_target["total_tasks"] = len(tasks_map)
    info_target["repo_id"] = repo_id

    # 写 meta
    write_info(info_target, target_root)
    _write_tasks(tasks_map, target_root)
    write_stats(stats_agg, target_root)
    if episode_entries:
        write_jsonlines(episode_entries, target_root / EPISODES_PATH)
    else:
        write_jsonlines([], target_root / EPISODES_PATH)
    if episode_stats_entries:
        write_jsonlines(episode_stats_entries, target_root / EPISODES_STATS_PATH)
    else:
        write_jsonlines([], target_root / EPISODES_STATS_PATH)

    # 重建空数据集结构（data/videos 目录在复制时已创建）
    (target_root / "data").mkdir(exist_ok=True)
    (target_root / "videos").mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并多个同构 LeRobot 数据集")
    parser.add_argument("--sources", nargs="+", required=True, help="源数据集根目录列表（含 meta/data/videos）")
    parser.add_argument("--target", required=True, help="合并后的目标目录")
    parser.add_argument("--repo-id", required=True, help="目标数据集 repo_id，用于写入 info.json")
    parser.add_argument("--overwrite", action="store_true", help="如目标目录已存在则覆盖")
    return parser.parse_args()


def main():
    args = parse_args()
    merge_datasets(args.sources, Path(args.target), repo_id=args.repo_id, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
