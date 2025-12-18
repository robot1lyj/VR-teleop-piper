#!/usr/bin/env python3
"""Plot telemetry records captured from Piper teleop sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8")


def _load_records(path: Path) -> List[dict[str, Any]]:
    records: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)
    return records


def _plot_joint_traces(records: List[dict[str, Any]], save_path: Path | None) -> None:
    if not records:
        raise SystemExit("No telemetry records found.")

    times = np.array([rec.get("ts_sent", rec.get("ts_ik", 0.0)) for rec in records], dtype=float)
    times -= times[0]

    q_ik = np.array([rec.get("q_ik", []) for rec in records], dtype=float)
    q_cmd = np.array([rec.get("q_cmd", []) for rec in records], dtype=float)
    dof = q_ik.shape[1]

    q_meas_list = []
    for rec in records:
        q_meas = rec.get("q_meas")
        if q_meas is None:
            q_meas_list.append([np.nan] * dof)
        else:
            q_meas_list.append(q_meas)
    q_meas = np.array(q_meas_list, dtype=float)

    dt_send = np.array([rec.get("dt_send", np.nan) for rec in records], dtype=float)

    q_ik_deg = np.degrees(q_ik)
    q_cmd_deg = np.degrees(q_cmd)
    q_meas_deg = np.degrees(q_meas)

    rows = int(np.ceil(dof / 2))
    fig, axes = plt.subplots(rows, 2, sharex=True, figsize=(12, 1.8 * rows + 2))
    axes = axes.flatten()

    for idx in range(dof):
        ax = axes[idx]
        ax.plot(times, q_ik_deg[:, idx], label="IK", linewidth=1.2)
        ax.plot(times, q_cmd_deg[:, idx], label="Command", linewidth=1.2)
        if not np.all(np.isnan(q_meas_deg[:, idx])):
            ax.plot(times, q_meas_deg[:, idx], label="Measured", linewidth=1.0)
        ax.set_title(f"Joint J{idx + 1}")
        ax.set_ylabel("Angle (deg)")
        ax.grid(True, alpha=0.3)
    bottom_start = (rows - 1) * 2
    bottom_indices = [idx for idx in range(bottom_start, bottom_start + 2) if idx < dof]
    for idx in bottom_indices:
        axes[idx].set_xlabel("Time (s)")
    for ax in axes[dof:]:
        fig.delaxes(ax)
    axes[0].legend(loc="upper right")
    fig.suptitle("Piper Teleop Joint Trace", fontsize=14)
    fig.text(0.5, 0.02, "All angles in degrees; time aligned to first sample.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))

    fig_dt, ax_dt = plt.subplots(figsize=(12, 2.4))
    ax_dt.plot(times, dt_send * 1000.0, color="#d97706", linewidth=1.5)
    ax_dt.set_title("Command Interval")
    ax_dt.set_xlabel("Time (s)")
    ax_dt.set_ylabel("dt_send (ms)")
    ax_dt.grid(True, alpha=0.3)
    fig_dt.tight_layout()

    if save_path:
        base = save_path
        base.parent.mkdir(parents=True, exist_ok=True)
        name_root = base.stem if base.suffix else base.name
        joints_path = base.parent / f"{name_root}_joints.png"
        dt_path = base.parent / f"{name_root}_dt.png"
        fig.savefig(joints_path, dpi=160)
        fig_dt.savefig(dt_path, dpi=160)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise Piper teleop telemetry JSONL logs.")
    parser.add_argument("telemetry_file", help="Path to telemetry JSONL file")
    parser.add_argument(
        "--save", dest="save", help="Save plots with given path prefix (without extension)"
    )
    args = parser.parse_args()

    records = _load_records(Path(args.telemetry_file))
    save_path = Path(args.save) if args.save else None
    _plot_joint_traces(records, save_path)


if __name__ == "__main__":
    main()
