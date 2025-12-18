"""使用 JointMitCtrl 回放 telemetry 关节轨迹（右臂示例）。

读取 JSONL 里的关节指令（q_cmd/q_ik），按时间戳重放到 Piper，
不改动现有管线，只作为 MIT 控制的独立测试入口。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

from piper_sdk import C_PiperInterface_V2


def _iter_commands(path: Path) -> Iterator[Tuple[float, List[float]]]:
    """逐条产出 (relative_ts, joints)；优先使用 q_cmd，没有则回退 q_ik。"""

    start_ts = None
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            joints: Sequence[float] | None = obj.get("q_cmd") or obj.get("q_ik")
            if joints is None:
                continue
            if len(joints) < 6:
                continue

            ts = obj.get("ts_ik") or obj.get("ts_sent") or obj.get("ts")
            if ts is None:
                continue
            if start_ts is None:
                start_ts = float(ts)
            yield float(ts) - float(start_ts), [float(v) for v in joints[:6]]


def _enable_arm(can_name: str) -> C_PiperInterface_V2:
    """上电并切换到 MIT 模式，返回已连接的接口实例。"""

    arm = C_PiperInterface_V2(can_name)
    arm.ConnectPort()
    while not arm.EnablePiper():
        time.sleep(0.01)
    arm.EnableArm(7)
    # 设置 MIT 模式，move_mode=0x04（MIT），mit_mode=0xAD
    arm.MotionCtrl_2(0x01, 0x04, 0, 0xAD)
    return arm


def _send_mit_frame(
    arm: C_PiperInterface_V2,
    joints: Iterable[float],
    kp: float,
    kd: float,
    vel_ref: float,
    torque_ref: float,
) -> None:
    """对 6 个关节逐一下发 MIT 指令。"""

    for idx, pos in enumerate(joints, start=1):
        arm.JointMitCtrl(idx, pos, vel_ref, kp, kd, torque_ref)


def main() -> None:
    parser = argparse.ArgumentParser(description="回放 telemetry 数据到 Piper（MIT 模式）")
    parser.add_argument(
        "--telemetry",
        default="output/telemetry_right_15s.jsonl",
        help="包含 q_cmd/q_ik 的 JSONL 路径，仅前 6 轴会被使用",
    )
    parser.add_argument("--can", default="can0", help="CAN 端口名")
    parser.add_argument("--kp", type=float, default=10.0, help="MIT kp 增益")
    parser.add_argument("--kd", type=float, default=0.8, help="MIT kd 增益")
    parser.add_argument(
        "--vel-ref",
        type=float,
        default=0.0,
        help="MIT 速度前馈（rad/s），默认为 0",
    )
    parser.add_argument(
        "--torque-ref",
        type=float,
        default=0.0,
        help="MIT 力矩前馈，默认为 0",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="时间缩放（<1 加快回放，>1 放慢），默认按原始节奏",
    )
    args = parser.parse_args()

    telemetry_path = Path(args.telemetry).expanduser()
    if not telemetry_path.exists():
        raise SystemExit(f"未找到 telemetry 文件: {telemetry_path}")

    commands = list(_iter_commands(telemetry_path))
    if not commands:
        raise SystemExit(f"文件中未找到有效的 q_cmd/q_ik: {telemetry_path}")

    print(f"加载 {len(commands)} 条指令，自 {telemetry_path}")

    arm = _enable_arm(args.can)
    print("已上电并切换 MIT 模式，开始回放…（Ctrl+C 退出）")

    start_wall = time.monotonic()
    for rel_ts, joints in commands:
        target_time = start_wall + rel_ts * args.time_scale
        while True:
            now = time.monotonic()
            if now >= target_time:
                break
            time.sleep(min(0.002, target_time - now))
        _send_mit_frame(arm, joints, args.kp, args.kd, args.vel_ref, args.torque_ref)

    print("回放完成。")


if __name__ == "__main__":
    main()
