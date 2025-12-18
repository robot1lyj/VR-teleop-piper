"""记录 VR 手柄 DataChannel 帧为轨迹文件，便于后续离线回放。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import time
from typing import Any, Dict, Iterable, List, Optional

import sys

# 将仓库根目录加入 sys.path，避免直接运行脚本时找不到模块
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vr_runtime.webrtc_endpoint import VRWebRTCServer


class TrajectoryRecorder:
    """接收 VR 手柄报文并将其保存为 JSONL。"""

    def __init__(
        self,
        allowed_hands: Iterable[str],
        output_path: pathlib.Path,
        *,
        auto_start: bool = False,
        auto_stop: bool = False,
        start_grip_threshold: int = 3,
    ) -> None:
        self.allowed_hands = set(allowed_hands)
        self.output_path = output_path
        self._start_time: Optional[float] = None
        self._file = self.output_path.open("w", encoding="utf-8")
        self._auto_start = bool(auto_start)
        self._auto_stop = bool(auto_stop)
        self._start_grip_threshold = max(1, int(start_grip_threshold))
        self._active = not self._auto_start  # 自动开录时等待握持触发
        self._stop_requested = False  # 由菜单键长按请求停止
        self._grip_frames: Dict[str, int] = {hand: 0 for hand in self.allowed_hands}

    def process_message(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """兼容 VRWebRTCServer 的处理接口，将手柄报文写入文件。"""

        now = time.time()
        if self._stop_requested:
            return []
        normalized = self._normalize_payload(payload)
        if not normalized:
            return []

        self._update_grip_state(normalized)

        if not self._active:
            if self._should_start():
                self._start_time = now
                self._active = True
                logging.getLogger(__name__).info("✅ 检测到握持信号，开始写入轨迹")
            else:
                return []

        if self._start_time is None:
            self._start_time = now
        if self._file.closed:
            self._file = self.output_path.open("a", encoding="utf-8")

        elapsed = now - self._start_time

        # 记录原始报文与归一化结果，方便后续离线回放
        record = {
            "timestamp": now,
            "elapsed": elapsed,
            "payload": normalized,
            "raw": payload,
        }
        json.dump(record, self._file, ensure_ascii=False)
        self._file.write("\n")
        self._file.flush()

        if self._auto_stop and self._check_stop(normalized):
            logging.getLogger(__name__).info("🛑 检测到停止信号，自动结束录制")
            self._stop_requested = True
            self._active = False
            self.close()

        logging.getLogger(__name__).debug("保存帧: elapsed=%.3f s", elapsed)
        # 录制阶段无需返回给服务器任何信息
        return []

    def reset(self) -> None:
        """连接断开时关闭文件句柄。"""
        self._start_time = None
        if not self._stop_requested:
            self._active = not self._auto_start
        for hand in self._grip_frames:
            self._grip_frames[hand] = 0

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """只保留参与录制的手柄键，既支持多手柄也支持单手柄格式。"""

        if "leftController" in payload or "rightController" in payload:
            return {
                hand: value
                for hand, value in payload.items()
                if hand in {"leftController", "rightController"}
                and hand.replace("Controller", "") in self.allowed_hands
            }

        hand = payload.get("hand")
        if hand in self.allowed_hands:
            return {hand: payload}
        return {}

    def _update_grip_state(self, normalized: Dict[str, Any]) -> None:
        for key, info in normalized.items():
            hand = key.replace("Controller", "")
            if hand not in self._grip_frames:
                continue
            grip_active = bool(info.get("gripActive", False))
            if grip_active:
                self._grip_frames[hand] += 1
            else:
                self._grip_frames[hand] = 0

    def _check_stop(self, normalized: Dict[str, Any]) -> bool:
        return any(bool(info.get("menuPressed")) for info in normalized.values())

    def _should_start(self) -> bool:
        if not self._auto_start:
            return False
        return any(frames >= self._start_grip_threshold for frames in self._grip_frames.values())


async def main() -> None:
    parser = argparse.ArgumentParser(description="记录 VR 手柄轨迹")
    parser.add_argument("output", help="保存轨迹的 JSONL 文件路径")
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket 信令监听地址")
    parser.add_argument("--port", type=int, default=8442, help="WebSocket 信令端口")
    parser.add_argument("--channel", default="controller", help="DataChannel 名称")
    parser.add_argument("--hands", choices=["both", "left", "right"], default="right")
    parser.add_argument("--no-stun", action="store_true")
    parser.add_argument("--stun", action="append", default=[], metavar="URL")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--auto-start", action="store_true", help="检测握持后自动开始录制")
    parser.add_argument("--auto-stop", action="store_true", help="检测菜单键长按后自动停止")
    parser.add_argument("--start-grip-threshold", type=int, default=3, help="连续握持帧数阈值")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if args.hands == "both":
        hands = {"left", "right"}
    else:
        hands = {args.hands}

    output_path = pathlib.Path(args.output)
    recorder = TrajectoryRecorder(
        allowed_hands=hands,
        output_path=output_path,
        auto_start=args.auto_start,
        auto_stop=args.auto_stop,
        start_grip_threshold=args.start_grip_threshold,
    )

    stun_servers = [] if args.no_stun else list(args.stun)
    server = VRWebRTCServer(
        host=args.host,
        port=args.port,
        pipeline=recorder,  # type: ignore[arg-type]
        channel_name=args.channel,
        stun_servers=stun_servers,
    )

    await server.start()
    logging.info("轨迹录制开始，输出文件: %s", output_path)
    try:
        while True:
            if recorder.stop_requested:
                break
            await asyncio.sleep(0.1)
    finally:
        await server.stop()
        recorder.close()
        logging.info("轨迹录制结束")


if __name__ == "__main__":
    asyncio.run(main())
