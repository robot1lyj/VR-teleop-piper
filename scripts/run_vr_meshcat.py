"""运行 VR 手柄 -> Piper Meshcat 遥操作演示。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np
import pinocchio as pin

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from webrtc_endpoint import VRWebRTCServer
from robot.ik import ArmIK
from robot.teleop import ArmTeleopSession, IncrementalPoseMapper


class TeleopPipeline:
    """兼容 VRWebRTCServer 的管线包装器，内部调用遥操作会话。"""

    def __init__(
        self,
        session: ArmTeleopSession,
        allowed_hands: Iterable[str],
        reference_translation: np.ndarray,
        reference_rotation: np.ndarray,
    ) -> None:
        self.session = session
        self.allowed_hands = set(allowed_hands)
        self.reference_translation = np.asarray(reference_translation, dtype=float).reshape(3)
        self.reference_rotation = np.asarray(reference_rotation, dtype=float).reshape(3, 3)
        self._apply_reference()

    def process_message(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将 VR 报文送入 TeleopSession，并以字典形式返回日志信息。"""

        results = self.session.handle_vr_payload(payload)
        summarized: List[Dict[str, Any]] = []
        for item in results:
            summarized.append(
                {
                    "hand": item.hand,
                    "success": item.success,
                    "info": item.info,
                    "gripper_closed": item.gripper_closed,
                    "target_translation": item.target.translation.tolist(),
                }
            )
        _log_results(results)
        return summarized

    def reset(self) -> None:
        """当连接断开时清理控制状态。"""

        for hand in list(self.allowed_hands):
            self.session.clear_reference_pose(hand)
        self._apply_reference()

    def _apply_reference(self) -> None:
        for hand in self.allowed_hands:
            self.session.set_reference_pose(hand, self.reference_translation, self.reference_rotation)


def _log_results(results: List[Any]) -> None:
    """统一打印 IK 结果，便于实时与离线共享逻辑。"""

    logger = logging.getLogger(__name__)
    for item in results:
        if getattr(item, "success", False):
            logger.info(
                "[%s] IK OK: joints=%s",
                item.hand,
                np.array2string(item.joints, precision=4) if item.joints is not None else "(none)",
            )
        else:
            logger.warning("[%s] IK Fail: %s", item.hand, item.info)


def _iter_trajectory(path: pathlib.Path) -> Iterator[Dict[str, Any]]:
    """逐行读取 JSONL 轨迹，提取 elapsed 与 payload 字段。"""

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logging.getLogger(__name__).warning("跳过无效轨迹行: %s", line[:80])
                continue
            payload = record.get("raw") or record.get("payload")
            if not payload:
                continue
            yield {
                "elapsed": float(record.get("elapsed", 0.0)),
                "payload": payload,
            }


def _replay_trajectory_sync(
    session: ArmTeleopSession,
    trajectory: List[Dict[str, Any]],
    speed: float,
    loop_playback: bool,
) -> None:
    logger = logging.getLogger(__name__)
    logger.info("开始离线轨迹回放，共 %d 帧，速度倍率 %.2f", len(trajectory), speed)
    speed = max(speed, 1e-6)
    try:
        while True:
            start = time.perf_counter()
            for frame in trajectory:
                target_time = start + frame["elapsed"] / speed
                remaining = target_time - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                results = session.handle_vr_payload(frame["payload"])
                _log_results(results)
            if not loop_playback:
                break
    finally:
        logger.info("轨迹回放完成")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VR 手柄 Meshcat 遥操作演示")
    parser.add_argument("--urdf", default="piper_description/urdf/piper_description.urdf", help="Piper URDF 路径")
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket 信令监听地址")
    parser.add_argument("--port", type=int, default=8442, help="WebSocket 信令端口")
    parser.add_argument("--channel", default="controller", help="DataChannel 名称")
    parser.add_argument("--scale", type=float, default=1.0, help="位置增量缩放")
    parser.add_argument("--no-meshcat", action="store_true", help="禁用 Meshcat 可视化，只做轨迹回放/IK")
    parser.add_argument("--hands", choices=["both", "left", "right"], default="right", help="参与遥操作的手柄")
    parser.add_argument("--no-stun", action="store_true", help="禁用 STUN，仅用于局域网")
    parser.add_argument("--stun", action="append", default=[], metavar="URL", help="额外 STUN 地址")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--no-collision", action="store_true", help="关闭 IK 自碰撞检测，避免模型几何缺失导致的崩溃")
    parser.add_argument("--replay", help="指定录制的 JSONL 轨迹文件进行离线回放")
    parser.add_argument("--replay-speed", type=float, default=1.0, help="回放速度倍率")
    parser.add_argument("--replay-loop", action="store_true", help="循环回放轨迹")
    return parser


def _resolve_hands(arg: str) -> set[str]:
    if arg == "both":
        return {"left", "right"}
    return {arg}


def build_session(args: argparse.Namespace) -> tuple[ArmTeleopSession, TeleopPipeline]:
    hands = _resolve_hands(args.hands)

    ik = ArmIK(
        urdf_path=args.urdf,
        use_meshcat=not args.no_meshcat,
        smooth_weight=0.05,
        position_weight=20.0,
        orientation_weight=20.0,
    )

    mapper = IncrementalPoseMapper(scale=args.scale, allowed_hands=hands)
    session = ArmTeleopSession(ik_solver=ik, mapper=mapper, check_collision=not args.no_collision)

    model = ik.reduced_robot.model
    data = ik.reduced_robot.data
    q_home = pin.neutral(model)
    pin.forwardKinematics(model, data, q_home)
    pin.updateFramePlacements(model, data)
    ee_id = model.getFrameId("ee")
    if ee_id < 0 or ee_id >= len(data.oMf):
        raise RuntimeError("无法找到名为 'ee' 的末端 Frame，请确认 ArmIK 初始化已刷新 model/data")
    base_pose = data.oMf[ee_id]
    for hand in hands:
        session.set_reference_pose(hand, base_pose.translation, base_pose.rotation)

    pipeline = TeleopPipeline(
        session=session,
        allowed_hands=hands,
        reference_translation=base_pose.translation,
        reference_rotation=base_pose.rotation,
    )
    return session, pipeline


async def run_live(args: argparse.Namespace, pipeline: TeleopPipeline) -> None:
    stun_servers = [] if args.no_stun else list(args.stun)
    server = VRWebRTCServer(
        host=args.host,
        port=args.port,
        pipeline=pipeline,  # type: ignore[arg-type]
        channel_name=args.channel,
        stun_servers=stun_servers,
    )

    await server.start()
    logging.info("Meshcat 遥操作会话已启动，等待 VR 手柄接入……")
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await server.stop()


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    session, pipeline = build_session(args)

    if args.replay:
        trajectory_path = pathlib.Path(args.replay)
        frames = list(_iter_trajectory(trajectory_path))
        if not frames:
            logging.error("轨迹文件为空或解析失败: %s", trajectory_path)
            raise SystemExit(1)
        logging.info("读取轨迹帧数：%d", len(frames))
        _replay_trajectory_sync(session, frames, args.replay_speed, args.replay_loop)
    else:
        asyncio.run(run_live(args, pipeline))
