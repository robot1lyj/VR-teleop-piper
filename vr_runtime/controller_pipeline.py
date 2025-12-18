"""WebRTC 信令入口，负责运行 VR 手柄姿态流水线。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import signal
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np

from .controller_state import ControllerState
from .webrtc_endpoint import VRWebRTCServer

logger = logging.getLogger(__name__)

# 数值运算的极小值阈，用于避免除 0 或不稳定的归一化。
_EPSILON = 1e-8


class ControllerPipeline:
    """纯姿态处理管线，负责将 VR 手柄数据转换为机械臂目标。"""

    def __init__(self, scale: float = 1.0, allowed_hands: Optional[Iterable[str]] = None) -> None:
        allowed = set(allowed_hands or {"left", "right"})
        invalid = allowed - {"left", "right"}
        if invalid:
            raise ValueError(f"Unsupported hand identifiers: {sorted(invalid)}")

        self.scale = scale
        self.allowed_hands: Set[str] = allowed
        # 每个手柄维持独立状态，便于在一次握持内累积偏移量。
        self.controllers: Dict[str, ControllerState] = {
            "left": ControllerState("left"),
            "right": ControllerState("right"),
        }

    def reset(self) -> None:
        """重置全部手柄的握持与触发状态。"""

        for controller in self.controllers.values():
            controller.reset_grip()
            controller.trigger_active = False

    def process_message(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理来自浏览器的数据通道消息并返回目标列表。"""

        goals: List[Dict[str, Any]] = []

        if "leftController" in payload or "rightController" in payload:
            if "leftController" in payload and "left" in self.allowed_hands:
                goal = self._handle_controller(self.controllers["left"], payload["leftController"])
                if goal:
                    goals.append(goal)
            if "rightController" in payload and "right" in self.allowed_hands:
                goal = self._handle_controller(self.controllers["right"], payload["rightController"])
                if goal:
                    goals.append(goal)
            return goals

        hand = payload.get("hand")
        if hand in self.allowed_hands:
            controller = self.controllers[hand]
            goal = self._handle_controller(controller, payload)
            if goal:
                goals.append(goal)

        return goals

    def _handle_controller(self, controller: ControllerState, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        position = payload.get("position")
        quaternion = payload.get("quaternion")
        grip_active = payload.get("gripActive", False)
        trigger = payload.get("trigger", 0)

        if position is None:
            return None

        trigger_active = trigger > 0.5
        if trigger_active != controller.trigger_active:
            controller.trigger_active = trigger_active
            logger.info("%s trigger %s", controller.hand, "ON" if trigger_active else "OFF")

        if not grip_active:
            if controller.grip_active:
                controller.reset_grip()
                logger.info("%s grip released", controller.hand)
            return None

        if not controller.grip_active:
            controller.grip_active = True
            controller.origin_position = np.array(
                [position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0)]
            )
            if quaternion:
                controller.origin_quaternion = _quaternion_from_payload(quaternion)
                controller.accumulated_quaternion = controller.origin_quaternion
            logger.info("%s grip engaged; origin locked", controller.hand)
            return None

        if quaternion:
            controller.accumulated_quaternion = _quaternion_from_payload(quaternion)

        relative = _compute_relative_position(position, controller.origin_position, self.scale)
        z_rot, x_rot = _extract_axis_angles(controller.accumulated_quaternion, controller.origin_quaternion)
        return _to_goal(controller, relative, z_rot, x_rot)


def _quaternion_from_payload(payload: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            payload.get("x", 0.0),
            payload.get("y", 0.0),
            payload.get("z", 0.0),
            payload.get("w", 1.0),
        ]
    )


def _compute_relative_position(current: Dict[str, float], origin: Optional[np.ndarray], scale: float) -> np.ndarray:
    """计算当前位置相对握持起点的位移，并乘以缩放系数。"""

    if origin is None:
        return np.zeros(3)

    delta = np.array([current.get("x", 0.0), current.get("y", 0.0), current.get("z", 0.0)]) - origin
    return delta * scale


def _normalize_quaternion(quat: Optional[np.ndarray]) -> np.ndarray:
    """返回单位化后的四元数，避免数值噪声扩大。"""

    if quat is None:
        return np.array([0.0, 0.0, 0.0, 1.0])

    norm = np.linalg.norm(quat)
    if norm < _EPSILON:
        return quat
    return quat / norm


def _quaternion_conjugate(quat: np.ndarray) -> np.ndarray:
    """求四元数的共轭，用于计算相对旋转。"""

    x, y, z, w = quat
    return np.array([-x, -y, -z, w])


def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """按数学定义相乘两个四元数（均为 x/y/z/w 顺序）。"""

    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def _quaternion_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """将单位四元数转换为 Rodrigues 旋转向量（弧度）。"""

    quat = _normalize_quaternion(quat)
    w = float(np.clip(quat[3], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    sin_half = math.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < _EPSILON:
        return np.zeros(3)
    axis = quat[:3] / sin_half
    return axis * angle


def _extract_axis_angles(current_quat: Optional[np.ndarray], origin_quat: Optional[np.ndarray]) -> tuple[float, float]:
    """从当前/初始四元数中解析 Z 轴和 X 轴的相对旋转角度。"""

    if current_quat is None or origin_quat is None:
        return 0.0, 0.0

    try:
        relative_quat = _quaternion_multiply(
            _normalize_quaternion(current_quat), _quaternion_conjugate(_normalize_quaternion(origin_quat))
        )
        rotvec = _quaternion_to_rotvec(relative_quat)
        z_rotation_deg = -math.degrees(rotvec[2])
        x_rotation_deg = math.degrees(rotvec[0])
        return z_rotation_deg, x_rotation_deg
    except Exception as exc:  # pragma: no cover - 数值边界情况
        logger.warning("Failed to extract axis angles: %s", exc)
        return 0.0, 0.0


def _to_goal(controller: ControllerState, relative: np.ndarray, z_rot: float, x_rot: float) -> Dict[str, Any]:
    """将当前状态转换为控制目标。"""

    return {
        "arm": controller.hand,
        "mode": "position",
        "target_position": relative.tolist(),
        "wrist_roll_deg": -z_rot,
        "wrist_flex_deg": -x_rot,
        "gripper_closed": not controller.trigger_active,
    }


async def _serve_forever(server: VRWebRTCServer) -> None:
    stop_event = asyncio.Event()

    def _stop(*_args: object) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:  # pragma: no cover - Windows 等平台
            logger.info("Signal handlers not supported for %s", sig)

    await server.start()
    try:
        await stop_event.wait()
    finally:
        await server.stop()


def run_vr_controller_stream() -> None:
    """命令行入口：解析参数并运行 WebRTC 信令服务器。"""

    parser = argparse.ArgumentParser(description="Standalone VR controller WebRTC stream logger")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8442, help="WebSocket 信令端口")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--hands",
        choices=["both", "left", "right"],
        default="both",
        help="选择姿态数据来源：双手柄或仅跟踪单侧手柄。",
    )
    parser.add_argument("--channel-name", default="controller", help="DataChannel 名称")
    parser.add_argument(
        "--stun",
        action="append",
        default=[],
        metavar="URL",
        help="可选 STUN 服务器地址（示例：stun:stun.l.google.com:19302）",
    )
    parser.add_argument(
        "--no-stun",
        action="store_true",
        help="禁用内置 STUN 列表，仅使用局域网 host-candidate。",
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if args.hands == "both":
        allowed_hands = {"left", "right"}
    else:
        allowed_hands = {args.hands}

    stun_servers = [] if args.no_stun else list(args.stun)

    pipeline = ControllerPipeline(scale=args.scale, allowed_hands=allowed_hands)
    server = VRWebRTCServer(
        host=args.host,
        port=args.port,
        pipeline=pipeline,
        channel_name=args.channel_name,
        stun_servers=stun_servers,
    )

    asyncio.run(_serve_forever(server))


if __name__ == "__main__":
    run_vr_controller_stream()
