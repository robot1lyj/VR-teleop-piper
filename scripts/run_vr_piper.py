"""VR 手柄 -> Piper 实机遥操作入口。

复用通用 VR/IK 管线求解关节角，再将指令下发到 Piper 硬件。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional

import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PIPER_CONFIG = ROOT_DIR / "configs" / "piper_teleop.json"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from robot.real.piper import PiperMotorsBus
from scripts.piper_pipeline import (
    PiperTeleopPipeline,
    extract_piper_hand_configs,
    sync_reference_with_robot,
)
from scripts.teleop_common import (
    TeleopPipeline,
    _create_config_parser,
    _load_config_file,
    build_arg_parser as build_teleop_arg_parser,
    build_session as build_teleop_session,
)
from vr_runtime.webrtc_endpoint import VRWebRTCServer


def build_parser(parent: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """生成 Piper 遥操作 CLI 解析器，包含 VR/IK 参数与硬件专属参数。"""

    parser = build_teleop_arg_parser(parent)
    parser.description = "VR 遥操作 Piper 实机（复用通用 IK 配置）"
    parser.add_argument(
        "--piper-config",
        default=str(DEFAULT_PIPER_CONFIG),
        help="Piper 机械臂硬件配置 JSON",
    )
    parser.add_argument(
        "--command-interval",
        type=float,
        default=1.0 / 90.0,
        help="连续关节指令之间的最小时间间隔（秒），默认 1/90≈0.0111；0 表示不限制",
    )
    parser.add_argument(
        "--gripper-open",
        type=float,
        default=0.04,
        help="遥操作张开时的夹爪目标值（米或 SDK 要求单位）",
    )
    parser.add_argument(
        "--gripper-closed",
        type=float,
        default=0.0,
        help="遥操作闭合时的夹爪目标值",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印关节目标，不触发真实写入/连接",
    )
    parser.add_argument(
        "--skip-home",
        action="store_true",
        help="启动时跳过 apply_calibration()，避免自动回到初始位姿",
    )
    parser.add_argument(
        "--joint-speed-limits-deg",
        type=float,
        nargs="+",
        default=[180.0, 195.0, 180.0, 225.0, 225.0, 225.0],
        help="关节最大速度限制(度/秒)，按 joint1..joint6 顺序提供",
    )
    parser.add_argument(
        "--joint-acc-limits-deg",
        type=float,
        nargs="+",
        default=[400.0, 400.0, 400.0, 450.0, 450.0, 450.0],
        help="关节最大加速度限制(度/秒^2)，按 joint1..joint6 顺序提供",
    )
    parser.add_argument(
        "--joint-error-deadband-deg",
        type=float,
        default=0.12,
        help="关节误差死区(度)，误差低于该值时视为零",
    )
    parser.add_argument(
        "--velocity-filter-window",
        type=int,
        default=5,
        help="关节速度前馈的滑动窗口（帧数），0 表示禁用",
    )
    parser.add_argument(
        "--telemetry-file",
        default="",
        help="Telemetry JSONL 输出文件路径，留空表示关闭",
    )
    parser.add_argument(
        "--telemetry-sample-measured",
        action="store_true",
        help="启用后在发送指令后同步读取实机关节用于日志",
    )
    # --- SharedMemory：控制/录制进程硬隔离 ---
    parser.add_argument(
        "--publish-shm",
        action="store_true",
        help="启用 SharedMemory 发布 q_cmd/q_meas/握持状态，供录制进程严格对齐读取（推荐本地录制使用）",
    )
    parser.add_argument(
        "--shm-name",
        default="piper_vr",
        help="SharedMemory 段名前缀，将创建 <name>_status/<name>_cmd_right/<name>_meas_right 等",
    )
    parser.add_argument(
        "--shm-cmd-capacity",
        type=int,
        default=4096,
        help="cmd ring buffer 容量（条数）；90Hz 下约可覆盖 45 秒",
    )
    parser.add_argument(
        "--shm-meas-capacity",
        type=int,
        default=2048,
        help="meas ring buffer 容量（条数）；30Hz 下约可覆盖 68 秒",
    )
    parser.add_argument(
        "--shm-meas-hz",
        type=float,
        default=60.0,
        help="measured 采样频率（Hz），在硬件写线程内采样并写入 SharedMemory；<=0 表示关闭",
    )
    return parser


def _resolve_piper_config(path_text: str) -> pathlib.Path:
    path = pathlib.Path(path_text).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path


def _load_piper_hand_overrides(config_path: pathlib.Path) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    config_data: Dict[str, Any] = {}
    if config_path.exists():
        config_data = _load_config_file(config_path)
    return extract_piper_hand_configs(config_data)


def _init_piper_buses(
    allowed_hands: List[str],
    config_path: pathlib.Path,
    hw_overrides: Dict[str, Dict[str, Any]],
    skip_home: bool,
    dry_run: bool,
) -> Dict[str, Optional[PiperMotorsBus]]:
    if dry_run:
        return {hand: None for hand in allowed_hands}

    bus_map: Dict[str, Optional[PiperMotorsBus]] = {}
    logger = logging.getLogger(__name__)
    for hand in allowed_hands:
        overrides = hw_overrides.get(hand)
        logger.info("[%s] 初始化 Piper 机械臂，配置: %s", hand, config_path)
        bus = PiperMotorsBus(config_path, overrides)
        logger.info("[%s] 使用 CAN 端口: %s", hand, bus.can_name)
        if not bus.connect(True):
            raise SystemExit(f"[{hand}] Piper 机械臂使能失败，请检查硬件与急停")
        if not skip_home:
            logger.info("[%s] 移动至初始姿态（apply_calibration）", hand)
            bus.apply_calibration()
        bus_map[hand] = bus
    return bus_map


def _sync_joint_reference(
    session,
    pipeline: PiperTeleopPipeline,
    bus_map: Dict[str, Optional[PiperMotorsBus]],
) -> None:
    logger = logging.getLogger(__name__)
    for hand, bus in bus_map.items():
        if bus is None:
            continue
        try:
            current = bus.read()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[%s] 读取实机关节失败：%s", hand, exc)
            continue
        joints_rad = np.array(
            [
                current["joint_1"],
                current["joint_2"],
                current["joint_3"],
                current["joint_4"],
                current["joint_5"],
                current["joint_6"],
            ],
            dtype=float,
        )
        logger.info(
            "[%s] 同步实机关节到 IK，起始角度(度)：%s",
            hand,
            ", ".join(f"{math.degrees(v):.2f}" for v in joints_rad),
        )
        try:
            sync_reference_with_robot(session, pipeline, joints_rad, hand)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[%s] 同步 IK 参考失败：%s", hand, exc)


def _safe_shutdown(bus_map: Dict[str, Optional[PiperMotorsBus]]) -> None:
    logger = logging.getLogger(__name__)
    for hand, bus in bus_map.items():
        if bus is None:
            continue
        try:
            logger.info("[%s] 退出遥操作：回到初始姿态待命", hand)
            bus.apply_calibration()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[%s] 回初始姿态失败: %s", hand, exc)


def build_piper_pipeline(
    args: argparse.Namespace,
    session,
    teleop_pipeline: TeleopPipeline,
    bus_map: Dict[str, Optional[PiperMotorsBus]],
    hand_pipeline_overrides: Dict[str, Dict[str, Any]],
    *,
    shm_status_writer: Any | None = None,
) -> PiperTeleopPipeline:
    return PiperTeleopPipeline(
        session=session,
        allowed_hands=teleop_pipeline.allowed_hands,
        reference_translation=teleop_pipeline.reference_translation,
        reference_rotation=teleop_pipeline.reference_rotation,
        buses=bus_map,
        command_interval=args.command_interval,
        gripper_open=args.gripper_open,
        gripper_closed=args.gripper_closed,
        dry_run=args.dry_run,
        joint_speed_limits_deg=args.joint_speed_limits_deg,
        joint_acc_limits_deg=args.joint_acc_limits_deg,
        joint_error_deadband_deg=args.joint_error_deadband_deg,
        telemetry_file=args.telemetry_file or None,
        telemetry_sample_measured=args.telemetry_sample_measured,
        velocity_filter_window=args.velocity_filter_window,
        hand_overrides=hand_pipeline_overrides,
        shm_status_writer=shm_status_writer,
    )


async def run_live(args: argparse.Namespace, pipeline: PiperTeleopPipeline) -> None:
    """在指定地址启动 WebRTC 服务并阻塞运行，直到收到 Ctrl+C。"""

    stun_servers = [] if args.no_stun else list(args.stun)
    server = VRWebRTCServer(
        host=args.host,
        port=args.port,
        pipeline=pipeline,  # type: ignore[arg-type]
        channel_name=args.channel,
        stun_servers=stun_servers,
    )

    await server.start()
    logging.info("Piper 遥操作服务已启动，等待 VR 手柄接入……")
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await server.stop()


def main(argv: Optional[List[str]] = None) -> None:
    """CLI 入口：解析配置、初始化 Piper 机械臂并运行 WebRTC 服务。"""

    argv = list(sys.argv[1:] if argv is None else argv)
    config_parent = _create_config_parser()
    config_parent.set_defaults(config=str(DEFAULT_PIPER_CONFIG))
    config_args, remaining = config_parent.parse_known_args(argv)

    config_path = pathlib.Path(config_args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (ROOT_DIR / config_path).resolve()

    config_data: Dict[str, Any] = {}
    if config_path.exists():
        config_data = _load_config_file(config_path)

    parser = build_parser(config_parent)
    if config_data:
        known = {action.dest for action in parser._actions}
        overrides: Dict[str, Any] = {}
        for key, value in config_data.items():
            if key in known:
                overrides[key] = value
        parser.set_defaults(**overrides)

    args = parser.parse_args(remaining)
    args.config = str(config_path)
    if args.piper_config == str(DEFAULT_PIPER_CONFIG) and args.config != str(DEFAULT_PIPER_CONFIG):
        args.piper_config = args.config

    piper_config_path = _resolve_piper_config(args.piper_config)
    hand_hw_overrides, hand_pipeline_overrides = _load_piper_hand_overrides(piper_config_path)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)
    logger.info("使用配置文件: %s", args.config)

    session, teleop_pipeline = build_teleop_session(args)
    allowed_hands = sorted(teleop_pipeline.allowed_hands)
    if not allowed_hands:
        raise SystemExit("配置未启用任何手柄，无法建立遥操作链路")

    bus_map = _init_piper_buses(
        allowed_hands,
        piper_config_path,
        hand_hw_overrides,
        skip_home=args.skip_home,
        dry_run=args.dry_run,
    )

    shm_objects: list[Any] = []
    shm_status_writer: Any | None = None
    if args.publish_shm:
        if args.dry_run:
            logger.warning("dry-run 模式下不启用 SharedMemory 发布（无硬件写线程）。")
        else:
            try:
                from vr_runtime.shm_ring import ShmRingWriter, ShmStatusWriter
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"导入 SharedMemory 协议失败：{exc}") from exc

            base = str(args.shm_name)
            shm_status_writer = ShmStatusWriter.create_or_replace(f"{base}_status")
            shm_objects.append(shm_status_writer)
            try:
                shm_status_writer.update(0)
            except Exception:
                pass

            for hand, bus in bus_map.items():
                if bus is None:
                    continue
                cmd_writer = ShmRingWriter.create_or_replace(
                    f"{base}_cmd_{hand}", capacity=int(args.shm_cmd_capacity)
                )
                meas_writer = ShmRingWriter.create_or_replace(
                    f"{base}_meas_{hand}", capacity=int(args.shm_meas_capacity)
                )
                shm_objects.extend([cmd_writer, meas_writer])
                bus.enable_shm_publish(
                    cmd_writer=cmd_writer,
                    meas_writer=meas_writer,
                    meas_hz=float(args.shm_meas_hz),
                )

            logger.info("SharedMemory 发布已启用：prefix=%s hands=%s", base, allowed_hands)

    pipeline = build_piper_pipeline(
        args,
        session,
        teleop_pipeline,
        bus_map,
        hand_pipeline_overrides,
        shm_status_writer=shm_status_writer,
    )

    if not args.dry_run:
        _sync_joint_reference(session, pipeline, bus_map)

    try:
        asyncio.run(run_live(args, pipeline))
    finally:
        try:
            pipeline.shutdown()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("关闭指令线程失败: %s", exc)

        _safe_shutdown(bus_map)
        # 先禁用总线线程的 shm 发布，避免 close/unlink 时并发写入。
        if shm_objects:
            for bus in bus_map.values():
                if bus is None:
                    continue
                try:
                    bus.enable_shm_publish(cmd_writer=None, meas_writer=None, meas_hz=0.0)
                except Exception:
                    pass
        # SharedMemory 清理：关闭并 unlink（若录制进程仍在运行，也不影响其已 attach 的句柄）。
        for obj in reversed(shm_objects):
            try:
                if hasattr(obj, "close"):
                    obj.close()
            except Exception:
                pass
            try:
                if hasattr(obj, "unlink"):
                    obj.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
