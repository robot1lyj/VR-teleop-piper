"""基于配置快速预览机械臂安装姿态的 Meshcat 工具。"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import Any

import numpy as np
import pinocchio as pin

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "run_vr_meshcat.json"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from robot.ik import ArmIK  # noqa: E402  # pylint: disable=wrong-import-position


def _ensure_path(path_str: str | pathlib.Path) -> pathlib.Path:
    """将可能为相对路径的字符串转换为绝对路径。"""

    path = pathlib.Path(path_str)
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    """读取 JSON 配置，失败时报错退出。"""

    if not path.exists():
        raise SystemExit(f"配置文件不存在: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # pragma: no cover - 仅用于诊断
        raise SystemExit(f"读取配置文件失败: {exc}")

    if not isinstance(data, dict):
        raise SystemExit("配置文件必须是 JSON 对象")
    return data


def _parse_mount_rpy_deg(value: Any | None) -> np.ndarray | None:
    """解析配置中的安装姿态（单位度），允许列表或逗号分隔字符串。"""

    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        items = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]

    if len(items) != 3:
        raise ValueError("mount_rpy_deg 需要提供 roll、pitch、yaw 三个角度")

    try:
        return np.array([float(item) for item in items], dtype=float)
    except ValueError as exc:  # pragma: no cover - 仅用于诊断
        raise ValueError("mount_rpy_deg 必须全部为浮点数") from exc


def _parse_home_q_deg(value: Any | None) -> np.ndarray | None:
    """解析自定义初始关节角（单位度）。"""

    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        items = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]

    if not items:
        return None

    try:
        return np.array([float(item) for item in items], dtype=float)
    except ValueError as exc:  # pragma: no cover - 仅用于诊断
        raise ValueError("home_q_deg 必须全部为浮点数") from exc


def _rotation_from_rpy_deg(rpy_deg: np.ndarray | None) -> np.ndarray:
    """将 RPY 角（度）转换为旋转矩阵，未指定时返回单位矩阵。"""

    if rpy_deg is None:
        return np.eye(3)

    rpy_deg = np.asarray(rpy_deg, dtype=float).reshape(3)
    rpy_rad = np.deg2rad(rpy_deg)

    if hasattr(pin.rpy, "rpyToMatrix"):
        return pin.rpy.rpyToMatrix(rpy_rad)
    if hasattr(pin.rpy, "matrixFromRpy"):
        return pin.rpy.matrixFromRpy(rpy_rad)
    utils = getattr(pin, "utils", None)
    if utils is not None and hasattr(utils, "rpyToMatrix"):
        return utils.rpyToMatrix(rpy_rad)
    raise RuntimeError("当前 Pinocchio 版本不支持 RPY 转换函数")


def _parse_mount_offset(value: Any | None) -> np.ndarray | None:
    """解析基座平移偏置（米）。"""

    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        items = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]

    if len(items) != 3:
        raise ValueError("mount_offset 需要提供 xyz 三个分量")

    try:
        return np.array([float(item) for item in items], dtype=float)
    except ValueError as exc:  # pragma: no cover - 仅用于诊断
        raise ValueError("mount_offset 必须全部为浮点数") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""

    parser = argparse.ArgumentParser(description="Meshcat 预览机械臂安装姿态")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="配置文件路径（JSON）")
    parser.add_argument("--urdf", help="可选：覆盖配置中的 URDF 路径")
    parser.add_argument("--mount-rpy-deg", help="可选：直接指定安装姿态，覆盖配置")
    parser.add_argument("--mount-offset", help="可选：直接指定平移偏置 (m)，覆盖配置，格式 'x,y,z'")
    parser.add_argument("--home-q-deg", help="可选：指定初始关节角（度），覆盖配置")
    parser.add_argument("--log-level", default="info", help="日志等级，如 info/debug")
    return parser


def main() -> None:
    """主逻辑：加载配置 -> 创建 ArmIK（启用 Meshcat）-> 应用安装姿态并展示。"""

    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)

    config_path = _ensure_path(args.config)
    config = _load_config(config_path)

    urdf_path = args.urdf or config.get("urdf", "piper_description/urdf/piper_description.urdf")
    urdf_path = _ensure_path(urdf_path)

    mount_override = args.mount_rpy_deg if args.mount_rpy_deg is not None else config.get("mount_rpy_deg")
    mount_rpy_deg = _parse_mount_rpy_deg(mount_override)
    offset_override = args.mount_offset if args.mount_offset is not None else config.get("mount_offset")
    mount_offset = _parse_mount_offset(offset_override)
    home_override = args.home_q_deg if args.home_q_deg is not None else config.get("home_q_deg")
    home_q_deg = _parse_home_q_deg(home_override)

    logger.info("使用配置文件: %s", config_path)
    logger.info("加载 URDF: %s", urdf_path)
    if mount_rpy_deg is None:
        logger.info("安装姿态: 水平（未额外旋转）")
    else:
        logger.info("安装姿态 RPY(度): [%.2f, %.2f, %.2f]", *mount_rpy_deg)
    if mount_offset is None:
        logger.info("安装平移: [0.00, 0.00, 0.00] m")
    else:
        logger.info("安装平移 (m): [%.3f, %.3f, %.3f]", *mount_offset)
    if home_q_deg is None:
        logger.info("初始关节角: 使用 URDF 中立姿态")
    else:
        logger.info("初始关节角(度): %s", ", ".join(f"{val:.2f}" for val in home_q_deg))

    # ArmIK 默认会创建 Meshcat 可视化器并显示中立位姿
    ik = ArmIK(urdf_path=str(urdf_path), use_meshcat=True)

    home_q_rad: np.ndarray | None = None
    if home_q_deg is not None:
        home_q_rad = np.deg2rad(home_q_deg)
        model = ik.reduced_robot.model
        if home_q_rad.shape[0] != model.nq:
            logger.error("home_q_deg 长度需为 %d，实际 %d", model.nq, home_q_rad.shape[0])
            home_q_rad = None
        else:
            ik.set_seed(home_q_rad)
            ik.q_last = home_q_rad.copy()

    if ik.use_meshcat:
        rotation = _rotation_from_rpy_deg(mount_rpy_deg)
        mount_hom = np.eye(4)
        mount_hom[:3, :3] = rotation
        if mount_offset is not None:
            mount_hom[:3, 3] = mount_offset
        try:
            ik.vis.viewer["pinocchio"].set_transform(mount_hom)
            if home_q_rad is not None:
                ik.vis.display(home_q_rad)
            if hasattr(ik, "_meshcat_base_transform"):
                ik._meshcat_base_transform = mount_hom
            logger.info("Meshcat 已应用安装姿态/初始关节角")
        except Exception as exc:  # pragma: no cover - Meshcat 异常仅用于调试
            logger.warning("Meshcat 根节点变换设置失败: %s", exc)
    elif home_q_rad is not None:
        logger.info("Meshcat 未启用，但已设置 IK 初始姿态")

    logger.info("Meshcat 预览已启动，可在浏览器中查看当前安装姿态")

    input("按 Enter 退出预览……\n")
    logger.info("已退出预览模式")


if __name__ == "__main__":
    main()
