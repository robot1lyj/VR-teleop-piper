"""VR 手柄二进制协议 v1 解包实现。"""

from __future__ import annotations

import struct
from typing import Any, Dict, Optional

MAGIC_VRP1 = 0x31505256  # "VRP1" little-endian
PACKET_SIZE_V1 = 84
HEADER_SIZE_V1 = 12
BLOCK_SIZE_V1 = 36
LEFT_OFFSET_V1 = HEADER_SIZE_V1
RIGHT_OFFSET_V1 = HEADER_SIZE_V1 + BLOCK_SIZE_V1


def _parse_block(view: memoryview, offset: int, hand: str) -> Dict[str, Any]:
    pos_x, pos_y, pos_z, qx, qy, qz, qw, trigger = struct.unpack_from("<8f", view, offset)
    grip_active = bool(view[offset + 32])
    menu_pressed = bool(view[offset + 33])
    return {
        "hand": hand,
        "position": {"x": float(pos_x), "y": float(pos_y), "z": float(pos_z)},
        "quaternion": {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)},
        "gripActive": grip_active,
        "trigger": float(trigger),
        "menuPressed": menu_pressed,
    }


def decode_frame(data: bytes) -> Optional[Dict[str, Any]]:
    """若 data 为 VRP1 v1 包则返回解析后的 payload，否则返回 None。"""

    if len(data) != PACKET_SIZE_V1:
        return None

    view = memoryview(data)
    try:
        magic, client_ts_ms, client_dt_ms, flags, _pad = struct.unpack_from("<IIHBB", view, 0)
    except struct.error:
        return None

    if magic != MAGIC_VRP1:
        return None

    left_present = bool(flags & 0x01)
    right_present = bool(flags & 0x02)
    if not (left_present or right_present):
        return None

    payload: Dict[str, Any] = {
        "client_ts": int(client_ts_ms),
        "client_dt": int(client_dt_ms),
    }
    if left_present:
        payload["leftController"] = _parse_block(view, LEFT_OFFSET_V1, "left")
    if right_present:
        payload["rightController"] = _parse_block(view, RIGHT_OFFSET_V1, "right")

    return payload

