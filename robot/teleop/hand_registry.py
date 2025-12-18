"""按手柄管理 IK 实例与默认选择的简单注册表。"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Tuple

from robot.ik import BaseArmIK


class HandRegistry:
    """维护每只手的 IK 求解器映射，提供统一的访问接口。"""

    def __init__(self, ik_map: Dict[str, BaseArmIK]) -> None:
        if not ik_map:
            raise ValueError("HandRegistry 需要至少一个 hand->IK 映射")
        self._ik_map = dict(ik_map)
        self.default_hand = sorted(self._ik_map.keys())[0]

    def get(self, hand: str | None = None) -> BaseArmIK:
        if hand is not None and hand in self._ik_map:
            return self._ik_map[hand]
        return self._ik_map[self.default_hand]

    def items(self) -> Iterator[Tuple[str, BaseArmIK]]:
        return iter(self._ik_map.items())

    def hands(self) -> Iterable[str]:
        return self._ik_map.keys()

    def __contains__(self, hand: str) -> bool:  # pragma: no cover - 辅助接口
        return hand in self._ik_map


__all__ = ["HandRegistry"]
