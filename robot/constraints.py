"""关节级硬约束与步长管理工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover - 构建环境缺少 pinocchio 时给出清晰提示
    raise RuntimeError("JointConstraintManager 依赖 pinocchio") from exc


def _to_float(value: float) -> float:
    """把任意数值转换成 float，遇到非法输入直接抛错。"""

    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 输入非法场景
        raise ValueError(f"无法将 {value!r} 转成浮点数") from exc


def _parse_pair(values: Iterable[float]) -> Tuple[float, float]:
    """读取长度至少为 2 的序列并返回 (low, high)。"""

    seq = list(values)
    if len(seq) < 2:
        raise ValueError("关节区间需要提供至少两个数值")
    low = _to_float(seq[0])
    high = _to_float(seq[1])
    if low > high:
        raise ValueError(f"下界 {low} 不能大于上界 {high}")
    return low, high


@dataclass
class _ConstraintConfig:
    """归一化后的约束配置。"""

    hard_lower: np.ndarray | None
    hard_upper: np.ndarray | None
    step_limits: np.ndarray | None
    filter_alpha: float | None


class JointConstraintManager:
    """统一管理逐关节硬约束与步长限制，避免在 IK 主流程里散落大量 if。"""

    @classmethod
    def from_config(cls, model: pin.Model, config: Dict) -> "JointConstraintManager | None":
        """根据配置构建管理器，没有有效约束时返回 None。"""

        if not config:
            return None

        parsed = cls._normalize_config(model, config)
        if (
            parsed.hard_lower is None
            and parsed.hard_upper is None
            and parsed.step_limits is None
            and parsed.filter_alpha is None
        ):
            return None
        return cls(model, parsed)

    def __init__(self, model: pin.Model, parsed: _ConstraintConfig) -> None:
        self.model = model
        self.nq = model.nq
        self._filter_alpha = parsed.filter_alpha
        self._filter_state: np.ndarray | None = None

        if parsed.hard_lower is not None:
            self._hard_lower = parsed.hard_lower.copy()
        else:
            self._hard_lower = np.full(self.nq, -np.inf, dtype=float)

        if parsed.hard_upper is not None:
            self._hard_upper = parsed.hard_upper.copy()
        else:
            self._hard_upper = np.full(self.nq, np.inf, dtype=float)

        if parsed.step_limits is not None:
            self._step_limits = parsed.step_limits.copy()
        else:
            self._step_limits = np.full(self.nq, np.inf, dtype=float)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def adjust_bounds(self, q_ref: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """将硬约束与步长限制融合进已有的上下界数组中。"""

        lower_adj = np.maximum(lower, self._hard_lower)
        upper_adj = np.minimum(upper, self._hard_upper)

        if q_ref is not None and np.any(np.isfinite(self._step_limits)):
            center = np.asarray(q_ref, dtype=float)
            if self._filter_alpha is not None:
                alpha = float(self._filter_alpha)
                if not 0.0 < alpha <= 1.0:
                    raise ValueError("filter_alpha 需在 (0, 1] 范围内")
                if self._filter_state is None:
                    self._filter_state = center.copy()
                else:
                    self._filter_state = (1.0 - alpha) * self._filter_state + alpha * center
                center = self._filter_state

            lower_adj = np.maximum(lower_adj, center - self._step_limits)
            upper_adj = np.minimum(upper_adj, center + self._step_limits)

        if np.any(lower_adj > upper_adj):  # pragma: no cover - 属于配置错误
            raise ValueError("关节约束冲突：下界超过上界")
        return lower_adj, upper_adj

    def update_after_solve(self, q_solution: np.ndarray) -> None:
        """在求解成功后刷新滤波状态。"""

        if self._filter_alpha is not None:
            self._filter_state = np.asarray(q_solution, dtype=float)

    # ------------------------------------------------------------------
    # 配置解析逻辑
    # ------------------------------------------------------------------
    @classmethod
    def _normalize_config(cls, model: pin.Model, config: Dict) -> _ConstraintConfig:
        """把用户配置转成长度为 nq 的 numpy 向量。"""

        nq = model.nq
        name_to_index = cls._build_joint_index_map(model)

        hard_lower = np.full(nq, -np.inf, dtype=float)
        hard_upper = np.full(nq, np.inf, dtype=float)
        hard_mask = np.zeros(nq, dtype=bool)

        step_limits = np.full(nq, np.inf, dtype=float)
        step_mask = np.zeros(nq, dtype=bool)

        def apply_limits(container: Dict, scale: float = 1.0) -> None:
            for key, value in container.items():
                idx = cls._resolve_index(name_to_index, key)
                low, high = _parse_pair(value)
                hard_lower[idx] = scale * low
                hard_upper[idx] = scale * high
                hard_mask[idx] = True

        def apply_step(container: Dict, scale: float = 1.0) -> None:
            for key, value in container.items():
                idx = cls._resolve_index(name_to_index, key)
                limit = abs(scale * _to_float(value))
                if limit <= 0:
                    raise ValueError("步长限制必须为正数")
                step_limits[idx] = limit
                step_mask[idx] = True

        hard_limits = config.get("hard_limits")
        if isinstance(hard_limits, dict):
            apply_limits(hard_limits, scale=1.0)

        hard_limits_deg = config.get("hard_limits_deg")
        if isinstance(hard_limits_deg, dict):
            apply_limits(hard_limits_deg, scale=np.pi / 180.0)

        step_limits_cfg = config.get("step_limits")
        if isinstance(step_limits_cfg, dict):
            apply_step(step_limits_cfg, scale=1.0)

        step_limits_deg = config.get("step_limits_deg")
        if isinstance(step_limits_deg, dict):
            apply_step(step_limits_deg, scale=np.pi / 180.0)

        filter_alpha = config.get("filter_alpha")
        if filter_alpha is not None:
            filter_alpha = _to_float(filter_alpha)
            if not 0.0 < filter_alpha <= 1.0:
                raise ValueError("filter_alpha 需处于 (0, 1] 区间")

        parsed = _ConstraintConfig(
            hard_lower=hard_lower if hard_mask.any() else None,
            hard_upper=hard_upper if hard_mask.any() else None,
            step_limits=step_limits if step_mask.any() else None,
            filter_alpha=filter_alpha,
        )
        return parsed

    @staticmethod
    def _build_joint_index_map(model: pin.Model) -> Dict[str, int]:
        """生成 joint 名称到 q 索引的映射，仅支持单自由度旋转/平移关节。"""

        mapping: Dict[str, int] = {}
        for jid, joint in enumerate(model.joints):
            if joint.nq != 1:
                continue
            name = model.names[jid]
            mapping[name] = joint.idx_q
        return mapping

    @staticmethod
    def _resolve_index(mapping: Dict[str, int], key: str | int) -> int:
        """接受 joint 名称或下标并返回 q 向量中的索引。"""

        if isinstance(key, int):
            idx = int(key)
        else:
            name = str(key)
            if name not in mapping:
                raise KeyError(f"未知关节名称 {name}")
            idx = mapping[name]
        if idx < 0:
            raise ValueError("索引不能为负数")
        return idx


__all__ = ["JointConstraintManager"]
