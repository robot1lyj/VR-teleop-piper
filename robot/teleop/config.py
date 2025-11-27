"""Typed configuration helpers for teleop pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np


def _as_rpy_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError("RPY 需要提供 3 个角度")
    return arr


@dataclass
class TeleopConfig:
    urdf: str
    hands: Set[str]
    scale: float
    mount_rpy_deg: Optional[np.ndarray] = None
    hand_mount_rpy_deg: Dict[str, np.ndarray] = field(default_factory=dict)
    hand_home_q_deg: Dict[str, np.ndarray] = field(default_factory=dict)
    home_q_deg: Optional[np.ndarray] = None
    joint_reg_weights: Optional[List[float]] = None
    joint_smooth_weights: Optional[List[float]] = None
    swivel_range_deg: Optional[float] = None
    trust_region: Any = None
    joint_constraints: Optional[Dict[str, Any]] = None
    hand_joint_constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pose_filter_window_sec: float = 0.0
    pose_filter_degree: int = 2
    pose_filter_min_samples: int = 15
    pose_filter_lookahead_sec: float = 0.02
    no_collision: bool = False

    @classmethod
    def from_args(
        cls,
        *,
        urdf: str,
        hands: Set[str],
        scale: float,
        mount_rpy_deg: Any,
        hand_mount_rpy_deg: Dict[str, Any],
        hand_home_q_deg: Dict[str, Any],
        home_q_deg: Any,
        joint_reg_weights: Any,
        joint_smooth_weights: Any,
        swivel_range_deg: Any,
        trust_region: Any,
        joint_constraints: Any,
        hand_joint_constraints: Dict[str, Any],
        pose_filter_window_sec: float,
        pose_filter_degree: int,
        pose_filter_min_samples: int,
        pose_filter_lookahead_sec: float,
        no_collision: bool,
    ) -> "TeleopConfig":
        mount_rpy = _as_rpy_array(mount_rpy_deg)
        home_q = None if home_q_deg is None else np.asarray(home_q_deg, dtype=float).reshape(-1)

        per_hand_rpy: Dict[str, np.ndarray] = {}
        for hand, val in hand_mount_rpy_deg.items():
            if hand not in {"left", "right"}:
                continue
            per_hand_rpy[hand] = _as_rpy_array(val)  # type: ignore[arg-type]

        per_hand_home: Dict[str, np.ndarray] = {}
        for hand, val in hand_home_q_deg.items():
            if hand not in {"left", "right"}:
                continue
            per_hand_home[hand] = np.asarray(val, dtype=float).reshape(-1)

        per_hand_constraints: Dict[str, Dict[str, Any]] = {}
        for hand, val in hand_joint_constraints.items():
            if hand not in {"left", "right"}:
                continue
            if isinstance(val, dict):
                per_hand_constraints[hand] = val

        return cls(
            urdf=str(urdf),
            hands=set(hands),
            scale=float(scale),
            mount_rpy_deg=mount_rpy,
            hand_mount_rpy_deg=per_hand_rpy,
            hand_home_q_deg=per_hand_home,
            home_q_deg=home_q,
            joint_reg_weights=None if joint_reg_weights is None else list(joint_reg_weights),
            joint_smooth_weights=None if joint_smooth_weights is None else list(joint_smooth_weights),
            swivel_range_deg=None if swivel_range_deg is None else float(swivel_range_deg),
            trust_region=trust_region,
            joint_constraints=joint_constraints,
            hand_joint_constraints=per_hand_constraints,
            pose_filter_window_sec=float(pose_filter_window_sec),
            pose_filter_degree=int(pose_filter_degree),
            pose_filter_min_samples=int(pose_filter_min_samples),
            pose_filter_lookahead_sec=float(pose_filter_lookahead_sec),
            no_collision=bool(no_collision),
        )

    def rotation_rpy_for(self, hand: str) -> Optional[np.ndarray]:
        if hand in self.hand_mount_rpy_deg:
            return self.hand_mount_rpy_deg[hand]
        return self.mount_rpy_deg

    def constraints_for(self, hand: str) -> Optional[Dict[str, Any]]:
        if hand in self.hand_joint_constraints:
            return self.hand_joint_constraints[hand]
        return self.joint_constraints

    def home_q_for(self, hand: str) -> Optional[np.ndarray]:
        if hand in self.hand_home_q_deg:
            return self.hand_home_q_deg[hand]
        return self.home_q_deg


__all__ = ["TeleopConfig"]
