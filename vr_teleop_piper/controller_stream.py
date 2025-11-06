"""兼容入口：保留旧模块路径，实际实现位于 vr_runtime.controller_pipeline。"""

from __future__ import annotations

from vr_runtime.controller_pipeline import ControllerPipeline, run_vr_controller_stream

__all__ = ["ControllerPipeline", "run_vr_controller_stream"]

if __name__ == "__main__":
    run_vr_controller_stream()
