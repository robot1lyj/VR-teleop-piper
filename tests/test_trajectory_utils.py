"""验证轨迹录制与回放工具的基本行为。"""

import pathlib

from scripts.record_vr_trajectory import TrajectoryRecorder
from scripts.run_vr_meshcat import _iter_trajectory


def test_record_and_iter(tmp_path: pathlib.Path) -> None:
    output_path = tmp_path / "traj.jsonl"
    recorder = TrajectoryRecorder(allowed_hands={"right"}, output_path=output_path)

    payload = {
        "rightController": {
            "position": {"x": 0.0, "y": 0.0, "z": -0.3},
            "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "gripActive": True,
            "trigger": 0.2,
        }
    }
    recorder.process_message(payload)
    recorder.close()

    frames = list(_iter_trajectory(output_path))
    assert len(frames) == 1
    frame = frames[0]
    assert isinstance(frame["elapsed"], float)
    assert "rightController" in frame["payload"]
