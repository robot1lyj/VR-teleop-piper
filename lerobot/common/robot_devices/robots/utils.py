# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Protocol

from lerobot.common.robot_devices.robots.configs import RobotConfig


def get_arm_id(name, arm_type):
    """Returns the string identifier of a robot arm. For instance, for a bimanual manipulator
    like Aloha, it could be left_follower, right_follower, left_leader, or right_leader.
    """
    return f"{name}_{arm_type}"


class Robot(Protocol):
    # TODO(rcadene, aliberts): Add unit test checking the protocol is implemented in the corresponding classes
    robot_type: str
    features: dict

    def connect(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def disconnect(self): ...


def make_robot_config(robot_type: str, **kwargs) -> RobotConfig:
    if robot_type == "piper_vr":
        from lerobot.common.robot_devices.robots.piper_vr import PiperVRRobotConfig

        return PiperVRRobotConfig(**kwargs)
    if robot_type == "piper_shm":
        from lerobot.common.robot_devices.robots.piper_shm import PiperShmRobotConfig

        return PiperShmRobotConfig(**kwargs)
    else:
        raise ValueError(f"Robot type '{robot_type}' is not available.")


def make_robot_from_config(config: RobotConfig):
    # 注意：按 type 延迟 import，避免录制进程无意 import 到 WebRTC/aiortc 等重模块。
    robot_type = config.type
    if robot_type == "piper_vr":
        from lerobot.common.robot_devices.robots.piper_vr import PiperVRRobot, PiperVRRobotConfig

        if isinstance(config, PiperVRRobotConfig):
            return PiperVRRobot(config)
    if robot_type == "piper_shm":
        from lerobot.common.robot_devices.robots.piper_shm import PiperShmRobot, PiperShmRobotConfig

        if isinstance(config, PiperShmRobotConfig):
            return PiperShmRobot(config)

    raise ValueError(f"Unsupported robot config type: {type(config)} (type={robot_type})")


def make_robot(robot_type: str, **kwargs) -> Robot:
    config = make_robot_config(robot_type, **kwargs)
    return make_robot_from_config(config)
