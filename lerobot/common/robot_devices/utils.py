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

import logging
import platform
import time

from lerobot.common.utils.utils import has_method


def busy_wait(seconds):
    if platform.system() == "Darwin":
        # On Mac, `time.sleep` is not accurate and we need to use this while loop trick,
        # but it consumes CPU cycles.
        # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        if seconds > 0:
            time.sleep(seconds)


def safe_disconnect(func):
    # TODO(aliberts): Allow to pass custom exceptions
    # (e.g. ThreadServiceExit, KeyboardInterrupt, SystemExit, UnpluggedError, DynamixelCommError)
    def wrapper(robot, *args, **kwargs):
        try:
            return func(robot, *args, **kwargs)
        except Exception as e:
            if robot.is_connected:
                logger = logging.getLogger(__name__)
                if has_method(robot, "run_calibration"):
                    try:
                        logger.warning("异常触发：先回到初始姿态再断开连接")
                        robot.run_calibration()
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.warning("异常回位失败: %s", exc)
                robot.disconnect()
            raise e

    return wrapper


def safe_recover(func):
    """在出现异常时优先尝试让机器人回到初始姿态，但不强制断电。"""

    def wrapper(robot, *args, **kwargs):
        try:
            return func(robot, *args, **kwargs)
        except Exception as exc:
            if robot.is_connected and has_method(robot, "run_calibration"):
                logger = logging.getLogger(__name__)
                try:
                    logger.warning("异常触发：执行 run_calibration() 回初始姿态")
                    robot.run_calibration()
                except Exception as recover_exc:  # pylint: disable=broad-except
                    logger.warning("异常回位失败: %s", recover_exc)
            raise exc

    return wrapper


class RobotDeviceNotConnectedError(Exception):
    """Exception raised when the robot device is not connected."""

    def __init__(
        self, message="This robot device is not connected. Try calling `robot_device.connect()` first."
    ):
        self.message = message
        super().__init__(self.message)


class RobotDeviceAlreadyConnectedError(Exception):
    """Exception raised when the robot device is already connected."""

    def __init__(
        self,
        message="This robot device is already connected. Try not calling `robot_device.connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)
