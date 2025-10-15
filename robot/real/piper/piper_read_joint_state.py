#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
# 读取机械臂消息并打印,需要先安装piper_sdk
from piper_sdk import *

# 测试代码
if __name__ == "__main__":
    piper = C_PiperInterface_V2("can_right")
    piper.ConnectPort()
    while True:
        import time

        joint_msg = piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state

        gripper_msg = piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state
        
        # 打印原始joint_state对象以查看其结构
        print(f"Joint State: {joint_state}")
        # 如果你知道具体的属性名，可以这样访问并除以1000
        # 例如，如果属性是joint_1, joint_2等:
        print(f"Joint 1: {joint_state.joint_1/1000}")
        print(f"Joint 2: {joint_state.joint_2/1000}")
        
        print(f"Gripper State: {gripper_state}")

        time.sleep(2)
        pass
