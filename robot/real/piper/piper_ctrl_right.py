#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
import time

from piper_sdk import *

if __name__ == "__main__":

    piper1 = C_PiperInterface_V2("can_right")
    piper1.ConnectPort()
    while( not piper1.EnablePiper()):
        time.sleep(0.01)
    factor = 1000 #1000*180/3.1415926
    # position = [ 0 , 0, 0,   0 ,0, 0,60]
    # position = [ -50, 150.0, -123.0, -100.0, -28.0,0,58]
    position = [-80, 170.0, -150.0, 0, 0, 0,0]
    
    joint_0 = round(position[0]*factor)
    joint_1 = round(position[1]*factor)
    joint_2 = round(position[2]*factor)
    joint_3 = round(position[3]*factor)
    joint_4 = round(position[4]*factor)
    joint_5 = round(position[5]*factor)
    joint_6 = round(position[6]*1000)

    piper1.ModeCtrl(0x01, 0x01, 30, 0x00)
    piper1.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
    piper1.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
    pass
