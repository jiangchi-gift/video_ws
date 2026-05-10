#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String

# 回调函数，收到消息自动执行
def back_func(msg):
    print("已经接收到消息：",msg.data)
    rospy.loginfo("接收内容：%s",msg.data)

def main():
    rospy.init_node("test_sub")
    # 话题名必须和发布者一模一样 my_topic
    rospy.Subscriber("my_topic",String,back_func)
    rospy.spin()

if __name__ == "__main__":
    main()