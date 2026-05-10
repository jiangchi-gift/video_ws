#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String

def main():
    rospy.init_node("test_pub")
    pub = rospy.Publisher("my_topic",String,queue_size=10)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        msg = String()
        msg.data = "第五人格是世界上最好玩的游戏"
        pub.publish(msg)
        print("第五人格是世界上最好玩的游戏")
        rate.sleep()

if __name__ == "__main__":
    main()