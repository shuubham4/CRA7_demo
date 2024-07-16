#!/usr/bin/env python3

import rospy
import actionlib
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion

def move_forward(condition):
    rospy.init_node('move_forward_node')

    client = actionlib.SimpleActionClient('husky/move_base', MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "husky/base"
    goal.target_pose.header.stamp = rospy.Time.now()

    if condition == 1:
        angle = math.radians(-40)
        goal.target_pose.pose = Pose(Point(11.0, -5.0, 0.0), Quaternion(0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)))
    elif condition == 2:
        goal.target_pose.pose = Pose(Point(10.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0))


    rospy.loginfo("Sending goal to move 2 meters forward")
    client.send_goal(goal)
    wait = client.wait_for_result()

    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
        result = client.get_result()
        rospy.loginfo("Goal reached!")

if __name__ == '__main__':
    try:
        move_forward(1)
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")
    try:
        move_forward(2)
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")
