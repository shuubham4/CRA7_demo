#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import subprocess

def callback(data):
    if data.data == 'jackal go to the final goal':
        rospy.loginfo("Received 'go to goal a', launching send_goals.launch")
        subprocess.Popen(["roslaunch", "outdoor_waypoint_nav", "send_goals.launch"])

    elif data.data == 'jackal go to the initial goal':
        rospy.loginfo("Received 'go to goal a', launching send_goals2.launch")
        subprocess.Popen(["roslaunch", "outdoor_waypoint_nav", "send_goals2.launch"])

    elif data.data == 'husky go to the final goal':
        rospy.loginfo("Received 'husky go to the final goal', launching send_goal.py")
        subprocess.Popen(["rosrun", "dvpg", "send_goal.py"])



def listener():
    rospy.init_node('goal_listener', anonymous=True)
    rospy.Subscriber('web_input', String, callback)
    rospy.loginfo("Listening to 'web_input' topic...")
    rospy.spin()

if __name__ == '__main__':
    listener()
