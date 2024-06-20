#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Joy
from sensor_msgs.msg import NavSatFix

file_path = '/home/mpsc/catkin_ws/src/outdoor_waypoint_nav/waypoint_files/points_outdoor.txt'

gps_status = False

def gps_callback(data):
	global gps_status
	if gps_status:
		gps_data = "%f %f" % (data.latitude, data.longitude)
		# print(gps_data)
		with open(file_path, 'w') as file:
			file.write(gps_data+"\n")
			file.write(gps_data)
		gps_status = False
		

def joy_CB(joy_msg):
	global gps_status
	if joy_msg.buttons[0] == 1:
		gps_status = True

def main():
	rospy.init_node('joy_launch_control')
	rospy.Subscriber("/bluetooth_teleop/joy", Joy, joy_CB)
	rospy.Subscriber("/gps/filtered", NavSatFix, gps_callback)
	rospy.spin()

if __name__ == '__main__':
	main()