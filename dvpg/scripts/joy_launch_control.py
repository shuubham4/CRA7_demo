#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Joy
from sensor_msgs.msg import NavSatFix
import time

file_path = '/home/mpsc/catkin_ws/src/CRA7_demo/outdoor_waypoint_nav/waypoint_files/points_outdoor.txt'

gps_status = 0

def gps_callback(data):
	global gps_status
	if gps_status == 1:
		gps_data = "%f %f" % (data.latitude, data.longitude)
		# print(gps_data)
		with open(file_path, 'w') as file:
			file.write(gps_data+"\n")
		gps_status = 0
	elif gps_status == 2:
		gps_data = "%f %f" % (data.latitude, data.longitude)
		# print(gps_data)
		with open(file_path, 'a') as file:
			file.write(gps_data+"\n")
		gps_status = 0
		time.sleep(4)
	elif gps_status == 3:
		gps_data = "%f %f" % (data.latitude, data.longitude)
		# print(gps_data)
		with open(file_path.replace('points_outdoor', 'points_outdoor2'), 'w') as file:
			file.write(gps_data+"\n")
		gps_status = 0
		time.sleep(4)
		

def joy_CB(joy_msg):
	global gps_status
	if joy_msg.buttons[0] == 1:
		gps_status = 1
	if joy_msg.buttons[2] == 1:
		gps_status = 2
	if joy_msg.buttons[1] == 1:
		gps_status = 3

def main():
	rospy.init_node('joy_launch_control')
	rospy.Subscriber("/bluetooth_teleop/joy", Joy, joy_CB)
	rospy.Subscriber("/gps/filtered", NavSatFix, gps_callback)
	rospy.spin()

if __name__ == '__main__':
	main()
