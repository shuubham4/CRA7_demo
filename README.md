# CRA7_demo
This is Demo project for CRA7_demo

## Connect multiple device 
Both client and server is setup with Ubuntu 18 and ROS

1. First connect all the device under same wifi network
2. On the server pc open the `.bashrc` file
3. Then find the followind lines
```
export ROS_MASTER_URI=http://master:11311
export ROS_IPV6=on
```
4. Change this line with following line
```
export ROS_MASTER_URI=http://[server_ip_address]:11311
export ROS_IP=[server_ip_address]
#export ROS_MASTER_URI=http://master:11311
#export ROS_IPV6=on
```
5. Similarly, on the clients robot open the `.bashrc` file and replace with:
```
export ROS_MASTER_URI=http://[server_ip_address]:11311
export ROS_IP=[client_ip_address]
#export ROS_MASTER_URI=http://master:11311
#export ROS_IPV6=on
```
## VNC Via SSH resources
https://www.techrepublic.com/article/how-to-connect-to-vnc-using-ssh/

## Catkin error (noetic with 20.04)
https://answers.ros.org/question/353111/following-installation-instructions-catkin_make-generates-a-cmake-error/

- Ros Noetic master can control melodic clients (checked).
- Data from two robots can be seen on Rviz at the same time under Noetic master.
- Turtlebot SLAM can't be run using the robot's ROS system right now. Will have to look into how to install gmapping inside of Turtlebot3.
- Backup plan: Treat the remote pc of turtlebot as a client of Noetic Rosmaster and take the topic from there while running SLAM.
