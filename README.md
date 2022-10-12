# CRA7_demo
This is Demo project for CRA7_demo

## Connect multiple device 
Both client and server is setup with Ubuntu 20.04 and ROS noetic

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

## Raspberry pi webcam server 
https://pimylifeup.com/raspberry-pi-webcam-server/
(done)

## Hector slam on RosBot2
https://automaticaddison.com/how-to-build-an-indoor-map-using-ros-and-lidar-based-slam/ (done)
see the source folder below to find out details and Read.md file

## Saving map using rosrun
```
rosrun map_server map_saver -f ~/map
```
## Task
Create rule for 11311 port and enable firewall.

## Run the final code
1. Copy all the codes from `src_pc` to `catkin_ws\src\`
2. Similarly copy all the codes to respective bots
3. Build the packages using `catkin_make`
4. First create the map Using
```
roslaunch dvpg map_creation.launch
```
5. Follow the map using following code
```
roslaunch dvpg map_follower.launch
```
