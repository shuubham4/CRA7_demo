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
Create rule for 11311 port and enbale firewall.
