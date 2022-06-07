# Install hector slam and run the code 

## Install

1. Install Qt4
```
sudo apt-get install qt4-qmake qt4-dev-tools
```
2. Copy the whole `hector_slam` folder to `/catkin_ws/src`
3. All the files are edited inside
4. Build the packages
```
catkin_make
```
5. Shutdown the bot
```
Sudo shutdown -h now
```

## Run the hector mapping
```
roslaunch rplidar_ros rplidar.launch
roslaunch hector_slam_launch tutorial.launch
```

## Save the map

```
rosrun map_server map_saver -f my_map
```
