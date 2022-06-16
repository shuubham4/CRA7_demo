## Navigation using turtlebot3
- Run 'roscore' in the master pc.
- Issue the following command to enable bringup

```
roslaunch turtlebot3_bringup turtlebot3_robot.launch
```
- Launch the Navigation with the following command

```
roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/map.yaml
```
## Tuning navigation Hyperparameters
Navigate to the following file to change parameters like cost scaling factor, rotation, velocity. etc.

``` 
turtlebot3_navigation/param/costmap_common_param_${TB3_MODEL}.yaml
```
