# ARS_Finetune

## Prerequisite 
ROS Noetic in the both Server and Edge node. To install ROS Noetic please follow thes steps: 
[ROS Noetic Installation Steps](http://wiki.ros.org/noetic/Installation)

## Installation The framework
1. Create a catkin workspace `mkdir -p catkin_ws/src`
2. Navigate to src folder `cd catkin_ws/src`
3. Clone the repo in the src folder `git clone https://github.com/mahmed10/ARS_Finetune.git`
4. Install all the rquired libraries `pip install -r ARS_Finetune/requirement.txt`
5. Make all the py excecutable `chmod +x ARS_Finetune/scripts/*.py` `chmod +x ARS_Finetune/src/ars_finetune/*.py`
6. provide pretrain weight in the checkpoints folder `gdown 'https://drive.google.com/uc?id=1kKPXTxAPg2iHejfQHIjiuGbEcQXqrLGl' -O ARS_Finetune/src/ars_finetune/checkpoints/`
7. Navigate to catkin_ws folder `cd ..`
8. Build the project with catkin_make `catkin_make`

## Run the Code
Open terminal, and type the following command
1. For Server `roslaunch ars_finetune server.launch`
2. For Edge `roslaunch ars_finetune robot.launch`
