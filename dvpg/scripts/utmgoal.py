#!/usr/bin/env python3
import rospy
from geographiclib.geodesic import Geodesic
from geometry_msgs.msg import PoseStamped

def gps_to_utm(lat, lon):
    # Replace with appropriate zone number
    geod = Geodesic.WGS84
    g = geod.Direct(lat, lon, 0, 0)
    return g['lat2'], g['lon2']

def publish_goal(lat, lon):
    utm_x, utm_y = gps_to_utm(lat, lon)
    goal = PoseStamped()
    goal.header.frame_id = "utm"
    goal.header.stamp = rospy.Time.now()
    goal.pose.position.x = utm_x
    goal.pose.position.y = utm_y
    goal.pose.orientation.w = 1  # Assuming facing forward
    print(goal)
    pub.publish(goal)

if __name__ == "__main__":
    rospy.init_node('gps_goal_publisher')
    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    # Example: publish a goal
    publish_goal(39.2604832, -76.7149824)