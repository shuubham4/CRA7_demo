#!/usr/bin/env python3

import rospy
import tf
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge

class LidarToImageMapper:
    def __init__(self):
        rospy.init_node('lidar_to_image_mapper', anonymous=True)

        # Parameters
        self.lidar_topic = rospy.get_param('~lidar_topic', '/velodyne_points')
        self.image_topic = rospy.get_param('~image_topic', '/axis/image_raw/compressed')
        self.lidar_frame = rospy.get_param('~lidar_frame', 'velodyne')
        self.camera_frame = rospy.get_param('~camera_frame', 'axis')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/axis/camera_info')

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        # Subscribers
        self.lidar_sub = rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)

        self.image = None

    def lidar_callback(self, msg):
        # Wait for the transform between lidar and camera frames
        try:
            self.listener.waitForTransform(self.camera_frame, self.lidar_frame, rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform(self.camera_frame, self.lidar_frame, rospy.Time(0))
            transform_matrix = self.listener.fromTranslationRotation(trans, rot)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to get transform")
            return

        points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(points))

        if self.image is not None:
            # Project points to image plane
            projected_image = self.project_lidar_to_image(points, transform_matrix)
            cv2.imshow('Lidar Mapped Image', projected_image)
            cv2.waitKey(1)

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def project_lidar_to_image(self, points, transform_matrix):
        # Apply transformation
        points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
        points_transformed = transform_matrix.dot(points_hom.T).T

        # Assuming intrinsic camera parameters (you may need to replace these with actual values)
        fx = 525.0  # Focal length in x direction
        fy = 525.0  # Focal length in y direction
        cx = 319.5  # Principal point in x direction
        cy = 239.5  # Principal point in y direction

        image_points = np.zeros((points_transformed.shape[0], 2), dtype=np.int32)
        for i, point in enumerate(points_transformed):
            if point[2] > 0:  # Check if the point is in front of the camera
                image_points[i] = [int(fx * point[0] / point[2] + cx), int(fy * point[1] / point[2] + cy)]

        for point in image_points:
            if 0 <= point[0] < self.image.shape[1] and 0 <= point[1] < self.image.shape[0]:
                cv2.circle(self.image, (point[0], point[1]), 2, (0, 0, 255), -1)

        return self.image

if __name__ == '__main__':
    try:
        mapper = LidarToImageMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
