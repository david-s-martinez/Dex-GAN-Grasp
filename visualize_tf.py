#!/usr/bin/env python

import rospy
import tf2_ros
import geometry_msgs.msg
import numpy as np
import tf
from iiwa_msgs.msg import CartesianPose

class TFBroadcaster:
    def __init__(self):
        rospy.init_node('tf_broadcaster')

        self.br1 = tf2_ros.TransformBroadcaster()
        self.br2 = tf2_ros.TransformBroadcaster()
        self.rate = rospy.Rate(100) 

        self.base_T_cam = np.array([[0.99993021, -0.00887332, -0.00779972, 0.31846705],
                                    [0.00500804, -0.2795885, 0.96010686, -1.10184744],
                                    [-0.01070005, -0.96007892, -0.27952455, 0.50819482],
                                    [0., 0., 0., 1.]])
        
        self.cartesian_pose = None

        rospy.Subscriber("/iiwa/state/CartesianPose", CartesianPose, self.pose_callback)

    def pose_callback(self, msg):
        self.cartesian_pose = msg.poseStamped.pose
        print(self.cartesian_pose)
        
    def broadcast_camera_transform(self):
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "base"
        t.child_frame_id = "camera"

        t.transform.translation.x = self.base_T_cam[0, 3]
        t.transform.translation.y = self.base_T_cam[1, 3]
        t.transform.translation.z = self.base_T_cam[2, 3]

        q = tf.transformations.quaternion_from_matrix(self.base_T_cam)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.br1.sendTransform(t)

    def broadcast_iiwa_transform(self):
        try:
            t = geometry_msgs.msg.TransformStamped()

            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "base"
            t.child_frame_id = "iiwa_pose"

            t.transform.translation.x = self.cartesian_pose.position.x
            t.transform.translation.y = self.cartesian_pose.position.y
            t.transform.translation.z = self.cartesian_pose.position.z

            t.transform.rotation.x = self.cartesian_pose.orientation.x
            t.transform.rotation.y = self.cartesian_pose.orientation.y
            t.transform.rotation.z = self.cartesian_pose.orientation.z
            t.transform.rotation.w = self.cartesian_pose.orientation.w

            self.br2.sendTransform(t)
        except:
            print('I did not get the pose')

    def broadcast_transforms(self):
        while not rospy.is_shutdown():
            self.broadcast_camera_transform()
            self.broadcast_iiwa_transform()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        broadcaster = TFBroadcaster()
        broadcaster.broadcast_transforms()
    except rospy.ROSInterruptException:
        pass
