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

        self.br = tf2_ros.TransformBroadcaster()
        self.rate = rospy.Rate(100)  # 100 Hz

        self.base_T_cam = np.array([[0.99993021, -0.00887332, -0.00779972, 0.31846705],
                                    [0.00500804, -0.2795885, 0.96010686, -1.10184744],
                                    [-0.01070005, -0.96007892, -0.27952455, 0.50819482],
                                    [0., 0., 0., 1.]])
        
        self.cartesian_pose = None
        self.inferred_pose = np.array([[0.64438732, -0.1737258,  -0.74470421,  0.6172552],
                                        [ -0.72986569, -0.43030012, -0.53116651,  0.03628537],
                                        [-0.228169,    0.88581105, -0.40407643,  0.3199517],
                                        [ 0.,          0. ,         0. ,         1.        ]])
        # TODO: grasp_pos_inter_palm = np.eye(4)
        # grasp_pos_inter_palm[:3,-1] = base_T_palm_np[:3,-1] - base_T_palm_np[:3,:3] @ inter_offset
        # grasp_pos_inter_palm[:3,:3] = base_T_palm_np[:3,:3]
        
        rospy.Subscriber("/iiwa/state/CartesianPose", CartesianPose, self.pose_callback)
        rospy.spin()

    def pose_callback(self, msg):
        self.cartesian_pose = msg.poseStamped.pose
        rospy.loginfo(self.cartesian_pose)
        self.broadcast_camera_transform()
        self.broadcast_iiwa_transform()
        self.broadcast_grasp_transform()

    def update_inferred_pose_tf(self, inferred_pose):
        self.inferred_pose = inferred_pose

    def broadcast_grasp_transform(self):
        if self.inferred_pose is not None:
            t = geometry_msgs.msg.TransformStamped()

            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "base"
            t.child_frame_id = "inferred_pose"

            t.transform.translation.x = self.inferred_pose[0, 3]
            t.transform.translation.y = self.inferred_pose[1, 3]
            t.transform.translation.z = self.inferred_pose[2, 3]

            q = tf.transformations.quaternion_from_matrix(self.inferred_pose)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            self.br.sendTransform(t)
        else:
            rospy.logwarn('Inferred pose not received')

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

        self.br.sendTransform(t)

    def broadcast_iiwa_transform(self):
        if self.cartesian_pose is not None:
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

            self.br.sendTransform(t)
        else:
            rospy.logwarn('Cartesian pose not received')

if __name__ == '__main__':
    try:
        broadcaster = TFBroadcaster()
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass
