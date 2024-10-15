#!/usr/bin/env python
# Code to control the robot and gripper in real world.
from std_msgs.msg import String
import rospy
from geometry_msgs.msg import PoseStamped 
from robotiq_3f_srvs.srv import Move # from https://github.com/david-s-martinez/robotiq
import json
import numpy as np
from iiwa_msgs.msg import CartesianPose # from https://github.com/IFL-CAMP/iiwa_stack

class RobotControl:
    def __init__(self):
        self.received_message = None
        self.robot_pose = None
        rospy.init_node('get_pick_pose')
        pick_goals_dict = self.listener()
        rospy.loginfo("Received the desired message: %s", pick_goals_dict)
        self.pose_sub = rospy.Subscriber("/iiwa/state/CartesianPose", CartesianPose, self.pose_callback)
        self.pose_pub = rospy.Publisher('/iiwa/command/CartesianPose', PoseStamped, queue_size=10)
        # Wait for the gripper services to become available
        rospy.wait_for_service('/robotiq_3f_gripper/close_hand')
        rospy.wait_for_service('/robotiq_3f_gripper/open_hand')

        # Create service proxies for the gripper services
        self.close_gripper = rospy.ServiceProxy('/robotiq_3f_gripper/close_hand', Move)
        self.open_gripper = rospy.ServiceProxy('/robotiq_3f_gripper/open_hand', Move)

        rospy.sleep(1)  # Allow some time for the publisher to set up
        self.main(pick_goals_dict)

    def move_to_pose(self, pose):
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = 'base_link'  # Set this to the appropriate frame
        ps.pose.position.x = pose['position']['x']
        ps.pose.position.y = pose['position']['y']
        ps.pose.position.z = pose['position']['z']
        ps.pose.orientation.x = pose['orientation']['x']
        ps.pose.orientation.y = pose['orientation']['y']
        ps.pose.orientation.z = pose['orientation']['z']
        ps.pose.orientation.w = pose['orientation']['w']
        self.pose_pub.publish(ps)

    
    def get_pick_pose(self, data):
        try:
            rospy.loginfo("I heard %s", data.data)
            # Check if the received message is the desired message
            self.received_message = data.data
        except:
            rospy.loginfo("Pick pose not published ")

    def listener(self):
        rospy.Subscriber("goal_pick_pose", String, self.get_pick_pose)
        
        # Keep looping until the desired message is received
        while not rospy.is_shutdown() and self.received_message is None:
            rospy.sleep(0.1)  # Sleep for a short time to avoid busy waiting

        # Return the received message
        pick_goals_dict = json.loads(self.received_message.replace("'" , '"'))
        self.received_message = None
        return pick_goals_dict
    
    def pose_callback(self,msg):
        self.robot_pose = msg.poseStamped.pose

    def main(self, pick_goals_dict):
        self.open_gripper()
        # Define the poses with positions and orientations in dictionaries
        home_pose = {
            'position': {'x': 0.373, 'y': -0.28, 'z': 0.7},
            'orientation': {'x': 0.0, 'y': 1.0, 'z': 0.0, 'w': 0.0}
        }
        rospy.loginfo("Moving to home position")
        self.move_to_pose(home_pose)
        rospy.sleep(3)  # Wait for the robot to reach the home position
       
        for i in range(len(pick_goals_dict)):
            pick_pose = pick_goals_dict[str(i)]['pick']
            inter_pose = pick_goals_dict[str(i)]['inter']

            pre_place_pose = {
                'position': {'x': 0.473, 'y': -0.38, 'z': 0.6},
                'orientation': {'x': 0.0, 'y': 1.0, 'z': 0.0, 'w': 0.0}
            }

            place_pose = {
                'position': {'x': 0.473, 'y': -0.38, 'z': 0.4},
                'orientation': {'x': 0.0, 'y': 1.0, 'z': 0.0, 'w': 0.0}
            }

            rospy.loginfo("Moving to pre pick position")
            self.move_to_pose(inter_pose)
            rospy.sleep(4)  # Wait for the robot to reach the pick position
            move_diff_x = abs(self.robot_pose.position.x - home_pose['position']['x'])
            move_diff_y = abs(self.robot_pose.position.y - home_pose['position']['y'])
            move_diff = np.sqrt(move_diff_x**2 + move_diff_y**2)
            print('move_diff:', move_diff)

            not_moved = move_diff < 0.01
            if not_moved:
                print('The robot is not moving')
                continue
            rospy.loginfo("Moving to pick position")
            self.move_to_pose(pick_pose)
            rospy.sleep(3)  # Wait for the robot to reach the pick position

            rospy.loginfo("Closing gripper")
            self.close_gripper()  # Close the gripper
            rospy.sleep(2)  # Wait for the robot to reach the home position

            rospy.loginfo("Moving to home position")
            self.move_to_pose(home_pose)
            rospy.sleep(3)  # Wait for the robot to reach the home position

            rospy.loginfo("Moving to pre place position")
            self.move_to_pose(pre_place_pose)
            rospy.sleep(3)  # Wait for the robot to reach the pre place position

            rospy.loginfo("Opening gripper")
            self.open_gripper()  # Open the gripper

            rospy.loginfo("Moving to home position")
            self.move_to_pose(home_pose)
            rospy.sleep(3)  # Wait for the robot to reach the home position

            rospy.loginfo("Pick and place operation completed")
            break

if __name__ == '__main__':
    while True:
        try:
            rc = RobotControl()

        except rospy.ROSInterruptException:
            pass
        