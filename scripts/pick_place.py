#!/usr/bin/env python

from std_msgs.msg import String
import rospy
from geometry_msgs.msg import PoseStamped
from robotiq_3f_srvs.srv import Move
import json

def move_to_pose(publisher, pose):
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
    publisher.publish(ps)

received_message = None

def get_pick_pose(data):
    try:
        global received_message
        rospy.loginfo("I heard %s", data.data)
        # Check if the received message is the desired message
        received_message = data.data
    except:
        rospy.loginfo("Pick pose not published ")

def listener():
    global received_message
    rospy.init_node('get_pick_pose')
    rospy.Subscriber("goal_pick_pose", String, get_pick_pose)
    
    # Keep looping until the desired message is received
    while not rospy.is_shutdown() and received_message is None:
        rospy.sleep(0.1)  # Sleep for a short time to avoid busy waiting

    # Return the received message
    pick_goals_dict = json.loads(received_message.replace("'" , '"'))
    return pick_goals_dict

def main(pick_goals_dict):
    rospy.init_node('pick_and_place_node', anonymous=True)

    pub = rospy.Publisher('/iiwa/command/CartesianPose', PoseStamped, queue_size=10)

    rospy.sleep(1)  # Allow some time for the publisher to set up

    # Define the poses with positions and orientations in dictionaries
    home_pose = {
        'position': {'x': 0.473, 'y': 0.00017, 'z': 0.6},
        'orientation': {'x': 0.0, 'y': 1.0, 'z': 0.0, 'w': 0.0}
    }

    pick_pose = pick_goals_dict['pick']
    inter_pose = pick_goals_dict['inter']

    pre_place_pose = {
        'position': {'x': 0.473, 'y': -0.38, 'z': 0.6},
        'orientation': {'x': 0.0, 'y': 1.0, 'z': 0.0, 'w': 0.0}
    }

    place_pose = {
        'position': {'x': 0.473, 'y': -0.38, 'z': 0.4},
        'orientation': {'x': 0.0, 'y': 1.0, 'z': 0.0, 'w': 0.0}
    }
    
    # Wait for the gripper services to become available
    rospy.wait_for_service('/robotiq_3f_gripper/close_hand')
    rospy.wait_for_service('/robotiq_3f_gripper/open_hand')

    # Create service proxies for the gripper services
    close_gripper = rospy.ServiceProxy('/robotiq_3f_gripper/close_hand', Move)
    open_gripper = rospy.ServiceProxy('/robotiq_3f_gripper/open_hand', Move)

    rospy.loginfo("Moving to home position")
    move_to_pose(pub, home_pose)
    rospy.sleep(3)  # Wait for the robot to reach the home position

    rospy.loginfo("Moving to pre pick position")
    move_to_pose(pub, inter_pose)
    rospy.sleep(3)  # Wait for the robot to reach the pick position

    rospy.loginfo("Moving to pick position")
    move_to_pose(pub, pick_pose)
    rospy.sleep(3)  # Wait for the robot to reach the pick position

    rospy.loginfo("Closing gripper")
    close_gripper()  # Close the gripper

    rospy.loginfo("Moving to home position")
    move_to_pose(pub, home_pose)
    rospy.sleep(3)  # Wait for the robot to reach the home position

    rospy.loginfo("Moving to pre place position")
    move_to_pose(pub, pre_place_pose)
    rospy.sleep(3)  # Wait for the robot to reach the pre place position

    rospy.loginfo("Moving to place position")
    move_to_pose(pub, place_pose)
    rospy.sleep(3)  # Wait for the robot to reach the place position

    rospy.loginfo("Opening gripper")
    open_gripper()  # Open the gripper

    rospy.loginfo("Moving to home position")
    move_to_pose(pub, home_pose)
    rospy.sleep(3)  # Wait for the robot to reach the home position

    rospy.loginfo("Pick and place operation completed")

if __name__ == '__main__':
    try:
        pick_goals_dict = listener()
        rospy.loginfo("Received the desired message: %s", pick_goals_dict)
        main(pick_goals_dict)

    except rospy.ROSInterruptException:
        pass
