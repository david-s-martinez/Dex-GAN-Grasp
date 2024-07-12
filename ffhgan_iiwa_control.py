import sys
import os

from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import time
import logging
import open3d as o3d
import copy
import cv2

# Add GraspInference to the path and import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','src'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import zmq
import numpy as np
import open3d as o3d
from time import time,sleep

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://10.3.100.77:5561")
flags=0
track=False


def get_grasp():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    # we need to write 127.0.0.1 instead of localhost. It does not work otherwise.
    # socket.bind("tcp://127.0.0.1:5562")
    # in case between two pc
    socket.bind("tcp://*:5561")
    flags=0
    copy=True
    track=False

    info = socket.recv_json(flags=flags)
    msgs = socket.recv_multipart()

    grasp_np = msgs[0]
    grasp_np_buffer = memoryview(grasp_np)
    grasp_np = np.frombuffer(grasp_np_buffer, dtype=info["grasp_np_dtype"]).reshape(info["grasp_np_shape"])

    print("Keys of dictionary sent from client: ", info.keys())
    print("Shape of obj_pcd sent in dictionary: ", info["grasp_np_shape"])
    print("Shape of obj_pcd sent after reconstruction: ", grasp_np.shape)



def vis_all_grasps(pcd,cam_T_grasps_np):
    obj_pcd_base = pcd.transform(base_T_cam)
    grasps_to_show = []
    for j in range(cam_T_grasps_np.shape[0]):
        cam_T_grasp_np = cam_T_grasps_np[j,:4,:]
        base_T_palm_np = np.matmul(base_T_cam, cam_T_grasp_np)
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        grasp_frame = grasp_frame.transform(base_T_palm_np)
        grasps_to_show.append(grasp_frame)
    grasps_to_show.append(obj_pcd_base)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for f in grasps_to_show:
        vis.add_geometry(f)
    vis.run()

# carboard
home_joints  = [0.23065982874125268, -0.8819193506051936, 1.6833640644137136, 2.4141621427013176, 1.5965623182504292, 0.2667926434916148, 1.5478464411064614]
moveJ_joints(home_joints)
print('arrived at home pose')

# Transformation from flange frame to hand palm frame
# TODO Change robotiq in panda flange flange to iiwa flange frame

R_panda = R.from_euler('xyz', [math.pi, 0, 0]).as_matrix()  # FFHNet was trained with Franka Panda
diana_fl_T_panda_fl = np.eye(4)
diana_fl_T_panda_fl[:3,:3] = R_panda
panda_fl_T_palm = np.eye(4)
panda_fl_T_palm[:3,:3] = np.array([[ 0.26749883,  0.14399234, -0.95273847],
                                    [ 0.        ,  0.98877108,  0.14943813],
                                    [ 0.96355819, -0.03997453,  0.26449511]])
panda_fl_T_palm[:3,-1] = np.array([0.02, 0, 0.06]).reshape((3,))
diana_fl_T_palm = np.matmul(diana_fl_T_panda_fl, panda_fl_T_palm)
palm_T_diana_fl = np.linalg.inv(diana_fl_T_palm)


base_T_cam = np.array([[ 0.99993021, -0.00887332 ,-0.00779972 , 0.31846705],
                    [ 0.00500804, -0.2795885  , 0.96010686 ,-1.10184744],
                    [-0.01070005, -0.96007892 ,-0.27952455 , 0.50819482],
                    [ 0.        ,  0.         , 0.          ,1.        ]])

inter_offset = np.array([0.15, 0, 0])

i = int(input('i=?'))
try:
    while True:
        ##### receive grasp to local pc
        cam_T_grasps_np = get_grasp()

        ############################

        for j in range(cam_T_grasps_np.shape[0]):

            # from IPython import embed; embed()
            print(f'try {j} th grasp')
            cam_T_grasp_np = cam_T_grasps_np[j,:4,:]
            joint_conf_np = cam_T_grasps_np[j,4:,:].reshape(-1)[:15]

            # Rotate grasp poses into end effector frame
            base_T_palm_np = np.matmul(base_T_cam, cam_T_grasp_np)
            base_T_flange = np.matmul(base_T_palm_np, palm_T_diana_fl)

            ###################### Collision filter #################
            # Calculate intermediate point above object to prevent fingers from pushing the object over
            base_T_flange_2 = deepcopy(base_T_flange)

            grasp_pos_inter_palm = np.eye(4)
            grasp_pos_inter_palm[:3,-1] = base_T_palm_np[:3,-1] - base_T_palm_np[:3,:3] @ inter_offset
            grasp_pos_inter_palm[:3,:3] = base_T_palm_np[:3,:3]
            y = input('choose?')
            if y == 'y':
                break
        ###
        base_T_flange = base_T_flange_2
        target_pose = base_T_flange[:3,-1].tolist()
        vxyz = cv2.Rodrigues(np.array(base_T_flange[:3,:3]))[0][:,0].tolist()
        target_pose_rv = target_pose + vxyz

        # no colision filter #
        print('base_T_flange,',base_T_flange)
        print('base_T_palm_np', base_T_palm_np)
        grasp_pos_inter = np.matmul(grasp_pos_inter_palm, palm_T_diana_fl)

        #######################

        target_pose = grasp_pos_inter[:3,-1].tolist()
        vxyz = cv2.Rodrigues(np.array(grasp_pos_inter[:3,:3]))[0][:,0].tolist()
        target_pose_rv_middle = target_pose + vxyz
        target_pose_rv_middle[2] += 0.05
        
        # moveL_pose(target_pose_rv_middle)
        moveJ_pose(target_pose_rv_middle)
        print('reached via pose')
        moveL_pose(target_pose_rv)

    #     # Close hand
        hand.move(joint_conf_np)
        sleep(3)

        # Lift object
        # lifting_distance = 0.08 #0.15
        # lift_diana(lifting_distance)
        target_pose_rv_middle[2] += 0.03
        moveJ_pose(target_pose_rv_middle)

        # robot move up
        sleep(3)
        hand.move(np.zeros(15))

        # go home
        DianaApi.moveJToTarget(home_joints, 0.2, 0.4, 0, 0, 0, ipAddress)

        i += 1
        input('Next grasp?')

except KeyboardInterrupt:
    pass
finally:
    socket.close()
    context.term()