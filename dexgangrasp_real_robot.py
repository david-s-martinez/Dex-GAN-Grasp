import sys
import os
import rospy
from std_msgs.msg import String
import tf
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import time
import logging
import open3d as o3d
import copy
import cv2
import torch
import argparse
from DexGanGrasp.config.config import Config
from DexGanGrasp.data.bps_encoder import BPSEncoder
from DexGanGrasp.models.dexgangrasp import DexGanGrasp
from DexGanGrasp.utils import utils, visualization, writer
from DexGanGrasp.utils.writer import Writer
import bps_torch.bps as b_torch

import tf.transformations
# Add GraspInference to the path and import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','src'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from inference.segmentation import PlaneSegmentation
from inference.realsense import RealSense
import zmq
import numpy as np
import open3d as o3d
from time import time,sleep

def divide_into_trans_quat(base_T_flange):
    flange_quat = tf.transformations.quaternion_from_matrix(base_T_flange)
    flange_trans = base_T_flange[:3,-1]
    return flange_trans, flange_quat

# Transformation from flange frame to hand palm framer
# tf flange 2 palm
# rosrun tf tf_echo /panda_link8 /palm_link_robotiq
# At time 0.000
# - Translation: [0.020, 0.000, 0.050]
# - Rotation: in Quaternion [0.000, -0.707, -0.000, 0.707]
#             in RPY (radian) [2.356, -1.571, -2.356]
#             in RPY (degree) [135.000, -90.000, -135.000]

flange_T_palm = np.array([[ 0.,  0., -1.,  0.020],
                            [-0.,  1.,  0.,  0.],
                            [ 1.,  0.,  0.,  0.050],
                            [ 0.,  0.,  0.,  1.]])
# camera
base_T_cam = np.array([[ 0.99993021, -0.00887332 ,-0.00779972 , 0.31846705],
                    [ 0.00500804, -0.2795885  , 0.96010686 ,-1.10184744],
                    [-0.01070005, -0.96007892 ,-0.27952455 , 0.50819482],
                    [ 0.        ,  0.         , 0.          ,1.        ]])

save_path = '/workspaces/inference_container/exp_images/'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
rs = RealSense(logger, save_path)
segment = PlaneSegmentation()

grasp_region_mask = np.zeros((720,1280),dtype=np.bool)
# grasp_region_mask[150:420, 150:600] = True  # single obj
grasp_region_mask[200:630, 530:930] = True  # cupboard grasping

inter_offset = np.array([0.16, 0, 0])
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.split(os.path.split(ROOT_PATH)[0])[0]
parser = argparse.ArgumentParser()

# Best GAN so far:
gen_path = "checkpoints/ffhgan/2024-03-10T17_31_55_ffhgan_lr_0.0001_bs_1000"
best_epoch = 32
parser.add_argument('--gen_path', default=gen_path, help='path to DexGenerator model')
parser.add_argument('--load_gen_epoch', type=int, default=best_epoch, help='epoch of DexGenerator model')
parser.add_argument('--eva_path', default='checkpoints/ffhevaluator/2024-06-23_ffhevaluator', help='path to DexEvaluator model')
parser.add_argument('--load_eva_epoch', type=int, default=30, help='epoch of DexEvaluator model')
parser.add_argument('--config', type=str, default='DexGanGrasp/config/config_dexgangrasp.yaml')

args = parser.parse_args()

load_path_gen = args.gen_path
load_path_eva = args.eva_path
load_epoch_gen = args.load_gen_epoch
load_epoch_eva = args.load_eva_epoch
config_path = args.config
config = Config(config_path)
cfg = config.parse()
dexgangrasp = DexGanGrasp(cfg)
print(dexgangrasp)
base_data_bath = os.path.join(ROOT_PATH,'data','real_objects')
dexgangrasp.load_dexgenerator(epoch=load_epoch_gen, load_path=load_path_gen)
dexgangrasp.load_dexevaluator(epoch=load_epoch_eva, load_path=load_path_eva)
path_real_objs_bps = os.path.join(base_data_bath, 'bps')

bps_path = 'models/basis_point_set.npy'
bps_np = np.load(bps_path)
bps = b_torch.bps_torch(custom_basis=bps_np)

grasp_pub = rospy.Publisher('goal_pick_pose', String, queue_size=10)
rospy.init_node('pose_pub')
rate = rospy.Rate(10) # 10hz
# i = int(input('i=?'))
i = 0
try:
    while True:
        color_image, depth_image, pcd, _ = rs.capture_image()
        rs.visualize_color(color_image)
        rs.visualize_depth(depth_image)

        pcd = segment.crop_pcd_with_bbox(pcd, grasp_region_mask)
        rs.visualize_pcd(pcd)
        pcd = rs.point_cloud_distance_removal(pcd)
        obj_pcd, normal_vector = segment.plane_seg_with_angle_constrain(pcd)
        rs.save_images(i, color_image, depth_image, pcd, obj_pcd)

        # crop depth in robot base with z > 0
        crop_pcd = copy.deepcopy(obj_pcd).transform(base_T_cam)
        crop_pcd_np = np.asarray(crop_pcd.points)
        crop_pcd_np = crop_pcd_np[crop_pcd_np[:,2] >0]
        crop_pcd = o3d.geometry.PointCloud()
        crop_pcd.points = o3d.utility.Vector3dVector(crop_pcd_np)
        obj_pcd = copy.deepcopy(crop_pcd).transform(np.linalg.inv(base_T_cam))

        obj_pcd_cam = deepcopy(obj_pcd)

        # Run Inference.
        obj_pcd_np = np.asarray(obj_pcd.points)
        pcd_np = np.asarray(pcd.points)
        pc_center = obj_pcd.get_center()
        obj_pcd.translate(-pc_center)
        points = np.asarray(obj_pcd.points)

        pc_tensor = torch.from_numpy(points)
        pc_tensor.to('cuda')
        enc_dict = bps.encode(pc_tensor)

        enc_np = enc_dict['dists'].cpu().detach().numpy()

        grasps = dexgangrasp.generate_grasps(enc_np, n_samples=400, return_arr=True)
        print(grasps)

        # Visualize sampled distribution
        obj_pcd_path = './obj.pcd'
        o3d.io.write_point_cloud(obj_pcd_path, obj_pcd)
        # visualization.show_generated_grasp_distribution(obj_pcd_path, grasps)
        filtered_grasps_2 = dexgangrasp.filter_grasps(enc_np, grasps, thresh=0.80)
        n_grasps_filt_2 = filtered_grasps_2['rot_matrix'].shape[0]

        print("n_grasps after filtering: %d" % n_grasps_filt_2)
        print("This means %.2f of grasps pass the filtering" % (n_grasps_filt_2 / 400))

        # Visulize filtered distribution
        visualization.show_generated_grasp_distribution(obj_pcd_path, filtered_grasps_2)

        NUM_GRASP = 10
        rot_matrix = filtered_grasps_2['rot_matrix'][:NUM_GRASP, :, :]
        transl = filtered_grasps_2['transl'][:NUM_GRASP, :]
        joint_conf = filtered_grasps_2['joint_conf'][:NUM_GRASP, :]
        #### Pick pose for flange:####
        grasps  = {}
        for j in range(NUM_GRASP):

            cam_T_palm = utils.hom_matrix_from_transl_rot_matrix(transl[j]+pc_center, rot_matrix[j])
            base_T_palm = np.matmul(base_T_cam, cam_T_palm)

            palm_T_flange=np.linalg.inv(flange_T_palm)
            base_T_flange = np.matmul(base_T_palm, palm_T_flange)

            ##### Intermediate pose for flange: ######
            base_T_palm_inter = np.eye(4)
            base_T_palm_inter[:3,-1] = base_T_palm[:3,-1] - base_T_palm[:3,:3] @ inter_offset
            base_T_palm_inter[:3,:3] = base_T_palm[:3,:3]
            base_T_flange_inter = np.matmul(base_T_palm_inter, palm_T_flange)

            print(base_T_flange_inter)
            print(base_T_flange)

            #### Decompose and send poses:
            flange_trans_inter, flange_quat_inter = divide_into_trans_quat(base_T_flange_inter)
            flange_trans_pick, flange_quat_pick = divide_into_trans_quat(base_T_flange)

            pick_goals_dict = {
                "inter":{
                    "position": {"x": flange_trans_inter[0], "y": flange_trans_inter[1], "z": flange_trans_inter[2]},
                    "orientation": {"x": flange_quat_inter[0], "y": flange_quat_inter[1], "z": flange_quat_inter[2], "w": flange_quat_inter[3]}
                },

                "pick":{
                    "position": {"x": flange_trans_pick[0], "y": flange_trans_pick[1], "z": flange_trans_pick[2]},
                    "orientation": {"x": flange_quat_pick[0], "y": flange_quat_pick[1], "z": flange_quat_pick[2], "w": flange_quat_pick[3]}
                }
            }
            grasps[str(j)] = pick_goals_dict
        
        grasp_pub.publish(str(grasps))
        rate.sleep()

        np.save("./base2flange_inferred.npy",base_T_flange)

        visualization.show_grasp_and_object(obj_pcd_path, palm_pose_centr, joint_conf,
                                            'meshes/robotiq_palm/robotiq-3f-gripper_articulated.urdf')
        
        a = input('Break loop? (y/n): ')
        if a == 'y':
            break

        print('got reply fro zeromq')
        i += 1

except KeyboardInterrupt:
    print('something broke')