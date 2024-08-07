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
from FFHNet.config.config import Config
from FFHNet.data.bps_encoder import BPSEncoder
from FFHNet.data.ffhevaluator_data_set import (FFHEvaluatorDataSet,
                                               FFHEvaluatorPCDDataSet)
from FFHNet.data.ffhgenerator_data_set import FFHGeneratorDataSet
from FFHNet.models.ffhgan import FFHGANet

from FFHNet.models.ffhnet import FFHNet

from FFHNet.models.networks import FFHGAN
from FFHNet.utils import utils, visualization, writer
from FFHNet.utils.writer import Writer
import bps_torch.bps as b_torch

import tf.transformations

# from bps_torch.utils import to_np
# Add GraspInference to the path and import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','src'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','vlpart'))

from FFHNet.utils.filter_grasps_given_mask import filter_grasps_given_mask, sort_grasps, filter_grasps_given_mask_offline
import zmq
import numpy as np
import open3d as o3d
from time import time,sleep


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
save_path = '/workspaces/inference_container/exp_images/'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

grasp_region_mask = np.zeros((720,1280),dtype=np.bool)
# grasp_region_mask[150:420, 150:600] = True  # single obj

grasp_region_mask[200:630, 530:930] = True  # cupboard grasping
mask_shape = (430,400,3)
# for bigger item
# grasp_region_mask[200:720, 430:1030] = True  # cupboard grasping
# mask_shape = (520,600,3)


base_T_cam = np.array([[ 0.99993021, -0.00887332 ,-0.00779972 , 0.31846705],
                    [ 0.00500804, -0.2795885  , 0.96010686 ,-1.10184744],
                    [-0.01070005, -0.96007892 ,-0.27952455 , 0.50819482],
                    [ 0.        ,  0.         , 0.          ,1.        ]])


inter_offset = np.array([0.16, 0, 0])
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.split(os.path.split(ROOT_PATH)[0])[0]
parser = argparse.ArgumentParser()

# # Best VAE so far:
gen_path = "checkpoints/ffhnet/ffhgenerator_bs5012"
best_epoch = 30

# Best GAN so far:
# gen_path = "checkpoints/ffhgan/2024-03-10T17_31_55_ffhgan_lr_0.0001_bs_1000"
# best_epoch = 32

# gen_path = "checkpoints/ffhgan/2024-03-15T15_20_19_ffhgan_lr_0.0001_bs_1000"
# best_epoch = 63


parser.add_argument('--gen_path', default=gen_path, help='path to FFHGenerator model')
parser.add_argument('--load_gen_epoch', type=int, default=best_epoch, help='epoch of FFHGenerator model')
# New evaluator:checkpoints/ffhevaluator/2024-06-23_ffhevaluator
parser.add_argument('--eva_path', default='checkpoints/ffhevaluator/2024-06-23_ffhevaluator', help='path to FFHEvaluator model')
parser.add_argument('--load_eva_epoch', type=int, default=30, help='epoch of FFHEvaluator model')
parser.add_argument('--config', type=str, default='FFHNet/config/config_ffhgan.yaml')

args = parser.parse_args()

load_path_gen = args.gen_path
load_path_eva = args.eva_path
load_epoch_gen = args.load_gen_epoch
load_epoch_eva = args.load_eva_epoch
config_path = args.config
config = Config(config_path)
cfg = config.parse()

if 'ffhgan' in gen_path:
    model = FFHGANet(cfg)
else:
    model = FFHNet(cfg)
# print(model)
base_data_bath = os.path.join(ROOT_PATH,'data','real_objects')
model.load_ffhgenerator(epoch=load_epoch_gen, load_path=load_path_gen)
model.load_ffhevaluator(epoch=load_epoch_eva, load_path=load_path_eva)

path_real_objs_bps = os.path.join(base_data_bath, 'bps')

bps_path = 'models/basis_point_set.npy'
bps_np = np.load(bps_path)
bps = b_torch.bps_torch(custom_basis=bps_np)

start = int(input('i=?'))
# i = 0
try:
    for i in range(start, 115, 1):

        color_name = 'color_' + str(i).zfill(4) + '.png'
        color2save = os.path.join(save_path, color_name)
        cropped_color_name = 'cropped_color_' + str(i).zfill(4) + '.png'
        cropped_color2save = os.path.join(save_path, cropped_color_name)
        depth_name = 'depth_' + str(i).zfill(4) + '.npy'
        depth_png_name = 'depth_' + str(i).zfill(4) + '.png'
        depth2save = os.path.join(save_path, depth_name)
        depth2save_png = os.path.join(save_path, depth_png_name)
        saved_pcd_obj_path = save_path + 'point_cloud_obj_' + str(i).zfill(4) + '.pcd'
        saved_pcd_path = save_path +  'point_cloud_' + str(i).zfill(4) + '.pcd'
        path2mask = os.path.join(save_path,'mask_' + str(i).zfill(4) +'.npy')
        masks = np.load(path2mask)
        if masks.ndim == 3:
            masks = masks[0]
        vlm_region_mask = np.zeros_like(grasp_region_mask,dtype=np.bool)

        vlm_region_mask[200:630, 530:930] = masks

        obj_pcd = o3d.io.read_point_cloud(saved_pcd_obj_path)
        
        pcd = o3d.io.read_point_cloud(saved_pcd_path)
        # Given camera intrinsic matrix
        camera_matrix = [
            [952.828, 0.0, 646.699],
            [0.0, 952.828, 342.637],
            [0.0, 0.0, 1.0]
        ]

        # Extract intrinsic parameters
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]

        # Image dimensions (replace with actual dimensions)
        width = 1280
        height = 720

        # Create Open3D PinholeCameraIntrinsic object
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(width, height, fx, fy, cx, cy)

        # Assuming an identity matrix for extrinsics
        extrinsics = np.eye(4)

        # Load depth and color images (replace with actual paths)
        depth_image_npy = np.load(depth2save)
        depth_image_cropped = np.where(grasp_region_mask, depth_image_npy, 0)
        depth_image_masked = np.where(vlm_region_mask, depth_image_cropped, 0)

        cv2.imwrite(depth2save_png, depth_image_masked)
        depth_image = o3d.io.read_image(depth2save_png)
        color_image = o3d.io.read_image(color2save)

        # cv2.imwrite(depth2save_png, depth_image_masked[200:630, 530:930])
        # depth_image = o3d.io.read_image(depth2save_png)
        # color_image = o3d.io.read_image(cropped_color2save)


        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image)

        # Create point cloud from RGBD image and camera intrinsic
        object_part_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

        # object_part_pcd= segment.crop_pcd_with_bbox(object_part_pcd, grasp_region_mask)

        # # Apply extrinsics (if necessary, here it's identity so no change)
        # object_part_pcd.transform(extrinsics)

        # Visualize point cloud
        o3d.visualization.draw_geometries([object_part_pcd])

        # crop depth in robot base with z > 0
        crop_pcd = copy.deepcopy(obj_pcd).transform(base_T_cam)
        crop_pcd_np = np.asarray(crop_pcd.points)
        crop_pcd_np = crop_pcd_np[crop_pcd_np[:,2] >0]
        crop_pcd = o3d.geometry.PointCloud()
        crop_pcd.points = o3d.utility.Vector3dVector(crop_pcd_np)
        o3d.visualization.draw_geometries([crop_pcd])
        obj_pcd = copy.deepcopy(crop_pcd).transform(np.linalg.inv(base_T_cam))

        obj_pcd_cam = deepcopy(obj_pcd)

        ####################### Run Inference  ##############

        # pcd_base = copy.deepcopy(pcd_raw).transform(base_T_cam)

        obj_pcd_np = np.asarray(obj_pcd.points)
        pcd_np = np.asarray(pcd.points)
        pc_center = obj_pcd.get_center()
        obj_pcd.translate(-pc_center)
        points = np.asarray(obj_pcd.points)

        pc_tensor = torch.from_numpy(points)
        pc_tensor.to('cuda')
        enc_dict = bps.encode(pc_tensor)

        enc_np = enc_dict['dists'].cpu().detach().numpy()


        grasps = model.generate_grasps(enc_np, n_samples=400, return_arr=True)
        # print(grasps)

        vis_grasps = deepcopy(grasps)
        #### For test base grasping ####
        # grasps in object point cloud center 
        obj_pcd_path = './obj.pcd'
        o3d.io.write_point_cloud(obj_pcd_path, obj_pcd)
        visualization.show_generated_grasp_distribution(obj_pcd_path, vis_grasps)

        part_pcd_np = np.asarray(object_part_pcd.points)
        sorted_grasp_indices, part_mean = filter_grasps_given_mask_offline(grasps, part_pcd_np, mask_shape, color2save, pc_center)
        grasps = sort_grasps(grasps, sorted_grasp_indices, sort_num=30)

        visualization.show_generated_grasp_distribution(obj_pcd_path, vis_grasps, mean_coord=part_mean)
        visualization.show_generated_grasp_distribution(obj_pcd_path, grasps)

        # # # Visualize sampled distribution
        # visualization.show_generated_grasp_distribution(obj_pcd_path, grasps)
        filtered_grasps_2 = model.filter_grasps(enc_np, grasps, thresh=-1)

        n_grasps_filt_2 = filtered_grasps_2['rot_matrix'].shape[0]

        print("n_grasps after filtering: %d" % n_grasps_filt_2)
        print("This means %.2f of grasps pass the filtering" % (n_grasps_filt_2 / 400))

        # Visulize filtered distribution
        # visualization.show_generated_grasp_distribution(obj_pcd_path, filtered_grasps_2)

        # for j in range(n_grasps_filt_2):
        # Get the grasp sample
        NUM_GRASP = 10
        rot_matrix = filtered_grasps_2['rot_matrix'][:NUM_GRASP, :, :]
        transl = filtered_grasps_2['transl'][:NUM_GRASP, :]
        # if transl[1] > -0.1:
        #     continue
        joint_conf = filtered_grasps_2['joint_conf'][:NUM_GRASP, :]
        # palm in frame wrt center of obj pcd

        #### Pick pose for flange:####
        # palm in frame wrt cam
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
                    "position": {"x": flange_trans_inter[0], "y": flange_trans_inter[1], "z": flange_trans_inter[2]+0.05},
                    "orientation": {"x": flange_quat_inter[0], "y": flange_quat_inter[1], "z": flange_quat_inter[2], "w": flange_quat_inter[3]}
                },

                "pick":{
                    "position": {"x": flange_trans_pick[0], "y": flange_trans_pick[1], "z": flange_trans_pick[2]+0.05},
                    "orientation": {"x": flange_quat_pick[0], "y": flange_quat_pick[1], "z": flange_quat_pick[2], "w": flange_quat_pick[3]}
                }
            }
            grasps[str(j)] = pick_goals_dict
            palm_pose_centr = utils.hom_matrix_from_transl_rot_matrix(transl[j], rot_matrix[j])
            visualization.show_grasp_and_object(obj_pcd_path, palm_pose_centr, joint_conf[j],
                                                'meshes/robotiq_palm/robotiq-3f-gripper_articulated.urdf')
        
        # grasp_pub.publish(str(grasps))
        # rate.sleep()

        np.save("./base2flange_inferred.npy",base_T_flange)

        #### send grasp to local pc
        a = input('Break loop? (y/n): ')
        if a == 'y':
            break
        # send_grasp(base_T_flange)
        print('got reply fro zeromq')

        # # TODO: add inference locally
        # # cam_T_grasps_np = send_pcd(obj_pcd_np,pcd_np)

        # # vis_all_grasps(pcd_raw,cam_T_grasps_np)
        i += 1


except KeyboardInterrupt:
    print('something broke')