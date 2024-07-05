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
import torch
import argparse
from FFHNet.config.config import Config
from FFHNet.data.bps_encoder import BPSEncoder
from FFHNet.data.ffhevaluator_data_set import (FFHEvaluatorDataSet,
                                               FFHEvaluatorPCDDataSet)
from FFHNet.data.ffhgenerator_data_set import FFHGeneratorDataSet
from FFHNet.models.ffhgan import FFHGANet
from FFHNet.models.networks import FFHGAN
from FFHNet.utils import utils, visualization, writer
from FFHNet.utils.writer import Writer
import bps_torch.bps as b_torch
# from bps_torch.utils import to_np
# Add GraspInference to the path and import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','src'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from segmentation import PlaneSegmentation
from realsense import RealSense

# Data collection
# from ffhflow_grasp_viewer import show_grasp_and_object_given_pcd

import zmq
import numpy as np
import open3d as o3d
from time import time,sleep

# context = zmq.Context()
# socket = context.socket(zmq.REQ)
# socket.connect("tcp://10.3.100.77:5561")
# flags=0
# track=False


# def send_grasp(grasp_np):
#     """
#     obj_pcd_np in camera orientation but self centered.
#     """

#     info_seg = dict(
#         grasp_np_dtype=str(grasp_np.dtype),
#         ograsp_np_shape=grasp_np.shape,
#     )
#     start = time()
#     socket.send_json(info_seg, flags | zmq.SNDMORE)
#     socket.send_multipart([grasp_np], flags | zmq.SNDMORE, copy=True, track=track)

#     # received_info_seg = socket.recv_json(flags=flags)
#     # response_seg = socket.recv_multipart()

#     # print('run ffhnet takes', time()-start)
#     # grasp_poses_buffer = memoryview(response_seg[1])
#     # grasp_poses = np.frombuffer(grasp_poses_buffer, dtype=received_info_seg["grasp_poses_dtype"]).reshape(received_info_seg["grasp_poses_shape"])
#     # print("The reply is: ", grasp_poses.shape)

#     return True


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
save_path = '~/Downloads'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
rs = RealSense(logger, save_path)
segment = PlaneSegmentation()

grasp_region_mask = np.zeros((720,1280),dtype=np.bool)
# grasp_region_mask[150:420, 150:600] = True  # single obj
grasp_region_mask[280:480, 650:950] = True  # cupboard grasping

base_T_cam = np.array([[ 0.99993021, -0.00887332 ,-0.00779972 , 0.31846705],
                    [ 0.00500804, -0.2795885  , 0.96010686 ,-1.10184744],
                    [-0.01070005, -0.96007892 ,-0.27952455 , 0.50819482],
                    [ 0.        ,  0.         , 0.          ,1.        ]])

inter_offset = np.array([0.15, 0, 0])
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.split(os.path.split(ROOT_PATH)[0])[0]
parser = argparse.ArgumentParser()
# # Best VAE so far:
# gen_path = "checkpoints/ffhnet/2023-09-01T01_16_11_ffhnet_lr_0.0001_bs_1000"
# best_epoch = 24

# Best GAN so far:
gen_path = "checkpoints/ffhgan/2024-03-10T17_31_55_ffhgan_lr_0.0001_bs_1000"
best_epoch = 32
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
ffhgan = FFHGANet(cfg)
print(ffhgan)
base_data_bath = os.path.join(ROOT_PATH,'data','real_objects')
ffhgan.load_ffhgenerator(epoch=load_epoch_gen, load_path=load_path_gen)
ffhgan.load_ffhevaluator(epoch=load_epoch_eva, load_path=load_path_eva)
path_real_objs_bps = os.path.join(base_data_bath, 'bps')

bps_path = './basis_point_set.npy'
bps_np = np.load(bps_path)
bps = b_torch.bps_torch(custom_basis=bps_np)


# i = int(input('i=?'))
try:
    while True:
        color_image, depth_image, pcd, _ = rs.capture_image()
        rs.visualize_color(color_image)
        rs.visualize_depth(depth_image)
        pcd_raw = deepcopy(pcd)
        pcd_raw = rs.point_cloud_distance_removal_by_input(pcd_raw)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()

        pcd = segment.crop_pcd_with_bbox(pcd, grasp_region_mask)
        pcd = rs.point_cloud_distance_removal(pcd)
        obj_pcd, normal_vector = segment.plane_seg_with_angle_constrain(pcd)
        # rs.save_images(i, color_image, depth_image, pcd, obj_pcd)

        # crop depth in robot base with z > 0
        crop_pcd = copy.deepcopy(obj_pcd).transform(base_T_cam)
        crop_pcd_np = np.asarray(crop_pcd.points)
        crop_pcd_np = crop_pcd_np[crop_pcd_np[:,2] >0]
        crop_pcd = o3d.geometry.PointCloud()
        crop_pcd.points = o3d.utility.Vector3dVector(crop_pcd_np)
        # o3d.visualization.draw_geometries([crop_pcd])
        obj_pcd = copy.deepcopy(crop_pcd).transform(np.linalg.inv(base_T_cam))

        obj_pcd_cam = deepcopy(obj_pcd)

        ####################### Run Inference  ##############
        pcd_base = copy.deepcopy(pcd_raw).transform(base_T_cam)
        obj_pcd_np = np.asarray(obj_pcd.points)
        pcd_np = np.asarray(pcd.points)
        pc_center = obj_pcd.get_center()
        obj_pcd.translate(-pc_center)
        points = np.asarray(obj_pcd.points)

        pc_tensor = torch.from_numpy(points)
        pc_tensor.to('cuda')
        enc_dict = bps.encode(pc_tensor)

        enc_np = enc_dict['dists'].cpu().detach().numpy()

        grasps = ffhgan.generate_grasps(enc_np, n_samples=400, return_arr=True)
        print(grasps)

        # # # Visualize sampled distribution
        obj_pcd_path = './obj.pcd'
        o3d.io.write_point_cloud(obj_pcd_path, obj_pcd)
        visualization.show_generated_grasp_distribution(obj_pcd_path, grasps)
        filtered_grasps_2 = ffhgan.filter_grasps(enc_np, grasps, thresh=0.90)
        n_grasps_filt_2 = filtered_grasps_2['rot_matrix'].shape[0]

        print("n_grasps after filtering: %d" % n_grasps_filt_2)
        print("This means %.2f of grasps pass the filtering" % (n_grasps_filt_2 / 400))

        # Visulize filtered distribution
        visualization.show_generated_grasp_distribution(obj_pcd_path, filtered_grasps_2)

    
        for j in range(n_grasps_filt_2):
            # Get the grasp sample
            rot_matrix = filtered_grasps_2['rot_matrix'][j, :, :]
            transl = filtered_grasps_2['transl'][j, :]
            # if transl[1] > -0.1:
            #     continue
            joint_conf = filtered_grasps_2['joint_conf'][j, :]
            # palm in frame wrt center of obj pcd
            palm_pose_centr = utils.hom_matrix_from_transl_rot_matrix(transl, rot_matrix)

            # palm in frame wrt cam
            cam_T_palm = utils.hom_matrix_from_transl_rot_matrix(transl+pc_center, rot_matrix)
            base_T_palm = np.matmul(base_T_cam, cam_T_palm)
            palm_T_flange=np.linalg.inv(flange_T_palm)
            base_T_flange = np.matmul(base_T_palm, palm_T_flange)
            print(base_T_flange)
            np.save('./base2flange_inferred.npy',base_T_flange)

            visualization.show_grasp_and_object(obj_pcd_path, palm_pose_centr, joint_conf,
                                                'meshes/robotiq_palm/robotiq-3f-gripper_articulated.urdf')
            a = input('Break loop? (y/n): ')
            if a == 'y':
                break
        # # TODO: add inference locally
        # # cam_T_grasps_np = send_pcd(obj_pcd_np,pcd_np)
        # print('got reply fro zeromq')

        # # vis_all_grasps(pcd_raw,cam_T_grasps_np)

        # ##### send grasp to local pc
        # # send_grasp(cam_T_grasps_np)

except KeyboardInterrupt:
    print('something broke')
# finally:
#     socket.close()
#     context.term()