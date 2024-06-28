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


i = int(input('i=?'))
try:
    while True:
        color_image, depth_image, pcd, _ = rs.capture_image()
        rs.visualize_color(color_image)
        rs.visualize_depth(depth_image)
        pcd_raw = deepcopy(pcd)
        pcd_raw = rs.point_cloud_distance_removal_by_input(pcd_raw)
        cv2.imshow("Display window", color_image)
        k = cv2.waitKey(1)
        print(f'get {i}th frame')
        if k == 27:
            break

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

        if True:
            for j in range(n_grasps_filt_2):
                # Get the grasp sample
                rot_matrix = filtered_grasps_2['rot_matrix'][j, :, :]
                transl = filtered_grasps_2['transl'][j, :]
                # if transl[1] > -0.1:
                #     continue
                joint_conf = filtered_grasps_2['joint_conf'][j, :]
                palm_pose_centr = utils.hom_matrix_from_transl_rot_matrix(transl, rot_matrix)
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

import logging
import os
import sys

import open3d as o3d
import numpy as np
import pyrealsense2 as rs

repo_root = os.path.dirname(__file__) + '/../'
sys.path.insert(0, os.path.join(repo_root, 'src'))
from realsense import RealSense

def signal_handler(self, signal, frame):
    print("====================================")
    print(" Ctrl C pressed! Script stops properly")
    self.stop()
    sys.exit(0)


def display_inlier_outlier(cloud, ind):
    """
    Args:
        cloud (open3d.geometry.PointCloud): open3d point cloud object
        ind (open3d.geometry.PointCloud): open3d point cloud object
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    # print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
                                    # zoom=0.3412,
                                    # front=[0.4257, -0.2125, -0.8795],
                                    # lookat=[2.6172, 2.0475, 1.532],
                                    # up=[-0.0694, -0.9768, 0.2024])


def get_angle_between_two_vec(vec1, vec2):
    # we assume they are normalized vector. TODO: add normalization check
    angle = np.arccos(np.dot(vec1, vec2))
    angle = np.rad2deg(angle)
    return angle

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

# parameteres
remove_statistical_outlier = True
vis = True
# If detected plane has normal with more than 45 deg compared to camera z axis.
plane_normal_threshold = 45

class PlaneSegmentation():
    def __init__(self) -> None:
        pass

    def crop_pcd_with_bbox(self, pcd, bbox):
        # -- Use bbox to crop depth/point cloud
        pcd_np = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)
        pcd_np = pcd_np.reshape(bbox.shape[0], bbox.shape[1], -1)
        pcd_colors = pcd_colors.reshape(bbox.shape[0], bbox.shape[1], -1)
        object_pcd_np = pcd_np[bbox]
        object_pcd_color_np = pcd_colors[bbox]
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_pcd_np)
        object_pcd.colors = o3d.utility.Vector3dVector(object_pcd_color_np)

        if vis:
            o3d.visualization.draw_geometries([object_pcd])

        if remove_statistical_outlier:
            print("Statistical oulier removal")
            cl, ind = object_pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
            if vis:
                display_inlier_outlier(object_pcd, ind)
        return object_pcd

    def pcd_distance_removal(self, pcd):
        pass

    def plane_seg_with_angle_constrain(self, pcd, remove_statistical_outlier=False):

        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                        ransac_n= 3,
                                                        num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        normal_vector = [a,b,c]/np.linalg.norm([a,b,c])

        # Here is the "correct" normal, which is supposed to orient towards camera
        # So we set angle constrains to filter out wrong plane segmentation.
        camera_z = np.array([0, 0, 1])
        angle = get_angle_between_two_vec(camera_z, normal_vector)
        if angle > plane_normal_threshold:
            print(f"Detected plane has normal {angle} deg, more than {plane_normal_threshold} deg")
            # return None, None
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        if vis:
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
            # o3d.visualization.draw_geometries([outlier_cloud])

        if remove_statistical_outlier:
            print("Statistical oulier removal")
            cl, ind = outlier_cloud.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
            display_inlier_outlier(outlier_cloud, ind)
            outlier_cloud = outlier_cloud.select_by_index(ind)
            o3d.visualization.draw_geometries([outlier_cloud])

        return outlier_cloud, normal_vector


if __name__ == "__main__":

    save_path = '/home/qf/Pictures/ffhflow_exp'
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    rs = RealSense(logger, save_path)
    segment = PlaneSegmentation()

    i = 0
    color_image, depth_image, pcd, _ = rs.capture_image()
    pcd = rs.point_cloud_distance_removal(pcd)
    rs.save_images(i, color_image, depth_image, pcd)
    inlier_pcd, normal_vector = segment.plane_seg_with_angle_constrain(pcd)

    # [480,640,3]
    # [480, 640] ,uint16, max 3586
    # pcd open3d



import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from time import time
import os
import subprocess

class RealSense():
    """simple warpper to realsense camera
    """
    def __init__(self, logger, save_path) -> None:

        self.pipeline = rs.pipeline()

        self.log = logger
        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        rs_conf = rs.config()
        # rs_conf.enable_device('211222064027')

        self.save_path = save_path

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = rs_conf.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        rs_conf.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
        rs_conf.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

        # Declare pointcloud object, for calculating pointclouds and texture mappings
        self.pc = rs.pointcloud()

        # Start streaming
        profile = self.pipeline.start(rs_conf)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        # print("Depth Scale is: " , depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away

        self.clipping_distance_max = 1.2  # 1.0 single objecg
        self.clipping_distance_min = 0.8
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.colorizer = rs.colorizer()

    def capture_image(self):
        """

        Returns:
            color_image(np.array):
            depth_image(np.array):
            pcd(open3d.geometry.PointCloud):
        """

        time1 = time()
        trials = 0
        while True:
            try:
                trials += 1
                frames = self.pipeline.wait_for_frames()
                if frames:
                    self.log.debug(f'tried {trials} times to capture a frame')
                    break
            except RuntimeError:
                continue

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Prepare the point cloud numpy array
        points = self.pc.calculate(aligned_depth_frame)
        w = rs.video_frame(aligned_depth_frame).width
        h = rs.video_frame(aligned_depth_frame).height
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(h, w, 3)

        # Get data of depth and color are most time-consuming part -> roughly 400ms
        time2 = time()
        depth_image = np.asanyarray(aligned_depth_frame.get_data()) # (480,640)
        color_image = np.asanyarray(color_frame.get_data()) # (480,640,3)

        # Convert point cloud array to open3d format.
        verts = verts.reshape((-1,3))

        # verts[:,2][verts[:,2]>self.clipping_distance] = 0
        # verts[:,2][verts[:,2]<0] = 0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)

        color_image_new = cv2.cvtColor(color_image,cv2.COLOR_RGB2BGR)
        colors = color_image_new / color_image_new.max()
        pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1,3))

        self.log.debug(f"time to get depth and color image {time()-time2}")

        return color_image, depth_image, pcd, time() - time1

    def depth_distance_removal(self, depth_image):
        depth_image[depth_image>self.clipping_distance_max] = 0
        return depth_image

    def point_cloud_distance_removal(self, pcd):
        pcd_np = np.asarray(pcd.points)
        pcd_colors_np = np.asarray(pcd.colors)
        new_pcd_np = pcd_np[pcd_np[:, 2] < self.clipping_distance_max]
        new_pcd_colors_np = pcd_colors_np[pcd_np[:, 2] < self.clipping_distance_max]

        new_pcd_np2 = new_pcd_np[new_pcd_np[:, 2] > self.clipping_distance_min]
        new_pcd_colors_np2 = new_pcd_colors_np[new_pcd_np[:, 2] > self.clipping_distance_min]

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_pcd_np2)
        new_pcd.colors = o3d.utility.Vector3dVector(new_pcd_colors_np2)
        return new_pcd

    def point_cloud_distance_removal_by_input(self, pcd, min=0.2,max=1.2):
        pcd_np = np.asarray(pcd.points)
        pcd_colors_np = np.asarray(pcd.colors)
        new_pcd_np = pcd_np[pcd_np[:, 2] < max]
        new_pcd_colors_np = pcd_colors_np[pcd_np[:, 2] < max]

        new_pcd_np2 = new_pcd_np[new_pcd_np[:, 2] > min]
        new_pcd_colors_np2 = new_pcd_colors_np[new_pcd_np[:, 2] > min]

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_pcd_np2)
        new_pcd.colors = o3d.utility.Vector3dVector(new_pcd_colors_np2)
        return new_pcd

    def save_grasp(self, i, grasp, joint_conf):
        """_summary_

        Args:
            i (_type_): _description_
            rot (np arr)
            transl (np arr)
        """
        grasp_name = 'grasp' + str(i).zfill(4) + '.npy'
        depth2save = os.path.join(self.save_path, grasp_name)
        joint_conf_name = 'joint_conf' + str(i).zfill(4) + '.npy'
        self.joint_conf2save = os.path.join(self.save_path, joint_conf_name)
        np.save(depth2save, grasp)
        np.save(self.joint_conf2save, joint_conf)

    def send_joint_conf(self):
        subprocess.run(["scp", self.joint_conf2save, "qf@SERVER:PATH"])

    def save_images(self, i, color_image, depth_image, pcd, obj_pcd=False) -> None:
        """save color/depth/point cloud data to configured path.

        Args:
            i (_type_): _description_
            color_image (_type_): _description_
            depth_image (_type_): _description_
            pcd (_type_): _description_
        """
        # Save to npy
        depth_name = 'depth_' + str(i).zfill(4) + '.npy'
        depth2save = os.path.join(self.save_path, depth_name)
        np.save(depth2save, depth_image)

        # # save the point cloud numpy array [w,h,3] as npy
        # depth_name = 'point_cloud_' + str(i).zfill(4) + '.npy'
        # depth2save = os.path.join(self.config["data"]["log_image_folder"], depth_name)
        # np.save(depth2save, verts)

        # save point cloud numpy array [wxh,3] as pcd
        pcd_name = 'point_cloud_' + str(i).zfill(4) + '.pcd'
        pcd2save = os.path.join(self.save_path, pcd_name)
        o3d.io.write_point_cloud(pcd2save, pcd)

        if obj_pcd is not False:
            pcd_name = 'point_cloud_obj_' + str(i).zfill(4) + '.pcd'
            pcd2save = os.path.join(self.save_path, pcd_name)
            o3d.io.write_point_cloud(pcd2save, obj_pcd)

        color_name = 'color_' + str(i).zfill(4) + '.png'
        color2save = os.path.join(self.save_path, color_name)
        cv2.imwrite(color2save, color_image)

    def get_colored_depth(self, depth_frame):
        depth_colormap = np.asanyarray(
            self.colorizer.colorize(depth_frame).get_data())
        return depth_colormap

    @staticmethod
    def visualize_color(color_image,waitkey=True):
        cv2.imshow('color_image',color_image)
        if waitkey:
            cv2.waitKey()

    @staticmethod
    def visualize_depth(depth_image):
        # depth image datatype check: float or int

        depth_image = np.expand_dims(depth_image,axis=2)
        depth_image = np.array(depth_image * 255, dtype=np.uint8)
        depth_image = np.concatenate((depth_image,depth_image,depth_image),axis=2)

        cv2.imshow('depth_image',depth_image)
        cv2.waitKey()

    @staticmethod
    def visualize_pcd(pcd):
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, origin])

    @staticmethod
    def visualize_grasp(pcd, grasp, grasp2=None):
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        grasp_frame = grasp_frame.transform(grasp)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        if grasp2 is not None:
            grasp_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            grasp_frame2 = grasp_frame2.transform(grasp2)
            o3d.visualization.draw_geometries([pcd, grasp_frame, grasp_frame2, origin])
        else:
            o3d.visualization.draw_geometries([pcd, grasp_frame, origin])

    def stop(self):
        self.pipeline.stop()

    def __del__(self):
        self.stop()
