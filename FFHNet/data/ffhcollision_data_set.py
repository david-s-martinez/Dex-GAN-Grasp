import os
import sys
from copy import deepcopy
from time import time

import h5py
import numpy as np
import open3d as o3d
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
from FFHNet.utils import utils, visualization
from FFHNet.utils.grasp_data_handler import GraspDataHandlerVae

path2objs_gazebo = "/home/yb/Projects/gazebo-objects/objects_gazebo"
class FFHCollDetrDataSet(Dataset):
    def __init__(self, cfg, eval=False) -> None:
        super().__init__()

        self.cfg = cfg

        # we have to manually switch train to eval during training since evaluation is needed after every epoch
        if eval:
            cfg["ds_name"] = "eval"
        else:
            cfg["ds_name"] = "train"

        # ratio for collision/noncollision
        # Please double check the ratio according to the dataset.
        self.coll_ratio = 0.5
        self.noncoll_ratio = 0.5

        # dataset related data
        self.ds_path = os.path.join(cfg["data_dir"], cfg["ds_name"])
        self.objs_names = self.get_objs_names(self.ds_path)
        self.objs_folder = os.path.join(self.ds_path, 'bps')

        # path to h5 data
        grasp_data_path = os.path.join(cfg["data_dir"], cfg["grasp_data_file_name"])
        # datahandler can provide grasp samples
        self.grasp_data_handler = GraspDataHandlerVae(grasp_data_path)

        # Extract infos about dataset fro csv file

        metadata = pd.read_csv(os.path.join(cfg["data_dir"], 'metadata.csv'))
        metadata_name_pos = metadata[metadata[cfg["ds_name"]] == 'X'].loc[:, ['Unnamed: 0', 'positive']]
        metadata_name_neg = metadata[metadata[cfg["ds_name"]] == 'X'].loc[:, ['Unnamed: 0', 'negative']]
        metadata_name_collision = metadata[metadata[cfg["ds_name"]] == 'X'].loc[:, ['Unnamed: 0', 'collision']]
        metadata_name_noncollision = metadata[metadata[cfg["ds_name"]] == 'X'].loc[:, ['Unnamed: 0', 'non_collision_not_executed']]

        self.num_collision_per_object = dict(
            zip(metadata_name_pos.iloc[:, 0], metadata_name_collision.iloc[:, 1].astype('int64')))
        # self.num_noncollision_per_object = dict(
        #     zip(metadata_name_pos.iloc[:, 0],
        #         metadata_name_pos.iloc[:, 1].astype('int64') + metadata_name_neg.iloc[:, 1].astype('int64')))
        self.num_noncollision_per_object = dict(
            zip(metadata_name_pos.iloc[:, 0], metadata_name_noncollision.iloc[:, 1].astype('int64')))

        # Build paths list and corr. labels
        self.bps_paths, self.labels = self.get_all_bps_paths_and_labels(
            self.objs_folder, self.num_noncollision_per_object)

        # camera pose for transforming the point cloud into correct frame
        self.camera_T_world = np.array([[1.00000000e+00, -2.18874845e-13, -1.44702345e-12, -4.80000000e-01],
                            [-1.44707624e-12, -2.95520207e-01, -9.55336489e-01, 9.46755955e-02],
                            [-2.18525541e-13,  9.55336489e-01, -2.95520207e-01, 9.15468088e-01],
                            [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
        self.world_T_camera = np.linalg.inv(self.camera_T_world)

        # Debug mode
        self.debug = False

    def get_objs_names(self, ds_path) -> list:
        """
        Args:
            ds_path (str): _description_

        Returns:
            list: list of object names
        """
        objs_folder = os.path.join(ds_path, 'bps')
        return [obj for obj in os.listdir(objs_folder) if '.' not in obj]

    def get_all_bps_paths_and_labels(self, objs_folder, noncollision_per_obj_dict):
        """ Creates a long list of paths to bps object files. Each bps for a respective object gets repeated
        as many times as there are success grasps. Based on this it is repeated for negative and collision grasps.

        Args:
            obj_folder (string): The path to the folder where the BPS lie
            noncollision_per_obj_dict (dict): A dict with one key for each object and val being the amount of x grasps per object

        Returns:
            bps_paths (list of strings): Long list of BPS paths.
            labels (list of strings): Indicating whether the grasp was successful, unsuccessful or in collision
        """
        paths = []
        labels = []
        for obj, n_noncoll in noncollision_per_obj_dict.items():
            n_total = int (n_noncoll // self.noncoll_ratio)
            n_coll = n_total - n_noncoll
            obj_path = os.path.join(objs_folder, obj)
            for f_name in os.listdir(obj_path):

                ########### Select specific dataset ############
                if f_name.split('.')[0].split('_')[-1] == 'single':
                    continue
                # elif f_name.split('.')[0].split('_')[-1] == 'multi':
                #     continue
                elif f_name.split('.')[0].split('_')[-1] == 'obstacle':
                    continue
                ################################################

                f_path = os.path.join(obj_path, f_name)
                if 'bps' in os.path.split(f_name)[1]:
                    # Paths
                    paths += n_coll * [f_path]
                    paths += n_noncoll * [f_path]
                    # Labels
                    labels += n_coll * ['collision']
                    labels += n_noncoll * ['noncollision']

        assert len(paths) == len(labels)
        return paths, labels

    def read_pose_pcd_transf_and_center(self, bps_path):
        """

        Args:
            bps_path (str):

        Returns:
            hom_matrix (array): transformation matrix
            object_mesh_world (array): transformation matrix
        """
        # pcd save path from bps save path
        base_path, bps_name = os.path.split(bps_path)
        pcd_name = bps_name.replace('bps', 'pcd')
        pcd_name = pcd_name.replace('.npy', '.pcd')
        path = os.path.join(base_path, pcd_name)

        # Extract object name from path
        head, pcd_file_name = os.path.split(path)
        pcd_name = pcd_file_name.split('.')[0]

        ########### Configure for different dataset ##########
        # for dataset with multi objects, pcd name ends with 'multi' which has to be removed.
        if pcd_name.find('_multi') != -1:
            pcd_name = pcd_name[:pcd_name.find('_multi')]
        if pcd_name.find('_obstacle') != -1:
            pcd_name = pcd_name[:pcd_name.find('_obstacle')]
        ######################################################

        obj = os.path.split(head)[1]

        # Read the corresponding transform in
        path = os.path.join(os.path.split(self.ds_path)[0], 'pcd_transforms.h5')

        # TODO: check here: frames here are generated in data augmentation,
        # which is not verified in verify_collision_label.py
        with h5py.File(path, 'r') as hdf:
            pos_quat_list = hdf[obj][pcd_name + '_mesh_to_centroid'][()]
            object_mesh_world = hdf[obj][pcd_name + '_mesh_to_world'][()]
            pcd_center = hdf[obj][pcd_name + '_multi_center'][()]

        # Transform the transform to numpy 4*4 array
        pose_matrix_mesh_to_centroid = utils.hom_matrix_from_pos_quat_list(pos_quat_list)
        object_matrix_mesh_to_world = utils.hom_matrix_from_pos_quat_list(object_mesh_world)
        return pose_matrix_mesh_to_centroid, object_matrix_mesh_to_world, pcd_center

    def get_pcd_centroid_T_world(self, pcd_center):
        # world_T_pcd_centroid[:3,:3] = world_rot_cam x cam_rot_centroid, where cam_rot_centroid is identical matrix
        # -> world_T_pcd_centroid[:3,:3] = world_rot_cam
        world_T_pcd_centroid = np.eye(4)
        world_T_pcd_centroid[:3,:3] = self.world_T_camera[:3,:3]
        world_T_pcd_centroid[:3,-1] = pcd_center

        pcd_centroid_T_world = np.linalg.inv(world_T_pcd_centroid)
        return pcd_centroid_T_world

    def __getitem__(self, idx):
        time1 = time()
        bps_path = self.bps_paths[idx]
        label = self.labels[idx]

        # the label for ffhcoll is 1 for collision and 0 for noncollision
        collision = 1 if label == 'collision' else 0

        # Load the bps encoding
        # in case the bps is not correct, rerun the script in bps_torch repo to overwrite bps file.
        base_path, bps_name = os.path.split(bps_path)
        obj_name = '_'.join(bps_name.split('_bps')[:-1])
        bps_obj = np.load(bps_path)

        # Read the corresponding transform between mesh_frame and object_centroid
        # TODO: this centr_T_mesh_pose is target obj center not all objects center, from data_augmentation_multi_obj.
        centr_T_mesh_pose, world_T_mesh_obj, pcd_center = self.read_pose_pcd_transf_and_center(bps_path)

        # Read in a grasp for a given object (in mesh frame)
        outcome = 'non_collision_not_executed' if label == 'noncollision' else 'collision'
        palm_pose_mesh, joint_conf, _ = self.grasp_data_handler.get_single_grasp_of_outcome(
            obj_name, outcome=outcome, random=True)
        mesh_T_palm_pose = utils.hom_matrix_from_pos_quat_list(palm_pose_mesh)


        # Transform plam pose from mesh frame to pcd centroid frame, so equal to bps encoded point cloud where it's self centered
        pcd_centroid_T_world = self.get_pcd_centroid_T_world(pcd_center)
        world_T_palm_pose = np.matmul(world_T_mesh_obj, mesh_T_palm_pose)
        # world_T_palm_pose = world_T_pcd_centroid x centroid_T_palm_pose
        centroid_T_palm_pose = np.matmul(pcd_centroid_T_world, world_T_palm_pose)

        # new_point_cloud_in_grasp_frame = world_T_grasp_pose * point_cloud_in_world
        world_T_grasp_pose = np.matmul(world_T_mesh_obj, mesh_T_palm_pose)

        # Turn the full 20 DoF into 15 DoF as every 4th joint is coupled with the third
        joint_conf = utils.reduce_joint_conf(joint_conf)

        # Extract rotmat and transl
        palm_rot_matrix = centroid_T_palm_pose[:3, :3]
        palm_transl = centroid_T_palm_pose[:3, 3]

        # Visualization
        if self.debug:
            # only look for noncollision grasps
            if label == 'noncollision':
                pcd_path = bps_path.replace('bps','pcd')
                pcd_path = pcd_path.replace('npy','pcd')
                pcd = o3d.io.read_point_cloud(pcd_path)

                origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.1)
                print("original pcd")
                o3d.visualization.draw_geometries([pcd, origin])

                # TODO: wrong world_T_mesh_obj in visualization
                visualization.show_dataloader_grasp_with_pcd_in_world_frame(obj_name, world_T_mesh_obj, world_T_grasp_pose, mesh_T_palm_pose
                            , pcd, path2objs_gazebo)
                print("label", label)
                print(palm_transl)
                print(joint_conf)

                # visualization.show_dataloader_grasp(bps_path, obj_name, centr_T_mesh_pose, mesh_T_palm_pose,
                #                                     centroid_T_palm_pose)
                # Visualize full hand config
                # TODO: bug in the following function

                pcd.transform(self.camera_T_world)
                pcd.translate(-1*pcd.get_center())
                visualization.show_grasp_and_object_given_pcd(pcd, centroid_T_palm_pose, joint_conf)

        # Build output dict
        data_out = {'rot_matrix': palm_rot_matrix,\
                    'transl': palm_transl,\
                    'joint_conf': joint_conf,\
                    'bps_object': bps_obj,\
                    'label': collision}

        # If we want to evaluate, also return the pcd path to load from for vis
        if self.cfg["ds_name"] == 'eval':
            data_out['pcd_path'] = bps_path.replace('bps', 'pcd').replace('npy', 'pcd')
            data_out['obj_name'] = obj_name

        # print("get one data sample takes", time()-time1)
        return data_out

    def __len__(self):
        return len(self.bps_paths)


class FFHCollDetrPCDDataSet(FFHCollDetrDataSet):
    def __init__(self, cfg, eval=False):
        super().__init__(cfg, eval=eval)
        # Build paths list and corr. labels
        self.objs_folder = os.path.join(self.ds_path, 'pcd')
        self.pcd_paths, self.labels = self.get_all_pcd_paths_and_labels(
            self.objs_folder, self.num_noncollision_per_object)
        self.NUM_POINTS = 2048
        self.debug = True

    def get_all_pcd_paths_and_labels(self, objs_folder, noncollision_per_obj_dict):
        """ Creates a long list of paths to bps object files. Each bps for a respective object gets repeated
        as many times as there are success grasps. Based on this it is repeated for negative and collision grasps.

        Args:
            obj_folder (string): The path to the folder where the BPS lie
            noncollision_per_obj_dict (dict): A dict with one key for each object and val being the amount of x grasps per object

        Returns:
            bps_paths (list of strings): Long list of BPS paths.
            labels (list of strings): Indicating whether the grasp was successful, unsuccessful or in collision
        """
        paths = []
        labels = []
        for obj, n_noncoll in noncollision_per_obj_dict.items():
            n_total = int(n_noncoll // self.noncoll_ratio)
            n_coll = n_total - n_noncoll
            obj_path = os.path.join(objs_folder, obj)
            for f_name in os.listdir(obj_path):

                ########### Select specific dataset ############
                if f_name.split('.')[0].split('_')[-1] == 'single':
                    continue
                # elif f_name.split('.')[0].split('_')[-1] == 'multi':
                #     continue
                elif f_name.split('.')[0].split('_')[-1] == 'obstacle':
                    continue
                ################################################

                f_path = os.path.join(obj_path, f_name)
                if 'dspcd' in os.path.split(f_name)[1]:
                    # Paths
                    paths += n_coll * [f_path]
                    paths += n_noncoll * [f_path]
                    # Labels
                    labels += n_coll * ['collision']
                    labels += n_noncoll * ['noncollision']

        assert len(paths) == len(labels)
        return paths, labels

    def read_pcd_transform(self, pcd_path):
        # pcd save path from pcd save path
        base_path, pcd_name = os.path.split(pcd_path)
        pcd_name = pcd_name.replace('_dspcd','_pcd')
        path = os.path.join(base_path, pcd_name)
        # Extract object name from path
        head, pcd_file_name = os.path.split(path)
        pcd_name = pcd_file_name.split('.')[0]
        obj = os.path.split(head)[1]

        # Read the corresponding transform in
        path = os.path.join(os.path.split(self.ds_path)[0], 'pcd_transforms.h5')
        with h5py.File(path, 'r') as hdf:
            pos_quat_list = hdf[obj][pcd_name + '_mesh_to_centroid'][()]

        # Transform the transform to numpy 4*4 array
        hom_matrix = utils.hom_matrix_from_pos_quat_list(pos_quat_list)
        return hom_matrix

    def read_pose_pcd_transf_and_center(self, pcd_path):
        """

        Args:
            bps_path (str):

        Returns:
            hom_matrix (array): transformation matrix
            object_mesh_world (array): transformation matrix
        """
        # pcd save path from bps save path
        base_path, pcd_name = os.path.split(pcd_path)
        pcd_name = pcd_name.replace('_dspcd','_pcd')
        path = os.path.join(base_path, pcd_name)

        # Extract object name from path
        head, pcd_file_name = os.path.split(path)
        pcd_name = pcd_file_name.split('.')[0]

        ########### Configure for different dataset ##########
        # for dataset with multi objects, pcd name ends with 'multi' which has to be removed.
        if pcd_name.find('_multi') != -1:
            pcd_name = pcd_name[:pcd_name.find('_multi')]
        if pcd_name.find('_obstacle') != -1:
            pcd_name = pcd_name[:pcd_name.find('_obstacle')]
        ######################################################

        obj = os.path.split(head)[1]

        # Read the corresponding transform in
        path = os.path.join(os.path.split(self.ds_path)[0], 'pcd_transforms.h5')
        with h5py.File(path, 'r') as hdf:
            pos_quat_list = hdf[obj][pcd_name + '_mesh_to_centroid'][()]
            object_mesh_world = hdf[obj][pcd_name + '_mesh_to_world'][()]
            pcd_center = hdf[obj][pcd_name + '_multi_center'][()]

        # Transform the transform to numpy 4*4 array
        # TODO: pose_matrix_mesh_to_centroid, what is centroid here mean?
        pose_matrix_mesh_to_centroid = utils.hom_matrix_from_pos_quat_list(pos_quat_list)
        object_matrix_mesh_to_world = utils.hom_matrix_from_pos_quat_list(object_mesh_world)
        return pose_matrix_mesh_to_centroid, object_matrix_mesh_to_world, pcd_center

    def _normalize_pc(self, points):
        centroid = np.mean(points, 0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance
        return points

    @staticmethod
    def pcd_transform(x,y):
        assert x.shape[1] == 3
        x_new = np.ones((x.shape[0],4))
        x_new[:,:3] = x
        x_new = np.matmul(y,x_new.T)
        return x_new.T[:,:3]

    def __getitem__(self, idx):
        pcd_path = self.pcd_paths[idx]
        label = self.labels[idx]

        # the label for ffhcoll is 1 for collision and 0 for noncollision
        collision = 1 if label == 'collision' else 0

        # Load the pcd
        base_path, pc_name = os.path.split(pcd_path)
        obj_name = '_'.join(pc_name.split('_dspcd')[:-1])
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd_arr = np.asarray(pcd.points)
        assert pcd_arr.shape == (self.NUM_POINTS,3)

        # Read the corresponding transform between mesh_frame and object_centroid
        # centr_T_mesh = self.read_pcd_transform(pcd_path)
        centr_T_mesh_pose, world_T_mesh_obj, pcd_center = self.read_pose_pcd_transf_and_center(pcd_path)

        # Read in a grasp for a given object (in mesh frame). If the label is hard-negative, read a positive one and shift it
        outcome = 'non_collision_not_executed' if label == 'noncollision' else 'collision'
        palm_pose_mesh, joint_conf, _ = self.grasp_data_handler.get_single_grasp_of_outcome(
            obj_name, outcome=outcome, random=True)
        mesh_T_palm_pose = utils.hom_matrix_from_pos_quat_list(palm_pose_mesh)

        # Transform plam pose from mesh frame to pcd centroid frame, so equal to bps encoded point cloud where it's self centered
        pcd_centroid_T_world = self.get_pcd_centroid_T_world(pcd_center)
        world_T_palm_pose = np.matmul(world_T_mesh_obj, mesh_T_palm_pose)
        # world_T_palm_pose = world_T_pcd_centroid x centroid_T_palm_pose
        centroid_T_palm_pose = np.matmul(pcd_centroid_T_world, world_T_palm_pose)

        # new_point_cloud_in_grasp_frame = world_T_grasp_pose * point_cloud_in_world
        world_T_grasp_pose = np.matmul(world_T_mesh_obj, mesh_T_palm_pose)

        # Turn the full 20 DoF into 15 DoF as every 4th joint is coupled with the third
        joint_conf = utils.reduce_joint_conf(joint_conf)

        # Extract rotmat and transl
        palm_rot_matrix = centroid_T_palm_pose[:3, :3]
        palm_transl = centroid_T_palm_pose[:3, 3]

        # Test visualize grasp
        if self.debug:
            if label == 'noncollision':
                # for vis comment the point cloud normalization part
                vis_pcd = o3d.geometry.PointCloud()
                vis_pcd.points = o3d.utility.Vector3dVector(pcd_arr)
                print(palm_transl)
                print(joint_conf)
                print("label", label)

                origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.1)
                print("original pcd")
                o3d.visualization.draw_geometries([vis_pcd, origin])

                # TODO: wrong world_T_mesh_obj in visualization
                visualization.show_dataloader_grasp_with_pcd_in_world_frame(obj_name, world_T_mesh_obj, world_T_grasp_pose, mesh_T_palm_pose
                            , vis_pcd, path2objs_gazebo)
                vis_pcd.transform(self.camera_T_world)
                vis_pcd.translate(-1*vis_pcd.get_center())
                visualization.show_grasp_and_object_given_pcd(vis_pcd, centroid_T_palm_pose, joint_conf)

        # normalize the pcd
        # TODO: check run time
        pcd_arr = self.pcd_transform(pcd_arr, pcd_centroid_T_world)
        pcd_arr = self._normalize_pc(pcd_arr)

        # Build output dict
        # For pytorch conv1 takes input of (B,C,L) batchsize, channel, length
        pcd_arr = pcd_arr.reshape((3,-1))
        data_out = {'rot_matrix': palm_rot_matrix,\
                    'transl': palm_transl,\
                    'joint_conf': joint_conf,\
                    'pcd_array': pcd_arr,\
                    'label': collision}

        # If we want to evaluate, also return the pcd path to load from for vis
        if self.cfg["ds_name"] == 'eval':
            data_out['pcd_path'] = pcd_path
            data_out['obj_name'] = obj_name

        return data_out

    def __len__(self):
        return len(self.pcd_paths)


class FFHCollDetrHandPCDDataSet(FFHCollDetrPCDDataSet):
    def __init__(self, cfg, eval=False):
        super().__init__(cfg, eval=eval)
        hand_pcd_arr = o3d.io.read_point_cloud(os.path.join(self.ds_path, '..', 'ds_hithand.pcd'))
        self.hand_pcd_arr = np.asarray(hand_pcd_arr.points)
        self.hand_base_T_palm_pose = np.array([[ 2.68490602e-01,  1.43867476e-01, -9.52478318e-01,
                                                2.00000000e-02],
                                            [ 9.00297098e-04,  9.88746286e-01,  1.49599368e-01,
                                                0.00000000e+00],
                                            [ 9.63281883e-01, -4.10235378e-02,  2.65339562e-01,
                                                6.00000000e-02],
                                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                                1.00000000e+00]])
        self.debug = True

    def __getitem__(self, idx):
        pcd_path = self.pcd_paths[idx]
        label = self.labels[idx]

        # the label for ffhcoll is 1 for collision and 0 for noncollision
        collision = 1 if label == 'collision' else 0

        # Load the pcd in world frame
        base_path, pc_name = os.path.split(pcd_path)
        obj_name = '_'.join(pc_name.split('_dspcd')[:-1])
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd_arr = np.asarray(pcd.points)
        assert pcd_arr.shape == (self.NUM_POINTS,3)

        ######### Get all transformation matrix ################
        # Read the corresponding transform between mesh_frame and object_centroid
        _, world_T_mesh_obj, pcd_center = self.read_pose_pcd_transf_and_center(pcd_path)

        # Read in a grasp for a given object (in mesh frame). If the label is hard-negative, read a positive one and shift it
        outcome = 'non_collision_not_executed' if label == 'noncollision' else 'collision'
        palm_pose_mesh, joint_conf, _ = self.grasp_data_handler.get_single_grasp_of_outcome(
            obj_name, outcome=outcome, random=True)
        mesh_T_palm_pose = utils.hom_matrix_from_pos_quat_list(palm_pose_mesh)

        # Transform plam pose from mesh frame to pcd centroid frame, so equal to bps encoded point cloud where it's self centered
        pcd_centroid_T_world = self.get_pcd_centroid_T_world(pcd_center)

        # TODO: Something this line is wrong, check mesh_T_palm_pose
        world_T_palm_pose = np.matmul(world_T_mesh_obj, mesh_T_palm_pose)
        # world_T_palm_pose = world_T_pcd_centroid x centroid_T_palm_pose
        centroid_T_palm_pose = np.matmul(pcd_centroid_T_world, world_T_palm_pose)

        # # new_point_cloud_in_grasp_frame = world_T_grasp_pose * point_cloud_in_world
        # world_T_grasp_pose = np.matmul(world_T_mesh_obj, mesh_T_palm_pose)
        centroid_T_mesh_obj = np.matmul(pcd_centroid_T_world, world_T_mesh_obj)
        palm_pose_T_centroid = np.linalg.inv(centroid_T_palm_pose)
        # hand_base_T_centroid = np.matmul(self.hand_base_T_palm_pose, palm_pose_T_centroid)
        hand_base_T_centroid = np.matmul(self.hand_base_T_palm_pose, palm_pose_T_centroid)
        centroid_T_hand_base = np.linalg.inv(hand_base_T_centroid)
        ############################################

        # transform pcd to centroid frame (camera orientation)
        pcd_arr = self.pcd_transform(pcd_arr, pcd_centroid_T_world)

        # transform hand pcd from self centroid to pcd centroid frame
        hand_pcd_in_centroid = self.pcd_transform(self.hand_pcd_arr, centroid_T_hand_base)

        def merge_pcd(x,y):
            x_new = np.ones((x.shape[0],4)) # [x,3]
            x_new[:,:3] = x
            y_new = np.zeros((x.shape[0],4))
            y_new[:,:3] = y
            return np.concatenate((x,y),axis=0)

        # here adding scene pcd with 4th feature of 1, hand pcd with 4th feature of 0
        def normalize_pc(points):
            centroid = np.mean(points, 0)
            points -= centroid
            furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
            points /= furthest_distance
            return points, centroid, furthest_distance

        # normalize pcd from scene and hand
        # pcd_arr, centroid, distance = normalize_pc(pcd_arr)
        # hand_pcd_in_centroid -= centroid
        # hand_pcd_in_centroid /= distance

        # merge pcd together into one
        all_pcd_arr = merge_pcd(pcd_arr, hand_pcd_in_centroid)

        # Test visualize grasp
        if self.debug:
            if label == 'noncollision':
                origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.1)
                print("original pcd")

                vis_pcd = o3d.geometry.PointCloud()
                vis_pcd.points = o3d.utility.Vector3dVector(all_pcd_arr)
                print("label", label)

                palm_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.1)
                palm_pose.transform(centroid_T_mesh_obj)
                hand_base = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.1)
                hand_base.transform(centroid_T_hand_base)
                o3d.visualization.draw_geometries([vis_pcd, origin, palm_pose, hand_base])

                # joint_conf = utils.reduce_joint_conf(joint_conf)
                # not working in ar-lx0137
                # visualization.show_grasp_and_object_given_pcd(vis_pcd, centroid_T_palm_pose, joint_conf)

        # Build output dict
        # For pytorch conv1 takes input of (B,C,L) batchsize, channel, length
        all_pcd_arr = all_pcd_arr.reshape((3,-1))
        data_out = {'pcd_array': all_pcd_arr,\
                    'label': collision}

        # If we want to evaluate, also return the pcd path to load from for vis
        if self.cfg["ds_name"] == 'eval':
            data_out['pcd_path'] = pcd_path
            data_out['obj_name'] = obj_name

        return data_out

    def __len__(self):
        return len(self.pcd_paths)

if __name__ == '__main__':
    from FFHNet.config.config import Config
    path = os.path.dirname(os.path.abspath(__file__))
    BASE_PATH = os.path.split(os.path.split(path)[0])[0]

    path = os.path.join(BASE_PATH, "FFHNet/config/config_pointnet_yb.yaml")
    config = Config(path)
    cfg = config.parse()
    # gds = FFHCollDetrDataSet(cfg)
    gds = FFHCollDetrHandPCDDataSet(cfg)
    i = 0
    time_start = time()
    while True:
        # i = np.random.randint(0, gds.__len__())
        gds.__getitem__(i)
        i += 1
        if i % 10000 == 0:
            print(time()-time_start)
