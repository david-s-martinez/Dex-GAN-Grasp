#!/bin/python
from __future__ import division
import os
import argparse
import time
import cv2
import open3d as o3d
import transforms3d.quaternions as quat
import torch
from torch.utils.data import DataLoader
import math
import numpy as np
from DexGanGrasp.config.config import Config
from DexGanGrasp.data.dexevaluator_data_set import DexEvaluatorDataSet, DexEvaluatorPCDDataSet
from DexGanGrasp.data.dexgenerator_data_set import DexGeneratorDataSet
from DexGanGrasp.utils.writer import Writer
from DexGanGrasp.utils import utils, visualization, writer
from DexGanGrasp.models.dexgangrasp import DexGanGrasp
from DexGanGrasp.utils.grasp_data_handler import GraspDataHandler
import csv
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

def save_batch_to_file(batch):
    """
    Saves the given batch of data to a file in PyTorch's `.pth` format.
    
    Args:
        batch (torch.Tensor or any serializable object): The data batch to be saved.

    The batch is saved to the file "data/eval_batch.pth".
    """
    torch.save(batch, "data/eval_batch.pth")


def load_batch(path):
    """
    Loads a batch of data from a file in PyTorch's `.pth` format.

    Args:
        path (str): The path to the file from which to load the data.

    Returns:
        torch.Tensor or any deserialized object: The data loaded from the file.
        
    The data is loaded to the GPU device `cuda:0`.
    """
    return torch.load(path, map_location="cuda:0")

def geodesic_distance_rotmats_pairwise_np(r1s, r2s):
    """Computes pairwise geodesic distances between two sets of rotation matrices.

    Args:
      r1s: [N, 3, 3] numpy array
      r2s: [M, 3, 3] numpy array

    Returns:
      [N, M] angular distances.
    """
    rot_rot_transpose = np.einsum('aij,bkj->abik', r1s, r2s, optimize=True) #[N,M,3,3]
    tr = np.trace(rot_rot_transpose, axis1=-2, axis2=-1) #[N,M]
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def euclidean_distance_points_pairwise_np(pt1, pt2):
    """_summary_

    Args:
        pt1 (_type_): [N, 3] numpy array, predicted grasp translation
        pts (_type_): [M, 3] numpy array, ground truth grasp translation

    Returns:
        dist_mat _type_: [N,M]
    """
    dist_mat = np.zeros((pt1.shape[0],pt2.shape[0]))
    for idx in range(pt1.shape[0]):
        deltas = pt2 - pt1[idx]
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        dist_mat[idx] = dist_2
    return dist_mat


def euclidean_distance_joint_conf_pairwise_np(joint1, joint2):
    """_summary_

    Args:
        joint1 (_type_): [N, num_joint] numpy array, predicted grasp joint config
        joint2 (_type_): [M, num_joint] numpy array, ground truth grasp joint config

    Returns:
        dist_mat _type_: [N,M]
    """
    dist_mat = np.zeros((joint1.shape[0],joint2.shape[0]))
    for idx in range(joint1.shape[0]):
        deltas = joint2 - joint1[idx]
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        dist_mat[idx] = dist_2
    return dist_mat

def magd_for_grasp_distribution(grasp1, grasp2):
    """
    Mean Absolute Grasp Deviation (MAGD).

    Args:
        grasp1 (dict): predicted grasp set
        grasp2 (dict): ground truth grasp set

    Returns:
        _type_: _description_
    """

    # Convert tensor to numpy if needed
    if torch.is_tensor(grasp1['rot_matrix']):
        grasp1['rot_matrix'] = grasp1['rot_matrix'].cpu().data.numpy()
        grasp1['transl'] = grasp1['transl'].cpu().data.numpy()
        grasp1['joint_conf'] = grasp1['joint_conf'].cpu().data.numpy()

    # calculate distance matrix
    transl_dist_mat = euclidean_distance_points_pairwise_np(grasp1['transl'], grasp2['transl'])
    rot_dist_mat = geodesic_distance_rotmats_pairwise_np(grasp1['rot_matrix'], grasp2['rot_matrix'])
    joint_dist_mat = euclidean_distance_joint_conf_pairwise_np(grasp1['joint_conf'], grasp2['joint_conf'])

    transl_loss = np.min(transl_dist_mat, axis=1)  # [N,1]
    rot_loss = np.zeros_like(transl_loss)
    joint_loss = np.zeros_like(transl_loss)

    cor_grasp_idxs = []
    # find corresponding grasp according to transl dist and add the rot/joint loss
    for idx in range(transl_loss.shape[0]):
        cor_grasp_idx = np.argmin(transl_dist_mat[idx])
        cor_grasp_idxs.append(cor_grasp_idx)
        rot_loss[idx] = rot_dist_mat[idx, cor_grasp_idx]
        joint_loss[idx] = joint_dist_mat[idx, cor_grasp_idx]

    # Calculate coverage. How many grasps are found in grasp2 set.
    unique_cor_grasp_idxs = sorted(set(cor_grasp_idxs), key=cor_grasp_idxs.index)
    coverage = len(unique_cor_grasp_idxs) / len(grasp2['transl'])

    return np.sum(transl_loss), np.sum(rot_loss), np.sum(joint_loss), coverage

def filter(dexgangrasp, obj_pcd_path, obj_bps, grasps, n_samples, is_discriminator=False, thresh_succ_list=[0.5, 0.75, 0.90], visualize=False):
    """
    Filters grasps based on success probabilities in multiple stages and optionally visualizes the filtering process.

    Args:
        dexgangrasp: grasp generation model
        obj_pcd_path (str): The file path to the object's point cloud data.
        obj_bps (object): Object's Basis Point Set (BPS).
        grasps (dict): A dictionary containing grasp configurations.
        n_samples (int): Total number of grasp samples to evaluate.
        is_discriminator (bool, optional): Whether to use a discriminator-based filtering method. Defaults to False.
        thresh_succ_list (list, optional): A list of thresholds for filtering grasps at different stages. Defaults to [0.5, 0.75, 0.90].
        visualize (bool, optional): Whether to visualize the grasp distributions at each filtering stage. Defaults to False.

    Returns:
        tuple: 
            - filtered_grasps_2 (dict): The final set of filtered grasps after all stages.
            - n_grasps_filt_2 (int): The number of grasps passing the final stage of filtering.

    Process:
        - The function filters the grasps in three stages, progressively applying stricter thresholds from `thresh_succ_list`.
        - At each stage, the number of grasps that pass the filtering is printed, and the percentage of remaining grasps relative to the total samples is computed.
        - If `visualize` is True, the filtered grasp distributions are visualized after each stage using the point cloud data.
    """
    
    if visualize:
        visualization.show_generated_grasp_distribution(obj_pcd_path, grasps)

    if is_discriminator:
        filter_func = dexgangrasp.filter_grasps_discriminator
    else:
        filter_func = dexgangrasp.filter_grasps

    ############### Stage 1 ################
    # Reject grasps with low probability
    filtered_grasps = filter_func(obj_bps, grasps, thresh=thresh_succ_list[0])
    n_grasps_filt = filtered_grasps['rot_matrix'].shape[0]

    print("n_grasps after filtering: %d" % n_grasps_filt)
    print("This means %.2f of grasps pass the filtering" % (n_grasps_filt / n_samples))

    # Visulize filtered distribution
    if visualize:
        visualization.show_generated_grasp_distribution(obj_pcd_path, filtered_grasps)

    ############### Stage 2 ################
    # Reject grasps with low probability
    filtered_grasps_1 = filter_func(obj_bps, grasps, thresh=thresh_succ_list[1])
    n_grasps_filt_1 = filtered_grasps_1['rot_matrix'].shape[0]

    print("n_grasps after filtering: %d" % n_grasps_filt_1)
    print("This means %.2f of grasps pass the filtering" % (n_grasps_filt_1 / n_samples))

    # Visulize filtered distribution
    if visualize:
        visualization.show_generated_grasp_distribution(obj_pcd_path, filtered_grasps_1)

    ############## Stage 3 ################
    # Reject grasps with low probability
    filtered_grasps_2 = filter_func(obj_bps, grasps, thresh=thresh_succ_list[2])
    n_grasps_filt_2 = filtered_grasps_2['rot_matrix'].shape[0]

    print("n_grasps after filtering: %d" % n_grasps_filt_2)
    print("This means %.2f of grasps pass the filtering" % (n_grasps_filt_2 / n_samples))

    return filtered_grasps_2, n_grasps_filt_2

def poses_to_transforms(pose_vectors):
    """
    Convert multiple 7D pose vectors into rotation matrices and translation vectors.

    Args:
        pose_vectors (numpy.array): Input pose vectors of shape (N, 7),
                                     where each row represents a pose vector [x, y, z, qx, qy, qz, qw].

    Returns:
        tuple: (rotation_matrices, translation_vectors)
               rotation_matrices (numpy.array): Array of rotation matrices of shape (N, 3, 3).
               translation_vectors (numpy.array): Array of translation vectors of shape (N, 3).
    """

    # Extract position and quaternion components from pose vectors
    positions = pose_vectors[:, :3]  # Shape (N, 3)
    quaternions = pose_vectors[:, 3:]  # Shape (N, 4)

    # Convert quaternions to rotation matrices
    rotation_matrices = np.empty((len(pose_vectors), 3, 3))
    for i in range(len(pose_vectors)):
        q = quaternions[i]  # Quaternion [qx, qy, qz, qw]
        R = quat.quat2mat(q)  # Convert quaternion to rotation matrix
        rotation_matrices[i] = R
    
    # Translation vectors
    translation_vectors = positions

    return rotation_matrices, translation_vectors

def main(
    config_path,
    load_epoch_eva,
    load_epoch_gen,
    load_path_eva,
    load_path_gen,
    show_individual_grasps=False,
    is_discriminator = False,
    is_filter = False
    ):
    """
    Main function to run the Mean Absolute Grasp Deviation (MAGD) metric.

    Args:
        config_path (str): Path to the configuration file.
        load_epoch_eva (int): Epoch number for loading the grasp evaluator model.
        load_epoch_gen (int): Epoch number for loading the grasp generator model.
        load_path_eva (str): File path to load the evaluator model weights.
        load_path_gen (str): File path to load the generator model weights.
        show_individual_grasps (bool, optional): Whether to visualize individual grasps during the process. Defaults to False.
        is_discriminator (bool, optional): Whether to apply discriminator-based filtering of grasps. Defaults to False.
        is_filter (bool, optional): Whether to filter generated grasps. Defaults to False.

    Returns:
        tuple:
            - transl_loss_sum (float): Sum of the translation losses for all grasps.
            - rot_loss_sum (float): Sum of the rotation losses for all grasps.
            - joint_loss_sum (float): Sum of the joint configuration losses for all grasps.
            - coverage_mean (float): Mean coverage of the grasps.
    """


    config = Config(config_path)
    cfg = config.parse()

    model = DexGanGrasp(cfg)

    if is_discriminator:
        thresh_succ_list=[0.15, 0.175, 0.20]
    else:
        thresh_succ_list=[0.5, 0.75, 0.90]

    print(model)
    
    base_data_bath = os.path.join(ROOT_PATH,'data','real_objects')
    model.load_dexgenerator(epoch=load_epoch_gen, load_path=load_path_gen)
    model.load_dexevaluator(epoch=load_epoch_eva, load_path=load_path_eva)

    dset_gen = DexGeneratorDataSet(cfg, eval=True)
    train_loader_gen = DataLoader(dset_gen,
                                    batch_size=64,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=cfg["num_threads"])
    grasp_data = dset_gen.grasp_data_handler

    if not os.path.isfile('data/eval_batch.pth'):
        for i, batch in enumerate(train_loader_gen):
            if i == 0:
                save_batch_to_file(batch)
                break

    transl_loss_sum = 0
    joint_loss_sum = 0
    rot_loss_sum = 0
    coverage_sum = 0
    num_nan_out = 0
    num_nan_transl = 0
    num_nan_rot = 0
    num_nan_joint = 0
    batch = load_batch('data/eval_batch.pth')
    print(batch.keys())

    for idx in range(len(batch['obj_name'])):
        pcd_filename = os.path.split(batch['pcd_path'][idx].replace("\\","/"))[1]
        pcd_path = os.path.join(cfg['data_dir'],"eval","pcd",batch['obj_name'][idx],pcd_filename)
        grasps_gt = dset_gen.get_grasps_from_pcd_path(pcd_path)
        grasps_gt['joint_conf'] = np.array(grasps_gt['joint_conf'])

        if show_individual_grasps:
            visualization.show_generated_grasp_distribution(pcd_path, grasps_gt)
        if is_filter:
            out = model.generate_grasps(
                batch['bps_object'][idx].cpu().data.numpy(), 
                n_samples=grasps_gt['joint_conf'].shape[0]*5, 
                return_arr=True,
                z_offset=z_offset
                )
            
            out , n_grasps_filt_2 = filter(
                                        model, 
                                        pcd_path, 
                                        batch['bps_object'][idx].cpu().data.numpy(), out, 
                                        grasps_gt['joint_conf'].shape[0], 
                                        is_discriminator = is_discriminator, 
                                        thresh_succ_list = thresh_succ_list,
                                        visualize = show_individual_grasps
                                        )
        else:
            out = model.generate_grasps(
                batch['bps_object'][idx].cpu().data.numpy(), 
                n_samples=grasps_gt['joint_conf'].shape[0], 
                return_arr=True
                )

        if show_individual_grasps:
            visualization.show_generated_grasp_distribution(pcd_path, out)
            
        transl_loss, rot_loss, joint_loss, coverage = magd_for_grasp_distribution(out, grasps_gt)
        if not math.isnan(transl_loss) and not math.isnan(rot_loss) and not math.isnan(joint_loss):
            transl_loss_sum += transl_loss
            rot_loss_sum += rot_loss
            joint_loss_sum += joint_loss
        else:
            if math.isnan(transl_loss):
                num_nan_transl += 1
            if math.isnan(rot_loss):
                num_nan_rot += 1
            if math.isnan(joint_loss):
                num_nan_joint += 1
            num_nan_out += 1
        coverage_sum += coverage

    coverage_mean = coverage_sum / len(batch['obj_name'])
    print('transl_loss_sum:', transl_loss_sum)
    print('rot_loss_sum:', rot_loss_sum)
    print('joint_loss_sum:', joint_loss_sum)
    print('coverage', coverage_mean)
    print(f'invalid output is: {num_nan_out}/{len(batch["obj_name"])}')
    print(f'invalid transl output is: {num_nan_transl}/{len(batch["obj_name"])}')
    print(f'invalid rot output is: {num_nan_rot}/{len(batch["obj_name"])}')
    print(f'invalid joint output is: {num_nan_joint}/{len(batch["obj_name"])}')

    return transl_loss_sum, rot_loss_sum, joint_loss_sum, coverage_mean

if __name__ == '__main__':
        
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    gen_path = "checkpoints/ffhgan/2024-03-10T17_31_55_ffhgan_lr_0.0001_bs_1000"
    best_epoch = 32
    is_discriminator = False
    is_filter = False

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

    if load_epoch_gen > 0:
        # Compute the metrics for a single epoch
        main(
        config_path,    
        load_epoch_eva,
        load_epoch_gen,
        load_path_eva,
        load_path_gen,
        show_individual_grasps = False, 
        is_discriminator=is_discriminator,
        is_filter=is_filter
        )
    else:
        # Compute the metrics for all available epochs
        save_freq = 3
        num_epochs = 90
        with open(load_path_gen + '_metrics.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "transl_loss_sum", "rot_loss_sum", "joint_loss_sum", "coverage_mean"])
            for epoch in range(save_freq, num_epochs, save_freq):
                print('Evaluating epoch:',epoch)
                load_epoch_gen = epoch
                transl_loss_sum, rot_loss_sum, joint_loss_sum, coverage_mean = main(
                                                                            config_path,    
                                                                            load_epoch_eva,
                                                                            load_epoch_gen,
                                                                            load_path_eva,
                                                                            load_path_gen,
                                                                            show_individual_grasps = False, 
                                                                            is_discriminator=is_discriminator,
                                                                            is_filter=is_filter
                                                                            )
                writer.writerow([epoch, transl_loss_sum, rot_loss_sum, joint_loss_sum, coverage_mean])