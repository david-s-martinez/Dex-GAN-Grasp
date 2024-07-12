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
from eval import run_eval_gan
from FFHNet.config.config import Config
from FFHNet.data.ffhevaluator_data_set import FFHEvaluatorDataSet, FFHEvaluatorPCDDataSet
from FFHNet.data.ffhgenerator_data_set import FFHGeneratorDataSet
from FFHNet.utils.writer import Writer
from FFHNet.utils import utils, visualization, writer
from FFHNet.models.ffhgan import FFHGANet
from FFHNet.models.ffhnet import FFHNet
from FFHNet.utils.grasp_data_handler import GraspDataHandlerVae
import csv
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

def save_batch_to_file(batch):
    torch.save(batch, "eval_batch.pth")

def load_batch(path):
    return torch.load(path, map_location="cuda:0")

def full_joint_conf_from_vae_joint_conf(vae_joint_conf):
    """Takes in the 15 dimensional joint conf output from VAE and repeats the 3*N-th dimension to turn dim 15 into dim 20.

    Args:
        vae_joint_conf (np array): dim(vae_joint_conf.position) = 15

    Returns:
        full_joint_conf (JointState): Full joint state with dim(full_joint_conf.position) = 20
    """
    # for split to run we have to have even joint dim so 15->16
    if vae_joint_conf.shape[0] == 16:
        vae_joint_conf = vae_joint_conf[:15]
    full_joint_pos = np.zeros(20)
    ix_full_joint_pos = 0
    for i in range(vae_joint_conf.shape[0]):
        if (i + 1) % 3 == 0:
            full_joint_pos[ix_full_joint_pos] = vae_joint_conf[i]
            full_joint_pos[ix_full_joint_pos + 1] = vae_joint_conf[i]
            ix_full_joint_pos += 2
        else:
            full_joint_pos[ix_full_joint_pos] = vae_joint_conf[i]
            ix_full_joint_pos += 1

    return full_joint_pos

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
    dist_mat = np.zeros((joint1.shape[0],joint2.shape[0]))
    for idx in range(joint1.shape[0]):
        deltas = joint2 - joint1[idx]
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        dist_mat[idx] = dist_2
    return dist_mat

def maad_for_grasp_distribution(grasp1, grasp2):
    """_summary_

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

    # Adapt format of joint conf from 15 dim to 20 dim and numpy array
    grasp2_joint_conf = grasp2['joint_conf']
    # grasp2_joint_conf = np.zeros((len(grasp2['joint_conf']),20))
    # for idx in range(len(grasp2['joint_conf'])):
    #     grasp2_joint_conf[idx] = grasp2['joint_conf'][idx]
    # pred_joint_conf_full = np.zeros((grasp1['pred_joint_conf'].shape[0], 20))
    # for idx in range(grasp1['pred_joint_conf'].shape[0]):
    #     pred_joint_conf_full[idx] = full_joint_conf_from_vae_joint_conf(grasp1['pred_joint_conf'][idx])
    # grasp1['pred_joint_conf'] = pred_joint_conf_full

    joint_dist_mat = euclidean_distance_joint_conf_pairwise_np(grasp1['joint_conf'], grasp2_joint_conf)

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

def filter(ffhgan, obj_pcd_path, obj_bps, grasps, n_samples, is_discriminator = False, thresh_succ_list = [0.5, 0.75, 0.90], visualize = False):
    if visualize:
        visualization.show_generated_grasp_distribution(obj_pcd_path, grasps)

    if is_discriminator:
        filter_func = ffhgan.filter_grasps_discriminator
    else:
        filter_func = ffhgan.filter_grasps

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

    return filtered_grasps_2 , n_grasps_filt_2

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
    is_gan = True,
    show_individual_grasps=False,
    is_discriminator = False,
    is_filter = True
    ):

    config = Config(config_path)
    cfg = config.parse()

    if is_gan:
        model = FFHGANet(cfg)
    else:
        model = FFHNet(cfg)

    if is_discriminator and is_gan:

        thresh_succ_list=[0.15, 0.175, 0.20]
    else:
        thresh_succ_list=[0.5, 0.75, 0.90]

    print(model)
    
    base_data_bath = os.path.join(ROOT_PATH,'data','real_objects')
    model.load_ffhgenerator(epoch=load_epoch_gen, load_path=load_path_gen)
    
    model.load_ffhevaluator(epoch=load_epoch_eva, load_path=load_path_eva)
    dset_gen = FFHGeneratorDataSet(cfg, eval=True)
    train_loader_gen = DataLoader(dset_gen,
                                    batch_size=64,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=cfg["num_threads"])
    grasp_data = dset_gen.grasp_data_handler

    if not os.path.isfile('eval_batch.pth'):
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
    batch = load_batch('eval_batch.pth')
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
                return_arr=True
                )
            out , n_grasps_filt_2 = filter(
                                        model, 
                                        pcd_path, 
                                        batch['bps_object'][idx].cpu().data.numpy(), out, 
                                        grasps_gt['joint_conf'].shape[0], 
                                        is_discriminator = is_discriminator and is_gan, 
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
            
        transl_loss, rot_loss, joint_loss, coverage = maad_for_grasp_distribution(out, grasps_gt)
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
    if True:
        torch.multiprocessing.set_start_method('spawn')
        parser = argparse.ArgumentParser()

        # # Best VAE so far:
        # gen_path = "checkpoints/ffhnet/ffhgenerator_bs5012"
        # best_epoch = 30
        # gen_path = "checkpoints/ffhnet/2024-06-07T14_35_31_ffhnet_lr_0.0001_bs_1000"
        # best_epoch = 19

        # # Best GAN so far:
        # gen_path = "checkpoints/ffhgan/2024-03-10T17_31_55_ffhgan_lr_0.0001_bs_1000"
        # best_epoch = 45

        # Experiment checkpoint:

        # gen_path = "checkpoints/ffhgan/2024-03-10T17_31_55_ffhgan_lr_0.0001_bs_1000"
        # best_epoch = 32
        gen_path = "checkpoints/ffhgan/2024-03-15T15_20_19_ffhgan_lr_0.0001_bs_1000"
        best_epoch = 63
        
        is_discriminator = False
        is_filter = False

        parser.add_argument('--gen_path', default=gen_path, help='path to FFHGenerator model')
        parser.add_argument('--load_gen_epoch', type=int, default=best_epoch, help='epoch of FFHGenerator model')

        # parser.add_argument('--eva_path', default='models/ffhevaluator', help='path to FFHEvaluator model')
        parser.add_argument('--eva_path', default='checkpoints/ffhevaluator/2024-06-23_ffhevaluator', help='path to FFHEvaluator model')
        parser.add_argument('--load_eva_epoch', type=int, default=30, help='epoch of FFHEvaluator model')
        parser.add_argument('--config', type=str, default='FFHNet/config/config_ffhgan.yaml')

        args = parser.parse_args()

        load_path_gen = args.gen_path
        load_path_eva = args.eva_path
        load_epoch_gen = args.load_gen_epoch
        load_epoch_eva = args.load_eva_epoch
        config_path = args.config
        
        if 'ffhgan' in gen_path:
            is_gan = True
        else:
            is_gan = False

        main(
        config_path,    
        load_epoch_eva,
        load_epoch_gen,
        load_path_eva,
        load_path_gen,
        is_gan = is_gan, 
        show_individual_grasps = True, 
        is_discriminator=is_discriminator,
        is_filter=is_filter)
        
        # with open(load_path_gen + '_metrics.csv', 'w') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(["epoch", "transl_loss_sum", "rot_loss_sum", "joint_loss_sum", "coverage_mean"])
        #     for epoch in range(1,53,1):
        #         print('Evaluating epoch:',epoch)
        #         load_epoch_gen = epoch
        #         transl_loss_sum, rot_loss_sum, joint_loss_sum, coverage_mean = main(config_path, load_epoch_gen, load_path_gen, is_gan = is_gan, show_individual_grasps = False)
        #         writer.writerow([epoch, transl_loss_sum, rot_loss_sum, joint_loss_sum, coverage_mean])