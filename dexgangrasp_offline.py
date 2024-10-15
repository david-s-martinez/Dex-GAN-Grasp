from __future__ import division

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

from DexGanGrasp.config.config import Config
from DexGanGrasp.data.bps_encoder import BPSEncoder
from DexGanGrasp.data.dexevaluator_data_set import (DexEvaluatorDataSet,
                                               DexEvaluatorPCDDataSet)
from DexGanGrasp.data.dexgenerator_data_set import DexGeneratorDataSet
from DexGanGrasp.models.dexgangrasp import DexGanGrasp
from DexGanGrasp.models.networks import DexGANGrasp
from DexGanGrasp.utils import utils, visualization
from DexGanGrasp.utils.writer import Writer
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.split(os.path.split(ROOT_PATH)[0])[0]
parser = argparse.ArgumentParser(
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', help='Path to template image.',
                    default='DexGanGrasp/config/config_ffhnet_vm_debug.yaml')
args = parser.parse_args()

def filter(dexgangrasp, obj_pcd_path, obj_bps, grasps, n_samples, is_discriminator=False, thresh_succ_list=[0.5, 0.75, 0.90], visualize=True):
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
    
def eval_dexgangrasp_sampling_and_filtering_real(config_path,
                                            load_epoch_eva,
                                            load_epoch_gen,
                                            load_path_eva,
                                            load_path_gen,
                                            n_samples=1000,
                                            thresh_succ=0.5,
                                            show_individual_grasps=False,
                                            thresh_succ_list = [0.5, 0.75, 0.90],
                                            is_discriminator = False):
    """
    Evaluates grasp sampling and filtering on real object data using the DexGANGrasp model.

    Process:
        - Loads the configuration, generator, and evaluator models.
        - Iterates over real object files, loading their BPS (Basis Point Set) representations and PCD (Point Cloud Data) files.
        - Generates grasp samples for each object and visualizes the initial distribution.
        - Applies filtering based on success thresholds and visualizes the filtered grasp distribution.
        - Optionally visualizes individual filtered grasps and allows user interaction for controlling visualization.

    Args:
        config_path (str): Path to the configuration file.
        load_epoch_eva (int): Epoch number to load the evaluator model.
        load_epoch_gen (int): Epoch number to load the generator model.
        load_path_eva (str): Path to load the evaluator model weights.
        load_path_gen (str): Path to load the generator model weights.
        n_samples (int): Number of grasp samples to generate. Default is 1000.
        thresh_succ (float): Success threshold for filtering grasps. Default is 0.5.
        show_individual_grasps (bool): If True, displays each individual grasp. Default is False.
        thresh_succ_list (list): List of success thresholds for multi-stage filtering. Default is [0.5, 0.75, 0.90].
        is_discriminator (bool): If True, uses discriminator-based filtering. Default is False.

    Returns:
        None
    """

    config = Config(config_path)
    cfg = config.parse()
    dexgangrasp = DexGanGrasp(cfg)
    print(dexgangrasp)
    base_data_bath = os.path.join(ROOT_PATH,'data','real_objects')
    dexgangrasp.load_dexgenerator(epoch=load_epoch_gen, load_path=load_path_gen)
    dexgangrasp.load_dexevaluator(epoch=load_epoch_eva, load_path=load_path_eva)
    path_real_objs_bps = os.path.join(base_data_bath, 'bps')
    for f_name in os.listdir(path_real_objs_bps):
        print(f_name)
        if input('Skip object? Press y: ') == 'y':
            continue
        # Paths to object and bps
        obj_bps_path = os.path.join(path_real_objs_bps, f_name)
        f_name_pcd = f_name.replace('.npy', '.pcd')
        obj_pcd_path = os.path.join(base_data_bath, 'object', f_name_pcd)

        obj_bps = np.load(obj_bps_path)
        grasps = dexgangrasp.generate_grasps(obj_bps, n_samples=n_samples, return_arr=True)

        # Visualize sampled distribution
        visualization.show_generated_grasp_distribution(obj_pcd_path, [])
        visualization.show_generated_grasp_distribution(obj_pcd_path, grasps)

        # Filter
        filtered_grasps_2 , n_grasps_filt_2 = filter(
                                                dexgangrasp, 
                                                obj_pcd_path, 
                                                obj_bps, grasps, 
                                                n_samples, 
                                                is_discriminator = is_discriminator, 
                                                thresh_succ_list = thresh_succ_list
                                                )

        print("n_grasps after filtering: %d" % n_grasps_filt_2)
        print("This means %.2f of grasps pass the filtering" % (n_grasps_filt_2 / n_samples))

        # Visulize filtered distribution
        visualization.show_generated_grasp_distribution(obj_pcd_path, filtered_grasps_2)

        if show_individual_grasps:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    gen_path = "checkpoints/ffhgan/2024-03-10T17_31_55_ffhgan_lr_0.0001_bs_1000"
    best_epoch = 32
    is_discriminator = False
    thresh_succ_list = [0.5, 0.75, 0.90]

    parser.add_argument('--gen_path', default=gen_path, help='path to DexGenerator model')
    parser.add_argument('--load_gen_epoch', type=int, default=best_epoch, help='epoch of DexGenerator model')
    # New evaluator:checkpoints/ffhevaluator/2024-06-23_ffhevaluator
    parser.add_argument('--eva_path', default='checkpoints/ffhevaluator/2024-06-23_ffhevaluator', help='path to DexEvaluator model')
    parser.add_argument('--load_eva_epoch', type=int, default=30, help='epoch of DexEvaluator model')
    parser.add_argument('--config', type=str, default='DexGanGrasp/config/config_dexgangrasp.yaml')

    args = parser.parse_args()

    load_path_gen = args.gen_path
    load_path_eva = args.eva_path
    load_epoch_gen = args.load_gen_epoch
    load_epoch_eva = args.load_eva_epoch
    config_path = args.config

    eval_dexgangrasp_sampling_and_filtering_real(config_path, 
                                            load_epoch_eva, 
                                            load_epoch_gen, 
                                            load_path_eva,
                                            load_path_gen, 
                                            show_individual_grasps= True, 
                                            thresh_succ_list = thresh_succ_list, 
                                            is_discriminator = is_discriminator
                                            )