from __future__ import division

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
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
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.split(os.path.split(ROOT_PATH)[0])[0]
parser = argparse.ArgumentParser(
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', help='Path to template image.',
                    default='FFHNet/config/config_ffhnet_vm_debug.yaml')
args = parser.parse_args()

def filter(ffhgan, obj_pcd_path, obj_bps, grasps, n_samples, is_discriminator = False, thresh_succ_list = [0.5, 0.75, 0.90] ):
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
    visualization.show_generated_grasp_distribution(obj_pcd_path, filtered_grasps)

    ############### Stage 2 ################
    # Reject grasps with low probability
    filtered_grasps_1 = filter_func(obj_bps, grasps, thresh=thresh_succ_list[1])
    n_grasps_filt_1 = filtered_grasps_1['rot_matrix'].shape[0]

    print("n_grasps after filtering: %d" % n_grasps_filt_1)
    print("This means %.2f of grasps pass the filtering" % (n_grasps_filt_1 / n_samples))

    # Visulize filtered distribution
    visualization.show_generated_grasp_distribution(obj_pcd_path, filtered_grasps_1)

    ############## Stage 3 ################
    # Reject grasps with low probability
    filtered_grasps_2 = filter_func(obj_bps, grasps, thresh=thresh_succ_list[2])
    n_grasps_filt_2 = filtered_grasps_2['rot_matrix'].shape[0]

    print("n_grasps after filtering: %d" % n_grasps_filt_2)
    print("This means %.2f of grasps pass the filtering" % (n_grasps_filt_2 / n_samples))

    return filtered_grasps_2 , n_grasps_filt_2
    
def eval_ffhnet_sampling_and_filtering_real(config_path,
                                            load_epoch_eva,
                                            load_epoch_gen,
                                            load_path_eva,
                                            load_path_gen,
                                            n_samples=1000,
                                            thresh_succ=0.5,
                                            show_individual_grasps=False,
                                            thresh_succ_list = [0.5, 0.75, 0.90],
                                            is_discriminator = False):
    config = Config(config_path)
    cfg = config.parse()
    ffhgan = FFHGANet(cfg)
    print(ffhgan)
    base_data_bath = os.path.join(ROOT_PATH,'data','real_objects')
    ffhgan.load_ffhgenerator(epoch=load_epoch_gen, load_path=load_path_gen)
    ffhgan.load_ffhevaluator(epoch=load_epoch_eva, load_path=load_path_eva)
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
        grasps = ffhgan.generate_grasps(obj_bps, n_samples=n_samples, return_arr=True)

        # Visualize sampled distribution
        visualization.show_generated_grasp_distribution(obj_pcd_path, [])
        visualization.show_generated_grasp_distribution(obj_pcd_path, grasps)

        # Filter
        filtered_grasps_2 , n_grasps_filt_2 = filter(
                                                ffhgan, 
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
def train():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help='Path to template image.',
                        default='FFHNet/config/config_ffhgan.yaml')
    args = parser.parse_args()

    # load configuration params
    config = Config(args.config)
    cfg = config.parse()

    # start cuda multiprocessing
    # TODO: discover the problem of cpu usage
    torch.multiprocessing.set_start_method('spawn')

    # Data for gen and eval and col is different. Gen sees only positive examples
    if cfg["train_ffhevaluator"]:
        if cfg["model"] == "ffhnet":
            dset_eva = FFHEvaluatorDataSet(cfg,eval=False)
        elif cfg["model"] == "pointnet":
            dset_eva = FFHEvaluatorPCDDataSet(cfg,eval=False)
        train_loader_eva = DataLoader(dset_eva,
                                      batch_size=cfg["batch_size"],
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=cfg["num_threads"])

    if cfg["train_ffhgenerator"]:
        dset_gen = FFHGeneratorDataSet(cfg)
        train_loader_gen = DataLoader(dset_gen,
                                      batch_size=cfg["batch_size"],
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=cfg["num_threads"])

    writer = Writer(cfg)

    ffhgan = FFHGANet(cfg)
    # if cfg["continue_train"]:
    #     if cfg["train_ffhevaluator"]:
    #         ffhgan.load_ffhevaluator(cfg["load_epoch"])
    #     if cfg["train_ffhgenerator"]:
    #         ffhgan.load_ffhgenerator(cfg["load_epoch"])
    #     start_epoch = cfg["load_epoch"] + 1
    # else:
    #     start_epoch = 1
    start_epoch = 1
    total_steps = 0
    epoch_start = time.time()

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        # === Generator ===
        
        # Initialize epoch / iter info
        prev_iter_end = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_loader_gen):
            # Update iter and total info
            cur_iter_start = time.time()
            total_steps += cfg["batch_size"]
            epoch_iter += cfg["batch_size"]

            # Measure time for data loading
            if total_steps % cfg["print_freq"] == 0:
                t_load_data = cur_iter_start - prev_iter_end

            # Update model one step, get losses
            loss_dict = ffhgan.update_ffhgan(data)

            # Log loss
            if total_steps % cfg["print_freq"] == 0:
                t_load = cur_iter_start - prev_iter_end  # time for data loading
                t_total = (time.time() - cur_iter_start) // 60
                # TODO: implement writer.
                # writer.print_current_train_loss(epoch, epoch_iter, loss_dict, t_total, t_load)
                # writer.plot_train_loss(loss_dict, epoch, epoch_iter, len(dset_gen))

            prev_iter_end = time.time()
            # End of data loading generator
if __name__ == '__main__':
    if True:
        parser = argparse.ArgumentParser()
        # # Best VAE so far:
        # gen_path = "checkpoints/ffhnet/2023-09-01T01_16_11_ffhnet_lr_0.0001_bs_1000"
        # best_epoch = 24

        # Best GAN so far:
        gen_path = "checkpoints/ffhgan/2024-03-15T15_20_19_ffhgan_lr_0.0001_bs_1000"
        best_epoch = 63
        # gen_path = "checkpoints/ffhgan/2024-03-10T17_31_55_ffhgan_lr_0.0001_bs_1000"
        # best_epoch = 32

        # is_discriminator = True
        # thresh_succ_list = [0.15, 0.20, 0.25]

        is_discriminator = False
        thresh_succ_list = [0.5, 0.75, 0.90]

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

        eval_ffhnet_sampling_and_filtering_real(config_path, 
                                                load_epoch_eva, 
                                                load_epoch_gen, 
                                                load_path_eva,
                                                load_path_gen, 
                                                show_individual_grasps= True, 
                                                thresh_succ_list = thresh_succ_list, 
                                                is_discriminator = is_discriminator
                                                )
    # train()