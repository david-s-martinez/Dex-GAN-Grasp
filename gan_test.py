from __future__ import division

import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

from FFHNet.config.config import Config
from FFHNet.data.bps_encoder import BPSEncoder
from FFHNet.data.ffhcollision_data_set import FFHCollDetrDataSet
from FFHNet.data.ffhevaluator_data_set import (FFHEvaluatorDataSet,
                                               FFHEvaluatorPCDDataSet)
from FFHNet.data.ffhgenerator_data_set import FFHGeneratorDataSet
from FFHNet.models.ffhnet import FFHNet
from FFHNet.models.networks import FFHGAN
from FFHNet.utils import utils, visualization, writer

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.split(os.path.split(ROOT_PATH)[0])[0]
parser = argparse.ArgumentParser(
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', help='Path to template image.',
                    default='FFHNet/config/config_ffhnet_vm_debug.yaml')
args = parser.parse_args()

def eval_ffhnet_sampling_and_filtering_real(config_path,
                                            load_epoch_eva,
                                            load_epoch_gen,
                                            load_path_eva,
                                            load_path_gen,
                                            n_samples=1000,
                                            thresh_succ=0.5,
                                            show_individual_grasps=False):
    config = Config(config_path)
    cfg = config.parse()
    ffhgan = FFHGAN(cfg)
    print(ffhgan)
    base_data_bath = os.path.join(ROOT_PATH,'data','real_objects')
    # ffhgan.load_ffhgenerator(epoch=load_epoch_gen, load_path=load_path_gen)
    # ffhgan.load_ffhevaluator(epoch=load_epoch_eva, load_path=load_path_eva)
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
        visualization.show_generated_grasp_distribution(obj_pcd_path, grasps)

        # ############### Stage 3 ################
        # # Reject grasps with low probability
        # filtered_grasps_2 = ffhgan.filter_grasps(obj_bps, grasps, thresh=0.90)
        # n_grasps_filt_2 = filtered_grasps_2['rot_matrix'].shape[0]

        # print("n_grasps after filtering: %d" % n_grasps_filt_2)
        # print("This means %.2f of grasps pass the filtering" % (n_grasps_filt_2 / n_samples))

        # # Visulize filtered distribution
        # visualization.show_generated_grasp_distribution(obj_pcd_path, filtered_grasps_2)

        # if show_individual_grasps:
        #     for j in range(n_grasps_filt_2):
        #         # Get the grasp sample
        #         rot_matrix = filtered_grasps_2['rot_matrix'][j, :, :]
        #         transl = filtered_grasps_2['transl'][j, :]
        #         # if transl[1] > -0.1:
        #         #     continue
        #         joint_conf = filtered_grasps_2['joint_conf'][j, :]
        #         palm_pose_centr = utils.hom_matrix_from_transl_rot_matrix(transl, rot_matrix)
        #         visualization.show_grasp_and_object(obj_pcd_path, palm_pose_centr, joint_conf)
        #         a = input('Break loop? (y/n): ')
        #         if a == 'y':
        #             break
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path', default='models/ffhgenerator', help='path to FFHGenerator model')
    parser.add_argument('--load_gen_epoch', type=int, default=10, help='epoch of FFHGenerator model')
    parser.add_argument('--eva_path', default='models/ffhevaluator', help='path to FFHEvaluator model')
    parser.add_argument('--load_eva_epoch', type=int, default=30, help='epoch of FFHEvaluator model')
    parser.add_argument('--config', type=str, default='FFHNet/config/config_ffhnet_yb.yaml')

    args = parser.parse_args()

    load_path_gen = args.gen_path
    load_path_eva = args.eva_path
    load_epoch_gen = args.load_gen_epoch
    load_epoch_eva = args.load_eva_epoch
    config_path = args.config

    eval_ffhnet_sampling_and_filtering_real(config_path, load_epoch_eva, load_epoch_gen, load_path_eva,
                                            load_path_gen, show_individual_grasps=True)