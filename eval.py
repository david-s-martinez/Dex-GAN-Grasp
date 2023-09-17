from __future__ import division

import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

from FFHNet.config.config import Config
from FFHNet.data.bps_encoder import BPSEncoder
from FFHNet.data.ffhevaluator_data_set import (FFHEvaluatorDataSet,
                                               FFHEvaluatorPCDDataSet)
from FFHNet.data.ffhgenerator_data_set import FFHGeneratorDataSet
from FFHNet.models.ffhnet import FFHNet
from FFHNet.models.ffhgan import FFHGANet
from FFHNet.utils import utils, visualization, writer

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.split(os.path.split(ROOT_PATH)[0])[0]
parser = argparse.ArgumentParser(
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', help='Path to template image.',
                    default='FFHNet/config/config_ffhnet_vm_debug.yaml')
args = parser.parse_args()


def update_mean_losses_gen(mean_losses, new_losses):
    mean_losses['total_loss_gen'] += new_losses['total_loss_gen'].detach().cpu().numpy()
    mean_losses['kl_loss'] += new_losses['kl_loss'].detach().cpu().numpy()
    mean_losses['transl_loss'] += new_losses['transl_loss'].detach().cpu().numpy()
    mean_losses['rot_loss'] += new_losses['rot_loss'].detach().cpu().numpy()
    mean_losses['conf_loss'] += new_losses['conf_loss'].detach().cpu().numpy()
    return mean_losses

def update_mean_losses(mean_losses, new_losses):
    for key in mean_losses.keys():
        mean_losses[key] += new_losses[key].detach().cpu().numpy()
    return mean_losses

def run_eval_gan(cfg, curr_epoch, ffhgan=None, epoch=-1, name=""):
    """Performs model evaluation on the eval set. Evaluates either only one or both the FFHGenerator, FFHEvaluator
    depending on the config settings.

    Args:
        eval_dir (str):
        curr_epoch (int):
        ffhgan (FFHNet, optional): The full FFHNet model. Defaults to None.
        epoch (int, optional): Epoch from which to load a model. Defaults to -1.
        name (str, optional): Name of the model to be loaded. Defaults to "".

    Returns:
        loss_dict (dict): A dict with the losses for FFHEvaluator and/or FFHGenerator, depending on cfg["train"]_* is set.
    """
    print('Running eval.')

    cfg["name"] = name

    loss_dict = {}

    if cfg["eval_ffhevaluator"]:
        if cfg["model"] == 'ffhnet':
            dset = FFHEvaluatorDataSet(cfg,eval=True)
        eval_loader = DataLoader(dset, batch_size=cfg["batch_size"], shuffle=False)
        eval_loss_dict_eva = run_eval_eva(ffhgan, eval_loader, curr_epoch, cfg["eval_dir"])
        loss_dict.update(eval_loss_dict_eva)

    if cfg["eval_ffhgenerator"]:
        eval_loss_dict_gen = run_eval_gan_gen(ffhgan, cfg)
        loss_dict.update(eval_loss_dict_gen)

    return loss_dict

def run_eval(cfg, curr_epoch, ffhnet=None, epoch=-1, name=""):
    """Performs model evaluation on the eval set. Evaluates either only one or both the FFHGenerator, FFHEvaluator
    depending on the config settings.

    Args:
        eval_dir (str):
        curr_epoch (int):
        ffhnet (FFHNet, optional): The full FFHNet model. Defaults to None.
        epoch (int, optional): Epoch from which to load a model. Defaults to -1.
        name (str, optional): Name of the model to be loaded. Defaults to "".

    Returns:
        loss_dict (dict): A dict with the losses for FFHEvaluator and/or FFHGenerator, depending on cfg["train"]_* is set.
    """
    print('Running eval.')

    cfg["name"] = name

    loss_dict = {}

    if ffhnet is None:
        ffhnet = FFHNet(cfg)
        ffhnet.load_ffhnet(cfg["load_epoch"])

    if cfg["eval_ffhevaluator"]:
        if cfg["model"] == 'ffhnet':
            dset = FFHEvaluatorDataSet(cfg,eval=True)
        elif cfg["model"] == 'pointnet':
            dset = FFHEvaluatorPCDDataSet(cfg)
        eval_loader = DataLoader(dset, batch_size=cfg["batch_size"], shuffle=False)
        eval_loss_dict_eva = run_eval_eva(ffhnet, eval_loader, curr_epoch, cfg["eval_dir"])
        loss_dict.update(eval_loss_dict_eva)

    if cfg["eval_ffhgenerator"]:
        # TODO add two cases
        eval_loss_dict_gen = run_eval_gen(ffhnet, cfg)
        loss_dict.update(eval_loss_dict_gen)

    return loss_dict


def run_eval_eva(ffhnet, dataloader, curr_epoch, eval_dir):
    print('Running eval for FFHEvaluator.')

    mean_losses = {
        'total_loss_eva': 0,
        'pos_acc': 0,
        'neg_acc': 0,
    }

    pred_labels = np.array([])
    gt_labels = np.array([])

    for i, data in enumerate(dataloader):
        loss_dict = ffhnet.eval_ffhevaluator_loss(data)
        pos_acc, neg_acc, pred_label, gt_label = ffhnet.eval_ffhevaluator_accuracy(data)

        mean_losses['total_loss_eva'] += loss_dict['total_loss_eva'].detach().cpu().numpy()
        mean_losses['pos_acc'] += pos_acc
        mean_losses['neg_acc'] += neg_acc

        pred_labels = np.append(pred_labels, pred_label)
        gt_labels = np.append(gt_labels, gt_label)

    for k, _ in mean_losses.items():
        mean_losses[k] /= (i + 1)

    # save the labels
    np.save(os.path.join(eval_dir, str(curr_epoch) + '_gt_labels.npy'), gt_labels)
    np.save(os.path.join(eval_dir, str(curr_epoch) + '_pred_labels.npy'), pred_labels)

    return mean_losses

def run_eval_gen(ffhnet, cfg):
    print('Running eval for FFHGenerator')
    dset = FFHGeneratorDataSet(cfg, eval = True)
    eval_loader = DataLoader(dset, batch_size=cfg["batch_size"], shuffle=False)

    mean_losses = {
        'total_loss_gen': 0,
        'kl_loss': 0,
        'transl_loss': 0,
        'rot_loss': 0,
        'conf_loss': 0,
    }

    for i, data in enumerate(eval_loader):
        loss_dict = ffhnet.eval_ffhgenerator_loss(data)
        mean_losses = update_mean_losses_gen(mean_losses, loss_dict)

    for k, _ in mean_losses.items():
        mean_losses[k] /= (i + 1)

    return mean_losses

def run_eval_gan_gen(ffhgan, cfg):
    print('Running eval for FFHGAN Generator')
    dset = FFHGeneratorDataSet(cfg, eval = True)
    eval_loader = DataLoader(dset, batch_size=cfg["batch_size"], shuffle=False)

    mean_losses = {
        'gen_loss_fake' : 0,
        'transl_loss': 0,
        'rot_loss' : 0,
        'conf_loss' : 0,
        'total_loss_gen' : 0
        }

    for i, data in enumerate(eval_loader):
        loss_dict = ffhgan.eval_ffhgan_generator_loss(data)
        if i % 100 == 0:
            print(i,'- Eval Loss:', loss_dict)
        mean_losses = update_mean_losses(mean_losses, loss_dict)

    for k, _ in mean_losses.items():
        mean_losses[k] /= (i + 1)

    return mean_losses


def eval_eva_accuracy(load_epoch_eva, load_path, show_confusion_matrix=False):
    config = Config()
    cfg = config.parse()
    ffhnet = FFHNet(cfg)
    ffhnet.load_ffhevaluator(load_path=load_path, epoch=load_epoch_eva, is_train=False)
    dset = FFHEvaluatorDataSet(cfg,eval=True)
    eval_loader = DataLoader(dset, batch_size=cfg["batch_size"], shuffle=False)

    mean_accuracy = {
        'pos_acc': 0,
        'neg_acc': 0,
    }

    pred_labels = np.array([])
    gt_labels = np.array([])

    for i, data in enumerate(eval_loader):
        #print(" %.2f percent  completed." % (100 * i * cfg["batch_size"] / len(dset)))
        pos_acc, neg_acc, pred_label, gt_label = ffhnet.eval_ffhevaluator_accuracy(data)

        mean_accuracy['pos_acc'] += pos_acc
        mean_accuracy['neg_acc'] += neg_acc

        pred_labels = np.append(pred_labels, pred_label)
        gt_labels = np.append(gt_labels, gt_label)

    print("\nMean accuracy in epoch: " + str(load_epoch_eva))
    for k, _ in mean_accuracy.items():
        mean_accuracy[k] /= (i + 1)
        print(k, mean_accuracy[k])

    if show_confusion_matrix:
        visualization.plot_confusion_matrix(1 - gt_labels,
                                            1 - pred_labels,
                                            classes=['success', 'failure'],
                                            normalize=True)

    return mean_accuracy, gt_labels, pred_labels

def eval_grasp_refinement(load_epoch_eva, method='gradient'):
    config = Config()
    cfg = config.parse()
    ffhnet = FFHNet(cfg)
    ffhnet.load_ffhevaluator(epoch=load_epoch_eva, is_train=False)
    dset = FFHEvaluatorDataSet(cfg, eval=True)
    eval_loader = DataLoader(dset, batch_size=cfg["batch_size"], shuffle=True)

    for i, data in enumerate(eval_loader):
        pcd_paths = data['pcd_path']
        refined_data, refined_success = ffhnet.refine_grasps(data, method)
        if cfg["vis_grasp_refinement"]:
            visualization.show_grasp_refinement(refined_data, refined_success, pcd_paths)


def eval_ffhnet_sampling_and_filtering(load_epoch_eva,
                                       load_epoch_gen,
                                       load_path_eva,
                                       load_path_gen,
                                       gazebo_obj_path,
                                       n_samples=1000,
                                       thresh_succ=0.5):
    config = Config()
    cfg = config.parse()
    ffhnet = FFHNet(cfg)
    ffhnet.load_ffhgenerator(epoch=load_epoch_gen, load_path=load_path_gen)
    ffhnet.load_ffhevaluator(epoch=load_epoch_eva, load_path=load_path_eva)
    ffhnet.FFHGenerator.eval()
    ffhnet.FFHEvaluator.eval()
    dset = FFHGeneratorDataSet(cfg)
    eval_loader = DataLoader(dset, batch_size=1, shuffle=True)

    total_sampled = 0
    total_after_filt = 0

    for i, data in enumerate(eval_loader):
        if data['obj_name'][0] != 'kit_HamburgerSauce':
            continue
        grasps = ffhnet.generate_grasps(data['bps_object'], n_samples=n_samples, return_arr=True)

        # Visualize sampled distribution
        visualization.show_generated_grasp_distribution(data['pcd_path'][0], grasps)

        # Reject grasps with low probability
        filtered_grasps = ffhnet.filter_grasps(data['bps_object'], grasps, thresh=thresh_succ)
        n_grasps_filt = filtered_grasps['rot_matrix'].shape[0]

        print("n_grasps after filtering: %d" % n_grasps_filt)
        print("This means %.2f of grasps pass the filtering" % (n_grasps_filt / n_samples))

        # Visulize filtered distribution
        visualization.show_generated_grasp_distribution(data['pcd_path'][0], filtered_grasps)

        ######### STAGE 2 0.75 ##############
        # Reject grasps with low probability
        filtered_grasps = ffhnet.filter_grasps(data['bps_object'], grasps, thresh=0.75)
        n_grasps_filt = filtered_grasps['rot_matrix'].shape[0]

        print("n_grasps after filtering: %d" % n_grasps_filt)
        print("This means %.2f of grasps pass the filtering" % (n_grasps_filt / n_samples))

        # Visulize filtered distribution
        visualization.show_generated_grasp_distribution(data['pcd_path'][0], filtered_grasps)

        ########### STAGE 3 0.9 #############
        # Reject grasps with low probability
        filtered_grasps = ffhnet.filter_grasps(data['bps_object'], grasps, thresh=0.9)
        n_grasps_filt = filtered_grasps['rot_matrix'].shape[0]

        print("n_grasps after filtering: %d" % n_grasps_filt)
        print("This means %.2f of grasps pass the filtering" % (n_grasps_filt / n_samples))

        # Visulize filtered distribution
        visualization.show_generated_grasp_distribution(data['pcd_path'][0], filtered_grasps)

        # Visualize ground truth object distribution
        visualization.show_ground_truth_grasp_distribution(data['obj_name'][0],
                                                           dset.grasp_data_path, gazebo_obj_path)
        total_sampled += n_samples
        total_after_filt += n_grasps_filt
        if i % 100 == 0:
            print("Total sampled grasps: %d" % total_sampled)
            print("Total grasps after filtering: %d" % total_after_filt)
            print("Average fraction: %.2f" % (total_after_filt / total_sampled))


def eval_ffheva_num_removed_grasps(config_path,
                                   load_epoch_eva,
                                   load_epoch_gen,
                                   load_path_eva,
                                   load_path_gen,
                                   n_samples=30000):
    config = Config(config_path)
    cfg = config.parse()
    ffhnet = FFHNet(cfg)
    ffhnet.load_ffhgenerator(epoch=load_epoch_gen, load_path=load_path_gen)
    ffhnet.load_ffhevaluator(epoch=load_epoch_eva, load_path=load_path_eva)
    dset = FFHGeneratorDataSet(cfg)
    eval_loader = DataLoader(dset, batch_size=1, shuffle=True)

    threshs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1]
    total_sampled = 0
    total_after_filt = np.zeros(len(threshs))

    for i, data in enumerate(eval_loader):
        grasps = ffhnet.generate_grasps(data['bps_object'], n_samples=n_samples, return_arr=True)

        # Evaluate grasps once
        p_success = ffhnet.evaluate_grasps(data['bps_object'], grasps)

        for i, thresh in enumerate(threshs):
            total_after_filt[i] += len(p_success[p_success > thresh])

        total_sampled += n_samples

        print("Total sampled grasps: %d" % total_sampled)
        print("Average fraction:")
        print(np.around(total_after_filt / total_sampled, 2))

        if i == 25:
            break


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
    ffhnet = FFHNet(cfg)
    base_data_bath = os.path.join(ROOT_PATH,'data','real_objects')
    ffhnet.load_ffhgenerator(epoch=load_epoch_gen, load_path=load_path_gen)
    ffhnet.load_ffhevaluator(epoch=load_epoch_eva, load_path=load_path_eva)
    path_real_objs_bps = os.path.join(base_data_bath, 'bps')
    for f_name in os.listdir(path_real_objs_bps):
        print(f_name)
        print("In train mode?: %d" % ffhnet.FFHEvaluator.training)
        print("In train mode?: %d" % ffhnet.FFHGenerator.training)
        if input('Skip object? Press y: ') == 'y':
            continue
        # Paths to object and bps
        obj_bps_path = os.path.join(path_real_objs_bps, f_name)
        f_name_pcd = f_name.replace('.npy', '.pcd')
        obj_pcd_path = os.path.join(base_data_bath, 'object', f_name_pcd)

        obj_bps = np.load(obj_bps_path)
        grasps = ffhnet.generate_grasps(obj_bps, n_samples=n_samples, return_arr=True)

        # Visualize sampled distribution
        visualization.show_generated_grasp_distribution(obj_pcd_path, grasps)

        ############### Stage 1 ################
        # Reject grasps with low probability
        filtered_grasps = ffhnet.filter_grasps(obj_bps, grasps, thresh=thresh_succ)
        n_grasps_filt = filtered_grasps['rot_matrix'].shape[0]

        print("n_grasps after filtering: %d" % n_grasps_filt)
        print("This means %.2f of grasps pass the filtering" % (n_grasps_filt / n_samples))

        # Visulize filtered distribution
        visualization.show_generated_grasp_distribution(obj_pcd_path, filtered_grasps)

        ############### Stage 2 ################
        # Reject grasps with low probability
        filtered_grasps_1 = ffhnet.filter_grasps(obj_bps, grasps, thresh=0.75)
        n_grasps_filt_1 = filtered_grasps_1['rot_matrix'].shape[0]

        print("n_grasps after filtering: %d" % n_grasps_filt_1)
        print("This means %.2f of grasps pass the filtering" % (n_grasps_filt_1 / n_samples))

        # Visulize filtered distribution
        visualization.show_generated_grasp_distribution(obj_pcd_path, filtered_grasps_1)

        ############### Stage 3 ################
        # Reject grasps with low probability
        filtered_grasps_2 = ffhnet.filter_grasps(obj_bps, grasps, thresh=0.90)
        n_grasps_filt_2 = filtered_grasps_2['rot_matrix'].shape[0]

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


def eval_ffhgenerator_qualitatively(load_epoch,
                                    load_path,
                                    gazebo_obj_path,
                                    n_samples=1000,
                                    show_individual_grasps=True):
    config = Config()
    cfg = config.parse()
    ffhnet = FFHNet(cfg)
    ffhnet.load_ffhgenerator(epoch=load_epoch, load_path=load_path)
    dset = FFHGeneratorDataSet(cfg)
    eval_loader = DataLoader(dset, batch_size=1, shuffle=False)

    for i, data in enumerate(eval_loader):
        print("Object:", data['obj_name'])
        #l = raw_input('Skip?: ')
        l = 'n'
        if l == 'y' or l == 'Y':
            continue
        grasps = ffhnet.generate_grasps(data['bps_object'], n_samples=n_samples, return_arr=True)

        # Visualize sampled distribution
        visualization.show_generated_grasp_distribution(data['pcd_path'][0], grasps)

        # Visualize ground truth object distribution
        visualization.show_ground_truth_grasp_distribution(data['obj_name'][0],
                                                           dset.grasp_data_path,
                                                           gazebo_obj_path)

        # Visualize individual grasps qualitatively
        if show_individual_grasps:
            for j in range(n_samples):
                # Get the grasp sample
                rot_matrix = grasps['rot_matrix'][j, :, :]
                transl = grasps['transl'][j, :]
                # if transl[1] < -0.08:
                #    continue
                joint_conf = grasps['joint_conf'][j, :]
                palm_pose_centr = utils.hom_matrix_from_transl_rot_matrix(transl, rot_matrix)
                visualization.show_grasp_and_object(data['pcd_path'][0], palm_pose_centr,
                                                    joint_conf,'meshes/robotiq_palm/robotiq-3f-gripper_articulated.urdf')


def eval_ffhgenerator_qualitatively_on_real_data(load_epoch,
                                                 load_path,
                                                 n_samples=1000,
                                                 show_individual_grasps=True):

    config = Config()
    cfg = config.parse()
    ffhnet = FFHNet(cfg)
    ffhnet.load_ffhgenerator(epoch=load_epoch, load_path=load_path)
    base_data_bath = os.path.join(ROOT_PATH,'data','real_objects')

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
        grasps = ffhnet.generate_grasps(obj_bps, n_samples=n_samples, return_arr=True)

        # Visualize sampled distribution
        visualization.show_generated_grasp_distribution(obj_pcd_path, grasps)
        # Visualize individual grasps qualitatively
        if show_individual_grasps:
            for j in range(n_samples):
                # Get the grasp sample
                rot_matrix = grasps['rot_matrix'][j, :, :]
                transl = grasps['transl'][j, :]
                # if transl[1] > -0.1:
                #     continue
                joint_conf = grasps['joint_conf'][j, :]
                palm_pose_centr = utils.hom_matrix_from_transl_rot_matrix(transl, rot_matrix)
                visualization.show_grasp_and_object(obj_pcd_path, palm_pose_centr, joint_conf, 
                                                    'meshes/robotiq_palm/robotiq-3f-gripper_articulated.urdf')


def eval_eva_acc_multiple_epochs(epochs, path):
    if not os.path.exists(path):
        os.mkdir(path)
    config = Config()
    cfg = config.parse()
    cfg["eval_dir"] = path
    cfg["save_dir"] = path
    cfg["name"] = os.path.split(path)[1]
    w = writer.Writer(cfg)
    for epoch in epochs:
        mean_accuracy, _, _ = eval_eva_accuracy(load_path=os.path.split(path)[0],
                                                                  load_epoch_eva=epoch,
                                                                  show_confusion_matrix=False)
        w.print_current_eval_loss(epoch, mean_accuracy)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--gen_path', default='models/ffhgenerator', help='path to FFHGenerator model')
    parser.add_argument('--gen_path', default='checkpoints/ffhnet/2023-09-01T01_16_11_ffhnet_lr_0.0001_bs_1000', help='path to FFHGenerator model')
    parser.add_argument('--load_gen_epoch', type=int, default=10, help='epoch of FFHGenerator model')
    parser.add_argument('--eva_path', default='models/ffhevaluator', help='path to FFHEvaluator model')
    parser.add_argument('--load_eva_epoch', type=int, default=30, help='epoch of FFHEvaluator model')
    parser.add_argument('--config', type=str, default='FFHNet/config/config_ffhnet_yb.yaml')
    # parser.add_argument('--config', type=str, default='FFHNet/config/config_ffhgan.yaml')

    args = parser.parse_args()

    load_path_gen = args.gen_path
    load_path_eva = args.eva_path
    load_epoch_gen = args.load_gen_epoch
    load_epoch_eva = args.load_eva_epoch
    config_path = args.config

    eval_ffhnet_sampling_and_filtering_real(config_path, load_epoch_eva, load_epoch_gen, load_path_eva,
                                            load_path_gen, show_individual_grasps=True)
    eval_ffheva_num_removed_grasps(config_path, load_epoch_eva, load_epoch_gen, load_path_eva, load_path_gen)
