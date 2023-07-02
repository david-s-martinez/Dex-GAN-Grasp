from __future__ import division

import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

from FFHNet.config.config import Config
from FFHNet.data.bps_encoder import BPSEncoder
from FFHNet.data.ffhcollision_data_set import FFHCollDetrDataSet, FFHCollDetrPCDDataSet

from FFHNet.models.ffhcolldetr import FFHNetCollDetr
from FFHNet.utils import utils, visualization, writer

path = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.split(os.path.split(path)[0])[0]

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


def run_eval(cfg, curr_epoch, ffhnet, eval_dir=False):
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


    loss_dict = {}


    if cfg["model"] == "ffhnet":
        dset = FFHCollDetrDataSet(cfg, eval=True)
    elif cfg["model"] == "pointnet":
        dset = FFHCollDetrPCDDataSet(cfg, eval=True)

    col_loader = DataLoader(dset, batch_size=cfg["batch_size"], shuffle=False)
    if eval_dir is not False:
        eval_loss_dict_col = run_eval_col(cfg, ffhnet, col_loader, curr_epoch, eval_dir)
    else:
        eval_loss_dict_col = run_eval_col(cfg, ffhnet, col_loader, curr_epoch, cfg["eval_dir"])
    loss_dict.update(eval_loss_dict_col)

    return loss_dict


def run_eval_col(cfg, ffhnet, dataloader, curr_epoch, eval_dir):
    print('Running eval for FFHColdetr.')

    mean_losses = {
        'bce_loss': 0,
        'pos_acc': 0,
        'neg_acc': 0,
        'acc': 0
    }

    pred_labels = np.array([])
    gt_labels = np.array([])

    for i, data in enumerate(dataloader):
        loss_dict = ffhnet.eval_ffhcolldetr_loss(data)
        pos_acc, neg_acc, acc, pred_label, gt_label = ffhnet.eval_ffhcolldetr_accuracy(data)

        mean_losses['bce_loss'] += loss_dict['bce_loss'].detach().cpu().numpy()
        mean_losses['pos_acc'] += pos_acc
        mean_losses['neg_acc'] += neg_acc
        mean_losses['acc'] += acc

        pred_labels = np.append(pred_labels, pred_label)
        gt_labels = np.append(gt_labels, gt_label)

        num_batches = len(dataloader)
        iter_print = num_batches//cfg["print_freq"]

        if (i+1) % iter_print == 0:
            print(f'eval: {(i+1) // iter_print} / {cfg["print_freq"]}, (acc: {acc}')

    for k, _ in mean_losses.items():
        mean_losses[k] /= (i + 1)

    # save the labels
    np.save(os.path.join(eval_dir, str(curr_epoch) + '_gt_labels.npy'), gt_labels)
    np.save(os.path.join(eval_dir, str(curr_epoch) + '_pred_labels.npy'), pred_labels)

    return mean_losses


def eval_col_accuracy(cfg, load_epoch_eva, load_path, thresh=0.5, show_confusion_matrix=False):

    ffhnet = FFHNet(cfg)
    ffhnet.load_ffhcolldetr(epoch=load_epoch_eva, load_path=load_path)
    dset = FFHCollDetrDataSet(cfg,eval=True)
    eval_loader = DataLoader(dset, batch_size=cfg["batch_size"], shuffle=False)

    mean_accuracy = {
        'pos_acc': 0,
        'neg_acc': 0,
    }

    pred_labels = np.array([])
    gt_labels = np.array([])
    print(len(eval_loader))
    for i, data in enumerate(eval_loader):
        #print(" %.2f percent  completed." % (100 * i * cfg["batch_size"] / len(dset)))
        pos_acc, neg_acc, acc, pred_label, gt_label = ffhnet.eval_ffhcolldetr_accuracy(data,thresh)

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



if __name__ == "__main__":
    load_path_gen = '/home/vm/hand_ws/src/ffhnet/models/ffhgenerator/'
    load_path_eva = '/home/vm/hand_ws/src/ffhnet/models/ffhevaluator/'
    load_path_col = '/home/vm/hand_ws/src/FFHNet-dev/checkpoints/test_model/'
    eval_dir = '/home/vm/hand_ws/src/FFHNet-dev/checkpoints/test_model/'
    load_epoch_gen = 10
    load_epoch_eva = 30
    load_epoch_col = 1

    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help='Path to template image.',
                        default='FFHNet/config/config_template.yaml')
    args = parser.parse_args()

    config = Config(args.config)
    cfg = config.parse()

    # eval_ffhnet_sampling_and_filtering_real(load_epoch_eva, load_epoch_gen, load_path_eva,
    #                                         load_path_gen)
    # eval_ffheva_num_removed_grasps(load_epoch_eva, load_epoch_gen, load_path_eva, load_path_gen)
    ffhnet = FFHNetCollDetr(cfg)
    ffhnet.load_ffhcolldetr(load_epoch_col, load_path_col)
    run_eval(cfg, load_epoch_col, ffhnet, eval_dir=eval_dir)
