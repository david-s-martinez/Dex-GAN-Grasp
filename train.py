from __future__ import division

import argparse
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from DexGanGrasp.config.config import Config
from DexGanGrasp.data.dexevaluator_data_set import DexEvaluatorDataSet, DexEvaluatorPCDDataSet
from DexGanGrasp.data.dexgenerator_data_set import DexGeneratorDataSet
from DexGanGrasp.utils.writer import Writer
from DexGanGrasp.models.dexgangrasp import DexGanGrasp

def update_mean_losses(mean_losses, new_losses):
    """
    Updates the cumulative mean losses by adding new losses from the current batch.

    Args:
        mean_losses (dict): Dictionary containing the cumulative mean losses for different metrics (e.g., 'gen_loss_fake', 'transl_loss', etc.).
        new_losses (dict): Dictionary containing the losses from the current batch.

    Returns:
        mean_losses (dict): Updated dictionary with cumulative losses after adding the new losses from the current batch.
    """
    for key in mean_losses.keys():
        mean_losses[key] += new_losses[key].detach().cpu().numpy()
    return mean_losses

def run_eval_gan_gen(dexgangrasp, cfg):
    """
    Runs evaluation for the DexGenerator model and computes average losses.

    Process:
        - Loads the evaluation dataset using the DexGeneratorDataSet with evaluation mode enabled.
        - Iterates through the evaluation data using the dataloader.
        - Computes various losses (generator fake loss, translation loss, rotation loss, confidence loss, total loss) for each batch.
        - Averages the computed losses across all batches and logs progress periodically.

    Args:
        dexgangrasp: The DexGanGrasp object containing the generator model.
        cfg (dict): Configuration dictionary with parameters such as batch size.

    Returns:
        mean_losses (dict): A dictionary containing:
            - 'gen_loss_fake' (float): The average generator fake loss.
            - 'transl_loss' (float): The average translation loss.
            - 'rot_loss' (float): The average rotation loss.
            - 'conf_loss' (float): The average confidence loss.
            - 'total_loss_gen' (float): The average total generator loss.
    """
    print('Running eval for DexGANGrasp Generator')
    dset = DexGeneratorDataSet(cfg, eval = True)
    eval_loader = DataLoader(dset, batch_size=cfg["batch_size"], shuffle=False)

    mean_losses = {
        'gen_loss_fake' : 0,
        'transl_loss': 0,
        'rot_loss' : 0,
        'conf_loss' : 0,
        'total_loss_gen' : 0
        }

    for i, data in enumerate(eval_loader):
        loss_dict = dexgangrasp.eval_dexgangrasp_generator_loss(data)
        if i % 100 == 0:
            print(i,'- Eval Loss:', loss_dict)
        mean_losses = update_mean_losses(mean_losses, loss_dict)

    for k, _ in mean_losses.items():
        mean_losses[k] /= (i + 1)

    return mean_losses

def run_eval_eva(dexgangrasp, dataloader, curr_epoch, eval_dir):
    """
    Runs evaluation for the DexEvaluator model and computes average losses and accuracy.

    Process:
        - Iterates through the evaluation dataset using the provided dataloader.
        - Computes the total evaluation loss and accuracy for positive and negative examples for each batch.
        - Averages the computed metrics across all batches.
        - Saves the predicted and ground truth labels for further analysis.

    Args:
        dexgangrasp: The DexGanGrasp object containing the DexEvaluator model.
        dataloader: Dataloader object for loading evaluation data.
        curr_epoch (int): The current epoch number, used for saving the evaluation results.
        eval_dir (str): Directory to save the evaluation results, including predicted and ground truth labels.

    Returns:
        mean_losses (dict): A dictionary containing:
            - 'total_loss_eva' (float): The average total evaluation loss.
            - 'pos_acc' (float): The average accuracy for positive examples.
            - 'neg_acc' (float): The average accuracy for negative examples.
    """

    print('Running eval for DexEvaluator.')

    mean_losses = {
        'total_loss_eva': 0,
        'pos_acc': 0,
        'neg_acc': 0,
    }

    pred_labels = np.array([])
    gt_labels = np.array([])

    for i, data in enumerate(dataloader):
        loss_dict = dexgangrasp.eval_dexevaluator_loss(data)
        pos_acc, neg_acc, pred_label, gt_label = dexgangrasp.eval_dexevaluator_accuracy(data)

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

def run_eval_gan(cfg, curr_epoch, dexgangrasp=None, epoch=-1, name=""):
    """Performs model evaluation on the eval set. Evaluates either only one or both the DexGenerator, DexEvaluator
    depending on the config settings.

    Args:
        eval_dir (str):
        curr_epoch (int):
        dexgangrasp (DexGanGrasp, optional): The full DexGanGrasp model. Defaults to None.
        epoch (int, optional): Epoch from which to load a model. Defaults to -1.
        name (str, optional): Name of the model to be loaded. Defaults to "".

    Returns:
        loss_dict (dict): A dict with the losses for DexEvaluator and/or DexGenerator, depending on cfg["train"]_* is set.
    """
    print('Running eval.')

    cfg["name"] = name

    loss_dict = {}

    if cfg["eval_ffhevaluator"]:
        if cfg["model"] == 'ffhnet':
            dset = DexEvaluatorDataSet(cfg,eval=True)
        eval_loader = DataLoader(dset, batch_size=cfg["batch_size"], shuffle=False)
        eval_loss_dict_eva = run_eval_eva(dexgangrasp, eval_loader, curr_epoch, cfg["eval_dir"])
        loss_dict.update(eval_loss_dict_eva)

    if cfg["eval_ffhgenerator"]:
        eval_loss_dict_gen = run_eval_gan_gen(dexgangrasp, cfg)
        loss_dict.update(eval_loss_dict_gen)

    return loss_dict

def main():
    """
    Main function to train DexGANGrasp generator and evaluator models.

    The function initializes argument parsing, loads configuration, and sets up data loaders for training both the grasp generator and evaluator models. It includes logging and saving mechanisms, and runs for a specified number of epochs.

    Process:
        - Loads configuration parameters from a YAML file.
        - Initializes CUDA multiprocessing to handle parallel data loading.
        - Trains the grasp evaluator and/or grasp generator based on configuration.
        - Logs the training progress, including losses and model weights, after every epoch.
        - Saves the model state at regular intervals.
        - Optionally evaluates the models on a validation dataset and logs evaluation losses.

    Args:
        --config (str): Path to the configuration YAML file. Defaults to 'DexGanGrasp/config/config_dexgangrasp.yaml'.
    
    Returns:
        None
    """

    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help='Path to template image.',
                        default='DexGanGrasp/config/config_dexgangrasp.yaml')
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
            dset_eva = DexEvaluatorDataSet(cfg,eval=False)
        elif cfg["model"] == "pointnet":
            dset_eva = DexEvaluatorPCDDataSet(cfg,eval=False)
        train_loader_eva = DataLoader(dset_eva,
                                      batch_size=cfg["batch_size"],
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=cfg["num_threads"])

    if cfg["train_ffhgenerator"]:
        dset_gen = DexGeneratorDataSet(cfg)
        train_loader_gen = DataLoader(dset_gen,
                                      batch_size=cfg["batch_size"],
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=cfg["num_threads"])

    writer = Writer(cfg)

    dexgangrasp = DexGanGrasp(cfg)
    if cfg["continue_train"]:
        if cfg["train_ffhevaluator"]:
            dexgangrasp.load_dexevaluator(cfg["load_epoch"])
        if cfg["train_ffhgenerator"]:
            dexgangrasp.load_dexgenerator(cfg["load_epoch"])
        start_epoch = cfg["load_epoch"] + 1
    else:
        start_epoch = 1
    total_steps = 0
    epoch_start = time.time()

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        # === Generator ===
        if dexgangrasp.train_dexgenerator:
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
                if i % cfg['gen_train_freq'] == 0:
                    loss_dict = dexgangrasp.update_dexgangrasp(data, is_train_gen= True)
                else:
                    loss_dict = dexgangrasp.update_dexgangrasp(data, is_train_gen = False)

                # Log loss
                if total_steps % cfg["print_freq"] == 0:
                    t_load = cur_iter_start - prev_iter_end  # time for data loading
                    t_total = (time.time() - cur_iter_start) // 60
                    writer.print_current_train_loss(epoch, epoch_iter, loss_dict, t_total, t_load)
                    writer.plot_train_loss(loss_dict, epoch, epoch_iter, len(dset_gen))

                prev_iter_end = time.time()
                # End of data loading generator

        # === Evaluator ===
        if dexgangrasp.train_dexevaluator:
            # Initialize epoch / iter info
            prev_iter_end = time.time()
            epoch_iter = 0

            # Evaluator training loop
            for i, data in enumerate(train_loader_eva):
                cur_iter_start = time.time()
                total_steps += cfg["batch_size"]
                epoch_iter += cfg["batch_size"]

                # Update model one step, get losses
                loss_dict = dexgangrasp.update_dexevaluator(data)

                # Log loss
                if total_steps % cfg["print_freq"] == 0:
                    t_load = cur_iter_start - prev_iter_end  # time for data loading
                    t_total = (time.time() - epoch_start) // 60
                    writer.print_current_train_loss(epoch, epoch_iter, loss_dict, t_total, t_load)
                    writer.plot_train_loss(loss_dict, epoch, epoch_iter, len(dset_eva))

                prev_iter_end = time.time()

        # === End of data loading for gen and eva ===
        # Save model after each epoch
        if epoch % cfg["save_freq"] == 0:
            print('Saving the model after epoch %d, iters %d' % (epoch, total_steps))
            if dexgangrasp.train_dexgenerator:
                dexgangrasp.save_dexgenerator(str(epoch), epoch)
            if dexgangrasp.train_dexevaluator:
                dexgangrasp.save_dexevaluator(str(epoch), epoch)

        # Some interesting prints
        epoch_diff = time.time() - epoch_start
        print('End of epoch %d / %d \t Time taken: %.3f min' %
              (epoch, cfg["num_epochs"], epoch_diff / 60))
        
        if epoch % cfg["save_freq"] == 0:
            # Eval model on eval dataset
            eval_loss_dict = run_eval_gan(cfg, epoch, dexgangrasp=dexgangrasp)
            writer.print_current_eval_loss(epoch, eval_loss_dict)
            writer.plot_eval_loss(eval_loss_dict, epoch)

        # Plot model weights and losses
        writer.plot_model_weights_gan(dexgangrasp, epoch)

        # End of epoch

    writer.close()

if __name__ == '__main__':
    main()
