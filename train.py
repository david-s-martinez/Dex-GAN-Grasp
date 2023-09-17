from __future__ import division

import argparse
import time
import torch
from torch.utils.data import DataLoader

from eval import run_eval
from FFHNet.config.config import Config
from FFHNet.data.ffhevaluator_data_set import FFHEvaluatorDataSet, FFHEvaluatorPCDDataSet
from FFHNet.data.ffhgenerator_data_set import FFHGeneratorDataSet
from FFHNet.models.ffhnet import FFHNet
from FFHNet.utils.writer import Writer


def main():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help='Path to template image.',
                        default='FFHNet/config/config_ffhnet_yb.yaml')
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

    ffhnet = FFHNet(cfg)
    if cfg["continue_train"]:
        if cfg["train_ffhevaluator"]:
            ffhnet.load_ffhevaluator(cfg["load_epoch"])
        if cfg["train_ffhgenerator"]:
            ffhnet.load_ffhgenerator(cfg["load_epoch"])
        start_epoch = cfg["load_epoch"] + 1
    else:
        start_epoch = 1
    total_steps = 0
    epoch_start = time.time()

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        # === Generator ===
        if ffhnet.train_ffhgenerator:
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

                # Perform Kl regularization scheme
                set_kl_term(ffhnet.FFHGenerator, epoch_iter, epoch, max_epoch=cfg["num_epochs"])

                # Update model one step, get losses
                loss_dict = ffhnet.update_ffhgenerator(data)

                # Log loss
                if total_steps % cfg["print_freq"] == 0:
                    t_load = cur_iter_start - prev_iter_end  # time for data loading
                    t_total = (time.time() - cur_iter_start) // 60
                    writer.print_current_train_loss(epoch, epoch_iter, loss_dict, t_total, t_load)
                    writer.plot_train_loss(loss_dict, epoch, epoch_iter, len(dset_gen))

                prev_iter_end = time.time()
                # End of data loading generator

        # === Evaluator ===
        if ffhnet.train_ffhevaluator:
            # Initialize epoch / iter info
            prev_iter_end = time.time()
            epoch_iter = 0

            # Evaluator training loop
            for i, data in enumerate(train_loader_eva):
                cur_iter_start = time.time()
                total_steps += cfg["batch_size"]
                epoch_iter += cfg["batch_size"]

                # Update model one step, get losses
                loss_dict = ffhnet.update_ffhevaluator(data)

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
            if ffhnet.train_ffhgenerator:
                ffhnet.save_ffhgenerator(str(epoch), epoch)
            if ffhnet.train_ffhevaluator:
                ffhnet.save_ffhevaluator(str(epoch), epoch)

        # Some interesting prints
        epoch_diff = time.time() - epoch_start
        print('End of epoch %d / %d \t Time taken: %.3f min' %
              (epoch, cfg["num_epochs"], epoch_diff / 60))

        if epoch % cfg["save_freq"] == 0:
            # Eval model on eval dataset
            eval_loss_dict = run_eval(cfg, epoch, ffhnet=ffhnet)
            writer.print_current_eval_loss(epoch, eval_loss_dict)
            writer.plot_eval_loss(eval_loss_dict, epoch)

        # Plot model weights and losses
        writer.plot_model_weights(ffhnet, epoch)

        # End of epoch

    writer.close()


def set_kl_term(FFHGenerator, iter, epoch, max_epoch=10, schedule='linear'):
    if schedule == 'cyclical':
        if iter <= 20e4:
            FFHGenerator.cfg["kl_coef"] = 0
        elif iter > 20e4 and iter <= 35e4:
            FFHGenerator.cfg["kl_coef"] = 0.5
        elif iter > 35e4:
            FFHGenerator.cfg["kl_coef"] = 1
    elif schedule == 'linear':
        if epoch <= 3:
            kl_weight = 0
        else:
            kl_weight = 1. / 5. * (epoch - 3.)
            if kl_weight > 1.:
                kl_weight = 1.
        FFHGenerator.kl_coef = kl_weight * FFHGenerator.cfg["kl_coef"]


if __name__ == '__main__':
    main()
