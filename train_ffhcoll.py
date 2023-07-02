import argparse
import time
import torch
from torch.utils.data import DataLoader

from eval_ffhcoll import run_eval
from FFHNet.config.config import Config
from FFHNet.data.ffhcollision_data_set import FFHCollDetrDataSet, FFHCollDetrPCDDataSet
from FFHNet.models.ffhcolldetr import FFHNetCollDetr
from FFHNet.utils.writer import Writer


def main():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help='Path to template image.',
                        default='FFHNet/config/config_pointnet_vm.yaml')
    args = parser.parse_args()

    # load configuration params
    config = Config(args.config)
    cfg = config.parse()

    # start cuda multiprocessing
    # TODO: discover the problem of cpu usage
    torch.multiprocessing.set_start_method('spawn')

    # Data for gen and eval and col is different. Gen sees only positive examples
    if cfg["train_ffhcolldetr"]:
        if cfg["model"] == "ffhnet":
            dset_eva = FFHCollDetrDataSet(cfg)
        elif cfg["model"] == "pointnet":
            dset_eva = FFHCollDetrPCDDataSet(cfg, eval=False)

        train_loader_col = DataLoader(dset_eva,
                                      batch_size=cfg["batch_size"],
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=cfg["num_threads"])

    writer = Writer(cfg)

    ffhnet = FFHNetCollDetr(cfg)
    if cfg["continue_train"] and cfg["train_ffhcolldetr"]:
        ffhnet.load_ffhcolldetr(cfg["load_epoch"])
        start_epoch = cfg["load_epoch"] + 1
    else:
        start_epoch = 1

    total_steps = 0
    epoch_start = time.time()

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        # === Collision Detector ===
        if ffhnet.train_ffhcolldetr:
            # Initialize epoch / iter info
            prev_iter_end = time.time()
            epoch_iter = 0
            num_batches = len(train_loader_col)
            iter_print = num_batches//cfg["print_freq"]

            # Evaluator training loop
            for i, data in enumerate(train_loader_col):
                cur_iter_start = time.time()
                total_steps += cfg["batch_size"]
                epoch_iter += cfg["batch_size"]

                # Update model one step, get losses
                loss_dict = ffhnet.update_ffhcolldetr(data)

                # Log loss
                if (i+1) % iter_print == 0:
                    t_load = cur_iter_start - prev_iter_end  # time for data loading
                    t_total = (time.time() - epoch_start) / 60
                    writer.print_current_train_loss(epoch, epoch_iter, loss_dict, t_total, t_load)
                    writer.plot_train_loss(loss_dict, epoch, epoch_iter, len(dset_eva))

                prev_iter_end = time.time()

                # End of data loading iter
        # Save model after each epoch
        if epoch % cfg["save_freq"] == 0:
            print('Saving the model after epoch %d, iters %d' % (epoch, total_steps))
            if ffhnet.train_ffhcolldetr:
                ffhnet.save_ffhcolldetr(str(epoch), epoch)

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


if __name__ == '__main__':
    main()
