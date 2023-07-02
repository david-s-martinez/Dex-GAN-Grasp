import torch
import os
from time import time
import numpy as np

import FFHNet.models.networks as networks
from FFHNet.utils.train_tools import EarlyStopping
from FFHNet.utils import utils
import FFHNet.models.losses as losses
from FFHNet.models.pointnet import PointNetEvaluator, PointNetGenerator, PointNetCollDetr
from FFHNet.models.networks import FFHCollDetr, FFHCollDetrTwoResBlock

class FFHNetCollDetr():
    def __init__(self, cfg) -> None:
        # set default float32 for training torch in GPU
        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)

        self.cfg = cfg

        # specify GPU or CPU
        if torch.cuda.is_available:
            self.device = torch.device('cuda:{}'.format(cfg["gpu_ids"][0]))
            torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')

        # init model
        if cfg["model"] == "ffhnet":
            self.FFHCollDetr = self.init_res_block_structure()
        elif cfg["model"] == "pointnet":
            self.FFHCollDetr = self.init_pointnet_structure()
        else:
            raise ValueError('Wrong configure model name of', cfg["model"])

        # init hyper params
        self.bce_weight = 10.
        self.kl_coef = cfg["kl_coef"]
        self.transl_coef = 100.
        self.rot_coef = 1.
        self.conf_coef = 10.
        self.train_ffhcolldetr = cfg["train_ffhcolldetr"]
        self.optim_ffhcolldetr = torch.optim.Adam(self.FFHCollDetr.parameters(),
                                                    lr=cfg["lr"],
                                                    betas=(cfg["beta1"], 0.999),
                                                    weight_decay=cfg["weight_decay"])
        self.scheduler_ffhcolldetr = networks.get_scheduler(self.optim_ffhcolldetr, cfg)
        self.estop_ffhcolldetr = EarlyStopping()

        # self.kl_loss, self.rec_pose_loss = define_losses('transl_rot_6D_l2')
        # self.L2_loss = torch.nn.MSELoss(reduction='mean')
        self.BCE_loss = torch.nn.BCELoss(reduction='mean')

        self.compute_eva_accuracy = losses.accuracy_evaluator

        # count params
        ffhcolldetr_vars = [var[1] for var in self.FFHCollDetr.named_parameters()]
        ffhcolldetr_n_params = sum(p.numel() for p in ffhcolldetr_vars if p.requires_grad)
        print("The ffhcolldetr has {:2.2f} parms".format(ffhcolldetr_n_params))

        self.file_path = os.path.dirname(os.path.abspath(__file__))
        self.logit_thresh = 0.5

    def init_res_block_structure(self):
        str = 'col'
        exec("globals()[str] = %s" % self.cfg["model_name"]+'(self.cfg)')
        col.to(self.device)

        # if self.cfg["is_train"]:
        #     self.init_net(col)
        return col

    def init_pointnet_structure(self):
        col = PointNetCollDetr(self.cfg)
        col.to(self.device)
        # Pytorch does init by default of He Kaiming methods
        # if self.cfg["is_train"]:
        #     self.init_net(col)
        return col

    # def init_net(self, net):
    #     init_type = net.cfg["weight_init_type"]
    #     init_gain = net.cfg["init_gain"]

    #     def init_func(m):
    #         classname = m.__class__.__name__
    #         # ?????????????????
    #         if hasattr(m, 'weight') and (classname.find('Conv') != -1
    #                                     or classname.find('Linear') != -1):
    #             if init_type == 'normal':
    #                 torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
    #             elif init_type == 'xavier':
    #                 torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
    #             elif init_type == 'kaiming':
    #                 torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    #             elif init_type == 'orthogonal':
    #                 torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
    #             else:
    #                 raise NotImplementedError(
    #                     'initialization method [%s] is not implemented' % init_type)
    #         elif classname.find('BatchNorm') != -1:
    #             torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
    #             torch.nn.init.constant_(m.bias.data, 0.0)

    #     net.apply(init_func)

    def compute_loss_ffhcolldetr(self, pred_success_p):
        """Computes the binary cross entropy loss between predicted success-label and true success"""
        bce_loss = self.bce_weight * \
            self.BCE_loss(pred_success_p, self.FFHCollDetr.gt_label)
        bce_loss_dict = {'bce_loss': bce_loss}
        return bce_loss, bce_loss_dict

    def eval_ffhcolldetr_accuracy(self, data, thresh=False):
        if thresh is False:
            thresh = self.logit_thresh
        logits = self.FFHCollDetr(data)  # network outputs logits

        # Turn the output logits into class labels. logits > thresh = 1, < thresh = 0
        pred_label = utils.class_labels_from_logits(logits, thresh)

        # Compute the accuracy for positive and negative class
        pos_acc, neg_acc, acc = self.compute_eva_accuracy(pred_label, self.FFHCollDetr.gt_label)

        # Turn the raw predictions into np arrays and return for confusion matrix
        pred_label_np = pred_label.detach().cpu().numpy()
        gt_label_np = self.FFHCollDetr.gt_label.detach().cpu().numpy()

        return pos_acc, neg_acc, acc, pred_label_np, gt_label_np

    def eval_ffhcolldetr_loss(self, data):
        self.FFHCollDetr.eval()

        with torch.no_grad():
            out = self.FFHCollDetr(data)
            _, loss_dict_ffhcolldetr = self.compute_loss_ffhcolldetr(out)

        return loss_dict_ffhcolldetr

    def filter_grasps_in_collision(self, bps, grasps, thresh=0.5, return_arr=True):
        """Takes in grasps generated by FFHGenerator, bps encoding of an object and removes grasps with predicted
        collision probability less than thresh.

        Args:
            bps (np array): Bps encoding of the object. n*4096
            grasps (dict): Dict holding the grasp information. keys: transl n*3, rot_matrix n*3*3, joint_conf n*15
            thresh (float, optional): Reject grasps with lower success p than this. Defaults to 0.5.

        """
        start = time.time()
        n_samples = grasps['rot_matrix'].shape[0]
        if len(bps.shape) > 1:
            bps = bps.squeeze()
        bps = np.tile(bps, (n_samples, 1))

        grasps['bps_object'] = bps
        grasps_t = utils.dict_to_tensor(grasps, device=self.device, dtype=self.dtype)

        p_noncollision = self.FFHCollDetr(grasps_t).squeeze()
        print(p_noncollision)
        print(p_noncollision.min())
        print(p_noncollision.max())
        filt_grasps = {}
        for k, v in grasps_t.items():
            filt_grasps[k] = v[p_noncollision > thresh]
            if return_arr:
                filt_grasps[k] = filt_grasps[k].cpu().detach().numpy()
        print("in total grasps:",n_samples)
        print("after filtering with collision",len(filt_grasps))

        #print("Filtering took: %.4f" % (time.time() - start))]

        return filt_grasps

    def load_ffhcolldetr(self, epoch, load_path=None):
        """Load ffhcolldetr from disk and set to eval or train mode.
        """
        if epoch == -1:
            path = os.path.split(os.path.split(self.file_path)[0])[0]
            dirs = sorted(os.path.listdir(os.path.join(path, 'checkpoints')))
            load_path = os.path.join(path, dirs[-1], str(epoch) + '_col_net.pt')
        else:
            if load_path is None:
                load_path = self.cfg["save_dir"]
            load_path = os.path.join(load_path, str(epoch) + '_col_net.pt')

        ckpt = torch.load(load_path, map_location=self.device)
        self.FFHCollDetr.load_state_dict(ckpt['ffhcolldetr_state_dict'])

        if self.cfg["is_train"]:
            self.optim_ffhcolldetr.load_state_dict(ckpt['optim_ffhcolldetr_state_dict'])
            self.scheduler_ffhcolldetr.load_state_dict(ckpt['scheduler_ffhcolldetr_state_dict'])
            self.cfg["load_epoch"] = ckpt['epoch']
            self.FFHCollDetr.train()
        else:
            self.FFHCollDetr.eval()

    def save_ffhcolldetr(self, net_name, epoch):
        """ Save ffhcolldetr to disk

        Args:
            net_name (str): The name of the model.
            epoch (int): Current epoch.
        """
        save_path = os.path.join(self.cfg["save_dir"], net_name + '_col_net.pt')
        if len(self.cfg["gpu_ids"]) > 1:
            ffhcolldetr_state_dict = self.FFHCollDetr.module.cpu().state_dict()
        else:
            ffhcolldetr_state_dict = self.FFHCollDetr.cpu().state_dict()
        torch.save(
            {
                'epoch': epoch,
                'ffhcolldetr_state_dict': ffhcolldetr_state_dict,
                'optim_ffhcolldetr_state_dict': self.optim_ffhcolldetr.state_dict(),
                'scheduler_ffhcolldetr_state_dict': self.scheduler_ffhcolldetr.state_dict(),
            }, save_path, _use_new_zipfile_serialization=False)

        if torch.cuda.is_available():
            self.FFHCollDetr.to(torch.device('cuda:{}'.format(self.cfg["gpu_ids"][0])))

    def update_ffhcolldetr(self, data):
        # Make sure net is in train mode
        self.FFHCollDetr.train()

        # zero the param gradients
        self.optim_ffhcolldetr.zero_grad()

        # Run forward pass of ffhcolldetr and predict grasp success
        out = self.FFHCollDetr(data)

        # Compute loss based on reconstructed data
        total_loss_ffhcolldetr, loss_dict_ffhcolldetr = self.compute_loss_ffhcolldetr(out)
        # Zero gradients, backprop new loss gradient, run one step
        total_loss_ffhcolldetr.backward()
        self.optim_ffhcolldetr.step()

        # Return loss
        return loss_dict_ffhcolldetr

# Add accuracy Accuracy = [TP + TN] / Total Population
