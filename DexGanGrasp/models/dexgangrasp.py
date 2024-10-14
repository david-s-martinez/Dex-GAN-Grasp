from __future__ import division
import numpy as np
import os
import time
import torch
from DexGanGrasp.utils.train_tools import EarlyStopping
from DexGanGrasp.utils import utils

import DexGanGrasp.models.losses as losses
import DexGanGrasp.models.networks as networks
from DexGanGrasp.models.networks import DexEvaluator, DexGANGrasp

def define_losses(loss_type_recon):
    """ This will return the loss functions. The KL divergence is fixed, but for the reconstruction loss
    different losses are possible. Currently [control_point_l1, rotation_6D_l2 are implemented]
    """
    kl_loss = losses.kl_divergence
    if loss_type_recon == 'control_point_l1':
        reconstruction_loss = losses.control_point_l1_loss
    elif loss_type_recon == 'transl_rot_6D_l2':
        reconstruction_loss = losses.transl_rot_6D_l2_loss
    else:
        raise Exception(
            'Requested loss not available, choose [control_point_l1, transl_rot_6D_l2]')
    return kl_loss, reconstruction_loss


def build_network(cfg, device, is_train):
    gen = DexGANGrasp(cfg)
    eva = DexEvaluator(cfg)
    if torch.cuda.is_available:
        gen.to(device)
        eva.to(device)
    if is_train:
        init_net(gen)
        init_net(eva)
    return gen, eva

def init_net(net):
    init_type = net.cfg["weight_init_type"]
    init_gain = net.cfg["init_gain"]

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

# TODO: refactor. This class is used in train.py. eval.py and visualization.py
class DexGanGrasp(object):
    """ Wrapper which houses the network blocks, the losses and training logic.
    """

    def __init__(self, cfg):
        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)
        self.cfg = cfg
        self.is_train = cfg["is_train"]
        self.is_wgan = cfg["is_wgan"]
        if torch.cuda.is_available:
            self.device = torch.device('cuda:{}'.format(cfg["gpu_ids"][0]))
            torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')

        # model, optimizer, scheduler_dexgenerator, losses
        if cfg["model"] == "ffhnet":
            self.DexGANGrasp, self.DexEvaluator = build_network(cfg, self.device, is_train=self.is_train)
        elif cfg["model"] == "dexgangrasp":
            self.DexGANGrasp, self.DexEvaluator = build_network(cfg, self.device, is_train=self.is_train)
        else:
            raise ValueError('Wrong configure model name of', cfg["model"])

        if self.is_train:
            self.bce_weight = cfg["bce_weight"]
            self.kl_coef = cfg["kl_coef"]
            self.transl_coef = 100.
            self.rot_coef = 1.
            self.conf_coef = 10.
            self.train_dexgenerator = cfg["train_ffhgenerator"]
            self.train_dexevaluator = cfg["train_ffhevaluator"]
            self.optim_dexgenerator = torch.optim.Adam(self.DexGANGrasp.parameters(),
                                                       lr=cfg["lr"],
                                                       betas=(cfg["beta1"], 0.999),
                                                       weight_decay=cfg["weight_decay"])
            self.optim_dexgangrasp_generator = torch.optim.Adam(self.DexGANGrasp.generator.parameters(),
                                                       lr=cfg["lr_gen"],
                                                       betas=(cfg["beta1"], 0.999),
                                                       weight_decay=cfg["weight_decay"])
            self.optim_dexgangrasp_discriminator = torch.optim.Adam(self.DexGANGrasp.discriminator.parameters(),
                                                       lr=cfg["lr_dis"],
                                                       betas=(cfg["beta1"], 0.999),
                                                       weight_decay=cfg["weight_decay"])
            self.optim_dexevaluator = torch.optim.Adam(self.DexEvaluator.parameters(),
                                                       lr=cfg["lr"],
                                                       betas=(cfg["beta1"], 0.999),
                                                       weight_decay=cfg["weight_decay"])
            self.scheduler_dexgenerator = networks.get_scheduler(self.optim_dexgenerator, cfg)
            self.scheduler_dexgangrasp_generator = networks.get_scheduler(self.optim_dexgangrasp_generator, cfg)
            self.scheduler_dexgangrasp_discriminator = networks.get_scheduler(self.optim_dexgangrasp_discriminator, cfg)
            self.scheduler_dexevaluator = networks.get_scheduler(self.optim_dexevaluator, cfg)
            self.estop_dexgenerator = EarlyStopping()
            self.estop_dexevaluator = EarlyStopping()

        self.kl_loss, self.rec_pose_loss = define_losses('transl_rot_6D_l2')
        self.L2_loss = torch.nn.MSELoss(reduction='mean')
        self.BCE_loss = torch.nn.BCELoss(reduction='mean')

        self.compute_eva_accuracy = losses.accuracy_evaluator

        # Wrap models if we use multi-gpu
        if len(cfg["gpu_ids"]) > 1:
            self.DexGANGrasp = torch.nn.DataParallel(self.DexGANGrasp, device_ids=cfg["gpu_ids"])
            self.DexEvaluator = torch.nn.DataParallel(self.DexEvaluator, device_ids=cfg["gpu_ids"])

        # count params
        dexgenerator_vars = [var[1] for var in self.DexGANGrasp.named_parameters()]
        dexgenerator_n_params = sum(p.numel() for p in dexgenerator_vars if p.requires_grad)
        print("The dexgenerator has {:2.2f} parms".format(dexgenerator_n_params))
        dexevaluator_vars = [var[1] for var in self.DexEvaluator.named_parameters()]
        dexevaluator_n_params = sum(p.numel() for p in dexevaluator_vars if p.requires_grad)
        print("The dexevaluator has {:2.2f} parms".format(dexevaluator_n_params))
        # TODO count flops as well
        self.file_path = os.path.dirname(os.path.abspath(__file__))
        self.logit_thresh = 0.5

    def compute_loss_dexevaluator(self, pred_success_p):
        """Computes the binary cross entropy loss between predicted success-label and true success"""
        bce_loss_val = self.bce_weight * \
            self.BCE_loss(pred_success_p, self.DexEvaluator.gt_label)
        loss_dict = {'total_loss_eva': bce_loss_val, 'bce_loss': bce_loss_val}
        return bce_loss_val, loss_dict
    
    def calculate_interp(self, real_data, fake_data):
        # Random weight term for interpolation between real and fake data
        if len(real_data.shape)==3:
            alpha = torch.randn((real_data.size(0), 1, 1), device=self.DexGANGrasp.device)
        else:
            alpha = torch.randn((real_data.size(0), 1), device=self.DexGANGrasp.device)
        # Get random interpolation between real and fake data
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
        return interpolates
    
    def calculate_grad_interp(self, interpolates, model_interpolates):
        grad_outputs = torch.ones(model_interpolates.size(), device=self.DexGANGrasp.device, requires_grad=False)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        return gradients
    
    def calculate_gradient_penalty(self, real_data, fake_data):
        """Calculates the gradient penalty loss for WGAN GP"""
        fake_rot_matrix = fake_data["rot_matrix"]
        fake_transl = fake_data["transl"]
        fake_joint_conf = fake_data["joint_conf"]

        real_rot_matrix = real_data["rot_matrix"].to(self.DexGANGrasp.device)
        real_transl = real_data["transl"].to(self.DexGANGrasp.device)
        real_joint_conf = real_data["joint_conf"].to(self.DexGANGrasp.device)
        interp_data = {
                "bps_object": real_data["bps_object"],
                "rot_matrix": self.calculate_interp(real_rot_matrix,fake_rot_matrix),
                "transl": self.calculate_interp(real_transl, fake_transl),
                "joint_conf": self.calculate_interp(real_joint_conf,fake_joint_conf)
            }

        interpolate_scores = self.DexGANGrasp.discriminator(interp_data)
        gradient_penaties = []

        for out in ["rot_matrix", "transl", "joint_conf"]:
            gradients = self.calculate_grad_interp(interp_data[out], interpolate_scores)
            gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
            gradient_penaties.append(gradient_penalty)
        return torch.mean(torch.stack(gradient_penaties))
    
    def compute_loss_dexgangrasp_discriminator_wass(self, real_score, fake_score, real_data, fake_data, penalty_gain = 10):
        """Computes the binary cross entropy loss between predicted real score and true real score"""
        bce_loss_real = torch.mean(real_score)
        bce_loss_fake = torch.mean(fake_score)
        gradient_penalty = self.calculate_gradient_penalty(real_data, fake_data)
        total_loss_disc = -bce_loss_real + bce_loss_fake + (gradient_penalty * penalty_gain)
        loss_dict = {
            'total_loss_disc': total_loss_disc, 
            'bce_loss_real': bce_loss_real, 
            'bce_loss_fake':bce_loss_fake
        }

        return total_loss_disc, loss_dict
    
    def compute_loss_dexgangrasp_discriminator(self, real_score, fake_score, real_data = None, fake_data_disc = None):
        """Computes the binary cross entropy loss between predicted real score and true real score"""
        bce_loss_real = self.bce_weight * self.BCE_loss(
                                                real_score, 
                                                torch.ones_like(real_score)
                                                )
        
        bce_loss_fake = self.bce_weight * self.BCE_loss(
                                                fake_score, 
                                                torch.zeros_like(fake_score)
                                                )
        
        total_loss_disc = (bce_loss_real + bce_loss_fake) / 2
        loss_dict = {
            'total_loss_disc': total_loss_disc, 
            'bce_loss_real': bce_loss_real, 
            'bce_loss_fake':bce_loss_fake
        }

        return total_loss_disc, loss_dict
    
    def compute_loss_dexgangrasp_generator_wass(self, real_data, fake_data, fake_score):
        """ The model should output a 6D representation of a rotation, which then gets mapped back to
        """
        # Pose loss, translation rotation
        gt_transl_rot_matrix = {
            'transl': real_data['transl'].to(self.DexGANGrasp.device).float(),
            'rot_matrix': real_data['rot_matrix'].to(self.DexGANGrasp.device).float()
        }
        transl_loss_val, rot_loss_val = self.rec_pose_loss(fake_data, gt_transl_rot_matrix,
                                                           self.L2_loss, self.device)
        transl_loss_val, rot_loss_val = transl_loss_val, rot_loss_val
        # Loss on joint angles
        conf_loss_val = self.L2_loss(
            fake_data['joint_conf'].to(self.DexGANGrasp.device).float(), 
            real_data['joint_conf'].to(self.DexGANGrasp.device).float()
        )

        # Wasserstein Generator fake loss

        # # L = -1/N * sum(1 - pred_score) if target label is always 0
        # gen_loss_fake = - torch.mean(torch.ones_like(fake_score) - fake_score)

        # L = -1/N * sum(pred_score) if target label is always 1 (fooling discriminator)
        gen_loss_fake = - torch.mean(fake_score)
        
        # Put all losses in one dict and weigh them individually
        loss_dict = {
            'gen_loss_fake' : gen_loss_fake,
            'transl_loss': self.transl_coef * transl_loss_val,
            'rot_loss': self.rot_coef * rot_loss_val,
            'conf_loss': self.conf_coef * conf_loss_val
        }
        total_loss = gen_loss_fake + (loss_dict['transl_loss'] + loss_dict['rot_loss'] + loss_dict['conf_loss'])
        loss_dict["total_loss_gen"] = total_loss
        self.last_loss_dict_gen = loss_dict
        
        return total_loss, loss_dict
    
    def compute_loss_dexgangrasp_generator(self, real_data, fake_data, fake_score):
        """ The model should output a 6D representation of a rotation, which then gets mapped back to
        """
        # Pose loss, translation rotation
        gt_transl_rot_matrix = {
            'transl': real_data['transl'].to(self.DexGANGrasp.device).float(),
            'rot_matrix': real_data['rot_matrix'].to(self.DexGANGrasp.device).float()
        }
        transl_loss_val, rot_loss_val = self.rec_pose_loss(fake_data, gt_transl_rot_matrix,
                                                           self.L2_loss, self.device)
        transl_loss_val, rot_loss_val = transl_loss_val, rot_loss_val
        # Loss on joint angles
        conf_loss_val = self.L2_loss(
            fake_data['joint_conf'].to(self.DexGANGrasp.device).float(), 
            real_data['joint_conf'].to(self.DexGANGrasp.device).float()
        )

        # Generator fake loss
        # BCE = -1/N * sum(target*log(pred_score)+(1-target)*log(1-pred_score))
        # BCE = -1/N * sum(log(pred_score)) if target label is always 1 (fooling discriminator)
        # BCE = -1/N * sum(log(1 - pred_score)) if target label is always 0
        gen_loss_fake = self.bce_weight * self.BCE_loss(
                                                fake_score, 
                                                torch.ones_like(fake_score)
                                                )
        # Put all losses in one dict and weigh them individually
        loss_dict = {
            'gen_loss_fake' : gen_loss_fake,
            'transl_loss': self.transl_coef * transl_loss_val,
            'rot_loss': self.rot_coef * rot_loss_val,
            'conf_loss': self.conf_coef * conf_loss_val
        }
        total_loss = gen_loss_fake + (loss_dict['transl_loss'] + loss_dict['rot_loss'] + loss_dict['conf_loss'])
        loss_dict["total_loss_gen"] = total_loss
        self.last_loss_dict_gen = loss_dict
        return total_loss, loss_dict

    def eval_dexevaluator_accuracy(self, data):
        logits = self.DexEvaluator(data)  # network outputs logits

        # Turn the output logits into class labels. logits > thresh = 1, < thresh = 0
        pred_label = utils.class_labels_from_logits(logits, self.logit_thresh)

        # Compute the accuracy for positive and negative class
        pos_acc, neg_acc, acc = self.compute_eva_accuracy(pred_label, self.DexEvaluator.gt_label)

        # Turn the raw predictions into np arrays and return for confusion matrix
        pred_label_np = pred_label.detach().cpu().numpy()
        gt_label_np = self.DexEvaluator.gt_label.detach().cpu().numpy()

        return pos_acc, neg_acc, pred_label_np, gt_label_np


    def eval_dexevaluator_loss(self, data):
        self.DexEvaluator.eval()

        with torch.no_grad():
            out = self.DexEvaluator(data)
            _, loss_dict_dexevaluator = self.compute_loss_dexevaluator(out)

        return loss_dict_dexevaluator
    
    def eval_dexgangrasp_generator_loss(self, real_data):
        self.DexGANGrasp.eval()
        with torch.no_grad():
            n_samples = real_data["bps_object"].shape[0]
            Zgen = torch.randn((n_samples, self.DexGANGrasp.latentD), dtype=self.DexGANGrasp.dtype, device=self.DexGANGrasp.device)
            # Zgen = torch.randn((n_samples, self.DexGANGrasp.latentD), dtype=self.DexGANGrasp.dtype)
            # Run forward pass of dexgenerator and reconstruct the data
            y_fake = self.DexGANGrasp.generator(Zgen, real_data["bps_object"].to(self.DexGANGrasp.device))
            fake_rot_6D = y_fake["rot_6D"]
            fake_transl = y_fake["transl"]
            fake_joint_conf = y_fake["joint_conf"]
            y_fake["rot_matrix"] = utils.rot_matrix_from_ortho6d(y_fake["rot_6D"])

            fake_data = {
                "bps_object": real_data["bps_object"],
                "rot_matrix": fake_rot_6D,
                # "rot_matrix": y_fake["rot_matrix"],
                "rot_6D": fake_rot_6D,
                "transl": fake_transl,
                "joint_conf": fake_joint_conf
            }
            # Train Generator
            fake_data_gen = fake_data
            fake_data_gen["rot_matrix"] = y_fake["rot_matrix"]
            fake_score_gen = self.DexGANGrasp.discriminator(fake_data_gen)
            # Compute loss based on reconstructed data
            real_data["rot_matrix"] = real_data["rot_matrix"].view(real_data["bps_object"].shape[0], -1)
            is_wgan = self.is_wgan
            gen_loss = self.compute_loss_dexgangrasp_generator_wass if is_wgan else self.compute_loss_dexgangrasp_generator
            _, loss_dict_dexgenerator = gen_loss(real_data, fake_data, fake_score_gen)
            
        return loss_dict_dexgenerator

    def evaluate_grasps(self, bps, grasps, thresh=0.5, return_arr=True):
        """Receives n grasps together with bps encodings of queried object and evaluates the probability of success.

        Args:
            bps (np array): [description]
            grasps (dict): Dict holding the grasp information
            thresh (float, optional): Reject grasps with lower success p than this. Defaults to 0.5.

        Returns:
            p_success [tensor or arr, n_samples*1]: Success probability of each grasp
        """
        n_samples = grasps['rot_matrix'].shape[0]
        if len(bps.shape) > 1:
            bps = bps.squeeze()
        bps = np.tile(bps, (n_samples, 1))

        grasps['bps_object'] = bps
        grasps_t = utils.dict_to_tensor(grasps, device=self.device, dtype=self.dtype)

        p_success = self.DexEvaluator(grasps_t).squeeze()

        if return_arr:
            p_success = p_success.cpu().detach().numpy()

        return p_success

    def filter_grasps(self, bps, grasps, thresh=0.5, return_arr=True):
        """ Takes in grasps generated by the DexGenerator for a bps encoding of an object and removes every grasp#
        with predicted success probability less than thresh

        Args:
            bps (np array): Bps encoding of the object. n*4096
            grasps (dict): Dict holding the grasp information. keys: transl n*3, rot_matrix n*3*3, joint_conf n*15
            thresh (float, optional): Reject grasps with lower success p than this. Defaults to 0.5.
        """
        # start = time.time()
        n_samples = grasps['rot_matrix'].shape[0]
        if len(bps.shape) > 1:
            bps = bps.squeeze()
        bps = np.tile(bps, (n_samples, 1))

        grasps['bps_object'] = bps
        grasps_t = utils.dict_to_tensor(grasps, device=self.device, dtype=self.dtype)

        p_success = self.DexEvaluator(grasps_t).squeeze()  # tensor([...])
        sorted_score, indices = p_success.sort(descending=True)

        if sorted_score[0] < thresh:
            raise ValueError("In total predicted ", n_samples, " grasps, but best score ",
                             sorted_score[0], " is still lower than thresh", thresh)

        indices = indices[sorted_score > thresh]
        sorted_score = sorted_score[sorted_score > thresh]

        filt_grasps = {}

        # sort the grasps according to score and remove the grasps with score < thresh
        for k, v in grasps_t.items():
            index = indices.clone()
            dim = 0

            # Dynamically adjust dimensions for sorting
            while len(v.shape) > len(index.shape):
                dim += 1
                index = index[..., None]
                index = torch.cat(v.shape[dim] * (index, ), dim)

            # Sort grasps
            filt_grasps[k] = torch.gather(input=v, dim=0, index=index)

        # Cast to python (if required)
        if return_arr:
            filt_grasps = {k: v.cpu().detach().numpy() for k, v in filt_grasps.items()}
        # print("Filtering took: %.4f" % (time.time() - start))

        return filt_grasps
    
    def filter_grasps_discriminator(self, bps, grasps, thresh=0.5, return_arr=True):
        """ Takes in grasps generated by the DexGenerator for a bps encoding of an object and removes every grasp#
        with predicted success probability less than thresh

        Args:
            bps (np array): Bps encoding of the object. n*4096
            grasps (dict): Dict holding the grasp information. keys: transl n*3, rot_matrix n*3*3, joint_conf n*15
            thresh (float, optional): Reject grasps with lower success p than this. Defaults to 0.5.
        """
        # start = time.time()
        n_samples = grasps['rot_matrix'].shape[0]
        if len(bps.shape) > 1:
            bps = bps.squeeze()
        bps = np.tile(bps, (n_samples, 1))

        grasps['bps_object'] = bps
        grasps_t = utils.dict_to_tensor(grasps, device=self.device, dtype=self.dtype)

        p_success = self.DexGANGrasp.discriminator(grasps_t).squeeze()  # tensor([...])
        sorted_score, indices = p_success.sort(descending=True)

        if sorted_score[0] < thresh:
            raise ValueError("In total predicted ", n_samples, " grasps, but best score ",
                             sorted_score[0], " is still lower than thresh", thresh)

        indices = indices[sorted_score > thresh]
        sorted_score = sorted_score[sorted_score > thresh]

        filt_grasps = {}

        # sort the grasps according to score and remove the grasps with score < thresh
        for k, v in grasps_t.items():
            index = indices.clone()
            dim = 0

            # Dynamically adjust dimensions for sorting
            while len(v.shape) > len(index.shape):
                dim += 1
                index = index[..., None]
                index = torch.cat(v.shape[dim] * (index, ), dim)

            # Sort grasps
            filt_grasps[k] = torch.gather(input=v, dim=0, index=index)

        # Cast to python (if required)
        if return_arr:
            filt_grasps = {k: v.cpu().detach().numpy() for k, v in filt_grasps.items()}
        # print("Filtering took: %.4f" % (time.time() - start))

        return filt_grasps

    def generate_grasps(self, bps, n_samples, return_arr=True, z_offset=0.025):
        """Samples n grasps either from combining given bps encoding with z or sampling from random normal distribution.

        Args:
            bps (np arr) 1*4096: BPS encoding of the segmented object point cloud.
            n_samples (int): How many samples sould be generated.
            return_arr (bool): Whether to return results as np arr or tensor.

        Returns:
            rot_matrix (tensor or array) n_samples*3*3: palm rotation matrix
            transl (tensor or array) n_samples*3: 3D palm translation
            joint_conf (tensor or array) n_samples*15: 15 dim finger configuration
        """
        # turn np arr to tensor and repeat n_samples times
        if len(bps.shape) > 1:
            bps = bps.squeeze()
        bps = np.tile(bps, (n_samples, 1))
        bps_tensor = torch.tensor(bps, dtype=self.dtype, device=self.device)

        return self.DexGANGrasp.generate_poses(bps_tensor, return_arr=return_arr, z_offset=z_offset)

    def improve_grasps_gradient_based(self, data, last_success):
        """Apply small gradient steps to improve an initial grasp.

        Args:
            data (dict): Keys being bps_object, rot_matrix, transl, joint_conf describing sensor observation and grasp pose.
            last_success (None): Only exists to have the same interface as sampling-based refinement.

        Returns:
            p_success (tensor): success probability of grasps in data.
        """
        p_success = self.DexEvaluator(data)
        p_success.squeeze().backward(torch.ones(p_success.shape[0]).to(self.device))

        diff = np.abs(p_success.cpu().data.numpy().squeeze() -
                      data['label'].cpu().data.numpy().squeeze())

        # Adjust the alpha so that it won't update more than 1 cm. Gradient is only valid in small neighborhood.
        norm_transl = torch.norm(data['transl'].grad, p=2, dim=-1).to(self.device)
        alpha = torch.min(0.01 / norm_transl, torch.tensor(1.0, device=self.device))

        # Take a small step on each variable
        data['transl'].data += data['transl'].grad * alpha[:, None]
        data['rot_matrix'].data += data['rot_matrix'].grad * alpha[:, None, None]
        data['joint_conf'].data += data['joint_conf'].grad * alpha[:, None]

        return p_success.squeeze(), None

    def improve_grasps_sampling_based(self, pcs, grasp_eulers, grasp_trans, last_success=None):
        with torch.no_grad():
            if last_success is None:
                grasp_pcs = utils.control_points_from_rot_and_trans(grasp_eulers, grasp_trans,
                                                                    self.device)
                last_success = self.grasp_evaluator.eval_dexevaluator_accuracy(pcs, grasp_pcs)

            delta_t = 2 * (torch.rand(grasp_trans.shape).to(self.device) - 0.5)
            delta_t *= 0.02
            delta_euler_angles = (torch.rand(grasp_eulers.shape).to(self.device) - 0.5) * 2
            perturbed_translation = grasp_trans + delta_t
            perturbed_euler_angles = grasp_eulers + delta_euler_angles
            grasp_pcs = utils.control_points_from_rot_and_trans(perturbed_euler_angles,
                                                                perturbed_translation, self.device)

            perturbed_success = self.grasp_evaluator.eval_dexevaluator_accuracy(pcs, grasp_pcs)
            ratio = perturbed_success / torch.max(last_success,
                                                  torch.tensor(0.0001).to(self.device))

            mask = torch.rand(ratio.shape).to(self.device) <= ratio

            next_success = last_success
            ind = torch.where(mask)[0]
            next_success[ind] = perturbed_success[ind]
            grasp_trans[ind].data = perturbed_translation.data[ind]
            grasp_eulers[ind].data = perturbed_euler_angles.data[ind]
            return last_success.squeeze(), next_success

    def load_dexevaluator(self, epoch, load_path=None):
        """Load dexevaluator from disk and set to eval or train mode.
        """
        if epoch == -1:
            path = os.path.split(os.path.split(self.file_path)[0])[0]
            dirs = sorted(os.path.listdir(os.path.join(path, 'checkpoints')))
            load_path = os.path.join(path, dirs[-1], str(epoch) + '_eva_net.pt')
        else:
            if load_path is None:
                load_path = self.cfg["load_path"]
            load_path = os.path.join(load_path, str(epoch) + '_eva_net.pt')

        ckpt = torch.load(load_path, map_location=self.device)
        self.DexEvaluator.load_state_dict(ckpt['ffhevaluator_state_dict'])

        if self.cfg["is_train"]:
            self.optim_dexevaluator.load_state_dict(ckpt['optim_ffhevaluator_state_dict'])
            self.scheduler_dexevaluator.load_state_dict(ckpt['scheduler_ffhevaluator_state_dict'])
            self.cfg["load_epoch"] = ckpt['epoch']
            self.DexEvaluator.train()
        else:
            self.DexEvaluator.eval()

    def load_dexgenerator(self, epoch, load_path=None):
        """Load dexgenerator from disk and set to eval or train mode
        """
        if epoch == -1:
            path = os.path.split(os.path.split(self.file_path)[0])[0]
            dirs = sorted(os.path.listdir(os.path.join(path, 'checkpoints')))
            load_path = os.path.join(path, dirs[-1], str(epoch) + '_gen_net.pt')
        else:
            if load_path is None:
                load_path = self.cfg["load_path"]
            load_path = os.path.join(load_path, str(epoch) + '_gen_net.pt')

        ckpt = torch.load(load_path, map_location=self.device)
        self.DexGANGrasp.load_state_dict(ckpt['ffhgenerator_state_dict'])

        if self.cfg["is_train"]:
            print("Load TRAIN mode")
            # TODO: load state dict for genertor and discriminator
            self.optim_dexgenerator.load_state_dict(ckpt['optim_ffhgenerator_state_dict'])
            self.scheduler_dexgenerator.load_state_dict(ckpt['scheduler_ffhgenerator_state_dict'])
            self.DexGANGrasp.train()
        else:
            print("Network in EVAL mode")
            self.DexGANGrasp.eval()

    def refine_grasps(self, data, refine_method, num_refine_steps=10, dtype=torch.float32):
        """ Refine sampled and ranked grasps.

        Args:
            data (dict): Keys being bps_object, rot_matrix, transl, joint_conf describing sensor observation and grasp pose.
            refine_method (str): Choose gradient or sampling based refinement method.
            num_refine_steps (int, optional): How many steps of gradient-based refinement to apply. Defaults to 10.

        Returns:
            refined: [description]
        """
        start = time.time()
        data = utils.data_dict_to_dtype(data, dtype)
        if refine_method == "gradient":
            refine_fn = self.improve_grasps_gradient_based

            # Wrap input in Variable class, this way gradients are computed
            data['rot_matrix'] = torch.autograd.Variable(data['rot_matrix'].to(self.device),
                                                         requires_grad=True)
            data['transl'] = torch.autograd.Variable(data['transl'].to(self.device),
                                                     requires_grad=True)
            data['joint_conf'] = torch.autograd.Variable(data['joint_conf'].to(self.device),
                                                         requires_grad=True)

        else:
            refine_fn = self.improve_grasps_sampling_based

        refined_success = []
        refined_data = []
        refined_data.append(utils.grasp_numpy_from_data_dict(data))
        last_success = None
        for i in range(num_refine_steps):
            p_success, last_success = refine_fn(data, last_success)
            refined_success.append(p_success.cpu().data.numpy())
            refined_data.append(utils.grasp_numpy_from_data_dict(data))

        # we need to run the success on the final refined grasps
        refined_success.append(self.DexEvaluator(data).squeeze().cpu().data.numpy())

        # print('Refinement took: ' + str(time.time() - start))

        return refined_data, refined_success

    def save_dexevaluator(self, net_name, epoch):
        """ Save dexevaluator to disk

        Args:
            net_name (str): The name of the model.
            epoch (int): Current epoch.
        """
        save_path = os.path.join(self.cfg["save_dir"], net_name + '_eva_net.pt')
        if len(self.cfg["gpu_ids"]) > 1:
            dexevaluator_state_dict = self.DexEvaluator.module.cpu().state_dict()
        else:
            dexevaluator_state_dict = self.DexEvaluator.cpu().state_dict()
        torch.save(
            {
                'epoch': epoch,
                'ffhevaluator_state_dict': dexevaluator_state_dict,
                'optim_ffhevaluator_state_dict': self.optim_dexevaluator.state_dict(),
                'scheduler_ffhevaluator_state_dict': self.scheduler_dexevaluator.state_dict(),
            }, save_path)

        if torch.cuda.is_available():
            self.DexEvaluator.to(torch.device('cuda:{}'.format(self.cfg["gpu_ids"][0])))

    def save_dexgenerator(self, net_name, epoch):
        """ Save dexgenerator to disk

        Args:
            net_name (str): The name of the model.
            epoch (int): Current epoch.
        """
        save_path = os.path.join(self.cfg["save_dir"], net_name + '_gen_net.pt')
        if len(self.cfg["gpu_ids"]) > 1:
            dexgenerator_state_dict = self.DexGANGrasp.module.cpu().state_dict()
        else:
            dexgenerator_state_dict = self.DexGANGrasp.cpu().state_dict()
        torch.save(
            {
                'epoch': epoch,
                'ffhgenerator_state_dict': dexgenerator_state_dict,
                'optim_ffhgenerator_state_dict': self.optim_dexgenerator.state_dict(),
                'scheduler_ffhgenerator_state_dict': self.scheduler_dexgenerator.state_dict(),

                'optim_ffhgan_generator_state_dict': self.optim_dexgangrasp_generator.state_dict(),
                'scheduler_ffhgan_generator_state_dict': self.scheduler_dexgangrasp_generator.state_dict(),

                'optim_ffhgan_discriminator_state_dict': self.optim_dexgangrasp_discriminator.state_dict(),
                'scheduler_ffhgan_discriminator_state_dict': self.scheduler_dexgangrasp_discriminator.state_dict(),
            }, save_path)

        if torch.cuda.is_available():
            self.DexGANGrasp.to(torch.device('cuda:{}'.format(self.cfg["gpu_ids"][0])))

    def update_estop(self, eval_loss_dict):
        """"[Never used]"

        Args:
            eval_loss_dict (dict): Dict with all the relevant losses
        """
        if self.train_dexevaluator:
            if self.estop_dexevaluator(eval_loss_dict['total_loss_eva']):
                self.train_dexevaluator = False
        if self.train_dexgenerator:
            if self.estop_dexgenerator(eval_loss_dict['total_loss_gen']):
                self.train_dexgenerator = False

    def update_learning_rate(self, eval_loss_dict):
        """update learning rate (called once every epoch)"""
        if self.train_dexevaluator:
            self.scheduler_dexevaluator.step(eval_loss_dict['total_loss_eva'])
            lr_eva = self.optim_dexevaluator.param_groups[0]['lr']
            print('learning rate evaluator = %.7f' % lr_eva)

        if self.train_dexgenerator:
            self.scheduler_dexgenerator.step(eval_loss_dict['total_loss_gen'])
            lr_gen = self.optim_dexgenerator.param_groups[0]['lr']
            print('learning rate generator = %.7f' % lr_gen)

    def update_dexevaluator(self, data):
        # Make sure net is in train mode
        self.DexEvaluator.train()

        # Run forward pass of dexevaluator and predict grasp success
        out = self.DexEvaluator(data)

        # Compute loss based on reconstructed data
        total_loss_dexevaluator, loss_dict_dexevaluator = self.compute_loss_dexevaluator(out)

        # Zero gradients, backprop new loss gradient, run one step
        self.optim_dexevaluator.zero_grad()
        total_loss_dexevaluator.backward()
        self.optim_dexevaluator.step()

        # Return loss
        return loss_dict_dexevaluator
    
    def update_dexgangrasp(self, real_data, is_train_gen = True):
        """ Receives a dict with all the input data to the dexgenerator, sets the model input and runs one complete update step.
        """
        # Loss selection:
        is_wgan = self.is_wgan
        disc_loss = self.compute_loss_dexgangrasp_discriminator_wass if is_wgan else self.compute_loss_dexgangrasp_discriminator
        gen_loss = self.compute_loss_dexgangrasp_generator_wass if is_wgan else self.compute_loss_dexgangrasp_generator
        
        # Make sure net is in train mode
        self.DexGANGrasp.train()

        #### Train Discriminator ####
        n_samples = real_data["bps_object"].shape[0]
        Zgen = torch.randn((n_samples, self.DexGANGrasp.latentD), dtype=self.DexGANGrasp.dtype, device=self.DexGANGrasp.device)
        # Run forward pass of dexgenerator and reconstruct the data
        y_fake = self.DexGANGrasp.generator(Zgen, real_data["bps_object"].to(self.DexGANGrasp.device))
        fake_rot_6D = y_fake["rot_6D"]
        fake_transl = y_fake["transl"]
        fake_joint_conf = y_fake["joint_conf"]
        y_fake["rot_matrix"] = utils.rot_matrix_from_ortho6d(y_fake["rot_6D"])

        fake_data_disc = {
            "bps_object": real_data["bps_object"],
            # "rot_matrix": fake_rot_6D.detach(),
            "rot_matrix": y_fake["rot_matrix"].detach(),
            # "rot_6D": fake_rot_6D.detach(),
            "transl": fake_transl.detach(),
            "joint_conf": fake_joint_conf.detach()
        }

        # Pass both real and fake data through the discriminator
        real_score = self.DexGANGrasp.discriminator(real_data)
        fake_score = self.DexGANGrasp.discriminator(fake_data_disc)
        # fake_score = self.DexGANGrasp.discriminator(fake_data)
        total_loss_disc, loss_dict_disc = disc_loss(real_score, fake_score, real_data, fake_data_disc)
        self.optim_dexgangrasp_discriminator.zero_grad()
        total_loss_disc.backward()
        self.optim_dexgangrasp_discriminator.step()

        #### Train Generator ####
        if is_train_gen:
            fake_data = {
                "bps_object": real_data["bps_object"],
                "rot_matrix": fake_rot_6D,
                # "rot_matrix": y_fake["rot_matrix"],
                "rot_6D": fake_rot_6D,
                "transl": fake_transl,
                "joint_conf": fake_joint_conf
            }
            fake_data_gen = fake_data
            fake_data_gen["rot_matrix"] = y_fake["rot_matrix"]
            fake_score_gen = self.DexGANGrasp.discriminator(fake_data_gen)
            # Compute loss based on reconstructed data
            real_data["rot_matrix"] = real_data["rot_matrix"].view(real_data["bps_object"].shape[0], -1)
            total_loss_gen, loss_dict_gen = gen_loss(real_data, fake_data, fake_score_gen)

            # Zero gradients, backprop new loss gradient, run one step
            self.optim_dexgangrasp_generator.zero_grad()
            total_loss_gen.backward()
            self.optim_dexgangrasp_generator.step()
        else:
            loss_dict_gen = self.last_loss_dict_gen
        # Return the loss
        # loss_dict = loss_dict_disc | loss_dict_gen
        loss_dict = loss_dict_disc.copy()
        loss_dict.update(loss_dict_gen)
        return loss_dict