import time
import numpy as np
import torch
from FFHNet.utils import utils
from torch import nn
from torch.optim import lr_scheduler

from FFHNet.utils import utils
from FFHNet.models import losses


def get_scheduler(optimizer, cfg):
    if cfg["lr_policy"] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg["lr_decay_iters"], gamma=0.1)
    elif cfg["lr_policy"] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=cfg["threshold_lr_policy_plateau"],
                                                   patience=cfg["patience_lr_policy_plateau"])
    else:
        raise NotImplementedError('Scheduler not implemented.')
    return scheduler


class ResBlock(nn.Module):
    def __init__(self, Fin, Fout, n_neurons=256):
        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout

class Generator(nn.Module):
    def __init__(self,
                 cfg,
                 n_neurons=512,
                 in_bps=4096,
                 in_pose=9 + 3,
                 dtype=torch.float32,
                 **kwargs):

        super(Generator, self).__init__()

        self.cfg = cfg
        in_pose += cfg["n_hand_joints"]
        self.latentD = cfg["latentD"]

        self.gen_bn1 = nn.BatchNorm1d(in_bps)
        self.gen_rb1 = ResBlock(self.latentD + in_bps, n_neurons)
        self.gen_rb2 = ResBlock(n_neurons + self.latentD + in_bps, n_neurons)

        self.gen_joint_conf = nn.Linear(n_neurons, cfg["n_hand_joints"])
        self.gen_rot = nn.Linear(n_neurons, 6)
        self.gen_transl = nn.Linear(n_neurons, 3)

        if self.cfg["is_train"]:
            print("Generator currently in TRAIN mode!")
            self.train()
        else:
            print("Generator currently in EVAL mode!")
            self.eval()

        self.dtype = dtype
        
    def forward(self, Zin, bps_object):

        bs = Zin.shape[0]
        o_bps = self.gen_bn1(bps_object.contiguous())

        X0 = torch.cat([Zin, o_bps], dim=1)
        X = self.gen_rb1(X0, True)
        X = self.gen_rb2(torch.cat([X0, X], dim=1), True)
        # print('x gen', X.shape)
        joint_conf = self.gen_joint_conf(X)
        rot_6D = self.gen_rot(X)
        transl = self.gen_transl(X)

        results = {"rot_6D": rot_6D, "transl": transl, "joint_conf": joint_conf, "z": Zin}

        return results
    
class Discriminator(nn.Module):
    def __init__(self,
                 cfg,
                 n_neurons=512,
                 in_bps=4096,
                 in_pose=9 + 3,
                 dtype=torch.float32,
                 **kwargs):

        super(Discriminator, self).__init__()

        self.cfg = cfg
        in_pose += cfg["n_hand_joints"]
        self.disc_bn1 = nn.BatchNorm1d(in_bps + in_pose)
        self.disc_rb1 = ResBlock(in_bps + in_pose, n_neurons)
        # why input in_bps again here?
        self.disc_rb2 = ResBlock(n_neurons + in_bps + in_pose, n_neurons)
        self.out_success = nn.Linear(n_neurons, 1)
        self.sigmoid = nn.Sigmoid()
        if self.cfg["is_train"]:
            print("Discriminator currently in TRAIN mode!")
            self.train()
        else:
            print("Discriminator currently in EVAL mode!")
            self.eval()

        self.dtype = dtype
        
    def set_input(self, data):
        """ Bring input tensors to correct dtype and device. Set whether gradient is required depending on
        we are in train or eval mode.
        """
        rot_matrix = data["rot_matrix"].to(dtype=self.dtype, device=self.device)
        transl = data["transl"].to(dtype=self.dtype, device=self.device)
        joint_conf = data["joint_conf"].to(dtype=self.dtype, device=self.device)
        bps_object = data["bps_object"].to(dtype=self.dtype, device=self.device).contiguous()

        self.rot_matrix = rot_matrix.requires_grad_(self.cfg["is_train"])
        self.transl = transl.requires_grad_(self.cfg["is_train"])
        self.joint_conf = joint_conf.requires_grad_(self.cfg["is_train"])
        self.bps_object = bps_object.requires_grad_(self.cfg["is_train"])

        self.rot_matrix = self.rot_matrix.view(self.bps_object.shape[0], -1)

    def forward(self, data):
        self.set_input(data)
        X = torch.cat([self.bps_object, self.rot_matrix, self.transl, self.joint_conf], dim=1)

        X0 = self.disc_bn1(X)
        X = self.disc_rb1(X0, True)
        X = self.disc_rb2(torch.cat([X0, X], dim=1), True)
        X = self.out_success(X)

        p_real = self.sigmoid(X)

        return p_real
    
class FFHGAN(nn.Module):
    def __init__(self,
                 cfg,
                 n_neurons=512,
                 in_bps=4096,
                 in_pose=9 + 3,
                 dtype=torch.float32,
                 **kwargs):

        super(FFHGAN, self).__init__()

        self.cfg = cfg
        in_pose += cfg["n_hand_joints"]
        self.device = torch.device('cuda:{}'.format(
            cfg["gpu_ids"][0])) if torch.cuda.is_available() else torch.device('cpu')

        self.latentD = cfg["latentD"]
        self.discriminator = Discriminator(
            cfg,
            n_neurons,
            in_bps,
            in_pose,
            dtype)
        self.discriminator.to(self.device)

        self.generator = Generator(
            cfg,
            n_neurons,
            in_bps,
            in_pose,
            dtype)
        self.generator.to(self.device)
        self.discriminator.device = self.device
        self.generator.device = self.device
        if self.cfg["is_train"]:
            print("FFHGAN currently in TRAIN mode!")
            self.train()
        else:
            print("FFHGAN currently in EVAL mode!")
            self.eval()

        self.dtype = dtype
    
    def forward(self, Zin, bps_object, real_data):
        # Generate fake data using the generator
        fake_results = self.generator(Zin, bps_object)
        fake_rot_6D = fake_results["rot_6D"]
        fake_transl = fake_results["transl"]
        fake_joint_conf = fake_results["joint_conf"]
        fake_data = {
            "bps_object": bps_object,
            "rot_matrix": fake_rot_6D,
            "transl": fake_transl,
            "joint_conf": fake_joint_conf
        }

        # Pass both real and fake data through the discriminator
        real_p_success = self.discriminator(real_data)
        fake_p_success = self.discriminator(fake_data)

        return fake_results, real_p_success, fake_p_success

    def generate_poses(self, bps_object, return_arr=False, seed=None, sample_uniform=False):
        """[summary]

        Args:
            bps_object (tensor): BPS encoding of object point cloud.
            return_arr (bool): Returns np array if True
            seed (int, optional): np random seed. Defaults to None.
        Returns:
            results (dict): keys being 1.rot_matrix, 2.transl, 3.joint_conf
        """
        start = time.time()
        n_samples = bps_object.shape[0]
        self.eval()
        with torch.no_grad():
            if not sample_uniform:
                Zgen = torch.randn((n_samples, self.latentD), dtype=self.dtype, device=self.device)
            else:
                Zgen = 8 * torch.rand(
                    (n_samples, self.latentD), dtype=self.dtype, device=self.device) - 4

        results = self.generator(Zgen, bps_object)

        # Transforms rot_6D to rot_matrix
        results['rot_matrix'] = utils.rot_matrix_from_ortho6d(results.pop('rot_6D'))

        if return_arr:
            for k, v in results.items():
                results[k] = v.cpu().detach().numpy()
        print("Sampling took: %.3f" % (time.time() - start))
        return results
    
    def generate_grasps(self, bps, n_samples, return_arr=True):
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

        return self.generate_poses(bps_tensor, return_arr=return_arr)

class FFHGenerator(nn.Module):
    def __init__(self,
                 cfg,
                 n_neurons=512,
                 in_bps=4096,
                 in_pose=9 + 3,
                 dtype=torch.float32,
                 **kwargs):

        super(FFHGenerator, self).__init__()

        self.cfg = cfg
        in_pose += cfg["n_hand_joints"]
        self.latentD = cfg["latentD"]

        self.enc_bn1 = nn.BatchNorm1d(in_bps + in_pose)
        self.enc_rb1 = ResBlock(in_bps + in_pose, n_neurons)
        # why input in_bps again here?
        self.enc_rb2 = ResBlock(n_neurons + in_bps + in_pose, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, self.latentD)
        self.enc_logvar = nn.Linear(n_neurons, self.latentD)
        self.do = nn.Dropout(p=.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_bps)
        self.dec_rb1 = ResBlock(self.latentD + in_bps, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + self.latentD + in_bps, n_neurons)

        self.dec_joint_conf = nn.Linear(n_neurons, cfg["n_hand_joints"])
        self.dec_rot = nn.Linear(n_neurons, 6)
        self.dec_transl = nn.Linear(n_neurons, 3)

        if self.cfg["is_train"]:
            print("FFHGenerator currently in TRAIN mode!")
            self.train()
        else:
            print("FFHGenerator currently in EVAL mode!")
            self.eval()

        self.dtype = dtype
        self.device = torch.device('cuda:{}'.format(
            cfg["gpu_ids"][0])) if torch.cuda.is_available() else torch.device('cpu')

    def decode(self, Zin, bps_object):

        bs = Zin.shape[0]
        o_bps = self.dec_bn1(bps_object)

        X0 = torch.cat([Zin, o_bps], dim=1)
        X = self.dec_rb1(X0, True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), True)

        joint_conf = self.dec_joint_conf(X)
        rot_6D = self.dec_rot(X)
        transl = self.dec_transl(X)

        results = {"rot_6D": rot_6D, "transl": transl, "joint_conf": joint_conf, "z": Zin}

        return results

    def encode(self, data):
        self.set_input(data)
        X = torch.cat([self.bps_object, self.rot_matrix, self.transl, self.joint_conf], dim=1)

        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0, True)
        X = self.enc_rb2(torch.cat([X0, X], dim=1), True)

        return self.enc_mu(X), self.enc_logvar(X)

    def forward(self, data):
        # Encode data, get mean and logvar
        mu, logvar = self.encode(data)

        std = logvar.exp().pow(0.5)
        q_z = torch.distributions.normal.Normal(mu, std)
        z = q_z.rsample()

        data_recon = self.decode(z, self.bps_object)
        results = {'mu': mu, 'logvar': logvar}
        results.update(data_recon)

        return results

    def generate_poses(self, bps_object, return_arr=False, seed=None, sample_uniform=False):
        """[summary]

        Args:
            bps_object (tensor): BPS encoding of object point cloud.
            return_arr (bool): Returns np array if True
            seed (int, optional): np random seed. Defaults to None.

        Returns:
            results (dict): keys being 1.rot_matrix, 2.transl, 3.joint_conf
        """
        start = time.time()
        n_samples = bps_object.shape[0]
        self.eval()
        with torch.no_grad():
            if not sample_uniform:
                Zgen = torch.randn((n_samples, self.latentD), dtype=self.dtype, device=self.device)
            else:
                Zgen = 8 * torch.rand(
                    (n_samples, self.latentD), dtype=self.dtype, device=self.device) - 4

        results = self.decode(Zgen, bps_object)

        # Transforms rot_6D to rot_matrix
        results['rot_matrix'] = utils.rot_matrix_from_ortho6d(results.pop('rot_6D'))

        if return_arr:
            for k, v in results.items():
                results[k] = v.cpu().detach().numpy()
        print("Sampling took: %.3f" % (time.time() - start))
        return results

    def set_input(self, data):
        """ Bring input tensors to correct dtype and device. Set whether gradient is required depending on
        we are in train or eval mode.
        """
        rot_matrix = data["rot_matrix"].to(dtype=self.dtype, device=self.device)
        transl = data["transl"].to(dtype=self.dtype, device=self.device)
        joint_conf = data["joint_conf"].to(dtype=self.dtype, device=self.device)
        bps_object = data["bps_object"].to(dtype=self.dtype, device=self.device).contiguous()

        self.rot_matrix = rot_matrix.requires_grad_(self.cfg["is_train"])
        self.transl = transl.requires_grad_(self.cfg["is_train"])
        self.joint_conf = joint_conf.requires_grad_(self.cfg["is_train"])
        self.bps_object = bps_object.requires_grad_(self.cfg["is_train"])

        self.rot_matrix = self.rot_matrix.view(self.bps_object.shape[0], -1)


class FFHEvaluator(nn.Module):
    def __init__(self,
                 cfg,
                 n_neurons=512,
                 in_bps=4096,
                 in_pose=9 + 3,
                 dtype=torch.float32,
                 **kwargs):
        super(FFHEvaluator, self).__init__()
        self.cfg = cfg

        self.bn1 = nn.BatchNorm1d(in_bps + in_pose)
        self.rb1 = ResBlock(in_bps + in_pose, n_neurons)
        self.rb2 = ResBlock(in_bps + in_pose + n_neurons, n_neurons)
        self.rb3 = ResBlock(in_bps + in_pose + n_neurons, n_neurons)
        self.out_success = nn.Linear(n_neurons, 1)
        self.dout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

        if self.cfg["is_train"]:
            print("FFHEvaluator currently in TRAIN mode!")
            self.train()
        else:
            print("FFHEvaluator currently in EVAL mode!")
            self.eval()
        self.dtype = torch.float32
        self.device = torch.device('cuda:{}'.format(
            cfg["gpu_ids"][0])) if torch.cuda.is_available() else torch.device('cpu')

    def set_input(self, data):
        self.rot_matrix = data["rot_matrix"].to(dtype=self.dtype, device=self.device)
        self.transl = data["transl"].to(dtype=self.dtype, device=self.device)
        self.bps_object = data["bps_object"].to(dtype=self.dtype, device=self.device).contiguous()
        if 'label' in data.keys():
            self.gt_label = data["label"].to(dtype=self.dtype, device=self.device).unsqueeze(-1)
        self.rot_matrix = self.rot_matrix.view(self.bps_object.shape[0], -1)

    def forward(self, data):
        """Run one forward iteration to evaluate the success probability of given grasps

        Args:
            data (dict): keys should be rot_matrix, transl, joint_conf, bps_object,

        Returns:
            p_success (tensor, batch_size*1): Probability that a grasp will be successful.
        """
        self.set_input(data)
        X = torch.cat([self.bps_object, self.rot_matrix, self.transl], dim=1)

        X0 = self.bn1(X)
        X = self.rb1(X0)
        X = self.dout(X)
        X = self.rb2(torch.cat([X, X0], dim=1))
        X = self.dout(X)
        X = self.rb3(torch.cat([X, X0], dim=1))
        X = self.dout(X)
        X = self.out_success(X)

        p_success = self.sigmoid(X)

        return p_success