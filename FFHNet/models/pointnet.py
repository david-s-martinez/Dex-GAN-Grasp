from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))
from FFHNet.models.networks import ResBlock


class PointNetGenerator(nn.Module):
    def __init__(self,
                 cfg,
                 n_neurons=512,
                 in_pcd=1024,
                 in_pose=9 + 3 + 15,
                 dtype=torch.float32,
                 **kwargs):

        super(PointNetGenerator, self).__init__()

        # [in_pcd x 3]
        self.conv1 = nn.Conv1d(in_channels=in_pcd,
                               out_channels=64, kernel_size=1)
        # [64 x 3]
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv5 = nn.Conv1d(
            in_channels=128, out_channels=1024, kernel_size=1) # [1024x3]

        self.fc1 = nn.Linear(in_features=1024, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=1024)

        # batch norm
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.cfg = cfg

        self.latentD = cfg["latentD"]

        self.grasp_bn1 = nn.BatchNorm1d(in_pose) # [27]
        self.grasp_rb1 = ResBlock(in_pose, n_neurons)
        # self.enc_rb2 = ResBlock(n_neurons + in_pose, n_neurons)
        self.grasp_rb2 = ResBlock(n_neurons, n_neurons)

        self.enc_rb1 = ResBlock(n_neurons + 1024, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, self.latentD)
        self.enc_logvar = nn.Linear(n_neurons, self.latentD)
        self.do = nn.Dropout(p=.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_pcd)
        self.dec_rb1 = ResBlock(self.latentD + in_pcd, n_neurons)

        self.dec_rb2 = ResBlock(n_neurons + self.latentD + in_pcd, n_neurons)

        self.dec_joint_conf = nn.Linear(n_neurons, 15)
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

    def decode(self, Zin, pcd_array):

        # decoder
        # x = F.relu(self.bn3(self.fc1(x)))
        # x = F.relu(self.bn3(self.fc2(x)))
        # reconstructed_points = self.fc3(x)

        bs = Zin.shape[0]
        x = F.relu(self.bn3(self.fc1(pcd_array)))
        x = F.relu(self.bn3(self.fc2(pcd_array)))

        o_bps = self.dec_bn1(x)

        X0 = torch.cat([Zin, o_bps], dim=1)
        X = self.dec_rb1(X0, True)
        # X = self.dec_rb2(torch.cat([X0, X], dim=1), True)
        X = self.dec_rb2(X, True)

        joint_conf = self.dec_joint_conf(X)
        rot_6D = self.dec_rot(X)
        transl = self.dec_transl(X)

        results = {"rot_6D": rot_6D, "transl": transl,
                   "joint_conf": joint_conf, "z": Zin}

        return results

    def encode(self,data):
        self.set_input(data)
        x = F.relu(self.bn1(self.conv1(self.pcd_array)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.bn3(self.conv5(x)))

        # do max pooling
        # what's x shape?? [batch, 1024,3]
        x = torch.max(x, 2, keepdim=True)[0] # [batch,1024,1]
        global_feat_pcd = x.view(-1, 1024) # view=reshapre, [batch, 1024]
        # get the global embedding

        X = torch.cat([self.rot_matrix,
                      self.transl, self.joint_conf], dim=1)
        X0 = self.grasp_bn1(X)
        X = self.grasp_rb1(X0, True)
        grasp_feat = self.grasp_rb2(X, True)

        feature_cat = torch.cat([global_feat_pcd, grasp_feat],dim=1)
        X = self.enc_rb1(feature_cat, True)
        return self.enc_mu(X), self.enc_logvar(X)

    def forward(self, data):
        # Encode data, get mean and logvar
        print("forward data size:", data.size())
        mu, logvar = self.encode(data)

        std = logvar.exp().pow(0.5)
        # what's the size of laten space distribution?
        q_z = torch.distributions.normal.Normal(mu, std)
        z = q_z.rsample()

        data_recon = self.decode(z, self.pcd_array)
        results = {'mu': mu, 'logvar': logvar}
        results.update(data_recon)

        return results

    def generate_poses(self, pcd_array, return_arr=False, seed=None, sample_uniform=False):
        """[summary]

        Args:
            pcd_array (tensor): BPS encoding of object point cloud.
            return_arr (bool): Returns np array if True
            seed (int, optional): np random seed. Defaults to None.

        Returns:
            results (dict): keys being 1.rot_matrix, 2.transl, 3.joint_conf
        """
        start = time.time()
        n_samples = pcd_array.shape[0]
        self.eval()
        with torch.no_grad():
            if not sample_uniform:
                Zgen = torch.randn((n_samples, self.latentD),
                                   dtype=self.dtype, device=self.device)
            else:
                Zgen = 8 * torch.rand(
                    (n_samples, self.latentD), dtype=self.dtype, device=self.device) - 4

        results = self.decode(Zgen, pcd_array)

        # Transforms rot_6D to rot_matrix
        results['rot_matrix'] = utils.rot_matrix_from_ortho6d(
            results.pop('rot_6D'))

        if return_arr:
            for k, v in results.items():
                results[k] = v.cpu().detach().numpy()
        print("Sampling took: %.3f" % (time.time() - start))
        return results

    def set_input(self, data):
        """ Bring input tensors to correct dtype and device. Set whether gradient is required depending on
        we are in train or eval mode.
        """
        rot_matrix = data["rot_matrix"].to(
            dtype=self.dtype, device=self.device)
        transl = data["transl"].to(dtype=self.dtype, device=self.device)
        joint_conf = data["joint_conf"].to(
            dtype=self.dtype, device=self.device)
        pcd_array = data["pcd_array"].to(
            dtype=self.dtype, device=self.device).contiguous()

        self.rot_matrix = rot_matrix.requires_grad_(self.cfg["is_train"])
        self.transl = transl.requires_grad_(self.cfg["is_train"])
        self.joint_conf = joint_conf.requires_grad_(self.cfg["is_train"])
        self.pcd_array = pcd_array.requires_grad_(self.cfg["is_train"])

        self.rot_matrix = self.rot_matrix.view(self.pcd_array.shape[0], -1)

class PointNetEvaluator(nn.Module):
    def __init__(self,
                 cfg,
                 n_neurons=512,
                 in_pcd=1024,
                 in_pose=9 + 3,
                 dtype=torch.float32,
                 **kwargs):
        super(PointNetEvaluator, self).__init__()
        self.cfg = cfg
        self.dtype = torch.float32
        self.device = torch.device('cuda:{}'.format(
            cfg["gpu_ids"][0])) if torch.cuda.is_available() else torch.device('cpu')

        self.conv1 = nn.Conv1d(in_channels=in_pcd,
                               out_channels=64, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        # self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv5 = nn.Conv1d(
            in_channels=128, out_channels=1024, kernel_size=1)

        # self.fc1 = nn.Linear(in_features=1024, out_features=1024)
        # self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        # self.fc3 = nn.Linear(in_features=1024, out_features=1024)

        # batch norm
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.grasp_bn1 = nn.BatchNorm1d(in_pose)
        self.grasp_rb1 = ResBlock(in_pose, n_neurons)
        # self.grasp_rb2 = ResBlock(n_neurons, n_neurons)
        self.total_rb1 = ResBlock(1024 + n_neurons, n_neurons)
        # self.total_rb2 = ResBlock(n_neurons, n_neurons)
        self.out_success = nn.Linear(n_neurons, 1)
        self.dout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

        if self.cfg["is_train"]:
            print("FFHEvaluator currently in TRAIN mode!")
            self.train()
        else:
            print("FFHEvaluator currently in EVAL mode!")
            self.eval()


    def set_input(self, data):
        self.rot_matrix = data["rot_matrix"].to(dtype=self.dtype, device=self.device)
        self.transl = data["transl"].to(dtype=self.dtype, device=self.device)
        self.pcd_array = data["pcd_array"].to(dtype=self.dtype, device=self.device).contiguous()
        if 'label' in data.keys():
            self.gt_label = data["label"].to(dtype=self.dtype, device=self.device).unsqueeze(-1)
        self.rot_matrix = self.rot_matrix.view(self.pcd_array.shape[0], -1)

    def forward(self, data):
        """Run one forward iteration to evaluate the success probability of given grasps

        Args:
            data (dict): keys should be rot_matrix, transl, joint_conf, pcd_array,

        Returns:
            p_success (tensor, batch_size*1): Probability that a grasp will be successful.
        """
        self.set_input(data)
        x = F.relu(self.bn1(self.conv1(self.pcd_array)))
        # x = F.relu(self.bn1(self.conv2(x)))
        # x = F.relu(self.bn1(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.bn3(self.conv5(x)))

        # do max pooling
        # x: [1000,1024,1]
        x = torch.max(x, 2, keepdim=True)[0]
        global_feat_pcd = x.view(-1, 1024)
        # get the global embedding

        X = torch.cat([self.rot_matrix,
                      self.transl], dim=1) # [1000,9] , [1000,3] -> [1000,12]
        X0 = self.grasp_bn1(X) # [1000,12]
        X = self.grasp_rb1(X0, True) # [1000,512]
        grasp_feat = X
        # grasp_feat = self.grasp_rb2(X, True)

        feature_cat = torch.cat([global_feat_pcd, grasp_feat],dim=1) # [1000,1536]
        X = self.total_rb1(feature_cat, True) # [1000,512]
        X = self.dout(X)
        # X = self.total_rb2(X)
        # X = self.dout(X) # [1000,512]

        X = self.out_success(X) # [1000,1]

        p_success = self.sigmoid(X)

        return p_success


class ResBlockOrigin(nn.Module):
    def __init__(self, Fin, Fout, n_neurons=256):
        super().__init__()

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

    def resblock(self,x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return x

    def forward(self, x):
        return x + self.resblock(x)


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetCollDetr(nn.Module):
    def __init__(self,
                 cfg,
                 in_pose=9 + 3,
                 feature_transform=False):
        super().__init__()
        self.cfg = cfg
        self.dtype = torch.float32
        self.device = torch.device('cuda:{}'.format(
            cfg["gpu_ids"][0])) if torch.cuda.is_available() else torch.device('cpu')

        self.dim_grasp_feat = 64
        self.dim_pcd_feat = 1024
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)

        self.fc1 = nn.Linear(in_features=self.dim_pcd_feat + self.dim_grasp_feat, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=1)

        self.grasp_bn1 = nn.BatchNorm1d(self.dim_grasp_feat)
        self.grasp_fc1 = nn.Linear(in_features=in_pose,out_features=self.dim_grasp_feat)
        self.grasp_rb1 = ResBlockOrigin(self.dim_grasp_feat,self.dim_grasp_feat)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

        if self.cfg["is_train"]:
            print("FFHCollDetr currently in TRAIN mode!")
            self.train()
        else:
            print("FFHCollDetr currently in EVAL mode!")
            self.eval()


    def set_input(self, data):
        self.rot_matrix = data["rot_matrix"].to(dtype=self.dtype, device=self.device)
        self.transl = data["transl"].to(dtype=self.dtype, device=self.device)
        self.pcd_array = data["pcd_array"].to(dtype=self.dtype, device=self.device).contiguous()
        if 'label' in data.keys():
            self.gt_label = data["label"].to(dtype=self.dtype, device=self.device).unsqueeze(-1)
        self.rot_matrix = self.rot_matrix.view(self.pcd_array.shape[0], -1)

    def forward(self, data):
        """Run one forward iteration to evaluate the success probability of given grasps

        Args:
            data (dict): keys should be rot_matrix, transl, joint_conf, pcd_array,

        Returns:
            p_success (tensor, batch_size*1): Probability that a grasp will be successful.
        """
        self.set_input(data)
        global_feat_pcd, trans, trans_feat = self.feat(self.pcd_array)


        X = torch.cat([self.rot_matrix,
                      self.transl], dim=1) # [1000,9] , [1000,3] -> [1000,12]
        X0 = self.grasp_bn1(self.grasp_fc1(X)) # [1000,12]
        grasp_feat = self.grasp_rb1(X0) # [1000,512]

        # concatenate the features from pcd and pose
        feature_cat = torch.cat([global_feat_pcd, grasp_feat],dim=1) # [1024,64]

        x = F.relu(self.bn1(self.fc1(feature_cat)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        p_success = F.log_softmax(x, dim=1)

        return p_success

class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
