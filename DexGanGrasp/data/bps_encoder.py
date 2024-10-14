""" This server only runs under python 3. It reads a segmented pcd from disk and also a BPS.
When it is called it will compute the BPS encoding of the object.
"""
import bps_torch.bps as b_torch
from bps_torch.utils import to_np
import numpy as np
import open3d as o3d
import torch
import os


class BPSEncoder():
    # os.path.join(rospy.get_param('ffhnet_path'), 'models/basis_point_set.npy')
    def __init__(self, cfg, bps_path,vis=False):
        self.bps_np = np.load(bps_path)
        self.device = torch.device('cuda:{}'.format(
            cfg["gpu_ids"][0])) if torch.cuda.is_available() else torch.device('cpu')
        print("bps encoder device", self.device)
        self.bps = b_torch.bps_torch(custom_basis=self.bps_np,device=self.device)
        self.VISUALIZE = vis

    @staticmethod
    def generate_new_bps(base_path):
        bps = b_torch.bps_torch(bps_type='random_uniform', n_bps_points=4096, radius=0.2, n_dims=3)
        # Save the "ground_truth" bps
        np.save(os.path.join(base_path, 'basis_point_set.npy'), to_np(bps.bps.squeeze()))

    def encode_pcd_with_bps(self, obj_pcd):
        # obj_pcd = o3d.io.read_point_cloud(self.pcd_path)
        obj_tensor = torch.from_numpy(np.asarray(obj_pcd.points))
        obj_tensor = obj_tensor.to(device=self.device).contiguous()

        enc_dict = self.bps.encode(obj_tensor)
        enc_np = enc_dict['dists'].cpu().detach().numpy()
        # np.save(self.enc_path, enc_np)

        if self.VISUALIZE:
            bps_pcd = o3d.geometry.PointCloud()
            bps_pcd.points = o3d.utility.Vector3dVector(self.bps_np)
            # bps_pcd.colors = o3d.utility.Vector3dVector(0.3 * np.ones(self.bps_np.shape))
            colors_np = np.zeros_like(self.bps_np)
            colors_np[:,1] = 1
            bps_pcd.colors = o3d.utility.Vector3dVector(colors_np)

            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1)
            o3d.visualization.draw_geometries([obj_pcd, bps_pcd, origin])

        # if all dists are greater, pcd is too far from bps
        assert enc_np.min() < 0.1, 'The pcd might not be in centered in origin !' + enc_np.min()

        # squeece shape from [x,1,y] to [x,y] to align the shape with other inputs to the FFHNEt
        enc_np = np.squeeze(enc_np)

        return enc_np


if __name__ == '__main__':
    BPSEncoder.generate_new_bps(base_path="/home/vm/new_data_full_lite")

    bps = BPSEncoder(bps_path='/home/vm/new_data_full_lite/basis_point_set.npy')
    pcd_path = '/home/vm/new_data_full_lite/point_clouds/bigbird_3m_high_tack_spray_adhesive/bigbird_3m_high_tack_spray_adhesive_pcd000_multi.pcd'
    obj_pcd = o3d.io.read_point_cloud(pcd_path)
    bps.encode_pcd_with_bps(obj_pcd)

    # vis for iros 2023
    path = '/home/vm/new_data_full/point_clouds/kit_BathDetergent/kit_BathDetergent_pcd008_multi.pcd'
    obj_pcd = o3d.io.read_point_cloud(path)
    obj_pcd.translate(-1*obj_pcd.get_center())
    bps = BPSEncoder(bps_path='/home/vm/new_data_full_lite/basis_point_set.npy',vis=True)
    bps.encode_pcd_with_bps(obj_pcd)

