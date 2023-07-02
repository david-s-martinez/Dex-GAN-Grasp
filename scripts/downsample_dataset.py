"""Downsample the original point cloud dataset into fixed shape point cloud of [1024,3]
"""
import numpy as np
import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
print(sys.path)
from FFHNet.data.ffhevaluator_data_set import FFHEvaluatorDataSet, FFHEvaluatorPCDDataSet
from FFHNet.data.ffhcollision_data_set import FFHCollDetrPCDDataSet
import open3d as o3d
from FFHNet.config.config import Config
path = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.split(path)[0]

path = os.path.join(REPO_PATH, "FFHNet/config/config_ffhcol_arlx0100_qf_pointnet.yaml")
config = Config(path)
cfg = config.parse()
# by default cfg["ds_name"] is 'train' for training set
# switch manually to 'eval' for evaluation set
#cfg["ds_name"] = 'eval'
gds = FFHCollDetrPCDDataSet(cfg, eval=False)
NUM_POINTS = 2048


def downsample_pcd(pcd, vis=False):
    # Downsample the point cloud
    # this requires open3d >= 0.17
    number_pcd = len(np.asarray(pcd.points))
    ds_pcd = pcd.farthest_point_down_sample(min(number_pcd, NUM_POINTS))
    pcd_arr = np.asarray(ds_pcd.points)  # shape [x,3]

    np.random.shuffle(pcd_arr)
    if pcd_arr.shape[0] < NUM_POINTS:
        empty_arr = np.zeros((NUM_POINTS, 3))
        empty_arr[:pcd_arr.shape[0]] = pcd_arr
        pcd_arr_fixed = empty_arr
    else:
        pcd_arr_fixed = pcd_arr[0:NUM_POINTS]

    assert pcd_arr_fixed.shape == (NUM_POINTS, 3)
    ds_pcd = o3d.geometry.PointCloud()
    ds_pcd.points = o3d.utility.Vector3dVector(pcd_arr_fixed)

    if vis:
        o3d.visualization.draw_geometries([pcd])
        o3d.visualization.draw_geometries([ds_pcd])

    return ds_pcd

# TODO: here bigbird dataset is not included

for obj in os.listdir(gds.objs_folder):
    path_to_obj = os.path.join(gds.objs_folder, obj)
    for pcd_name in os.listdir(path_to_obj):
        pcd_path = os.path.join(path_to_obj, pcd_name)
        pcd = o3d.io.read_point_cloud(pcd_path)
        downsampled_pcd = downsample_pcd(pcd, vis=False)
        pcd_new_path = pcd_path.replace('_pcd', '_dspcd')
        print(pcd_new_path)
        o3d.io.write_point_cloud(pcd_new_path, downsampled_pcd)

