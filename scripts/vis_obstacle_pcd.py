import open3d as o3d
import numpy as np
import os
from time import time

def get_dummy_bps(bps_path):
    return np.load(bps_path)

def get_obstacle_obj_pcd(pcd_path,vis=False):
    multi_pcd = o3d.io.read_point_cloud(pcd_path)
    single_pcd_path = pcd_path.replace('multi','single')
    obstacle_pcd_path = pcd_path.replace('multi','obstacle')

    single_pcd = o3d.io.read_point_cloud(single_pcd_path)
    obstacle_pcd = o3d.io.read_point_cloud(obstacle_pcd_path)

    multi_np = np.asarray(multi_pcd.points)
    obstacle_np = np.asarray(obstacle_pcd.points)
    if vis:
        colors_np = np.zeros_like(obstacle_np)
        colors_np[:,0] = 1
        obstacle_pcd.colors = o3d.utility.Vector3dVector(colors_np)
        o3d.visualization.draw_geometries([obstacle_pcd,single_pcd])
        o3d.visualization.draw_geometries([multi_pcd])

    return True

if __name__ == "__main__":
    # path_to_pcd = '/home/vm/new_data_full/train/pcd'
    # path_to_pcd = '/home/vm/new_data_full/eval/pcd'
    path_to_pcd = '/home/vm/new_data_full/train/pcd'

    for name in os.listdir(path_to_pcd):
        path_to_obj = os.path.join(path_to_pcd,name)
        for pcd in os.listdir(path_to_obj):
            if pcd.find('multi') != -1:
                pcd_path = os.path.join(path_to_obj,pcd)
                get_obstacle_obj_pcd(pcd_path,vis=True)



    """
    deleted points: -12526 by in total: 38363
save to path: /home/vm/new_data_full/train/pcd/kit_StrawberryPorridge/kit_StrawberryPorridge_pcd045_obstacle.pcd

    /home/vm/new_data_full/train/pcd/kit_CoffeeFilters2/kit_CoffeeFilters2

    """