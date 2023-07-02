"""Generate obstacle point cloud = multi point cloud - single point cloud.

"""

import open3d as o3d
import numpy as np
import os
from time import time

def get_dummy_bps(bps_path):
    return np.load(bps_path)

def get_obstacle_obj_pcd(pcd_path,vis=False):
    multi_pcd = o3d.io.read_point_cloud(pcd_path)
    single_pcd_path = pcd_path.replace('multi','single')
    single_pcd = o3d.io.read_point_cloud(single_pcd_path)

    multi_np = np.asarray(multi_pcd.points)
    total_len = multi_np.shape[0]
    single_np = np.asarray(single_pcd.points)
    obj_len = single_np.shape[0]
    # time1 = time()
    for i in range(single_np.shape[0]):
        idx = np.where(multi_np==single_np[i,:])
        if idx[0].shape == (0,):
            continue
        else:
            multi_np = np.delete(multi_np,idx[0][0],0)

    print(f'deleted points: {(total_len - multi_np.shape[0] - obj_len)} by in total: {total_len}')
    # print('time:',time()-time1)
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(multi_np)
    if vis:
        colors_np = np.zeros_like(multi_np)
        colors_np[:,0] = 1
        new_pcd.colors = o3d.utility.Vector3dVector(colors_np)
        o3d.visualization.draw_geometries([new_pcd,single_pcd])
        o3d.visualization.draw_geometries([multi_pcd])

    return new_pcd

if __name__ == "__main__":
    # path_to_pcd = '/home/vm/new_data_full/train/pcd'
    # path_to_pcd = '/home/vm/new_data_full/eval/pcd'
    path_to_pcd = '/home/vm/new_data_full/eval/bps'

    for name in os.listdir(path_to_pcd):
        path_to_obj = os.path.join(path_to_pcd,name)
        for pcd in os.listdir(path_to_obj):
            if pcd.find('multi') != -1:
                pcd_path = os.path.join(path_to_obj,pcd)
                new_bps = get_dummy_bps(pcd_path)
                # new_pcd = get_obstacle_obj_pcd(pcd_path,vis=False)
                obstacle_pcd_path = pcd_path.replace('multi','obstacle')
                print('save to path:',obstacle_pcd_path)
                # o3d.io.write_point_cloud(obstacle_pcd_path,new_pcd)
                print(obstacle_pcd_path)
                np.save(obstacle_pcd_path,new_bps)



    """
    deleted points: -12526 by in total: 38363
save to path: /home/vm/new_data_full/train/pcd/kit_StrawberryPorridge/kit_StrawberryPorridge_pcd045_obstacle.pcd

    /home/vm/new_data_full/train/pcd/kit_CoffeeFilters2/kit_CoffeeFilters2

    """