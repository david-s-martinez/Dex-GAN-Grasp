import os
import open3d as o3d
import numpy as np
import h5py


def main(base_path, data_set_names):
    h5_path = os.path.join(base_path, 'pcd_transforms.h5')

    for dset in data_set_names:
        pcd_folder = os.path.join(base_path, dset, 'pcd')
        objs = [obj for obj in os.listdir(pcd_folder) if '.' not in obj]
        for obj_full in objs:
            print("Processing object: ", obj_full)
            pcd_obj_path = os.path.join(pcd_folder, obj_full)
            pcds = [f for f in os.listdir(pcd_obj_path) if 'dspcd' in f]
            for pcd_name in pcds:
                # there are npy files about pcd center saved in the same folder, which has to be skipped.
                if pcd_name.find('center') != -1:
                    continue
                # single pcd data has to be skipped.
                if pcd_name.find('single') != -1:
                    continue
                pcd_path = os.path.join(pcd_obj_path, pcd_name)
                pcd = o3d.io.read_point_cloud(pcd_path)
                pcd_center = pcd.get_center().reshape((3))
                num_str = pcd_path.split('pcd')[-2][:-1]
                data_name = obj_full + '_pcd' + num_str + '_center'
                print(data_name)
                with h5py.File(h5_path, 'a') as pcd_h5:
                    try:
                        pcd_h5[obj_full].create_dataset(data_name, data=pcd_center)
                    except ValueError:
                        data = pcd_h5[obj_full][data_name]      # load the data
                        data[...] = pcd_center


if __name__ == '__main__':

    base_path = '/data/hdd1/qf/hithand_data/collision_only_data_with_ground'
    # bps_path = '/home/vm/new_data_full/basis_point_set.npy'
    main(base_path, data_set_names=['train', 'eval'])
