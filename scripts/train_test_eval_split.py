#!/usr/bin/python3
""" This script uses the information about which objects belong to the train/test/val
datasets in the metadata.csv file to split all the obejct pointclouds into their respective folders.

Lastest changed: Lisha Zhou
"""

import os
import shutil
import pandas as pd

def main(metadata_csv_path, src_pc_path, dst_base_path):
    """A small script used to split different dataset"""

    df = pd.read_csv(metadata_csv_path)
    for split in ['train', 'test', 'eval']:
        # Get the object names for this split
        print(f'------------Start to process {split} data ----------')
        obj_names = list(df[df[split] == 'X'].iloc[:, 0].values)
        # Create dst folder
        dst_split_pc_folder = os.path.join(dst_base_path, split, 'point_clouds')
        if not os.path.exists(dst_split_pc_folder):
            os.makedirs(dst_split_pc_folder)
        # copy each object from src to dst
        for obj in obj_names:
            src_obj_folder = os.path.join(src_pc_path, obj)
            dst_obj_folder = os.path.join(dst_split_pc_folder, obj)
            #os.mkdir(dst_obj_folder)
            # copy entire src folder with files to dst
            try:
                shutil.copytree(src_obj_folder, dst_obj_folder)
                # print(obj + "is copied to " + dst_obj_folder)
            except FileNotFoundError:
                print(obj + " not found, skipped")
            except Exception as err:
                print(err)
        print(f'------------End process {split} data ----------')


if __name__ == '__main__':
    # "Hyperparameters"
    # metadata_csv_path = '/home/vm/ffhnet-dataset/ffhnet_data/metadata.csv'
    # src_pc_path = '/home/vm/ffhnet-dataset/ffhnet_data/point_clouds'
    # dst_base_path = '/home/vm/ffhnet-dataset/ffhnet_data/'

    metadata_csv_path = '/home/dm/panda_ws/final_data/metadata.csv'
    src_pc_path = '/home/dm/panda_ws/final_data/point_clouds'
    dst_base_path = '/home/dm/panda_ws/final_data/'

    main(metadata_csv_path, src_pc_path, dst_base_path)
    # For testing
    # main('/mnt/f/lisha_work/test_data/metadata.csv', '/mnt/f/lisha_work/test_data/point_clouds',\
    #     '/mnt/f/lisha_work/test_data')
