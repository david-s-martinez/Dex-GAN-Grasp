from __future__ import division
import cv2
import h5py
import os
import numpy as np
from FFHNet.utils.utils import hom_matrix_from_pos_quat_list
import argparse

MD = 'metadata'
RS = 'recording_sessions'
RS1 = 'recording_session_0001'
GT = 'grasp_trials'
G = 'grasps'
GSL = 'grasp_success_label'
C = 'collision'
NC = 'no_collision'


class GraspDataHandlerVae:
    def __init__(self, file_path):
        print("GraspDataHandlerVae",file_path)
        assert os.path.exists(file_path)
        self.file_path = file_path

    def get_grasps_for_object(self, obj_name, outcome='positive'):
        """ Returns either all grasps for an outcome in [positive, negative, collision, all].
        All means all outcomes are combined and returned.
        """
        def grasps_for_outcome(file_path, outcome):
            if outcome == 'collision':
                joint_preshape_name = "desired_preshape_joint_state"
            else:
                joint_preshape_name = "true_preshape_joint_state"

            palm_poses = []
            joint_confs = []
            with h5py.File(file_path, 'r') as hdf:
                outcomes_gp = hdf[obj_name][outcome]
                for i, grasp in enumerate(outcomes_gp.keys()):
                    grasp_gp = outcomes_gp[grasp]
                    palm_poses.append(grasp_gp["desired_preshape_palm_mesh_frame"][()])
                    joint_confs.append(grasp_gp[joint_preshape_name][()])
                num_pos = i

            return palm_poses, joint_confs, num_pos

        if outcome == 'all':
            palm_poses = []
            joint_confs = []
            num_g = 0
            for oc in ['collision', 'negative', 'positive']:
                palms, joints, num = grasps_for_outcome(self.file_path, oc)
                palm_poses += palms
                joint_confs += joints
                num_g += num
            return palm_poses, joint_confs, num_g
        elif outcome in ['collision', 'negative', 'positive']:
            return grasps_for_outcome(self.file_path, outcome)
        else:
            raise Exception("Wrong outcome. Choose [positive, negative, collision, all]")

    def get_num_success_per_object(self):
        num_success_per_object = {}
        with h5py.File(self.file_path, 'r') as hdf:
            for obj in hdf.keys():
                num_success_per_object[obj] = len(hdf[obj]['positive'].keys())

        return num_success_per_object

    def get_single_successful_grasp(self, obj_name, random=True, idx=None):
        return self.get_single_grasp_of_outcome(obj_name, 'positive', random=random, idx=idx)

    def get_single_grasp_of_outcome(self, obj_name, outcome, random=True, idx=None):
        with h5py.File(self.file_path, 'r') as hdf:
            grasp_gp = hdf[obj_name][outcome]
            grasp_ids = list(grasp_gp.keys())
            if random:
                idx = np.random.randint(0, len(grasp_ids))
            else:
                idx = idx

            if outcome == 'collision' or 'non_collision_not_executed':
                joint_preshape_name = "desired_preshape_joint_state"
            else:
                # only executed grasps have this data
                joint_preshape_name = "true_preshape_joint_state"

            palm_pose = grasp_gp[grasp_ids[idx]]["desired_preshape_palm_mesh_frame"][()]
            joint_conf = grasp_gp[grasp_ids[idx]][joint_preshape_name][()]

        return palm_pose, joint_conf, None

    def get_single_noncollision_grasp(self, obj_name, idx=None):
        """ return a random grasp from grasps labeled either with positive or nagetive.
        These grasps are all not labeled with collision.
        """
        with h5py.File(self.file_path, 'r') as hdf:
            grasp_gp = hdf[obj_name]['positive']
            grasp_ids = list(grasp_gp.keys())
            grasp_gp_2 = hdf[obj_name]['negative']
            grasp_ids_2 = list(grasp_gp_2.keys())

            idx = np.random.randint(0, len(grasp_ids) + len(grasp_ids_2))
            joint_preshape_name = "true_preshape_joint_state"

            if idx < len(grasp_ids):
                palm_pose = grasp_gp[grasp_ids[idx]]["desired_preshape_palm_mesh_frame"][()]
                joint_conf = grasp_gp[grasp_ids[idx]][joint_preshape_name][()]
            elif idx >= len(grasp_ids):
                idx -= len(grasp_ids)
                palm_pose = grasp_gp_2[grasp_ids_2[idx]]["desired_preshape_palm_mesh_frame"][()]
                joint_conf = grasp_gp_2[grasp_ids_2[idx]][joint_preshape_name][()]
                grasp_gp = grasp_gp_2
                grasp_ids = grasp_ids_2
            return palm_pose, joint_conf, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path',  help='path to dataset h5 file')
    args = parser.parse_args()

    file_path = args.file_path

    gdhvae = GraspDataHandlerVae(file_path=file_path)
    gdhvae.get_single_grasp_of_outcome(obj_name='kit_GreenSaltCylinder', outcome='collision')

