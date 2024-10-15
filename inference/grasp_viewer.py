
# Run this in FFHNet env
import os
import sys
sys.path.append('/home/dm/panda_ws/inference_container/Multifinger-Net-dev')

import numpy as np
import pyrender
from DexGanGrasp.utils import utils
from urdfpy import URDF
from copy import deepcopy
import trimesh

BASE_PATH = '/home/qian.feng/FFHNet-dev'

def show_grasp_and_object_given_pcd(pcd, centr_T_palm, joint_conf):
    """Visualize the grasp object and the hand relative to it. Here the pcd is already transformed to grasp center

    Args:
        pcd: point cloud i open3d format, in world/base frame
        centr_T_palm (4*4 array): Homogeneous transform that describes the grasp (palm pose) w.r.t to point cloud centroid.
        joint_conf (15 or 20*1 array): 15 or 20 dimensional joint configuration
    """
    # path = os.path.dirname(os.path.abspath(__file__))
    # BASE_PATH = os.path.split(path)[0]

    robot = URDF.load(os.path.join(
        BASE_PATH, 'meshes/hithand_palm/hithand.urdf'))

    # get the full joint config
    if joint_conf.shape[0] == 15:
        joint_conf_full = utils.full_joint_conf_from_partial_joint_conf(
            joint_conf)
    elif joint_conf.shape[0] == 20:
        joint_conf_full = joint_conf
    else:
        raise Exception('Joint_conf has the wrong size in dimension one: %d. Should be 15 or 20' %
                        joint_conf.shape[0])
    # if you want to show hand pregrasp pose with finger straight.
    joint_conf_full = np.zeros(20)
    cfg_map = utils.get_hand_cfg_map(joint_conf_full)

    # compute fk for meshes and links
    fk = robot.visual_trimesh_fk(cfg=cfg_map)
    fk_link = robot.link_fk()
    assert robot.links[2].name == 'palm_link_hithand'  # link 2 must be palm
    # get the transform from base to palm
    hand_base_T_palm = fk_link[robot.links[2]]

    # Compute the transform from base to object centroid frame
    palm_T_centr = np.linalg.inv(centr_T_palm)
    hand_base_T_centr = np.matmul(hand_base_T_palm, palm_T_centr)

    # Turn open3d pcd into pyrender mesh or load trimesh from path
    pcd_center_T_world = np.eye(4)
    pcd_center_T_world[:3,-1] = -1*pcd.get_center()
    copy_pcd = deepcopy(pcd)
    copy_pcd.translate(-1*pcd.get_center())
    centr_T_hand_base = np.linalg.inv(hand_base_T_centr)
    pcd_centr_T_hand_base = np.matmul(pcd_center_T_world, centr_T_hand_base)

    pts = np.asarray(copy_pcd.points)

    # obj_geometry = pyrender.Mesh.from_points(pts,
    #                                              colors=np.tile([55, 55, 4], (pts.shape[0], 1)))
    # obj_geometry = pyrender.Mesh.from_points(pts,
    #                                              colors=np.asarray(copy_pcd.colors))

    sm = trimesh.creation.uv_sphere(radius=0.001)
    sm.visual.vertex_colors = [1.0, 0.0, 0.0]
    tfs = np.tile(np.eye(4), (len(pts), 1, 1))
    tfs[:,:3,3] = pts
    obj_geometry = pyrender.Mesh.from_trimesh(sm, poses=tfs)

    # Construct a scene
    scene = pyrender.Scene(bg_color=(255,255,255))
    # scene = pyrender.Scene(bg_color=(1,1,1))

    # Translate everything to pcd center
    # Add the robot to the scene
    for tm in fk:
        pose = fk[tm]
        pose = np.matmul(pcd_centr_T_hand_base, pose)
        mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
        scene.add(mesh, pose=pose)

    # Add cloud to scene
    # centr_T_base
    scene.add(obj_geometry, pose=np.eye(4))

    # Add more light to scene
    pose_light = np.eye(4)
    pose_light[:3, 3] = [-0.5, 0, 0]
    scene.add(pyrender.PointLight(intensity=10), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0.5, 0, 0]
    scene.add(pyrender.PointLight(intensity=10), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0, 0.9, 0]
    scene.add(pyrender.PointLight(intensity=10), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0, -0.9, 0]
    scene.add(pyrender.PointLight(intensity=10), pose=pose_light)

    T_view_1 = np.array([[0.38758592, 0.19613444, -0.90072662, -0.54629509],
                         [0.34160963, -0.93809507, -0.05727561, -0.12045398],
                         [-0.85620091, -0.28549766, -0.43059386, -0.25333053], [0., 0., 0., 1.]])
    T_view_2 = np.array([[0.38043475, 0.20440112, -0.90193658, -0.48869244],
                         [0.36146523, -0.93055351, -0.05842123, -0.11668246],
                         [-0.85124161, -0.30379325, -0.4278988, -0.22640526], [0., 0., 0., 1.]])

    # View the scene
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    # print(scene.scale)
    nc = pyrender.Node(camera=cam, matrix=T_view_2)
    scene.add_node(nc)

    viewer = pyrender.Viewer(scene, viewer_flags={
                    "fullscreen": False}, use_raymond_lighting=True)


if __name__ == '__main__':
    import numpy as np
    import open3d as o3d
    # pcd_np = np.load('/home/yb/Documents/ffhflow_grasp/pcd_path.npy')
    # pcd = o3d.io.read_point_cloud(str(pcd_np))
    # centr_T_palm = np.load('/home/yb/Documents/ffhflow_grasp/centr_T_palm.npy')
    # centr_T_palm[-1,-1] = 1
    # joint_conf = np.zeros(15)
    # show_grasp_and_object_given_pcd(pcd, centr_T_palm, joint_conf)


    data_path = '/home/dm/panda_ws/inference_container/Multifinger-Net-dev/data/real_objects/object'
    pcd_path = os.path.join(data_path, 'cheez_it.pcd')
    pcd = o3d.io.read_point_cloud(pcd_path)

    grasp_path = os.path.join(data_path, 'grasp0000.npy')
    centr_T_palm = np.load(grasp_path)
    joint_conf = np.zeros(15)
    show_grasp_and_object_given_pcd(pcd, centr_T_palm, joint_conf)