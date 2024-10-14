from __future__ import division

import collections
import colorsys
import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from DexGanGrasp.utils.grasp_data_handler import GraspDataHandler

try:
    from urdfpy import URDF
except:
    print("[WARNING] urdfpy (ROS) not installed.")
import open3d as o3d
import pandas as pd
import pyrender
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import trimesh
from DexGanGrasp.config.config import Config
from DexGanGrasp.utils import utils

path = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.split(os.path.split(path)[0])[0]


def find_max_min_success(p_success_list, idx):
    """Find the max and min success in a list of success obtained from refinement. Each item in success list
    contains batch_size many grasps. Idx indicates which grasp should be evaluated.

    Args:
        p_success_list (list of array len_list*batch_size*1): List of arrays where each entry is the success probability for all batch_size grasps at specific refinement step.
        idx (int): An integer indicating which grasp should be checked
    """
    succ_min, succ_max = 1, 0
    for succ_batch in p_success_list:
        if succ_batch[idx] < succ_min:
            succ_min = succ_batch[idx]
        if succ_batch[idx] > succ_max:
            succ_max = succ_batch[idx]

    return succ_max, succ_min


def get_mesh_path(obj_name, gazebo_obj_path):
    obj_split = obj_name.split('_')
    dset = obj_split[0]
    obj = '_'.join(obj_split[1:])
    if dset == 'kit':
        file_name = obj + '_25k_tex.obj'
    elif dset == 'bigbird':
        file_name = 'optimized_tsdf_texture_mapped_mesh.obj'
    elif dset == 'ycb':
        file_name = ''
    else:
        raise Exception('Unknown dataset name.')
    path = os.path.join(gazebo_obj_path, dset, obj, file_name)
    return path


def get_mesh_for_object(obj_name, gazebo_obj_path, from_trimesh=False):
    """Loads the mesh for the object and returns it

    Args:
        obj_name (str): Name of thje object

    Raises:
        Exception: If the the dataset from the object stems is not known

    Returns:
        mesh (o3d.TriangleMesh): The o3d triangle mesh object.
    """
    path = get_mesh_path(obj_name, gazebo_obj_path)

    if from_trimesh:
        mesh = trimesh.load_mesh(path)
    else:
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh.paint_uniform_color([1, 0, 0])
    return mesh


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # font = {'serif': 'Times', 'family': 'serif', 'weight': 'bold', 'size': 8}
    # matplotlib.rc('font', **font)
    plt.style.use(['science'])
    matplotlib.rcParams.update({'font.size': 8})
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    np.set_printoptions(precision=3)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.savefig('confusion_matrix.pdf')
    plt.show()
    return ax


def rgb_color_array_from_h_value(h_min, h_max, n_colors):
    h_vals = np.linspace(h_min, h_max, n_colors).astype(int)
    rgb_colors = []
    for h in h_vals:
        rgb = colorsys.hsv_to_rgb(h, 0., 1.)
        rgb_colors.append(rgb)
    return rgb_colors


def show_camera(camera_model_path=os.path.join(BASE_PATH, 'meshes/camera/d415.stl')):
    mesh = o3d.io.read_triangle_mesh(camera_model_path)
    colors_shape = np.asarray(mesh.vertex_colors).shape
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.tile([0.3, 0.3, 0.3], (colors_shape[0], 1)))
    mesh.compute_vertex_normals()

    orig = mesh.create_coordinate_frame(size=0.04)
    T = np.eye(4)
    T[:2, :2] = np.array([[-1, 0], [0, -1]])
    orig.transform(T)

    o3d.visualization.draw_geometries([mesh, orig])


def show_grasp_refinement(data_list, p_success_list, pcd_paths, grasp_idx=-1):
    """Loads the object point cloud and visualizes the refinement steps.

    Args:
        data_list (list of dicts with array values): List of dicts with keys being palm orientation, translation, joint conf and
        p_success_list (list of np arrays): Indicates the success of each grasp in data_list.
        pcd_path (list of strings): Paths to load the object pcd from.
        grasp_idx (int): Number between 0 and batch_size -1 indicating which grasp should be visualized. Defaults to -1 which means random grasp is chosen.
    """
    if grasp_idx == -1:
        idx = np.random.randint(0, data_list[0]['transl'].shape[0] - 1)
    else:
        idx = grasp_idx
    pcd = o3d.io.read_point_cloud(pcd_paths[idx])
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)

    succ_max, succ_min = find_max_min_success(p_success_list, idx)

    grasps = [origin, pcd]
    for data, succ in zip(data_list, p_success_list):
        grasp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)

        # find h-value proportional to success. min success should be h=0, max_success should be h=0.33
        # (y2 -y1) / (x2 -x1) = (1 - 0) / (suc_max - suc_min) * (suc - suc_min)
        h_val = 0.33 * ((1 / (succ_max - succ_min)) * (succ[idx] - succ_min))
        print(succ[idx])
        # Choose color proportional to grasp success
        grasp.paint_uniform_color(colorsys.hsv_to_rgb(h_val, 1, 1))

        print(data['transl'][idx, :])
        # Get homogenous transform
        grasp_hom = utils.hom_matrix_from_transl_rot_matrix(data['transl'][idx, :],
                                                            data['rot_matrix'][idx, :, :])

        # Transform grasp
        grasp.transform(grasp_hom)

        grasps.append(grasp)

    o3d.visualization.draw_geometries(grasps)


def show_grasps_o3d_viewer(self, palm_poses, object_pcd_path):
    """Visualize the sampled grasp poses in open3d along with the object point cloud.

    Args:
        palm_poses (list of np array, n*4*4): Sampled poses.
        object_pcd_path (str): Path to object pcd.
    """
    frames = []
    for pose in palm_poses:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            0.01).transform(pose)
        frames.append(frame)

    # visualize
    orig = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01)
    frames.append(orig)

    obj = o3d.io.read_point_cloud(object_pcd_path)
    frames.append(obj)
    o3d.visualization.draw_geometries(frames)


def show_dataloader_grasp(bps_path, obj, centr_T_mesh, palm_pose_mesh, palm_pose_centr, gazebo_obj_path,rendered_pcd=False):
    """
    """
    # Load the actual object mesh and turn it into a point cloud
    obj_split = obj.split('_')
    dset = obj_split[0]
    obj_name = '_'.join(obj_split[1:])
    if dset == 'kit':
        file_name = obj_name + '_25k_tex.obj'
    elif dset == 'bigbird':
        file_name = 'optimized_tsdf_texture_mapped_mesh.obj'
    elif dset == 'ycb':
        file_name = ''
    else:
        raise Exception('Unknown dataset name.')
    path = os.path.join(gazebo_obj_path, dset, obj_name, file_name)
    mesh_pc = o3d.io.read_triangle_mesh(path)
    mesh_pc.compute_vertex_normals()
    mesh_pc.paint_uniform_color(np.array([0.2, 0.2, 0.2]))

    # Load the rendered point cloud
    if rendered_pcd is None:
        rendered_pcd = utils.load_rendered_pcd(bps_path)

    # Construct the frames
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
    centr_frame = copy.deepcopy(mesh_frame)
    palm_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)

    # we get centr_T_mesh, but need mesh_T_centr to transform centr_p_centr to mesh_p_centr, i.e. the centroid frame w.r.t to mesh frame
    centr_frame.transform(np.linalg.inv(centr_T_mesh))
    palm_frame.transform(palm_pose_mesh)
    rendered_pcd.transform(np.linalg.inv(centr_T_mesh))
    #print("Palm pose in centroid frame:")
    # print(palm_pose_centr)

    o3d.visualization.draw_geometries(
        [rendered_pcd, mesh_pc, mesh_frame, centr_frame, palm_frame])
    # o3d.visualization.draw_geometries([rendered_pcd, mesh_pc, centr_frame, palm_frame])

def show_dataloader_grasp_with_pcd_in_world_frame(obj,
                                                  world_T_mesh_obj,
                                                  world_T_palm_pose,
                                                  mesh_T_palm_pose,
                                                  rendered_pcd,
                                                  path2objs_gazebo):
    """

    Args:
        obj (str): object name
        world_T_mesh_obj (array): _description_
        world_T_palm_pose (array): _description_
        mesh_T_palm_pose (array): _description_
        rendered_pcd (_type_): open3d point cloud data

    """
    # Load the actual object mesh and turn it into a point cloud
    obj_split = obj.split('_')
    dset = obj_split[0]
    obj_name = '_'.join(obj_split[1:])
    if dset == 'kit':
        file_name = obj_name + '_25k_tex.obj'
    elif dset == 'bigbird':
        file_name = 'optimized_tsdf_texture_mapped_mesh.obj'
    elif dset == 'ycb':
        file_name = ''
    else:
        raise Exception('Unknown dataset name.')
    path = os.path.join(path2objs_gazebo, dset, obj_name, file_name)
    mesh_pc = o3d.io.read_triangle_mesh(path)
    mesh_pc.compute_vertex_normals()
    mesh_pc.paint_uniform_color(np.array([0.2, 0.2, 0.2]))

    # Construct the frames
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    centr_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    palm_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    palm_frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)

    # we get centr_T_mesh, but need mesh_T_centr to transform centr_p_centr to mesh_p_centr, i.e. the centroid frame w.r.t to mesh frame
    centr_frame.transform(world_T_mesh_obj)
    print(centr_frame.get_center())
    palm_frame.transform(world_T_palm_pose)
    palm_frame_mesh.transform(mesh_T_palm_pose)
    # rendered_pcd.transform(np.linalg.inv(centr_T_mesh))
    #print("Palm pose in centroid frame:")
    # print(palm_pose_centr)

    o3d.visualization.draw_geometries(
        [rendered_pcd, mesh_pc, centr_frame, mesh_frame, palm_frame, palm_frame_mesh])
    # o3d.visualization.draw_geometries([rendered_pcd, mesh_pc, centr_frame, palm_frame])



def show_generated_grasp_distribution(pcd_path,
                                      grasps,
                                      highlight_idx=-1,
                                      custom_vis=True,
                                      mean_coord=False,
                                      save_ix=0):
    """Visualizes the object point cloud together with the generated grasp distribution.

    Args:
        path (str): Path to the object pcd
        grasps (dict): contains arrays rot_matrix [n*3*3], palm transl [n*3], joint_conf [n*15]
    """
    frames = []
    if grasps:
        n_samples = grasps['rot_matrix'].shape[0]
        for i in range(n_samples):
            rot_matrix = grasps['rot_matrix'][i, :, :]
            transl = grasps['transl'][i, :]
            palm_pose_centr = utils.hom_matrix_from_transl_rot_matrix(
                transl, rot_matrix)
            if i == highlight_idx:
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.065).transform(
                    palm_pose_centr)
            else:
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01).transform(
                    palm_pose_centr)
            frames.append(frame)

        if mean_coord is not False:
            origin_mean = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2).translate(
                    mean_coord.reshape(3,))
            frames.append(origin_mean)
        # visualize
        ## If you want to add origin, this
        # orig = o3d.geometry.TriangleMesh.create_coordinate_frame(0.001)
        # orig = orig.scale(5, center=orig.get_center())
        # frames.append(orig)

    obj = o3d.io.read_point_cloud(pcd_path)
    #obj = obj.voxel_down_sample(0.002)
    obj.paint_uniform_color([230. / 255., 230. / 255., 10. / 255.])
    obj.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=100))
    frames.append(obj)
    
    if custom_vis:
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.07)
        frames.append(origin)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for f in frames:
            vis.add_geometry(f)

        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(os.path.join(BASE_PATH,"DexGanGrasp/config/view_point.json"))
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.get_render_option().load_from_json(os.path.join(BASE_PATH,"DexGanGrasp/config/render_opt.json"))
        vis.run()
        vis.destroy_window()
        # vis.get_render_option().save_to_json("/home/vm/hand_ws/src/DexGanGrasp/render_opt.json")
        # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # o3d.io.write_pinhole_camera_parameters("/home/vm/hand_ws/src/DexGanGrasp/view_point.json",
        #                                        param)
        # l = raw_input("Save image?: ")
        # if l == 'y':
        #     vis.capture_screen_image("/home/vm/Pictures/{}.png".format(save_ix))

    else:
        o3d.visualization.draw_geometries(frames)


def show_individual_ground_truth_grasps(obj_name, grasp_data_path, outcome='positive'):
    # Get mesh for object
    mesh_path = get_mesh_path(obj_name)

    # Get the ground truth grasps
    data_handler = GraspDataHandler(file_path=grasp_data_path)
    palm_poses, joint_confs, num_succ = data_handler.get_grasps_for_object(obj_name,
                                                                           outcome=outcome)

    # Display the grasps in a loop
    for i, (palm_pose, joint_conf) in enumerate(zip(palm_poses, joint_confs)):
        palm_hom = utils.hom_matrix_from_pos_quat_list(palm_pose)
        th = joint_conf[16]
        joint_conf = np.zeros(20)
        joint_conf[16] = th
        show_grasp_and_object(mesh_path, palm_hom, joint_conf)
        print(joint_conf)


def show_ground_truth_grasp_distribution(obj_name, grasp_data_path, gazebo_obj_path):
    """Shows all the ground truth positive grasps for an object.

    Args:
        obj_name (str): Name of the object, in the format datasetname_objectname
    """
    # Get the object mesh
    mesh = get_mesh_for_object(obj_name, gazebo_obj_path)

    # Get all the grasps
    data_handler = GraspDataHandler(file_path=grasp_data_path)

    palm_poses, _, num_succ = data_handler.get_grasps_for_object(
        obj_name, outcome='positive')

    print("Successful ones:", num_succ)

    frames = []
    for i in range(0, len(palm_poses)):
        palm_hom = utils.hom_matrix_from_pos_quat_list(palm_poses[i])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            0.01).transform(palm_hom)
        frames.append(frame)

    # visualize
    orig = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01)
    frames.append(mesh)
    frames.append(orig)
    o3d.visualization.draw_geometries(frames)

def show_grasp_and_object_given_pcd(pcd, centr_T_palm, joint_conf):
    """Visualize the grasp object and the hand relative to it. Here the pcd is already transformed to grasp center

    Args:
        pcd: point cloud i open3d format, in world/base frame
        centr_T_palm (4*4 array): Homogeneous transform that describes the grasp (palm pose) w.r.t to point cloud centroid.
        joint_conf (15 or 20*1 array): 15 or 20 dimensional joint configuration
    """
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

    # pcd.translate(-1*pcd.get_center())
    pts = np.asarray(pcd.points)

    obj_geometry = pyrender.Mesh.from_points(pts,
                                                 colors=np.tile([55, 55, 4], (pts.shape[0], 1)))

    # Construct a scene
    scene = pyrender.Scene()

    centr_T_hand_base = np.linalg.inv(hand_base_T_centr)
    # Add the robot to the scene
    for tm in fk:
        pose = fk[tm]
        pose = np.matmul(centr_T_hand_base, pose)
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

    pyrender.Viewer(scene, viewer_flags={
                    "fullscreen": False}, use_raymond_lighting=True)


def show_grasp_and_object(path, palm_T_centr, joint_conf, urdf_path = 'meshes/hithand_palm/hithand.urdf'):
    """Visualize the grasp object and the hand relative to it

    Args:
        path (str): Path to bps or pointcloud or mesh of object
        palm_T_centr (4*4 array): Homogeneous transform that describes the grasp (palm pose) w.r.t to object centroid.
        joint_conf (15 or 20*1 array): 15 or 20 dimensional joint configuration
    """
    robot = URDF.load(os.path.join(
        BASE_PATH, urdf_path))

    # get the full joint config
    if joint_conf.shape[0] == 15:
        joint_conf_full = utils.full_joint_conf_from_partial_joint_conf(
            joint_conf)
    elif joint_conf.shape[0] == 20 or joint_conf.shape[0] == 12:
        joint_conf_full = joint_conf
    else:
        raise Exception('Joint_conf has the wrong size in dimension one: %d. Should be 12, 15 or 20' %
                        joint_conf.shape[0])
    cfg_map = utils.get_hand_cfg_map(joint_conf_full)

    # compute fk for meshes and links
    fk = robot.visual_trimesh_fk(cfg=cfg_map)
    fk_link = robot.link_fk()
    if joint_conf.shape[0] == 12:
        assert robot.links[0].name == 'palm_link_robotiq'  # link 0 must be palm
        # get the transform from base to palm
        palm_T_base = fk_link[robot.links[0]]
    else:
        assert robot.links[2].name == 'palm_link_hithand'  # link 2 must be palm
        # get the transform from base to palm
        palm_T_base = fk_link[robot.links[2]]

    # Compute the transform from base to object centroid frame
    centr_T_palm = np.linalg.inv(palm_T_centr)
    centr_T_base = np.matmul(palm_T_base, centr_T_palm)

    # Turn open3d pcd into pyrender mesh or load trimesh from path
    if 'bps' in path or 'pcd' in path:
        obj_pcd = utils.load_rendered_pcd(path)
        pts = np.asarray(obj_pcd.points)
        obj_geometry = pyrender.Mesh.from_points(pts,
                                                 colors=np.tile([55, 55, 4], (pts.shape[0], 1)))
    else:
        mesh = trimesh.load_mesh(path)
        obj_geometry = pyrender.Mesh.from_trimesh(mesh,
                                                  material=pyrender.MetallicRoughnessMaterial(
                                                      emissiveFactor=[
                                                          255, 0, 0],
                                                      doubleSided=True,
                                                      baseColorFactor=[255, 0, 0, 1]))

    # Construct a scene
    scene = pyrender.Scene(bg_color=(255,255,255))

    base_T_centr = np.linalg.inv(centr_T_base)
    # Add the robot to the scene
    for tm in fk:
        pose = fk[tm]
        pose = np.matmul(base_T_centr, pose)
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

    pyrender.Viewer(scene, viewer_flags={
                    "fullscreen": False}, use_raymond_lighting=True)


def render_hand_in_configuration(cfg=np.zeros(20)):
    path = os.path.dirname(os.path.abspath(__file__))
    robot = URDF.load(os.path.join(
        BASE_PATH, 'meshes/hithand_palm/hithand.urdf'))

    cfg_map = utils.get_hand_cfg_map(cfg)

    # compute fk for meshes and links
    fk = robot.visual_trimesh_fk(cfg=cfg_map)

    # Construct a scene
    scene = pyrender.Scene()

    # Add the robot to the scene
    for tm in fk:
        pose = fk[tm]
        mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
        scene.add(mesh, pose=pose)

    # Add more light to scene
    pose_light = np.eye(4)
    pose_light[:3, 3] = [-0.5, 0, 0]
    scene.add(pyrender.PointLight(intensity=6), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0.5, 0, 0]
    scene.add(pyrender.PointLight(intensity=6), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0, 0.9, 0]
    scene.add(pyrender.PointLight(intensity=6), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0, -0.9, 0]
    scene.add(pyrender.PointLight(intensity=6), pose=pose_light)

    # View the scene
    pyrender.Viewer(scene, use_raymond_lighting=True)


def show_pcd_and_bps(pcd_path):
    obj_pcd = o3d.io.read_point_cloud(pcd_path)

    bps = np.load('/home/vm/basis_point_set.npy')
    bps_pcd = o3d.geometry.PointCloud()
    bps_pcd.points = o3d.utility.Vector3dVector(bps)
    bps_pcd.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([obj_pcd, bps_pcd])


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def show_dist_histogram():
    # colors
    #clist = [(0, "green"), (0.6, "yellow"), (0.9, "orange"), (1, "red")]
    clist = [(0, "blue"), (0.5, "green"), (1, "yellow")]
    rvb = mcolors.LinearSegmentedColormap.from_list("", clist)
    N = 128

    # Generate color map, sort data get the index "mixing" apply to color and give to barh
    data = np.random.uniform(0.01, 0.1, (N)) + \
        np.abs(np.random.normal(0, 0.01, (N)))
    data = np.sort(data)
    x = np.arange(N).astype(float)
    t = x / N
    data, t = shuffle_in_unison(data, t)
    plt.barh(x, data, color=rvb(t), height=1.0)
    plt.axis('off')
    plt.show()
    plt.savefig("test.png", bbox_inches='tight')


def plot_coverage_success_curve():
    x = [0.05, 0.28, 0.55, 0.73, 0.83, 1]  # coverage
    y = [0.9, 0.84, 0.8, 0.75, 0.7, 0.61]  # grasping success

    plt.style.use(['science', 'grid'])
    matplotlib.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.autoscale(tight=True)
    pparam = dict(title='Coverage-Success Curve',
                  xlabel='Coverage', ylabel='Grasping Success')
    ax.set(**pparam)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])

    plt.show()
    save_path = os.path.join(
        '/home/vm/Documents/thesis/master_thesis/figures/chapter_05/eva_eval/coverage_success_curve.pdf'
    )
    fig.savefig(save_path)


def plot_threshold_success_curve():
    x = [0.0, 0.5, 0.7, 0.9, 0.95]  # coverage
    y = [0.61, 0.7, 0.75, 0.82, 0.91]  # grasping success
    plt.style.use(['science', 'grid'])
    matplotlib.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.autoscale(tight=True)
    pparam = dict(title='Grasping Success over Threshold',
                  xlabel='Threshold',
                  ylabel='Grasping Success')
    ax.set(**pparam)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])

    plt.show()
    save_path = os.path.join(
        '/home/vm/Documents/thesis/master_thesis/figures/chapter_05/eva_eval/threshold_success_curve.pdf'
    )
    fig.savefig(save_path)


def plot_2D_gaussian():
    N = 100
    X = np.linspace(-2, 2, N)
    Y = np.linspace(-2, 2, N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([0., 0.])
    Sigma = np.array([[1., 0], [0, 1.]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2, ))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos."""

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2 * np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

        return np.exp(-fac / 2) / N

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)

    # plot using subplots
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    clist = [(0, "green"), (0.3, "yellow"), (0.7, "orange"), (1, "red")]
    cmap = mcolors.LinearSegmentedColormap.from_list("my_map", clist, N)
    ax1.plot_surface(X,
                     Y,
                     Z,
                     rstride=1,
                     cstride=1,
                     linewidth=0,
                     antialiased=False,
                     cmap=cmap,
                     alpha=1)
    # plt.axes('off')
    ax1.set_axis_off()
    plt.show()


def plot_filtered_grasps_per_threshold(path_to_csv):
    """Plots a graph where the x-axis is the threshold of the DexEvaluator evaluation, below which grasps are rejected
    and y-axis is the percentage of reamining grasps

    Args:
        path_to_csv (str): Path to a csv where each column is threshold and reamining grasps percentage.
    """
    df = pd.read_csv(path_to_csv)
    x = df['thresh'].to_numpy()
    y = df['average_remaining_grasps'].to_numpy()

    plt.style.use(['science', 'grid'])
    matplotlib.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.autoscale(tight=True)
    pparam = dict(title='Ratio of Grasps above Threshold',
                  xlabel='Success Threshold ($th$)',
                  ylabel='\% of Grasps: $p(s) > th$')
    ax.set(**pparam)
    plt.show()

    save_path = os.path.join(
        '/home/vm/Documents/thesis/master_thesis/figures/chapter_05/eva_eval/grasps_above_thresh.pdf'
    )
    fig.savefig(save_path)