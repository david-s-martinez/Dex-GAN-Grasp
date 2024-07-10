import os
import numpy as np
import cv2
import open3d as o3d
# path2img = '/home/qf/Documents/test_vlpart/image_20240705_181108.png'
# path2mask = '/home/qf/Documents/test_vlpart/mask_20240705_181108.npy'
# mask_np = np.load(path2mask)[0]
# im = cv2.imread(path2img)
# pcd_np = np.random.rand(im.shape[0], im.shape[1], 3)
# obj_pcd_np = pcd_np[mask_np==True]
 
def euclidean_distance_points_pairwise_np(pt1, pt2,L1=False):
    """_summary_
 
    Args:
        pt1 (_type_): [N, 3] numpy array, predicted grasp translation
        pts (_type_): [M, 3] numpy array, ground truth grasp translation
 
    Returns:
        dist_mat _type_: [N,M]
    """
    dist_mat = np.zeros((pt1.shape[0],pt2.shape[0]))
    for idx in range(pt1.shape[0]):
        deltas = abs(pt2 - pt1[idx])
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        if not L1:
            dist_mat[idx] = dist_2
        else:
            dist_mat[idx] = np.sqrt(dist_2)
    return dist_mat


def filter_grasps_given_mask(grasps, obj_pcd_np, mask_shape, image_path, pc_center):

    directory = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)
    index = file_name[file_name.find('color_') + 6:file_name.find('color_') + 10]
    path2mask = os.path.join(directory,'mask_' + index +'.npy')
    masks = np.load(path2mask)

    if masks.ndim == 3:
        masks = masks[0]
    # print(grasps)
    grasps_transl = grasps['transl']
    # filter point cloud
    part_pcd_np = obj_pcd_np.reshape(mask_shape)[masks]

    part_pcd_np -= pc_center
    obj_pcd_np -= pc_center

    part_pcd_np = part_pcd_np[abs(part_pcd_np[:,2])<0.2]
    part_pcd_np = part_pcd_np[abs(part_pcd_np[:,1])<0.2]
    part_pcd_np = part_pcd_np[abs(part_pcd_np[:,0])<0.2]

    part_pcd = o3d.geometry.PointCloud()
    part_pcd.points = o3d.utility.Vector3dVector(part_pcd_np)
    part_pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([255,0,0]),(part_pcd_np.shape[0],1)))

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_pcd_np)

    obj_pcd_path = './obj.pcd'
    obj_pcd = o3d.io.read_point_cloud(obj_pcd_path)
    part_pcd_mean = part_pcd_np.mean(axis=0).reshape(1,3)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    origin.translate(part_pcd_mean.reshape(3,1))
    o3d.visualization.draw_geometries([part_pcd, obj_pcd, origin])


    transl_dist_mat = euclidean_distance_points_pairwise_np(grasps_transl, part_pcd_mean)

    min_transl_per_grasp = np.min(transl_dist_mat, axis=1)  # [N,1]
    sorted_grasp_indices = np.argsort(min_transl_per_grasp)
    # sorted_grasp_indices = np.argsort(min_transl_per_grasp)

    print('check sorting on distance matrix')
    print(min_transl_per_grasp[sorted_grasp_indices])
    
    return sorted_grasp_indices, part_pcd_mean
    # here sorted grasps according to distance
    # grasps_transl[sorted_grasp_indices]

def sort_grasps(grasps, sorted_grasp_indices, sort_num):
    grasps['transl'] = grasps['transl'][sorted_grasp_indices][:sort_num]
    grasps['joint_conf'] = grasps['joint_conf'][sorted_grasp_indices][:sort_num]
    grasps['rot_matrix'] = grasps['rot_matrix'][sorted_grasp_indices][:sort_num]
    return grasps


# if __name__ == "__main__":
