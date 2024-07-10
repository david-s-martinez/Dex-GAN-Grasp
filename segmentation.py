import logging
import os
import sys

import open3d as o3d
import numpy as np
import pyrealsense2 as rs

repo_root = os.path.dirname(__file__) + '/../'
sys.path.insert(0, os.path.join(repo_root, 'src'))
from realsense import RealSense

def signal_handler(self, signal, frame):
    print("====================================")
    print(" Ctrl C pressed! Script stops properly")
    self.stop()
    sys.exit(0)


def display_inlier_outlier(cloud, ind):
    """
    Args:
        cloud (open3d.geometry.PointCloud): open3d point cloud object
        ind (open3d.geometry.PointCloud): open3d point cloud object
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    # print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
                                    # zoom=0.3412,
                                    # front=[0.4257, -0.2125, -0.8795],
                                    # lookat=[2.6172, 2.0475, 1.532],
                                    # up=[-0.0694, -0.9768, 0.2024])


def get_angle_between_two_vec(vec1, vec2):
    # we assume they are normalized vector. TODO: add normalization check
    angle = np.arccos(np.dot(vec1, vec2))
    angle = np.rad2deg(angle)
    return angle

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

# parameteres
remove_statistical_outlier = True
vis = False
# If detected plane has normal with more than 45 deg compared to camera z axis.
plane_normal_threshold = 45

class PlaneSegmentation():
    def __init__(self) -> None:
        pass

    def crop_pcd_with_bbox(self, pcd, bbox):
        # -- Use bbox to crop depth/point cloud
        pcd_np = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)
        pcd_np = pcd_np.reshape(bbox.shape[0], bbox.shape[1], -1)
        pcd_colors = pcd_colors.reshape(bbox.shape[0], bbox.shape[1], -1)
        object_pcd_np = pcd_np[bbox]
        object_pcd_color_np = pcd_colors[bbox]
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_pcd_np)
        object_pcd.colors = o3d.utility.Vector3dVector(object_pcd_color_np)

        if vis:
            o3d.visualization.draw_geometries([object_pcd])

        if remove_statistical_outlier:
            print("Statistical oulier removal")
            cl, ind = object_pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
            if vis:
                display_inlier_outlier(object_pcd, ind)
        return object_pcd

    def pcd_distance_removal(self, pcd):
        pass

    def plane_seg_with_angle_constrain(self, pcd, remove_statistical_outlier=False):

        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                        ransac_n= 3,
                                                        num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        normal_vector = [a,b,c]/np.linalg.norm([a,b,c])

        # Here is the "correct" normal, which is supposed to orient towards camera
        # So we set angle constrains to filter out wrong plane segmentation.
        camera_z = np.array([0, 0, 1])
        angle = get_angle_between_two_vec(camera_z, normal_vector)
        if angle > plane_normal_threshold:
            print(f"Detected plane has normal {angle} deg, more than {plane_normal_threshold} deg")
            # return None, None
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        vis = True
        if vis:
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
            # o3d.visualization.draw_geometries([outlier_cloud])

        if remove_statistical_outlier:
            print("Statistical oulier removal")
            cl, ind = outlier_cloud.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
            display_inlier_outlier(outlier_cloud, ind)
            outlier_cloud = outlier_cloud.select_by_index(ind)
            o3d.visualization.draw_geometries([outlier_cloud])

        return outlier_cloud, normal_vector


if __name__ == "__main__":

    save_path = '/home/qf/Pictures/ffhflow_exp'
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    rs = RealSense(logger, save_path)
    segment = PlaneSegmentation()

    i = 0
    color_image, depth_image, pcd, _ = rs.capture_image()
    pcd = rs.point_cloud_distance_removal(pcd)
    rs.save_images(i, color_image, depth_image, pcd)
    inlier_pcd, normal_vector = segment.plane_seg_with_angle_constrain(pcd)

    # [480,640,3]
    # [480, 640] ,uint16, max 3586
    # pcd open3d



