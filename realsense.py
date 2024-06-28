import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from time import time
import os
import subprocess

class RealSense():
    """simple warpper to realsense camera
    """
    def __init__(self, logger, save_path) -> None:

        self.pipeline = rs.pipeline()

        self.log = logger
        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        rs_conf = rs.config()
        # rs_conf.enable_device('211222064027')

        self.save_path = save_path

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = rs_conf.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        rs_conf.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
        rs_conf.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

        # Declare pointcloud object, for calculating pointclouds and texture mappings
        self.pc = rs.pointcloud()

        # Start streaming
        profile = self.pipeline.start(rs_conf)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        # print("Depth Scale is: " , depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away

        self.clipping_distance_max = 1.2  # 1.0 single objecg
        self.clipping_distance_min = 0.8
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.colorizer = rs.colorizer()

    def capture_image(self):
        """

        Returns:
            color_image(np.array):
            depth_image(np.array):
            pcd(open3d.geometry.PointCloud):
        """

        time1 = time()
        trials = 0
        while True:
            try:
                trials += 1
                frames = self.pipeline.wait_for_frames()
                if frames:
                    self.log.debug(f'tried {trials} times to capture a frame')
                    break
            except RuntimeError:
                continue

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Prepare the point cloud numpy array
        points = self.pc.calculate(aligned_depth_frame)
        w = rs.video_frame(aligned_depth_frame).width
        h = rs.video_frame(aligned_depth_frame).height
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(h, w, 3)

        # Get data of depth and color are most time-consuming part -> roughly 400ms
        time2 = time()
        depth_image = np.asanyarray(aligned_depth_frame.get_data()) # (480,640)
        color_image = np.asanyarray(color_frame.get_data()) # (480,640,3)

        # Convert point cloud array to open3d format.
        verts = verts.reshape((-1,3))

        # verts[:,2][verts[:,2]>self.clipping_distance] = 0
        # verts[:,2][verts[:,2]<0] = 0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)

        color_image_new = cv2.cvtColor(color_image,cv2.COLOR_RGB2BGR)
        colors = color_image_new / color_image_new.max()
        pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1,3))

        self.log.debug(f"time to get depth and color image {time()-time2}")

        return color_image, depth_image, pcd, time() - time1

    def depth_distance_removal(self, depth_image):
        depth_image[depth_image>self.clipping_distance_max] = 0
        return depth_image

    def point_cloud_distance_removal(self, pcd):
        pcd_np = np.asarray(pcd.points)
        pcd_colors_np = np.asarray(pcd.colors)
        new_pcd_np = pcd_np[pcd_np[:, 2] < self.clipping_distance_max]
        new_pcd_colors_np = pcd_colors_np[pcd_np[:, 2] < self.clipping_distance_max]

        new_pcd_np2 = new_pcd_np[new_pcd_np[:, 2] > self.clipping_distance_min]
        new_pcd_colors_np2 = new_pcd_colors_np[new_pcd_np[:, 2] > self.clipping_distance_min]

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_pcd_np2)
        new_pcd.colors = o3d.utility.Vector3dVector(new_pcd_colors_np2)
        return new_pcd

    def point_cloud_distance_removal_by_input(self, pcd, min=0.2,max=1.2):
        pcd_np = np.asarray(pcd.points)
        pcd_colors_np = np.asarray(pcd.colors)
        new_pcd_np = pcd_np[pcd_np[:, 2] < max]
        new_pcd_colors_np = pcd_colors_np[pcd_np[:, 2] < max]

        new_pcd_np2 = new_pcd_np[new_pcd_np[:, 2] > min]
        new_pcd_colors_np2 = new_pcd_colors_np[new_pcd_np[:, 2] > min]

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_pcd_np2)
        new_pcd.colors = o3d.utility.Vector3dVector(new_pcd_colors_np2)
        return new_pcd

    def save_grasp(self, i, grasp, joint_conf):
        """_summary_

        Args:
            i (_type_): _description_
            rot (np arr)
            transl (np arr)
        """
        grasp_name = 'grasp' + str(i).zfill(4) + '.npy'
        depth2save = os.path.join(self.save_path, grasp_name)
        joint_conf_name = 'joint_conf' + str(i).zfill(4) + '.npy'
        self.joint_conf2save = os.path.join(self.save_path, joint_conf_name)
        np.save(depth2save, grasp)
        np.save(self.joint_conf2save, joint_conf)

    def send_joint_conf(self):
        subprocess.run(["scp", self.joint_conf2save, "qf@SERVER:PATH"])

    def save_images(self, i, color_image, depth_image, pcd, obj_pcd=False) -> None:
        """save color/depth/point cloud data to configured path.

        Args:
            i (_type_): _description_
            color_image (_type_): _description_
            depth_image (_type_): _description_
            pcd (_type_): _description_
        """
        # Save to npy
        depth_name = 'depth_' + str(i).zfill(4) + '.npy'
        depth2save = os.path.join(self.save_path, depth_name)
        np.save(depth2save, depth_image)

        # # save the point cloud numpy array [w,h,3] as npy
        # depth_name = 'point_cloud_' + str(i).zfill(4) + '.npy'
        # depth2save = os.path.join(self.config["data"]["log_image_folder"], depth_name)
        # np.save(depth2save, verts)

        # save point cloud numpy array [wxh,3] as pcd
        pcd_name = 'point_cloud_' + str(i).zfill(4) + '.pcd'
        pcd2save = os.path.join(self.save_path, pcd_name)
        o3d.io.write_point_cloud(pcd2save, pcd)

        if obj_pcd is not False:
            pcd_name = 'point_cloud_obj_' + str(i).zfill(4) + '.pcd'
            pcd2save = os.path.join(self.save_path, pcd_name)
            o3d.io.write_point_cloud(pcd2save, obj_pcd)

        color_name = 'color_' + str(i).zfill(4) + '.png'
        color2save = os.path.join(self.save_path, color_name)
        cv2.imwrite(color2save, color_image)

    def get_colored_depth(self, depth_frame):
        depth_colormap = np.asanyarray(
            self.colorizer.colorize(depth_frame).get_data())
        return depth_colormap

    @staticmethod
    def visualize_color(color_image,waitkey=True):
        cv2.imshow('color_image',color_image)
        if waitkey:
            cv2.waitKey()

    @staticmethod
    def visualize_depth(depth_image):
        # depth image datatype check: float or int

        depth_image = np.expand_dims(depth_image,axis=2)
        depth_image = np.array(depth_image * 255, dtype=np.uint8)
        depth_image = np.concatenate((depth_image,depth_image,depth_image),axis=2)

        cv2.imshow('depth_image',depth_image)
        cv2.waitKey()

    @staticmethod
    def visualize_pcd(pcd):
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, origin])

    @staticmethod
    def visualize_grasp(pcd, grasp, grasp2=None):
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        grasp_frame = grasp_frame.transform(grasp)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        if grasp2 is not None:
            grasp_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            grasp_frame2 = grasp_frame2.transform(grasp2)
            o3d.visualization.draw_geometries([pcd, grasp_frame, grasp_frame2, origin])
        else:
            o3d.visualization.draw_geometries([pcd, grasp_frame, origin])

    def stop(self):
        self.pipeline.stop()

    def __del__(self):
        self.stop()
