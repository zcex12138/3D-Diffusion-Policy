# reference implementation: https://github.com/mattcorsaro1/mj_pc
# with personal modifications


import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
from typing import List
import open3d as o3d

"""
Generates numpy rotation matrix from quaternion

@param quat: w-x-y-z quaternion rotation tuple

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    # This function is lifted directly from scipy source code
    #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
        2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
        2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat

"""
Generates numpy rotation matrix from rotation matrix as list len(9)

@param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""
def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat

"""
Generates numpy transformation matrix from position list len(3) and 
    numpy rotation matrix

@param pos:     list len(3) containing position
@param rot_mat: 3x3 rotation matrix as numpy array

@return t_mat:  4x4 transformation matrix as numpy array
"""
def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat

"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# 
# and combines them into point clouds
"""
Class that renders depth images in MuJoCo, processes depth images from
    a single camera, converts them to point clouds, and processes the point
    clouds
"""
class PointCloudGenerator(object):
    """
    单相机点云生成器
    
    @param model:       MuJoCo model object
    @param viewer:      MuJoCo viewer object
    @param cam_name:    相机名称 (字符串)
    @param img_size:    图像尺寸
    """
    def __init__(self, model, viewer, cam_name, img_size=84, filter_geom_id=0):
        super(PointCloudGenerator, self).__init__()

        self.model = model
        self.viewer = viewer

        # this should be aligned with rgb
        self.img_width = img_size
        self.img_height = img_size

        self.cam_name = cam_name
        self.filter_geom_id = filter_geom_id
        # get camera id
        self.cam_id = self.model.camera(self.cam_name).id
        fovy = math.radians(self.model.cam_fovy[self.cam_id])
        f = self.img_height / (2 * math.tan(fovy / 2))
        cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
        self.cam_mat = cam_mat

    # Render and process an image
    def captureImage(self, camera_id, capture_depth=True):
        _, depth = self.viewer.render_segment_depth(camera_id, self.filter_geom_id)
        if capture_depth:
            depth = self.depthimg2Meters(depth)
            return depth
        else:
            rgb_img = self.viewer.render_rgb_cam("rgb_array", camera_id, False)
            return rgb_img
        
    def generateCroppedPointCloud(self, save_img_dir=None):
        """
        生成裁剪后的点云 (单相机版本)
        """
        # Render and optionally save image from camera
        color_img = self.captureImage(self.cam_id, capture_depth=False)
        depth = self.captureImage(self.cam_id, capture_depth=True)
        
        # If directory was provided, save color and depth images
        if save_img_dir != None:
            self.saveImg(depth, save_img_dir, "depth_test")
            self.saveImg(color_img, save_img_dir, "color_test")

        # 使用新的函数生成点云
        point_cloud = self.generatePointCloudFromImages(
            color_img=color_img,
            depth=depth,
            use_rgb=True
        )
        
        return point_cloud, depth
     
    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    def horizontalFlip(self, img):
        return np.flip(img, axis=1)
    

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file
    def saveImg(self, img, filepath, filename):
        normalized_image = img/img.max()*255
        normalized_image = normalized_image.astype(np.uint8)
        im = PIL_Image.fromarray(normalized_image)
        im.save(filepath + '/' + filename + ".jpg")

    def generatePointCloudFromImages(self, color_img, depth, use_rgb=True):
        """
        从现有的color_img和depth生成点云 (单相机版本)
        
        Args:
            color_img: RGB图像 (H, W, 3)
            depth: 深度图像 (H, W)
            use_rgb: 是否使用RGB颜色信息
        
        Returns:
            point_cloud: 点云数据 (N, 3) 或 (N, 6) [如果use_rgb=True]
        """
        # 获取相机内参矩阵
        cam_mat = self.cam_mat
        
        # 转换相机内参矩阵为Open3D格式
        od_cammat = cammat2o3d(cam_mat, self.img_width, self.img_height)
        
        # 将深度图像转换为Open3D格式
        od_depth = o3d.geometry.Image(depth)
        
        # 从深度图像生成点云
        o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)
        
        # 获取点云坐标
        point_cloud_points = np.asarray(o3d_cloud.points)
        
        # 计算世界坐标系变换矩阵
        cam_body_id = self.model.cam_bodyid[self.cam_id]  # 使用指定的相机ID
        cam_pos = self.model.body_pos[cam_body_id]
        c2b_r = rotMatList2NPRotMat(self.model.cam_mat0[self.cam_id])  # 使用指定的相机ID
        # MuJoCo相机坐标系到世界坐标系的变换
        b2w_r = quat2Mat([0, 1, 0, 0])
        c2w_r = np.matmul(c2b_r, b2w_r)
        c2w = posRotMat2Mat(cam_pos, c2w_r)
        
        # 应用坐标变换
        transformed_cloud = o3d_cloud.transform(c2w)
        point_cloud_points = np.asarray(transformed_cloud.points)
        
        # 处理颜色信息
        if use_rgb and color_img is not None:
            # 正确的方法：使用Open3D的RGBD图像功能来确保颜色和深度的正确对应
            od_color = o3d.geometry.Image(color_img.astype(np.uint8))
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                od_color, od_depth, depth_scale=1.0, depth_trunc=1000.0, convert_rgb_to_intensity=False
            )
            
            # 从RGBD图像创建点云，这样颜色和深度会自动正确对应
            o3d_cloud_with_color = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, od_cammat)
            
            # 应用相同的坐标变换
            transformed_cloud_with_color = o3d_cloud_with_color.transform(c2w)
            
            # 获取带颜色的点云数据
            point_cloud_points = np.asarray(transformed_cloud_with_color.points)
            point_cloud_colors = np.asarray(transformed_cloud_with_color.colors)
            
            # 将颜色从[0,1]范围转换为[0,255]范围以保持一致性
            point_cloud_colors = (point_cloud_colors * 255).astype(np.uint8)
            
            # 连接位置和颜色信息
            point_cloud = np.concatenate((point_cloud_points, point_cloud_colors), axis=1)
        else:
            # 只返回位置信息
            point_cloud = point_cloud_points
        
        return point_cloud

    def generatePointCloudFromImagesBatch(self, color_images, depth_images, use_rgb=True):
        """
        批量从现有的color_img和depth生成点云 (单相机版本)
        
        Args:
            color_images: RGB图像列表 [(H, W, 3), ...]
            depth_images: 深度图像列表 [(H, W), ...]
            use_rgb: 是否使用RGB颜色信息
        
        Returns:
            point_clouds: 点云数据列表 [(N, 3) 或 (N, 6), ...]
        """
        if len(color_images) != len(depth_images):
            raise ValueError("color_images和depth_images的长度必须相同")
        
        point_clouds = []
        for i, (color_img, depth) in enumerate(zip(color_images, depth_images)):
            if i % 100 == 0:
                print(f"处理进度: {i}/{len(color_images)}")
            
            point_cloud = self.generatePointCloudFromImages(
                color_img, depth, use_rgb=use_rgb
            )
            point_clouds.append(point_cloud)
        
        return point_clouds
