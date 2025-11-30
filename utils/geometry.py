import numpy as np
import open3d as o3d
import cv2

def create_point_cloud(color_image, depth_image, fx, fy, cx, cy):
    """
    Generate point cloud from RGB-D.
    从 RGB-D 生成点云。
    """
    rows, cols = depth_image.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    z = depth_image / 1000.0
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color_image.reshape(-1, 3) / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def estimate_focal_length(width: int):
    """
    Estimate focal length based on image width.
    基于图像宽度估计焦距。
    """
    # Matches S0 logic: estimated_focal_length = int(30/36*width)
    return int(30/36 * width)

def get_intrinsics(image_path):
    """
    Get camera intrinsics.
    获取相机内参。
    """
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    f = estimate_focal_length(width)
    return f, f, width/2, height/2, width, height

