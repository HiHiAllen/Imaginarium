import numpy as np
import cv2
from PIL import Image
import open3d as o3d
from scipy import ndimage
import time
import os
from scipy import stats
from scipy.ndimage import median_filter
from tqdm import tqdm
import argparse
import json
import re
import multiprocessing
from functools import partial
from sklearn.cluster import DBSCAN
# from scipy.spatial import cKDTree

def create_point_cloud(color_image, depth_image):
    # 设置相机内参
    height, width = color_image.shape[:2]  # 修改这里
    # UE5相机焦距30mm，传感器尺寸36mm
    estimated_focal_length = int(30/36*width)
    # estimated_focal_length = int(50/36*width)
    fx = estimated_focal_length
    fy = estimated_focal_length
    cx = width / 2
    cy = height / 2
    
    # 创建坐标网格
    rows, cols = depth_image.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    
    # 计算3D点
    z = depth_image / 1000.0  # 将深度从毫米转换为米
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy
    
    # 创建点云
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color_image.reshape(-1, 3) / 255.0  # 归一化颜色值
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def create_point_cloud_downsample(color_image, depth_image, downsample_size=518):
    resized_color_image = color_image.resize((downsample_size, downsample_size), Image.LANCZOS)
    resized_pred = depth_image.resize((downsample_size, downsample_size), Image.NEAREST)
    color_image = np.array(color_image)
    height, width = color_image.shape[:2]
    # UE5相机焦距30mm，传感器尺寸36mm
    estimated_focal_length = int(30/36*width)
    # estimated_focal_length = int(50/36*width)
    fx = estimated_focal_length
    fy = estimated_focal_length

    scale_x = downsample_size / width
    scale_y = downsample_size / height
    fx *= scale_x
    fy *= scale_y
    cx = cy = downsample_size/2
    
    resized_pred = np.array(resized_pred)
    z = resized_pred / 1000.0  # 将深度从毫米转换为米
    
    # 创建坐标网格
    rows, cols = resized_pred.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    
    # 计算3D点
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy
    
    # 创建点云
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = np.array(resized_color_image).reshape(-1, 3) / 255.0  # 归一化颜色值
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def process_image_and_create_point_cloud(color_image_path, depth_image_path, downsample_size=518):
    color_image = Image.open(color_image_path).convert('RGB')
    depth_image = Image.open(depth_image_path)
    width, height = color_image.size

    # 计算相机内参
    estimated_focal_length = int(30/36*width)
    # estimated_focal_length = int(50/36*width)
    fx = fy = estimated_focal_length
    cx = width / 2
    cy = height / 2

    # 创建点云
    pcd = create_point_cloud_downsample(color_image, depth_image, downsample_size)

    return pcd, (fx, fy, cx, cy), (width, height)

def calculate_center_xyz(center_x, center_y, center_depth, fx, fy, cx, cy):
    # 将深度从毫米转换为米
    center_depth_m = center_depth / 1000.0

    # 计算相机坐标系下的 XYZ 坐标
    X = (center_x - cx) * center_depth_m / fx
    Y = (center_y - cy) * center_depth_m / fy
    Z = center_depth_m

    return X, Y, Z

def estimate_obj_depth(depth_image_path, obj_mask_path):
    depth_image = np.array(Image.open(depth_image_path)).astype(np.float32)
    obj_mask = np.array(Image.open(obj_mask_path))
    obj_mask = (obj_mask > 128).astype(np.uint8)

    # 1. 预处理
    masked_depth = depth_image * obj_mask
    masked_depth[obj_mask == 0] = np.nan

    # 2. 去除物体外轮廓
    kernel = np.ones((5,5), np.uint8)
    eroded_mask = cv2.erode(obj_mask, kernel, iterations=2)
    dilated_mask = cv2.dilate(obj_mask, kernel, iterations=2)
    contour_mask = dilated_mask - eroded_mask
    masked_depth[contour_mask > 0] = np.nan

    # 3. 去噪和异常值移除
    valid_depths = masked_depth[~np.isnan(masked_depth)]
    if len(valid_depths) == 0:
        return None, None, None, None

    percentile_5 = np.percentile(valid_depths, 5)
    percentile_95 = np.percentile(valid_depths, 95)
    masked_depth[(masked_depth < percentile_5) | (masked_depth > percentile_95)] = np.nan

    # 4. 中值滤波去噪
    masked_depth_filtered = ndimage.median_filter(masked_depth, size=3)

    # 5. 确定几何中心
    y, x = np.where(eroded_mask > 0)  # 使用腐蚀后的mask来计算中心
    center_y, center_x = int(np.mean(y)), int(np.mean(x))

    # 6. 深度估计
    valid_depths = masked_depth_filtered[~np.isnan(masked_depth_filtered)]
    mean_depth = np.mean(valid_depths)
    median_depth = np.median(valid_depths)
    min_depth = np.min(valid_depths)
    max_depth = np.max(valid_depths)

    # 7. 深度分布分析
    hist, bin_edges = np.histogram(valid_depths, bins=50)
    most_common_depth = bin_edges[np.argmax(hist)]

    # 8. 综合深度估计
    estimated_depth = {
        'mean': mean_depth,
        'median': median_depth,
        'min': min_depth,
        'max': max_depth,
        'most_common': most_common_depth,
        'center': masked_depth_filtered[center_y, center_x]
    }

    # 使用加权平均作为物体中心的深度
    weights = [0.4, 0.3, 0.2, 0.1]  # 权重可以根据需要调整
    center_depth = (
        weights[0] * estimated_depth['median'] +
        weights[1] * estimated_depth['mean'] +
        weights[2] * estimated_depth['most_common'] +
        weights[3] * estimated_depth['center']
    )
    
    return estimated_depth, (center_x, center_y), masked_depth_filtered, center_depth

def get_min_bound_box_pca(pcd):
    points = np.asarray(pcd.points)
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigenvectors, eigenvalues, _ = np.linalg.svd(cov)
    
    # 将点云投影到主轴上
    projected = np.dot(centered, eigenvectors.T)
    
    # 计算在主轴上的最小和最大坐标
    min_point = np.min(projected, axis=0)
    max_point = np.max(projected, axis=0)
    
    # 计算边长
    sizes = max_point - min_point
    
    # 计算中心点
    center = mean + np.dot((min_point + max_point) / 2, eigenvectors)
    
    # 创建 OrientedBoundingBox
    obb = o3d.geometry.OrientedBoundingBox(center, eigenvectors.T, sizes)
    
    return obb, center

def get_min_bound_box_optimized_pca(pcd):
    points = np.asarray(pcd.points)
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigenvectors, eigenvalues, _ = np.linalg.svd(cov)
    
    # 尝试8种不同的方向组合
    directions = [eigenvectors, -eigenvectors]
    min_volume = float('inf')
    best_obb = None
    best_center = None
    
    for dx in directions[0]:
        for dy in directions[1]:
            dz = np.cross(dx, dy)
            R = np.column_stack((dx, dy, dz))
            
            rotated_points = np.dot(centered, R)
            min_point = np.min(rotated_points, axis=0)
            max_point = np.max(rotated_points, axis=0)
            sizes = max_point - min_point
            volume = np.prod(sizes)
            
            if volume < min_volume:
                min_volume = volume
                center = mean + np.dot(R, (min_point + max_point) / 2)
                best_obb = o3d.geometry.OrientedBoundingBox(center, R, sizes)
                best_center = center
    
    return best_obb, best_center

def get_min_bound_box_convex_hull(pcd):
    hull, _ = pcd.compute_convex_hull()
    hull_points = np.asarray(hull.vertices)
    
    # 计算凸包的主成分
    mean = np.mean(hull_points, axis=0)
    centered = hull_points - mean
    cov = np.cov(centered.T)
    eigenvectors, _, _ = np.linalg.svd(cov)
    
    # 将凸包点投影到主轴上
    projected = np.dot(centered, eigenvectors.T)
    
    # 计算在主轴上的最小和最大坐标
    min_point = np.min(projected, axis=0)
    max_point = np.max(projected, axis=0)
    
    # 计算边长
    sizes = max_point - min_point
    
    # 计算中心点
    center = mean + np.dot((min_point + max_point) / 2, eigenvectors)
    
    # 创建 OrientedBoundingBox
    obb = o3d.geometry.OrientedBoundingBox(center, eigenvectors.T, sizes)
    
    return obb, center

def get_min_bound_box_rotation_search(pcd):
    points = np.asarray(pcd.points)
    min_volume = float('inf')
    best_obb = None
    best_center = None
    
    for angle_x in np.linspace(0, np.pi, 18):
        for angle_y in np.linspace(0, np.pi, 18):
            for angle_z in np.linspace(0, np.pi, 18):
                R = o3d.geometry.get_rotation_matrix_from_xyz((angle_x, angle_y, angle_z))
                rotated_points = np.dot(points, R)
                
                min_point = np.min(rotated_points, axis=0)
                max_point = np.max(rotated_points, axis=0)
                sizes = max_point - min_point
                volume = np.prod(sizes)
                
                if volume < min_volume:
                    min_volume = volume
                    center_rotated = (min_point + max_point) / 2
                    center = np.dot(center_rotated, R.T)  # 将中心点旋转回原始坐标系
                    best_obb = o3d.geometry.OrientedBoundingBox(center, R, sizes)
                    best_center = center
    
    return best_obb, best_center

def get_rotation_matrix_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    one_minus_cos = 1 - cos_theta
    
    x, y, z = axis
    R = np.array([
        [cos_theta + x * x * one_minus_cos, x * y * one_minus_cos - z * sin_theta, x * z * one_minus_cos + y * sin_theta],
        [y * x * one_minus_cos + z * sin_theta, cos_theta + y * y * one_minus_cos, y * z * one_minus_cos - x * sin_theta],
        [z * x * one_minus_cos - y * sin_theta, z * y * one_minus_cos + x * sin_theta, cos_theta + z * z * one_minus_cos]
    ])
    return R


def get_obb_size_from_axes(obb):
    axes = np.array(obb.R).T  # 获取OBB的主轴
    points = np.asarray(obb.get_box_points())
    sizes = []
    for axis in axes:
        projected = np.dot(points - obb.center, axis)
        size = np.max(projected) - np.min(projected)
        sizes.append(size)
    return np.array(sizes)

def get_min_bound_box_rotation_search_fast(pcd, floor_transform_matrix=None):
    # print(f"使用凸包来过滤外部点之前, 点的数量: {len(pcd.points)}")
    
    try:
        hull, _ = pcd.compute_convex_hull()
        hull_pcd = o3d.geometry.PointCloud()
        hull_pcd.points = hull.vertices
        # print(f"使用凸包来过滤外部点之后, 点的数量: {len(hull_pcd.points)}")
    except RuntimeError as e:
        # print(f"计算凸包时出错: {e}")
        # print("使用原始点云进行旋转搜索")
        hull_pcd = pcd
    
    points = np.asarray(hull_pcd.points)
    min_volume = float('inf')
    best_obb = None
    
    if floor_transform_matrix is not None:
        # Extract the z-axis from the floor transform matrix
        z_axis = floor_transform_matrix[:3, 2]
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Define a rotation matrix around the z-axis
        def get_rotation_matrix_around_z(angle):
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
            return R

        angle_range = np.linspace(0, np.pi, 18)  # Full rotation around z-axis
        for angle in angle_range:
            R_z = get_rotation_matrix_around_z(angle)
            R = floor_transform_matrix[:3, :3].dot(R_z)
            rotated_points = np.dot(points, R)
            
            min_point = np.min(rotated_points, axis=0)
            max_point = np.max(rotated_points, axis=0)
            sizes = max_point - min_point
            volume = np.prod(sizes)
            
            if volume < min_volume:
                min_volume = volume
                center_rotated = (min_point + max_point) / 2
                center = np.dot(center_rotated, R.T)
                best_obb = o3d.geometry.OrientedBoundingBox(center, R, sizes)
    else:
        angle_x_range = np.linspace(0, np.pi, 18)
        angle_y_range = np.linspace(0, np.pi, 18)
        angle_z_range = np.linspace(0, np.pi, 18)
        
        for angle_x in angle_x_range:
            for angle_y in angle_y_range:
                for angle_z in angle_z_range:
                    R = o3d.geometry.get_rotation_matrix_from_xyz((angle_x, angle_y, angle_z))
                    rotated_points = np.dot(points, R)
                    
                    min_point = np.min(rotated_points, axis=0)
                    max_point = np.max(rotated_points, axis=0)
                    sizes = max_point - min_point
                    volume = np.prod(sizes)
                    
                    if volume < min_volume:
                        min_volume = volume
                        center_rotated = (min_point + max_point) / 2
                        center = np.dot(center_rotated, R.T)
                        best_obb = o3d.geometry.OrientedBoundingBox(center, R, sizes)

    return hull_pcd, best_obb


def remove_statistical_outliers(points, k=20, z_max=2):
    z_scores = stats.zscore(points)
    return points[np.abs(z_scores) < z_max]

def check_local_consistency(depth_map, window_size=5, threshold=0.1):
    # 计算中值滤波
    local_median = median_filter(depth_map, size=window_size, mode='nearest')
    
    # 创建一个掩码来标记不一致的值
    mask = np.abs(depth_map - local_median) > threshold
    result = np.where(mask, np.nan, depth_map)
    
    return result

def remove_scattered_points(depth_map, eps=0.1, min_samples=5):
    # 创建有效点的坐标列表
    rows, cols = np.where(~np.isnan(depth_map))
    points = np.column_stack((rows, cols, depth_map[rows, cols]))
    
    # 应用DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    
    # 创建新的深度图，只保留聚类点
    new_depth_map = np.full_like(depth_map, np.nan)
    clustered_points = points[labels != -1]
    new_depth_map[clustered_points[:, 0].astype(int), clustered_points[:, 1].astype(int)] = clustered_points[:, 2]
    
    return new_depth_map

def estimate_obj_depth_obb_faster(depth_mm, obj_mask, wall_floor_pose=None):
    # time_0 = time.time()
    obj_mask = (obj_mask > 128).astype(np.uint8)
    # color_image = np.array(Image.open(color_image_path))

    # 1. 预处理
    masked_depth = depth_mm * obj_mask
    masked_depth[obj_mask == 0] = np.nan
    # time_1 = time.time()
    # print(f'预处理 花费时间: {time_1-time_0}s')
    
    # 2. 去除物体外轮廓
    kernel = np.ones((3,3), np.uint8)
    eroded_mask = cv2.erode(obj_mask, kernel, iterations=1)
    dilated_mask = cv2.dilate(obj_mask, kernel, iterations=1)
    contour_mask = dilated_mask - eroded_mask
    masked_depth[contour_mask > 0] = np.nan
    # time_2 = time.time()
    # print(f'去除物体外轮廓 花费时间: {time_2-time_1}s')
    
    # 3. 异常值移除，保留1-99
    valid_depths = masked_depth[~np.isnan(masked_depth)]
    if len(valid_depths) == 0:
        return None, None, None, None
    percentile_5 = np.percentile(valid_depths, 1)
    percentile_95 = np.percentile(valid_depths, 99)
    masked_depth[(masked_depth < percentile_5) | (masked_depth > percentile_95)] = np.nan
    # time_3 = time.time()
    # print(f'异常值移除，保留1 花费时间: {time_3-time_2}s')
    
    # 4. 双边滤波去噪
    masked_depth_filtered = cv2.bilateralFilter(masked_depth.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)
    # time_4 = time.time()
    # print(f'双边滤波去噪 花费时间: {time_4-time_3}s')
    
    # 5. 局部深度一致性检查： 检查每个点的邻域，如果某个点的深度值与其邻域的平均深度值相差太大，则将其视为异常点。
    filtered_depth = check_local_consistency(masked_depth_filtered, window_size=5, threshold=0.1)
    # time_5 = time.time()
    # print(f'局部深度一致性检查： 花费时间: {time_5-time_4}s')
    
    if not np.all(np.isnan(filtered_depth)):
        # 6. 基于密度的聚类算法来过滤掉离群点  DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 
        filtered_depth = remove_scattered_points(filtered_depth, eps=3, min_samples=3)
        # time_6 = time.time()
        # print(f'基于密度的聚类算法来过滤掉离群点 花费时间: {time_6-time_5}s')   # <0.2s
    # 检查是否所有点都被过滤掉了
    if np.all(np.isnan(filtered_depth)):
        pass
        # print("Warning: All points were filtered out by 局部深度一致性检查和DBSCAN. Using original depth map.")
    else:
        masked_depth_filtered = filtered_depth
   
    if np.all(np.isnan(masked_depth_filtered)):
        masked_depth_filtered = masked_depth.astype(np.float32)
        
    # 创建点云
    time_7 = time.time()
    height, width = depth_mm.shape
    estimated_focal_length = int(30/36*width)
    # estimated_focal_length = int(50/36*width)
    fx = fy = estimated_focal_length
    cx, cy = width / 2, height / 2

    y, x = np.where(~np.isnan(masked_depth_filtered))
    z = masked_depth_filtered[y, x] / 1000.0  # 转换为米
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy

    points = np.stack((x_3d, y_3d, z), axis=-1)

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # time_8 = time.time()
    # print(f'创建Open3D点云对象 花费时间: {time_8-time_7}s')
    
    # 计算最小包围框
    if wall_floor_pose is not None and 'floor_0' in wall_floor_pose and 'matrix' in wall_floor_pose['floor_0']:
        floor_transform_matrix = np.array(wall_floor_pose['floor_0']['matrix'])
        # print(f'启用scene graph优化位置和scale !!!')
    else:
        floor_transform_matrix = None
        # print(f'未启用scene graph优化位置和scale !!!')

    # obb, center = get_min_bound_box_pca(pcd)  # PCA方法
    # obb, center = get_min_bound_box_optimized_pca(pcd)  # 优化的PCA方法
    # obb, center = get_min_bound_box_convex_hull(pcd)  # 凸包方法
    # obb, center = get_min_bound_box_rotation_search(pcd)  # 旋转搜索方法, 最准，但时间很长 60s
    hull_pcd, obb = get_min_bound_box_rotation_search_fast(pcd, floor_transform_matrix)  # 旋转搜索方法, 最准，但时间很长 60s
    # time_9 = time.time()
    # print(f'计算最小包围框 花费时间: {time_9-time_8}s')
    
    # 8. 深度估计
    center = obb.center
    center_depth = center[2] * 1000  # 转回毫米

    estimated_depth = {
        'mean': np.mean(z) * 1000,
        'median': np.median(z) * 1000,
        'min': np.min(z) * 1000,
        'max': np.max(z) * 1000,
        'center': center_depth
    }

    return estimated_depth, obb, pcd, hull_pcd

def downsample_point_cloud(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size)

def create_point_cloud(color_image, depth_image, fx, fy, cx, cy):
    height, width = depth_image.shape
    y, x = np.where(~np.isnan(depth_image))
    z = depth_image[y, x] / 1000.0  # 转换为米
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy
    points = np.stack((x_3d, y_3d, z), axis=-1)
    colors = color_image[y, x] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def create_full_connection_line_set(obb, color):
    import itertools
    bbox_points = np.asarray(obb.get_box_points())
    num_points = len(bbox_points)
    
    # Generate all possible point pairs
    bbox_lines = list(itertools.combinations(range(num_points), 2))
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(bbox_points),
        lines=o3d.utility.Vector2iVector(bbox_lines)
    )
    line_set.paint_uniform_color(color)
    return line_set

def sample_lines_as_point_cloud(line_set, num_samples=2000):
    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)
    colors = np.asarray(line_set.colors)

    sampled_points = []
    sampled_colors = []

    for line, color in zip(lines, colors):
        start_point = points[line[0]]
        end_point = points[line[1]]
        
        # 计算 alpha 的值
        alphas = np.linspace(0, 1, num_samples)
        
        # 通过广播计算 sampled_points
        interpolated_points = (1 - alphas[:, None]) * start_point + alphas[:, None] * end_point
        
        # 将 interpolated_points 和颜色添加到结果列表
        sampled_points.append(interpolated_points)
        sampled_colors.append(np.full((num_samples, 3), color))  # 假设 color 是 RGB 格式
        
    # 将列表转换为 numpy 数组
    sampled_points = np.vstack(sampled_points)
    sampled_colors = np.vstack(sampled_colors)
    
    return sampled_points, sampled_colors

def calculate_compensate_size_on_axis(obb_axes, obb_sizes, parent_thickness, closest_axis_index, normal):
    closest_axis = obb_axes[closest_axis_index]
    
    # 如果closest_axis与normal同向，直接返回obb_sizes之和的一半
    if np.allclose(closest_axis, normal) or np.allclose(closest_axis, -normal):
        return (obb_sizes[closest_axis_index] + parent_thickness) / 2
    
    half_sizes = obb_sizes / 2
    # closest_axis和normal是xzy坐标系, corner_offsets是xyz坐标系，需要转换
    corner_offsets = np.array([
        [half_sizes[0], half_sizes[2], half_sizes[1]],
        [half_sizes[0], half_sizes[2], -half_sizes[1]],
        [half_sizes[0], -half_sizes[2], half_sizes[1]],
        [half_sizes[0], -half_sizes[2], -half_sizes[1]],
        [-half_sizes[0], half_sizes[2], half_sizes[1]],
        [-half_sizes[0], half_sizes[2], -half_sizes[1]],
        [-half_sizes[0], -half_sizes[2], half_sizes[1]],
        [-half_sizes[0], -half_sizes[2], -half_sizes[1]],
    ])
    projections = np.dot(corner_offsets, closest_axis)
    positive_projections = projections[projections > 0]
    max_projection = np.max(positive_projections) if positive_projections.size > 0 else 0
    
    # 减去父物体的厚度, 这里不希望让物体obb减小  根据先验来看 本来就obb就估计的偏小
    return max(0, max_projection - parent_thickness / 2)


def refine_obb(obb, wall_floor_pose, parent, against_wall):
    obb_axes = np.array(obb.R).T
    obb_sizes = np.array(obb.extent)
    obb_center = np.array(obb.center)

    parent_info = wall_floor_pose[parent]
    obb_center, obb_sizes = adjust_obb_size_and_center(obb_center, obb_sizes, np.array(parent_info["matrix"]), obb_axes)
    
    # # 被遮挡物体的靠墙判断还不够准，暂时关闭against_wall的优化
    # if against_wall in wall_floor_pose and against_wall != parent:
    #     wall_info = wall_floor_pose[against_wall]
    #     obb_center, obb_sizes = adjust_obb_size_and_center(obb_center, obb_sizes, np.array(wall_info["matrix"]), obb_axes)
    return o3d.geometry.OrientedBoundingBox(obb_center, obb.R, obb_sizes)

def adjust_obb_size_and_center(obb_center, obb_sizes, transform_matrix, obb_axes):
    normal = transform_matrix[:3, 2]
    closest_axis_index = np.argmax(np.abs(np.dot(obb_axes, normal)))
    closest_axis = obb_axes[closest_axis_index]

    if np.dot(closest_axis, normal) > 0:
        closest_axis = -closest_axis

    center_to_center_distance = np.dot(obb_center - transform_matrix[:3, 3], normal)

    # 计算补偿大小
    parent_thickness = 0 # 认为墙的厚度是0
    compensate_size_on_axis = calculate_compensate_size_on_axis(obb_axes, obb_sizes, parent_thickness, closest_axis_index, normal)
    additional_length = max(0, np.abs(center_to_center_distance) - compensate_size_on_axis)

    # 更新OBB尺寸和中心
    obb_sizes[closest_axis_index] += additional_length
    obb_center += closest_axis * (additional_length / 2)

    return obb_center, obb_sizes

def refine_obb_with_parent_along_floor(obb, parent_obb, floor_normal):
    obb_axes = np.array(obb.R).T
    obb_sizes = np.array(obb.extent)
    obb_center = np.array(obb.center)
    
    parent_obb_sizes = np.array(parent_obb.extent)
    parent_obb_center = np.array(parent_obb.center)
    
    closest_axis_index = np.argmax(np.abs(np.dot(obb_axes, floor_normal)))
    closest_axis = obb_axes[closest_axis_index]
    
    if np.dot(closest_axis, floor_normal) > 0:
        closest_axis = -closest_axis
        
    # 计算两个obb中心沿着floor_normal的长度
    center_to_center_distance = np.dot(obb_center - parent_obb_center, floor_normal)
    parent_thickness = parent_obb_sizes[closest_axis_index]
    compensate_size_on_axis = calculate_compensate_size_on_axis(obb_axes, obb_sizes, parent_thickness, closest_axis_index, floor_normal)
    additional_length = max(0, np.abs(center_to_center_distance) - compensate_size_on_axis)

    # 更新OBB尺寸和中心
    obb_sizes[closest_axis_index] += additional_length
    obb_center += closest_axis * (additional_length / 2)

    return o3d.geometry.OrientedBoundingBox(obb_center, obb.R, obb_sizes)
    
def refine_all_obbs_with_scene_graph(obbs, names, wall_floor_pose, scene_graph):
    refined_obbs_dict = {}
    floor_normal = np.array(wall_floor_pose['floor_0']['matrix'])[:3, 2]
    processed_objects = set()

    def process_object(name, obb):
        if name in processed_objects:
            return  # 已经处理过，直接返回

        processed_objects.add(name)
        obb_scene_graph = scene_graph.get(name, {})
        parent = obb_scene_graph.get("supported")
        against_wall = obb_scene_graph.get("againstWall")

        # Treat certain parents as floor_0
        if parent and re.match(r'^(floor|rug|carpet)_\d+', parent):
            parent = 'floor_0'

        # 目前只处理一级摆放物体
        if not re.match(r'^(wall|floor)_\d+', parent):
            refined_obbs_dict[name] = obb
            return

        if parent in wall_floor_pose:
            refined_obb = refine_obb(obb, wall_floor_pose, parent, against_wall)
            refined_obbs_dict[name] = refined_obb
        elif parent in refined_obbs_dict:
            parent_obb = refined_obbs_dict[parent]
            refined_obb = refine_obb_with_parent_along_floor(obb, parent_obb, floor_normal)
            refined_obbs_dict[name] = refined_obb
        else:
            # 父物体还未处理，先处理父物体
            if parent not in processed_objects:
                parent_index = names.index(parent)
                process_object(parent, obbs[parent_index])
            
            # 再次尝试处理当前物体
            if parent in refined_obbs_dict:
                parent_obb = refined_obbs_dict[parent]
                refined_obb = refine_obb_with_parent_along_floor(obb, parent_obb, floor_normal)
                refined_obbs_dict[name] = refined_obb
            else:
                # 如果父物体仍然无法处理，可能存在循环依赖，直接使用原始 OBB
                print(f"Warning: Unable to process {name} due to circular dependency or missing parent. Using original OBB.")
                refined_obbs_dict[name] = obb

    for name, obb in zip(names, obbs):
        if name not in processed_objects:
            process_object(name, obb)

    return [refined_obbs_dict[name] for name in names]

def compute_axis_mapping(bbox_points, obb_size, epsilon=1e-6):
    # 计算所有点对之间的距离
    distances = np.linalg.norm(bbox_points[:, None] - bbox_points, axis=2)
    
    # 创建一个字典，用于存储每个尺寸对应的轴
    size_to_axis = {}
    used_axes = set()

    for size in obb_size:
        # 找到与当前size最接近的距离
        closest_distance_idx = np.argmin(np.abs(distances - size))
        i, j = np.unravel_index(closest_distance_idx, distances.shape)
        
        # 计算这对点之间的向量
        axis = bbox_points[i] - bbox_points[j]
        
        # 标准化向量，添加一个小的epsilon值以避免除以零
        norm = np.linalg.norm(axis)
        if norm < epsilon:
            continue  # 跳过长度接近零的向量
        axis = axis / (norm + epsilon)
        
        # 找到这个向量最接近的世界坐标轴
        world_axis = np.argmax(np.abs(axis))
        
        # 如果这个轴已经被使用，尝试次优的轴
        if world_axis in used_axes:
            sorted_axes = np.argsort(np.abs(axis))[::-1]
            for ax in sorted_axes:
                if ax not in used_axes:
                    world_axis = ax
                    break
            else:
                continue  # 如果找不到未使用的轴，跳过这个尺寸
        
        size_to_axis[size] = world_axis
        used_axes.add(world_axis)

    if len(size_to_axis) != 3:
        # 如果找不到三个唯一的轴，使用默认映射
        print("Warning: Could not find three unique axes. Using default mapping.")
        return [0, 1, 2]

    # 创建轴映射
    axis_mapping = [size_to_axis[size] for size in obb_size]
    
    return axis_mapping

def generate_pcd_mask(save_path, obj_pcds, names, scene_pcd, floor_matrix, project_plane='xoy', fill_mask=False):
    # Convert 4x4 matrix to 6-element pose vector (translation + normal)
    normal_vector = floor_matrix[:3, 2]
    point_on_plane = scene_pcd.get_center()
    floor_pose = list(point_on_plane) + list(normal_vector)

    x_range_full, y_range_full = get_pcd_range(scene_pcd, floor_pose, project_plane)
    mask_list = []
    
    for id, pcd in enumerate(obj_pcds):
        project_pcd, _, _ = project_points_to_floor(pcd, floor_pose, project_plane)
        mask = create_mask_from_projected_pcd(project_pcd, x_range_full, y_range_full, resolution=512, fill_mask=fill_mask)
        mask_list.append(mask)
        cv2.imwrite(os.path.join(save_path, f'{names[id]}_{project_plane}.png'), mask)

    if project_plane == 'xoy':
        s0_pcd_mask_dir = save_path
        s0_dir = os.path.dirname(s0_pcd_mask_dir)
        save_parent_folder = os.path.dirname(s0_dir)
        input_dir = os.path.join(save_parent_folder, 'S1_scene_parsing_results')
        
        if not os.path.isdir(input_dir):
            print(f"Warning: Wall PCD directory not found at {input_dir}, skipping wall mask generation.")
            return mask_list

        for file in os.listdir(os.path.join(input_dir, 'wall_pcd')):
            if not re.match(r'^wall_\d+\.pcd$', file):
                continue  # 跳过不匹配的文件
            name_without_extension = file.split('.')[0]
            pcd = o3d.io.read_point_cloud(os.path.join(input_dir, 'wall_pcd', file))
            projected_points, _, _ = project_points_to_floor(pcd, floor_pose, project_plane)
            mask = create_mask_from_projected_pcd(projected_points, x_range_full, y_range_full, resolution=512, fill_mask=fill_mask)
            cv2.imwrite(os.path.join(save_path, f'{name_without_extension}_{project_plane}.png'), mask)

    full_project_pcd, _, _ = project_points_to_floor(scene_pcd, floor_pose, project_plane)
    full_mask = create_mask_from_projected_pcd(full_project_pcd, x_range_full, y_range_full, resolution=512)
    cv2.imwrite(os.path.join(save_path, f'full_{project_plane}.png'), full_mask)

    return mask_list

def create_mask_from_projected_pcd(projected_points, x_range, y_range, resolution=512,fill_mask=False):
    # 将投影点缩放到掩膜分辨率内
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    #对齐坐标系长度
    x_len = x_max - x_min
    y_len = y_max - y_min
    max_len = max(x_len,y_len)
    if x_len<max_len:
        middle = (x_max+x_min)/2
        x_max = middle + max_len/2 
        x_min = middle - max_len/2
    if y_len<max_len:
        middle = (y_max+y_min)/2
        y_max = middle + max_len/2 
        y_min = middle - max_len/2  
        
    # 使用分辨率将坐标映射到掩膜网格
    x_scaled = ((projected_points[:, 0] - x_min) / (x_max - x_min) * resolution).astype(int)
    y_scaled = ((projected_points[:, 1] - y_min) / (y_max - y_min) * resolution).astype(int)
    
    # 创建掩膜
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    
    # 标记掩膜上的投影点
    for x, y in zip(x_scaled, y_scaled):
        if 0 <= x < resolution and 0 <= y < resolution:
            mask[y, x] = 255
    

    if fill_mask:
        # 膨胀操作
        kernel = np.ones((2, 2), np.uint8)  # 定义膨胀核大小
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)  # 膨胀操作
        mask = cv2.erode(dilated_mask, kernel, iterations=1)  # 腐蚀操作
        
        # 1. 提取掩码的轮廓点
        points = np.column_stack(np.where(mask > 0))  # 获取所有前景像素的坐标
        points = points[:, ::-1]  # 将坐标从 (y, x) 转换为 (x, y)

        # 2. 计算凸包
        hull = cv2.convexHull(points)  # 计算凸包

        # 3. 创建一个空白掩码用于绘制凸包
        hull_mask = np.zeros_like(mask)

        # 4. 绘制凸包并填充内部
        mask = cv2.drawContours(hull_mask, [hull], 0, 255, -1)  # thickness=-1 表示填充内部

    return mask

def transform_to_floor_pose(pcd, floor_pose):
    pcd = o3d.geometry.PointCloud(pcd)
    
    translation = np.array(floor_pose[:3])
    normal = np.array(floor_pose[3:])      
    
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(normal, z_axis)
    rotation_angle = np.arccos(np.dot(normal, z_axis) / (np.linalg.norm(normal) * np.linalg.norm(z_axis)))

    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

    pcd.translate(-translation)
    pcd.rotate(rotation_matrix, center=(0, 0, 0))
    
    return pcd

def project_points_to_floor(pcd, floor_pose, project_plane='xoy'):
    transformed_pcd = transform_to_floor_pose(pcd, floor_pose)
    points = np.asarray(transformed_pcd.points)

    if points.size == 0:
        print("Warning: Empty point cloud")
        return np.array([]), [0, 0], [0, 0]

    projected_points = None

    if project_plane == 'xoy':
        projected_points = points[:, [0,1]]
        projected_points[:,1] = -projected_points[:,1]
    elif project_plane == 'yoz':
        projected_points = points[:, [0,2]]
        projected_points[:,1] = -projected_points[:,1]
    else:
        raise ValueError(f"Unsupported projection plane: {project_plane}")

    projected_points = np.array(projected_points).reshape(-1, 2)

    if projected_points.size == 0:
        print("Warning: No points after projection")
        return np.array([]), [0, 0], [0, 0]

    x_range = [np.min(projected_points[:,0]), np.max(projected_points[:,0])]
    y_range = [np.min(projected_points[:,1]), np.max(projected_points[:,1])]

    if x_range[0] == x_range[1] or y_range[0] == y_range[1]:
        print("Warning: Invalid range detected")
        return projected_points, [0, 1], [0, 1]
    
    return projected_points, x_range, y_range

def get_pcd_range(pcd, floor_pose, project_plane='xoy'):
    """
    获取点云在投影平面上的范围，用于生成掩膜。
    """
    transformed_pcd = transform_to_floor_pose(pcd, floor_pose)
    points = np.asarray(transformed_pcd.points)

    if points.size == 0:
        print("Warning: Empty point cloud for range calculation.")
        return [0, 1], [0, 1]

    projected_points = None

    if project_plane == 'xoy':
        # 投影到xoy平面，忽略z轴
        projected_points = points[:, [0,1]]
        projected_points[:,1] = -projected_points[:,1]
    elif project_plane == 'yoz':
        # 投影到yoz平面，忽略x轴
        projected_points = points[:, [0,2]]
        projected_points[:,1] = -projected_points[:,1]

    else:
        raise ValueError(f"Unsupported projection plane: {project_plane}")

    # 确保 projected_points 是一个 2D numpy 数组
    projected_points = np.array(projected_points).reshape(-1, 2)

    if projected_points.size == 0:
        print("Warning: No points after projection for range calculation.")
        return [0, 1], [0, 1]

    x_range = [np.min(projected_points[:,0]), np.max(projected_points[:,0])]
    y_range = [np.min(projected_points[:,1]), np.max(projected_points[:,1])]

    # 检查范围是否有效
    if x_range[0] == x_range[1] or y_range[0] == y_range[1]:
        print("Warning: Invalid range detected for range calculation.")
        return [0, 1], [0, 1]  # 返回一个小的有效范围
    
    return x_range, y_range
        
def vis_obb(obbs, obb_color_list=None, scene_pcd=None):
    #vis
    obb_vis = []
    
    if obb_color_list is None:
        obb_color_list = [np.random.rand(3) for _ in range(len(obbs))]
        
    for obb,color in zip(obbs,obb_color_list):
        # 获取 OBB 的 8 个角点坐标和边连接信息
        corners = np.asarray(obb.get_box_points())  # 获取 8 个角点的坐标
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面四条边
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面四条边
            [0, 4], [1, 5], [2, 6], [3, 7]   # 连接上下四条边
        ]

        # 创建 LineSet 来手动显示包围盒边框
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)  # 设置顶点
        line_set.lines = o3d.utility.Vector2iVector(lines)  # 设置边的连接信息
        # 设置线条颜色
        line_set.paint_uniform_color(color)  
        obb_vis.append(line_set)
    
    if scene_pcd is not None:
        obb_vis.append(scene_pcd)
        
    o3d.visualization.draw_geometries(obb_vis)

def _process_single_obb_fitting(obj_mask_path, depth_mm, wall_floor_pose):
    # Set environment variables to limit OpenMP threads in child processes to avoid deadlock/contention
    os.environ["OMP_NUM_THREADS"] = "1"
    
    obj_mask = np.array(Image.open(obj_mask_path))
    _, obb, pcd, hull_pcd = estimate_obj_depth_obb_faster(depth_mm, obj_mask, wall_floor_pose)
    
    result = None
    if obb is not None:
        name = (obj_mask_path.split('/')[-1]).split('_mask.png')[0]
        
        color = np.random.rand(3)
        line_set_before = create_full_connection_line_set(obb, color)
        sampled_points_before, sampled_colors_before = sample_lines_as_point_cloud(line_set_before)
        
        # Extract necessary data for reconstruction to avoid pickling Open3D objects
        obb_data = {
            'center': np.asarray(obb.center),
            'R': np.asarray(obb.R),
            'extent': np.asarray(obb.extent)
        }
        
        pcd_data = {
            'points': np.asarray(pcd.points),
            'colors': np.asarray(pcd.colors) if pcd.has_colors() else None
        }

        result = {
            'obb_data': obb_data,
            'name': name,
            'pcd_data': pcd_data,
            'color': color,
            'sampled_points_before': sampled_points_before,
            'sampled_colors_before': sampled_colors_before
        }
    return result

def cal_and_visualize_scene_obj_bbox_fitting(depth_image_path, obj_mask_path_list, color_image_path, save_path=None, wall_floor_pose=None, scene_graph=None, generate_pcd_mask_plane = [], visualize_combined_pcd=False):
    # 创建场景彩色点云
    image = cv2.imread(color_image_path)
    color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = color_image.shape[:2]
    
    estimated_focal_length = int(30/36*width)
    # estimated_focal_length = int(50/36*width)
    fx = estimated_focal_length
    fy = estimated_focal_length
    cx = width / 2
    cy = height / 2
    
    depth_mm = np.array(Image.open(depth_image_path)).astype(np.float32)
    scene_pcd = create_point_cloud(color_image, depth_mm, fx, fy, cx, cy)
    scene_pcd = downsample_point_cloud(scene_pcd, voxel_size=0.01)
    
    all_sampled_points = []
    all_sampled_colors = []
    center_list = []
    obb_size_list = []
    obb_rotation_list = []
    obb_points_list = []
    
    obb_list = []
    refine_pcd = []
    
    all_sampled_points_before = []
    all_sampled_colors_before = []

    
    obbs = []
    names = []
    obb_color_list = []
    obj_pcds = []

    # Parallel processing
    num_processes = min(4, len(obj_mask_path_list))
    process_func = partial(_process_single_obb_fitting, depth_mm=depth_mm, wall_floor_pose=wall_floor_pose)
    
    # Use list(tqdm(...)) to show progress of submission/processing if we wanted, but map blocks.
    # To show progress with map, we can use imap.
    print(f"Starting parallel OBB fitting with {num_processes} processes...")
    results = []
    
    # Use 'spawn' context to avoid deadlocks with Open3D/OpenCV/NumPy
    try:
        ctx = multiprocessing.get_context('spawn')
    except ValueError:
        # Fallback for systems that don't support spawn (unlikely for Linux/Windows/macOS modern python)
        ctx = multiprocessing
        
    with ctx.Pool(processes=num_processes) as pool:
        for res in tqdm(pool.imap(process_func, obj_mask_path_list), total=len(obj_mask_path_list)):
            results.append(res)

    for res in results:
        if res is not None:
            # Reconstruct Open3D objects
            obb_data = res['obb_data']
            obb = o3d.geometry.OrientedBoundingBox(obb_data['center'], obb_data['R'], obb_data['extent'])
            
            pcd_data = res['pcd_data']
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_data['points'])
            if pcd_data['colors'] is not None:
                pcd.colors = o3d.utility.Vector3dVector(pcd_data['colors'])

            obbs.append(obb)
            names.append(res['name'])
            obj_pcds.append(pcd)
            obb_color_list.append(res['color'])
            all_sampled_points_before.append(res['sampled_points_before'])
            all_sampled_colors_before.append(res['sampled_colors_before'])
    
    if wall_floor_pose is not None:
        #project to floor plane
        save_path = depth_image_path.replace('S0_geometry_pred_results', 'S1_scene_parsing_results').replace('depth.png', 'pcd_mask')
        floor_matrix =  np.array(wall_floor_pose['floor_0']['matrix'])

        for plane in generate_pcd_mask_plane:
            generate_pcd_mask(save_path,obj_pcds,names,scene_pcd,floor_matrix,project_plane=plane,fill_mask=True)
        #vis
        # vis_obb(obbs, obb_color_list, scene_pcd)

    # Refine all OBBs together
    if scene_graph is not None:
        print(f'根据scene graph摆放关系和层级, 迭代优化obb', flush=True)
        obbs = refine_all_obbs_with_scene_graph(obbs, names, wall_floor_pose, scene_graph)
        #vis
        # vis_obb(obbs, obb_color_list, scene_pcd)

    # 根据obb和wall_floor_pose，绘制一个顶视布局图  投影到地面上  且图片的xy方向与wall对齐
    for obb in obbs:
        if obb is not None:
            center = obb.center
            center_list.append(center.tolist())
            rotation = obb.R
            
            obb_rotation_list.append(rotation.tolist())
            
            obb_size = obb.extent
            bbox_points = np.asarray(obb.get_box_points())
            # 保存8个角点坐标
            obb_points_list.append(bbox_points.tolist())
        
            # 已知obb_size list, [len_1, len_2, len_3], 根据obb的八个顶点坐标 bbox_points, 对这个obb_size重新排序, 使其和世界坐标系对应
            axis_mapping = compute_axis_mapping(bbox_points, obb_size)
            # 由于blender坐标系和深度估计坐标系有差别, 将xyz转为xzy
            reordered_sizes = [obb_size[axis_mapping.index(i)] for i in [0,2,1]]
            obb_size_list.append(reordered_sizes)

            color = np.random.rand(3)
            line_set = create_full_connection_line_set(obb, color)
            sampled_points, sampled_colors = sample_lines_as_point_cloud(line_set)
            all_sampled_points.append(sampled_points)
            all_sampled_colors.append(sampled_colors)
            

    # # Combine all sampled points and colors for before refine
    # all_sampled_points_before = np.vstack(all_sampled_points_before)
    # all_sampled_colors_before = np.vstack(all_sampled_colors_before)

    # # Create a point cloud for before refine
    # before_refine_bbox_pcd = o3d.geometry.PointCloud()
    # before_refine_bbox_pcd.points = o3d.utility.Vector3dVector(all_sampled_points_before)
    # before_refine_bbox_pcd.colors = o3d.utility.Vector3dVector(all_sampled_colors_before)

    # combined_pcd_before = scene_pcd + before_refine_bbox_pcd

    # Save the combined point cloud before refine
    # save_path_before = save_path.replace('.ply', '_before_refine.ply')
    # o3d.io.write_point_cloud(save_path_before, combined_pcd_before)
    # print(f"Combined point cloud before refine saved: {save_path_before}")

    if visualize_combined_pcd: # 可视化组合点云
        # Combine all sampled points and colors for after refine
        all_sampled_points = np.vstack(all_sampled_points)
        all_sampled_colors = np.vstack(all_sampled_colors)

        # Create a point cloud for after refine
        bbox_pcd = o3d.geometry.PointCloud()
        bbox_pcd.points = o3d.utility.Vector3dVector(all_sampled_points)
        bbox_pcd.colors = o3d.utility.Vector3dVector(all_sampled_colors)

        combined_pcd = scene_pcd + bbox_pcd

        # Save the combined point cloud after refine
        combined_pcd_save_path = save_path = depth_image_path.replace('S0_geometry_pred_results', 'S1_scene_parsing_results').replace('depth.png', 'combined_pcd.ply')
        o3d.io.write_point_cloud(combined_pcd_save_path, combined_pcd)
        print(f"Combined point cloud saved: {combined_pcd_save_path}")

    return center_list, obb_rotation_list, obb_size_list, obb_points_list