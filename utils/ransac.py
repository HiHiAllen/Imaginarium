import open3d as o3d
import numpy as np
import copy
import cv2
import pickle
import re
from sklearn.cluster import DBSCAN
import time
from scipy.ndimage import median_filter
import json
import os

# cp /opt/data/private/allenxmzhu/3dLayOut/gigapose/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu
# rm /usr/lib/x86_64-linux-gnu/libstdc++.so.6
# ln -s /opt/data/private/allenxmzhu/3dLayOut/gigapose/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/libstdc++.so.6

def get_min_bound_box_rotation_search_fast(pcd):
    print(f"使用凸包来过滤外部点之前, 点的数量: {len(pcd.points)}")
    
    try:
        # 计算凸包
        hull, _ = pcd.compute_convex_hull()
        # 转换为点云对象
        hull_pcd = o3d.geometry.PointCloud()
        hull_pcd.points = hull.vertices
        print(f"使用凸包来过滤外部点之后, 点的数量: {len(hull_pcd.points)}")
    except RuntimeError as e:
        print(f"计算凸包时出错: {e}")
        print("使用原始点云进行旋转搜索")
        hull_pcd = pcd
    
    points = np.asarray(hull_pcd.points)
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
    
    return hull_pcd, best_obb, best_center

def get_camera_intrinsics(color_cv_image):
    color_image = cv2.cvtColor(color_cv_image, cv2.COLOR_BGR2RGB)
    height, width = color_image.shape[:2]  # 修改这里
    estimated_focal_length = int(30/36*width)
    # estimated_focal_length = int(50/36*width)
    fx = estimated_focal_length
    fy = estimated_focal_length
    cx = width / 2
    cy = height / 2
    return fx, fy, cx, cy

def create_point_cloud(color_image, depth_image, fx, fy, cx, cy):
    # 创建坐标网格
    rows, cols = depth_image.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    
    # 计算3D点
    z = depth_image / 1000.0  # 将深度从毫米转换为米
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy
    
    # 创建有效点的掩码（过滤掉NaN值）
    valid_mask = ~np.isnan(z)
    
    # 应用掩码过滤无效点
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    z_valid = z[valid_mask]
    points = np.stack((x_valid, y_valid, z_valid), axis=-1)
    
    # 过滤对应的颜色值
    colors = color_image.reshape(-1, 3) / 255.0  # 归一化颜色值
    colors_valid = colors[valid_mask.reshape(-1)]
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_valid)
    
    return pcd

def fit_floor_pcd(floor_pcd, distance_threshold = 0.1, ransac_n = 3, num_iterations = 5000):
    floor_normal = None
    floor_cloud = None
    found_floor = False  # 新增一个标志变量
    print("\n正在寻找地面...")

    while True:
        if len(floor_pcd.points) < ransac_n:
            print("点云中的点不足以进行RANSAC平面分割。")
            break  # 跳出循环

        plane_model, inliers = floor_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        plane_cloud = floor_pcd.select_by_index(inliers)
        [a, b, c, d] = plane_model
        normal_vector = np.array([a, b, c])
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        y_axis = np.array([0, 1, 0])

        alignment_with_y = abs(np.dot(normal_vector, y_axis))
        if alignment_with_y > 0.7:  # 与Y轴接近平行
            found_floor = True  # 更新标志变量
            floor_normal = normal_vector
            floor_cloud = plane_cloud
            floor_cloud.paint_uniform_color([0.6, 0.3, 0])  # 棕色
            print(f"找到地面！")
            print(f"- 与Y轴的平行度: {alignment_with_y:.3f}")
            print(f"- 方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
            break  # 找到地面后跳出循环

        if not found_floor:
            # 只有在没有找到地面的情况下，才移除当前平面的点
            floor_pcd = floor_pcd.select_by_index(inliers, invert=True)
    # 可视化点云
    # o3d.visualization.draw_geometries([floor_cloud])
    return floor_cloud, floor_normal

def fit_walls_pcd_and_pred_xyz(point_cloud, floor_normal, distance_threshold = 0.1, ransac_n = 3, num_iterations = 5000, search_nums=4, known_wall_nums=False):
    # 定义颜色（根据平面数量动态生成）
    base_colors = [
        [1, 0, 0],    # 红色
        [0, 1, 0],    # 绿色
        [0, 0, 1],    # 蓝色
        [1, 1, 0],    # 黄色
        [1, 0, 1]     # 紫色
    ]
    COLOR = ["red", "green", "blue", "yellow", "purple"]
    # 如果平面数量超过预定义颜色，则循环使用
    colors = [base_colors[i % len(base_colors)] for i in range(search_nums)]

    # 存储所有平面的点云
    wall_clouds_list = []
    wall_normal_vectors_list = []
    wall_center_list = []
    remaining_cloud = copy.deepcopy(point_cloud)

    # 第二步：寻找墙面
    print("\n正在寻找墙面...")
    for i in range(search_nums): 
        if len(remaining_cloud.points) < ransac_n:
            print("点云中的点不足以进行RANSAC平面分割。")
            break  # 跳出循环
        
        # 检测平面
        plane_model, inliers = remaining_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        # 提取平面点云
        plane_cloud = remaining_cloud.select_by_index(inliers)
        
        # # 0. 异常处理：点云数量检查
        # if len(plane_cloud.points) < 5000:
        #     print(f"平面 {i+1} 点数过少 ({len(plane_cloud.points)}), 跳过")
        #     # 即使跳过，也要从剩余点云中移除，防止重复检测
        #     remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)
        #     continue
        
        # 检查平面是否与地面垂直
        [a, b, c, d] = plane_model
        normal_vector = np.array([a, b, c])
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        
        alignment_with_floor = abs(np.dot(normal_vector, floor_normal))
        if alignment_with_floor < 0.3:  # 允许一定的误差, (17.5度,其实可以更小一点)
            # 为平面设置颜色
            current_color = colors[i]
            plane_cloud.paint_uniform_color(current_color)

            # 获取OBB以检查尺寸
            _, best_obb, wall_center = get_min_bound_box_rotation_search_fast(plane_cloud)
            
            # 1. 异常处理：检测与其他墙面的夹角
            # 如果新检测到的墙面与已知墙面的夹角过小（例如小于30度），则认为是重复检测或噪声
            is_valid_angle = True
            for existing_normal in wall_normal_vectors_list:
                # 计算两个法向量的夹角余弦值
                angle_cos = abs(np.dot(normal_vector, existing_normal))
                # 限制在 [0, 1] 范围内以避免数值误差
                angle_cos = min(angle_cos, 1.0)
                angle_rad = np.arccos(angle_cos)
                angle_deg = np.degrees(angle_rad)
                
                # 如果夹角小于30度，则拒绝
                if angle_deg < 30:
                     print(f"平面 {i+1} 与已知墙面的夹角为 {angle_deg:.2f} 度 (<30度), 被视为重复/噪声, 跳过")
                    #  is_valid_angle = False
                    #  break
            
            if not is_valid_angle:
                 remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)
                 continue

            if known_wall_nums: # 如果已知墙的个数就不用计算下面密度比了
                print(f"平面 {i+1} 是墙面 - 与Y轴的垂直度: {alignment_with_floor:.3f}")
                print(f"方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
                print(f"平面颜色: {COLOR[i % len(base_colors)]}")
                # _, _, wall_center = get_min_bound_box_rotation_search_fast(plane_cloud) # 已经计算过了
                print(f"平面xyz为: {wall_center}")
                
                wall_clouds_list.append(plane_cloud)
                wall_normal_vectors_list.append(normal_vector)
                wall_center_list.append(wall_center)
                
                # 更新剩余点云 (修复bug: known_wall_nums为True时原来会跳过此步)
                remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)
                continue
                
                
            # 获取全部点云的点坐标（使用原始point_cloud而不是plane_cloud）
            points = np.asarray(point_cloud.points)
            
            # 计算每个点到平面的距离
            distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
            
            # 过滤掉abs距离小于0.2的点
            valid_points = points[np.abs(distances) >= 0.2]
            valid_distances = distances[np.abs(distances) >= 0.2]
            
            # 分别统计平面前后的点数
            front_points = valid_points[valid_distances > 0]
            back_points = valid_points[valid_distances < 0]
            
            # 计算密度比（前面点数 / 后面点数）
            raw_density_ratio = len(front_points) / len(back_points) if len(back_points) > 0 else float('inf')
            inverse_ratio = 1 / raw_density_ratio if raw_density_ratio != 0 else float('inf')
            density_ratio = min(raw_density_ratio, inverse_ratio)
            
            if density_ratio <= 0.05:  # 只有密度比大于等于0.2才被认为是墙面

                print(f"平面 {i+1} 是墙面 - 与Y轴的垂直度: {alignment_with_floor:.3f}")
                # print(f"方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
                # print(f"最小密度比: {density_ratio:.2f}")
                print(f"平面颜色: {COLOR[i % len(base_colors)]}")
                # _, _, wall_center = get_min_bound_box_rotation_search_fast(plane_cloud) # 已经计算过了
                print(f"平面xyz为: {wall_center}")
                
                wall_clouds_list.append(plane_cloud)
                wall_normal_vectors_list.append(normal_vector)
                wall_center_list.append(wall_center)
            else:
                print(f"平面 {i+1} 两侧点云密度比不满足要求 ({density_ratio:.2f}), 不被视为墙面")
        else:
            print(f"非墙面平面 {i+1} - 与Y轴的垂直度: {alignment_with_floor:.3f}")
            # print(f"方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        
        # 更新剩余点云
        remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)
        
    return wall_clouds_list, wall_normal_vectors_list, wall_center_list
    

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

def preprocess_depth(depth_image, obj_mask):
    # 1. 预处理
    if np.max(obj_mask) == 255:
        masked_depth = depth_image * (obj_mask / 255.0)
    elif np.max(obj_mask) == 1:
        masked_depth = depth_image * obj_mask
    return masked_depth

def create_custom_coordinate_frame_as_point_cloud(origin, normal, size=0.1, num_points=1200):
    # 归一化法向量
    normal = normal / np.linalg.norm(normal)

    # 创建一个初始坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    # 计算旋转矩阵，使 z 轴与法向量对齐
    z_axis = normal
    x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # 创建变换矩阵
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = origin

    # 应用变换
    coordinate_frame.transform(transformation_matrix)

    # 转换为点云
    coordinate_frame_pcd = coordinate_frame.sample_points_poisson_disk(number_of_points=num_points)

    # 确保颜色值在 [0, 1] 范围内
    if coordinate_frame_pcd.has_colors():
        colors = np.asarray(coordinate_frame_pcd.colors)
        np.clip(colors, 0, 1, out=colors)

    return coordinate_frame_pcd, transformation_matrix

def generate_mask_from_point_cloud(point_cloud, depth_image_shape, fx, fy, cx, cy):
    mask = np.zeros(depth_image_shape, dtype=np.uint8)
    points = np.asarray(point_cloud.points)

    for point in points:
        x, y, z = point
        if z > 0:  # 确保深度为正
            u = int((x * fx / z) + cx)
            v = int((y * fy / z) + cy)
            if 0 <= u < depth_image_shape[1] and 0 <= v < depth_image_shape[0]:
                mask[v, u] = 255

    return mask

def fill_and_close_mask(mask):
    # 定义内核大小
    kernel = np.ones((10, 10), np.uint8)

    # 先进行填充操作
    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 然后进行闭操作
    closed_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)

    return closed_mask

def find_floor(depth_image, floor_mask_img, color_image, voxel_size, fx, fy, cx, cy, distance_threshold, ransac_n, num_iterations):
    if not np.all(floor_mask_img == 0):
        # 如果有地面mask信息，就先匹配地面
        floor_masked_depth_filtered = preprocess_depth(depth_image, floor_mask_img)
    else:
        floor_masked_depth_filtered = depth_image  # 没有就直接用原图的depth_image
    
    floor_pcd = create_point_cloud(color_image, floor_masked_depth_filtered, fx, fy, cx, cy)
    # 添加降采样步骤
    floor_pcd = floor_pcd.voxel_down_sample(voxel_size)
    # floor_pcd.transform(flip_transform)
    floor_cloud, floor_normal = fit_floor_pcd(floor_pcd, distance_threshold, ransac_n, num_iterations)
    # 拟合地面的几何中心
    _, _, floor_center = get_min_bound_box_rotation_search_fast(floor_cloud)  # 旋转搜索方法, 最准，但时间很长 60s

    # 生成掩码
    unfilled_floor_mask = generate_mask_from_point_cloud(floor_cloud, depth_image.shape, fx, fy, cx, cy)
    filled_floor_mask = fill_and_close_mask(unfilled_floor_mask)
    return floor_cloud, floor_normal, floor_center, filled_floor_mask, unfilled_floor_mask
        

def find_walls(floor_normal, depth_image, walls_mask_img, color_image, voxel_size, fx, fy, cx, cy, max_wall_num, distance_threshold, ransac_n, num_iterations):
    # 再检测其他的墙
    masked_depth_filtered = preprocess_depth(depth_image, walls_mask_img)
    scene_pcd = create_point_cloud(color_image, masked_depth_filtered, fx, fy, cx, cy)
    scene_pcd = scene_pcd.voxel_down_sample(voxel_size)
    # scene_pcd.transform(flip_transform)
    wall_clouds_list, wall_normal_vectors_list, wall_center_list = fit_walls_pcd_and_pred_xyz(scene_pcd, floor_normal, distance_threshold, ransac_n, num_iterations, search_nums=5)
    # 此处先人为设置实际的墙的个数,取前几个
    wall_clouds_list = wall_clouds_list[:max_wall_num]
    wall_normal_vectors_list = wall_normal_vectors_list[:max_wall_num]
    wall_center_list = wall_center_list[:max_wall_num]
    # 生成掩码
    unfilled_wall_masks = []
    filled_wall_masks = []
    for wall_cloud in wall_clouds_list:
        unfilled_wall_mask = generate_mask_from_point_cloud(wall_cloud, depth_image.shape, fx, fy, cx, cy)
        filled_wall_mask = fill_and_close_mask(unfilled_wall_mask)
        unfilled_wall_masks.append(unfilled_wall_mask)
        filled_wall_masks.append(filled_wall_mask)
    print(f'共识别到 {len(wall_center_list)} 面墙体', flush=True)
    return [(wall, norm_vct, wall_center, filled_wall_mask, unfilled_wall_mask) for wall, norm_vct, wall_center, filled_wall_mask, unfilled_wall_mask in zip(wall_clouds_list, wall_normal_vectors_list, wall_center_list, filled_wall_masks, unfilled_wall_masks)]

def find_walls_with_known_wall_nums(wall_num, floor_normal, depth_image, walls_mask_img, color_image, voxel_size, fx, fy, cx, cy, distance_threshold, ransac_n, num_iterations):
    # 再检测其他的墙
    masked_depth_filtered = preprocess_depth(depth_image, walls_mask_img)
    scene_pcd = create_point_cloud(color_image, masked_depth_filtered, fx, fy, cx, cy)
    scene_pcd = scene_pcd.voxel_down_sample(voxel_size)
    # scene_pcd.transform(flip_transform)
    wall_clouds_list, wall_normal_vectors_list, wall_center_list = fit_walls_pcd_and_pred_xyz(scene_pcd, floor_normal, distance_threshold, ransac_n, num_iterations, search_nums=5, known_wall_nums=True)
    # 此处先人为设置实际的墙的个数,取前几个
    wall_clouds_list = wall_clouds_list[:wall_num]
    wall_normal_vectors_list = wall_normal_vectors_list[:wall_num]
    wall_center_list = wall_center_list[:wall_num]
    # 生成掩码
    wall_masks = []
    for wall_cloud in wall_clouds_list:
        wall_mask = generate_mask_from_point_cloud(wall_cloud, depth_image.shape, fx, fy, cx, cy)
        wall_mask = fill_and_close_mask(wall_mask)
        wall_masks.append(wall_mask)
    return [(wall, norm_vct, wall_center, wall_mask) for wall, norm_vct, wall_center, wall_mask in zip(wall_clouds_list, wall_normal_vectors_list, wall_center_list, wall_masks)]

def get_mask_bbox(mask_img):
    # 找到非零像素的坐标
    coords = cv2.findNonZero(mask_img)
    # 获取边界框
    x, y, w, h = cv2.boundingRect(coords)
    # 计算 x_max 和 y_max
    x_max = x + w
    y_max = y + h
    return [x, y, x_max, y_max]

def create_world_coordinate_frame(size=0.5, num_points=1200):
    # 创建初始坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    # 转换为点云
    coordinate_frame_pcd = coordinate_frame.sample_points_poisson_disk(number_of_points=num_points)

    return coordinate_frame_pcd

def save_results(plane_clouds_and_its_norm_vct, new_detection_res, background_mask, save_dir):
    floor_cloud, floor_normal, floor_center, filled_floor_mask, unfilled_floor_mask = plane_clouds_and_its_norm_vct[0]
    wall_normal_vectors_list = []
    wall_center_list = []
    filled_wall_masks_list = []
    unfilled_wall_masks_list = []
    # 通常地面的法向量应该朝向上方（即朝向相机的反方向）。如果相机位于地面上方拍摄，地面的法向量应该指向相机的方向。
    if np.dot(floor_normal, np.array([0, -1, 0])) < 0:
        floor_normal = -floor_normal
    # 墙面的法向量通常应该指向房间内部，即指向相机的方向。可以通过比较法向量与相机视点之间的关系来确定其方向。(wall, norm_vct, wall_center, filled_wall_mask, unfilled_wall_mask)
    for _,wall_normal,wall_center,filled_wall_mask, unfilled_wall_mask in plane_clouds_and_its_norm_vct[1:]:
        if np.dot(wall_normal, -wall_center) < 0:
            wall_normal_vectors_list.append(-wall_normal)
        else:
            wall_normal_vectors_list.append(wall_normal)
        wall_center_list.append(wall_center)
        filled_wall_masks_list.append(filled_wall_mask)
        unfilled_wall_masks_list.append(unfilled_wall_mask)

    all_point_clouds = []
    for pcd,_,_,_,_ in plane_clouds_and_its_norm_vct:
        all_point_clouds.append(pcd)
        
    # 创建并添加表示地面法向量的坐标系点云
    floor_coordinate_frame_pcd, floor_transformation_matrix = create_custom_coordinate_frame_as_point_cloud(floor_center, floor_normal, size=0.5)
    all_point_clouds.append(floor_coordinate_frame_pcd)

    # 对于每堵墙，创建并添加表示法向量的坐标系点云
    wall_transformation_matrices = []
    for i, (wall_center, wall_normal) in enumerate(zip(wall_center_list, wall_normal_vectors_list)):
        wall_coordinate_frame_pcd, wall_transformation_matrix = create_custom_coordinate_frame_as_point_cloud(wall_center, wall_normal, size=0.5)
        all_point_clouds.append(wall_coordinate_frame_pcd)
        wall_transformation_matrices.append(wall_transformation_matrix)
        # 输出变换矩阵
        print(f"Wall {i} Transformation Matrix:\n", wall_transformation_matrix)

    # 输出地面坐标系的变换矩阵
    print("Floor Transformation Matrix:\n", floor_transformation_matrix)

    # 合并所有点云
    world_coordinate_frame_pcd = create_world_coordinate_frame(size=0.5)
    all_point_clouds.append(world_coordinate_frame_pcd)

    combined_point_cloud = all_point_clouds[0]
    for pcd in all_point_clouds[1:]:
        combined_point_cloud += pcd

    # 保存合并后的点云
    o3d.io.write_point_cloud(os.path.join(save_dir,"combined_scene_and_coordinate_frames.ply"), combined_point_cloud)
    
    # 这里还需要保存未被填充的ground和wall的mask, 但由于不想对前面已有流程产生影响, 这里单独返回ground和wall的mask结果
    unfilled_wall_and_ground_masks = {}
        
    floor_walls_pose = {}
    # 填充floor
    filled_floor_mask = filled_floor_mask*background_mask
    floor_bbox = get_mask_bbox(filled_floor_mask)
    new_detection_res['boxes'].append(floor_bbox)
    new_detection_res['categorys'].append('floor_0')
    new_detection_res['scores'].append(1.0)
    new_detection_res['masks'].append(filled_floor_mask)
    
    unfilled_wall_and_ground_masks['floor_0'] = unfilled_floor_mask*background_mask
    
    floor_walls_pose['floor_0'] = {
            'normal': floor_normal.tolist(),
            'matrix': floor_transformation_matrix.tolist(),
    }
    # 填充walls
    for i,(normal, matrix, filled_wall_mask, unfilled_wall_mask) in enumerate(zip(wall_normal_vectors_list, wall_transformation_matrices, filled_wall_masks_list, unfilled_wall_masks_list)):
        filled_wall_mask = filled_wall_mask*background_mask
        bbox = get_mask_bbox(filled_wall_mask)
        label = f'wall_{i}'
        new_detection_res['boxes'].append(bbox)
        new_detection_res['categorys'].append(label)
        new_detection_res['scores'].append(1.0)
        new_detection_res['masks'].append(filled_wall_mask)
        
        unfilled_wall_and_ground_masks[label] = unfilled_wall_mask*background_mask
        
        floor_walls_pose[label] = {
            'normal': normal.tolist(),
            'matrix': matrix.tolist(),
    }
    
    print(f'墙和地面已经提取完毕')
    return new_detection_res, floor_walls_pose, unfilled_wall_and_ground_masks

def estimate_floor_and_walls(color_image, detection_res, depth_image, save_dir, max_wall_num=3):
    '''
    input:
        #(不使用gpt了) --wall_num : 图片中存在的墙的个数，（不计地面和天花板）, 询问gpt获得
        --color_image : 图片
        --detection_res : gdino输入的所有数据-pickle结果
        --depth_image : 图片的深度图
        --save_dir : 中间结果保存地址
    output:
        --final_detection_res: 把墙和地面的检测结果加入输入的detection_res之中
        --floor_walls_pose: dict形式, 记录了墙和地面的法向和相对于相机的位姿
    '''
    if isinstance(color_image, str):
        color_image = cv2.imread(color_image)
    if isinstance(detection_res, str):
        with open(detection_res, 'rb') as f:
            detection_res = pickle.load(f)
    if isinstance(depth_image, str):
        depth_image = cv2.imread(depth_image, cv2.IMREAD_UNCHANGED)
            
    fx, fy, cx, cy = get_camera_intrinsics(color_image)
    depth_image = depth_image.astype(np.float32)
    
    # 初始化全白的输出掩码图像
    mask_img = np.ones_like(depth_image, dtype=np.uint8) * 255
    floor_mask_img = np.zeros_like(depth_image, dtype=np.uint8)

    # 创建新的检测结果
    new_detection_res = {'boxes': [], 'categorys': [], 'scores': [], 'masks': []}
    
    # 初始化不符合条件的掩码图像
    background_mask = np.ones_like(mask_img, dtype=np.uint8)

    # 遍历检测结果
    for bbox, label, score, mask in zip(detection_res['boxes'], detection_res['categorys'], detection_res['scores'], detection_res['masks']):
        # 将 mask 转换为 NumPy 数组
        mask_array = np.array(mask)

        # 如果标签不符合模式
        if not re.match(r'^(ground|wall|floor)_\d+', label):    # 不去预测天花板
            # 将不符合条件的掩码区域设置为0
            mask_img[mask_array.astype(bool)] = 0
            # 添加到新的检测结果中
            new_detection_res['boxes'].append(bbox)
            new_detection_res['categorys'].append(label)
            new_detection_res['scores'].append(score)
            new_detection_res['masks'].append(mask)
            # 处理一下毯子之类的
            background_mask[mask_array.astype(bool)] = 0
            
        if re.match(r'^(ground|floor|carpet|rug)_\d+', label):
            mask_img[mask_array.astype(bool)] = 0
            floor_mask_img[mask_array.astype(bool)] = 255
            
    voxel_size = 0.01  # 体素大小，单位为米，可以调整这个值来控制降采样程度
    # RANSAC参数
    distance_threshold = 0.1
    ransac_n = 3
    num_iterations = 5000
    
    plane_clouds_and_its_norm_vct = []
    floor_cloud, floor_normal, floor_center, filled_floor_mask, unfilled_floor_mask = find_floor(depth_image, floor_mask_img, color_image, voxel_size, fx, fy, cx, cy, distance_threshold, ransac_n, num_iterations)
    # 将地面添加到平面列表中
    plane_clouds_and_its_norm_vct.append((floor_cloud, floor_normal, floor_center, filled_floor_mask, unfilled_floor_mask))

    assert floor_normal.any()
    walls_mask_img = cv2.bitwise_and(mask_img, cv2.bitwise_not(filled_floor_mask))  # 把地面的mask去除
    wall_res = find_walls(floor_normal, depth_image, walls_mask_img, color_image, voxel_size, fx, fy, cx, cy, max_wall_num, distance_threshold, ransac_n, num_iterations)

    save_wall_dir = os.path.join(save_dir, 'wall_pcd')
    os.makedirs(save_wall_dir,exist_ok=True)

    #save wall pcd
    for idx,item in enumerate(wall_res):
        pcd = item[0]
        # 保存点云数据
        o3d.io.write_point_cloud(os.path.join(save_wall_dir,f'wall_{idx}.pcd'), pcd)

    # 将墙添加到平面列表中
    plane_clouds_and_its_norm_vct.extend(wall_res)

    # 保存结果
    final_detection_res, floor_walls_pose, unfilled_wall_and_ground_masks = save_results(plane_clouds_and_its_norm_vct, new_detection_res, background_mask, save_dir)
    return final_detection_res, floor_walls_pose, unfilled_wall_and_ground_masks
