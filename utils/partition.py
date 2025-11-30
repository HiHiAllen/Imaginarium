import json
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import shutil
import random
import colorsys
import pickle
from pathlib import Path
from skimage.morphology import dilation, disk
import re
import multiprocessing
from functools import partial
from utils.obb import cal_and_visualize_scene_obj_bbox_fitting

def prepare_output_folder(output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

def load_data(object_bboxes_json_path, object_bboxes_wo_text_bbox_json_path, image_path):
    object_bboxes = json.load(open(object_bboxes_json_path, 'r'))
    object_bboxes_wo_text_bbox = json.load(open(object_bboxes_wo_text_bbox_json_path, 'r'))
    img = Image.open(image_path)
    return object_bboxes, object_bboxes_wo_text_bbox, img

def calculate_center_and_area(bbox):
    x_min, y_min, x_max, y_max = bbox
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
    area = (x_max - x_min) * (y_max - y_min)
    return center, area

def extract_centers_and_areas(object_bboxes):
    centers = []
    areas = []
    keys = list(object_bboxes.keys())
    for key in keys:
        bbox = object_bboxes[key]
        center, area = calculate_center_and_area(bbox)
        centers.append(center)
        areas.append(area)
    return np.array(centers), np.array(areas), keys

def perform_kmeans_clustering(centers, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(centers)
    return kmeans.labels_

def calculate_cluster_areas(labels, areas):
    cluster_areas = defaultdict(float)
    for i, label in enumerate(labels):
        cluster_areas[label] += areas[i]
    return cluster_areas

def save_clustered_image(centers, labels, img, output_path):
    draw = ImageDraw.Draw(img)
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    for i, center in enumerate(centers):
        x, y = center
        label = labels[i]
        color = colors[label % len(colors)]
        draw.ellipse((x-5, y-5, x+5, y+5), fill=color, outline=color)
    img.save(output_path)

def get_cluster_bounding_box(cluster_bboxes):
    x_min = min(bbox[0] for bbox in cluster_bboxes)
    y_min = min(bbox[1] for bbox in cluster_bboxes)
    x_max = max(bbox[2] for bbox in cluster_bboxes)
    y_max = max(bbox[3] for bbox in cluster_bboxes)
    return (x_min, y_min, x_max, y_max)

def overlap(bbox1, bbox2):
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

def generate_distinct_colors(num_colors,avoid_colors=None):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        lightness = random.uniform(0.5, 0.7)
        saturation = random.uniform(0.7, 1.0)
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        int_rgb = tuple(int(c * 255) for c in rgb)
        
        if avoid_colors is not None :
            if len(avoid_colors) != 0:
                while list(int_rgb) in avoid_colors:
                    hue = random.uniform(0, 1)
                    lightness = random.uniform(0.5, 0.7)
                    saturation = random.uniform(0.7, 1.0)
                    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
                    int_rgb = tuple(int(c * 255) for c in rgb)

        colors.append(int_rgb)
            
    random.shuffle(colors)
    return colors

def draw_bboxes(draw, cluster_keys, cluster_bboxes, color_map, cluster_bbox):
    offset_x, offset_y, x2, y2 = cluster_bbox
    crop_width = x2 - offset_x
    crop_height = y2 - offset_y

    for key, bbox in zip(cluster_keys, cluster_bboxes):
        color = color_map[key]
        # 将全图中的坐标转换为裁剪图像中的坐标
        x0, y0, x1, y1 = [coord - offset_x if idx % 2 == 0 else coord - offset_y for idx, coord in enumerate(bbox)]
        
        # 在裁剪后的图像上绘制边界框
        draw.rectangle([max(0, x0), max(0, y0), min(crop_width, max(0,x1)), min(crop_height, max(0,y1))], outline=color, width=2)

        # 检查并绘制被截断的部分
        if x0 < 0:  # 左边被截断
            draw.line([(0, max(0, y0)), (0, min(crop_height, y1))], fill=color, width=2)
        if y0 < 0:  # 上边被截断
            draw.line([(max(0, x0), 0), (min(crop_width, x1), 0)], fill=color, width=2)
        if x1 > crop_width:  # 右边被截断
            draw.line([(crop_width, max(0, y0)), (crop_width, min(crop_height, y1))], fill=color, width=2)
        if y1 > crop_height:  # 下边被截断
            draw.line([(max(0, x0), crop_height), (min(crop_width, x1), crop_height)], fill=color, width=2)

def draw_contur(draw, cluster_keys ,cluster_bboxes, color_map, cluster_bbox,mask_folder, mask_list=None, mask_json=None):
    offset_x, offset_y, x2, y2 = cluster_bbox
    crop_width = x2 - offset_x
    crop_height = y2 - offset_y
    #读取mask并绘制轮廓
        
    if mask_list is None:
        with open(os.path.join(mask_folder, 'masks.pkl'), "rb") as f:
            mask_list = pickle.load(f)
    if mask_json is None:
        with open(os.path.join(mask_folder, 'result.json'), "rb") as f:
            mask_json = json.load(f)
            
    for key, bbox in zip(cluster_keys, cluster_bboxes):
        color = color_map[key]
        # 将全图中的坐标转换为裁剪图像中的坐标
        x0, y0, x1, y1 = [coord - offset_x if idx % 2 == 0 else coord - offset_y for idx, coord in enumerate(bbox)]
        
        mask_index = mask_json["categorys"].index(key)
        mask = np.array(mask_list[mask_index])

        # 设置阈值，亮度大于此值的像素设为 255，其他设为 0
        threshold = 128
        _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 绘制轮廓
        for contour in contours:
            # 将轮廓点转换为 PIL 的点格式
            points = []
            for point in contour:
                x,y = point[0][0], point[0][1]
                # 将全图中的坐标转换为裁剪图像中的坐标
                x = x - offset_x
                y = y - offset_y
                #滤除不在其中的轮廓点
                if x<0:
                    x = 0
                if x>crop_width:
                    x = crop_width
                if y<0:
                    y = 0
                if y>crop_height:
                    y = crop_height
                points.append((x,y))

            draw.line(points+[points[0]], fill=color, width=2)  # 绘制闭合轮廓
            
        
def draw_labels(draw, cluster_keys, cluster_bboxes, font, drawn_text_positions, color_map, offset_x, offset_y, img_width, img_height):
    labels_positions = []
    for key, bbox in zip(cluster_keys, cluster_bboxes):
        color = color_map[key]
        x0, y0, x1, y1 = [coord - offset_x if idx % 2 == 0 else coord - offset_y for idx, coord in enumerate(bbox)]
        text = key
        left, top, right, bottom = font.getbbox(text)
        text_width = right - left
        text_height = bottom - top
        text_x, text_y = x0, y0

        # 确保文本初始位置在图像内
        if text_x + text_width > img_width:
            text_x = img_width - text_width
        if text_y + text_height > img_height:
            text_y = img_height - text_height

        # 尝试在所有方向上移动文本以避免重叠
        for dx, dy in [(5, 5), (-5, -5), (5, -5), (-5, 5), (5, 0), (-5, 0), (0, 5), (0, -5)]:
            found_spot = False
            for attempt in range(10):  # 最多尝试10次
                bbox = [text_x, text_y, text_x + text_width, text_y + text_height]
                if not any([overlap(bbox, pos) for pos in drawn_text_positions]) and \
                   text_x >= 0 and text_y >= 0 and \
                   text_x + text_width <= (img_width+5) and text_y + text_height <= (img_height+5):
                    found_spot = True
                    break  # 找到了不重叠且在图像内的位置
                text_x += dx
                text_y += dy
                # 确保文本不会移出图片边界
                text_x = max(0, min(text_x, img_width - text_width))
                text_y = max(0, min(text_y, img_height - text_height))

            if found_spot:
                break  # 如果找到合适的位置，则停止尝试其他方向

        drawn_text_positions.append([text_x, text_y, text_x + text_width, text_y + text_height])
        labels_positions.append((key, text_x, text_y))
        for offset in [(1, 1), (-1, -1), (1, -1), (-1, 1), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            draw.text((text_x + offset[0], text_y + offset[1]), text, fill="black", font=font)
        draw.text((text_x, text_y), text, fill=color, font=font)
    return labels_positions

def draw_labels_mask(draw, cluster_keys, masks, font, drawn_text_positions, color_map, offset_x, offset_y, img_width, img_height):
    labels_positions = []
    for key, mask in zip(cluster_keys, masks):
        color = color_map[key]
        #找到左上角的点
        x0, y0, w, h = cv2.boundingRect(mask)
        x0,y0 = x0 - offset_x, y0 - offset_y

        text = key
        left, top, right, bottom = font.getbbox(text)
        text_width = right - left
        text_height = bottom - top
        text_x, text_y = x0, y0

        # 确保文本初始位置在图像内
        if text_x + text_width > img_width:
            text_x = img_width - text_width
        if text_y + text_height > img_height:
            text_y = img_height - text_height

        # 尝试在所有方向上移动文本以避免重叠
        for dx, dy in [(5, 5), (-5, -5), (5, -5), (-5, 5), (5, 0), (-5, 0), (0, 5), (0, -5)]:
            found_spot = False
            for attempt in range(10):  # 最多尝试10次
                bbox = [text_x, text_y, text_x + text_width, text_y + text_height]
                if not any([overlap(bbox, pos) for pos in drawn_text_positions]) and \
                   text_x >= 0 and text_y >= 0 and \
                   text_x + text_width <= (img_width+5) and text_y + text_height <= (img_height+5):
                    found_spot = True
                    break  # 找到了不重叠且在图像内的位置
                text_x += dx
                text_y += dy
                # 确保文本不会移出图片边界
                text_x = max(0, min(text_x, img_width - text_width))
                text_y = max(0, min(text_y, img_height - text_height))

            if found_spot:
                break  # 如果找到合适的位置，则停止尝试其他方向

        drawn_text_positions.append([text_x, text_y, text_x + text_width, text_y + text_height])
        labels_positions.append((key, text_x, text_y))
        for offset in [(1, 1), (-1, -1), (1, -1), (-1, 1), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            draw.text((text_x + offset[0], text_y + offset[1]), text, fill="black", font=font)
        draw.text((text_x, text_y), text, fill=color, font=font)
    return labels_positions


def find_optimal_clusters(centers, min_clusters=8, max_clusters=12):
    sse = []
    for k in range(min_clusters, max_clusters + 1):
        if k > len(centers):
            break  # 如果簇数超过样本数，停止计算
        kmeans = KMeans(n_clusters=k, random_state=0).fit(centers)
        sse.append(kmeans.inertia_)

    if not sse:
        return len(centers)  # 如果没有计算任何簇数，返回样本数

    optimal_clusters = np.diff(np.diff(sse)).argmin() + min_clusters + 1
    return min(optimal_clusters, len(centers))  # 确保返回值不超过样本数


def expand_bbox_for_labels(cluster_bbox, cluster_keys, cluster_bboxes, font):
    x_min, y_min, x_max, y_max = cluster_bbox
    for key, bbox in zip(cluster_keys, cluster_bboxes):
        text = key
        left, top, right, bottom = font.getbbox(text)
        text_width = right - left
        text_height = bottom - top
        x0, y0 = bbox[0], bbox[1]
        if x0 + text_width > x_max:
            x_max = x0 + text_width
        if y0 + text_height > y_max:
            y_max = y0 + text_height
        if x0 < x_min:
            x_min = x0
        if y0 < y_min:
            y_min = y0
    return (x_min, y_min, x_max, y_max)


## 这个除了在区域内画出聚类结果中的keys物体，还要标出其他物体，并且这些物体的名字必须要在图中可见
def process_clusters(object_bboxes, object_bboxes_wo_text_bbox, img, labels, keys, output_folder, num_clusters, font_path):
    result_data = {}
    for cluster_id in range(num_clusters):
        font = ImageFont.truetype(font_path, 20)
        
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_keys = [keys[i] for i in cluster_indices]
        cluster_bboxes = [object_bboxes[key] for key in cluster_keys]
        cluster_bboxes_wo_text_bbox = [object_bboxes_wo_text_bbox[key] for key in cluster_keys]
        cluster_bbox = get_cluster_bounding_box(cluster_bboxes)
        cluster_bbox = expand_bbox_for_labels(cluster_bbox, cluster_keys, cluster_bboxes, font)
        
        expand_x = img.width * 0.05
        expand_y = img.height * 0.05
        
        cluster_bbox = (
            max(0, cluster_bbox[0] - expand_x),
            max(0, cluster_bbox[1] - expand_y),
            min(img.width, cluster_bbox[2] + expand_x),
            min(img.height, cluster_bbox[3] + expand_y)
        )
        
        cropped_img = img.crop(cluster_bbox)
        draw = ImageDraw.Draw(cropped_img)
        
        font_size = int(min(cropped_img.size) / 70)  # 可以调整这个比例
        font_size = max(15, font_size)
        font = ImageFont.truetype(font_path, font_size)
        
        colors = generate_distinct_colors(len(object_bboxes))
        color_map = {key: colors[i] for i, key in enumerate(object_bboxes.keys())}
        
       # 绘制不在 cluster_keys 中的其他对象
        other_keys = []
        other_bboxes_wo_text_bbox = []
        
        for key in object_bboxes.keys():
            if key not in cluster_keys:
                bbox = object_bboxes[key]
                if (bbox[2] > cluster_bbox[0] and bbox[0] < cluster_bbox[2] and
                    bbox[3] > cluster_bbox[1] and bbox[1] < cluster_bbox[3]):
                    other_keys.append(key)
                    other_bboxes_wo_text_bbox.append(object_bboxes_wo_text_bbox[key])
                    
        #bbox anot version             
        # draw_bboxes(draw, other_keys, other_bboxes_wo_text_bbox, color_map, cluster_bbox)
        # draw_bboxes(draw, cluster_keys, cluster_bboxes_wo_text_bbox, color_map, cluster_bbox)
        
        #mask anot version
        # 给定的路径
        path = Path(output_folder)
        # 去除最后一个目录
        mask_folder = str(path.parent)
        draw_contur(draw, other_keys, other_bboxes_wo_text_bbox, color_map, cluster_bbox,mask_folder)
        draw_contur(draw, cluster_keys, cluster_bboxes_wo_text_bbox, color_map, cluster_bbox,mask_folder)
        
        
        drawn_text_positions = []
        cluster_labels_positions = draw_labels(draw, cluster_keys, cluster_bboxes_wo_text_bbox, font, drawn_text_positions, color_map, cluster_bbox[0], cluster_bbox[1], cropped_img.width, cropped_img.height)
        labels_positions = draw_labels(draw, other_keys, other_bboxes_wo_text_bbox, font, drawn_text_positions, color_map, cluster_bbox[0], cluster_bbox[1], cropped_img.width, cropped_img.height)
        cropped_output_path = os.path.join(output_folder, f'cluster_{cluster_id}.png')
        cropped_img.save(cropped_output_path)
        
        cluster_labels_positions.sort(key=lambda x: (x[2], x[1]))
        result_data[f'cluster_{cluster_id}'] = {
            "objects": [{"key": key, "bbox": object_bboxes_wo_text_bbox[key]} for key, _, _ in cluster_labels_positions],
            "image_path": cropped_output_path
        }
    
    return result_data

def save_result_data(result_data, output_json_path):
    with open(output_json_path, 'w') as f:
        json.dump(result_data, f, indent=4)

def cluster_bounding_boxes(object_bboxes_json_path, object_bboxes_wo_text_bbox_json_path, image_path, output_folder, font_path, min_clusters=8, max_clusters=12):
    prepare_output_folder(output_folder)
    object_bboxes, object_bboxes_wo_text_bbox, img = load_data(object_bboxes_json_path, object_bboxes_wo_text_bbox_json_path, image_path)
    centers, areas, keys = extract_centers_and_areas(object_bboxes)
    num_clusters = find_optimal_clusters(centers, min_clusters, max_clusters) if min_clusters != max_clusters else max_clusters
    print(f'寻找的最佳聚类数量为{num_clusters}')
    labels = perform_kmeans_clustering(centers, num_clusters)
    initial_output_path = os.path.join(output_folder, 'initial_clustering.png')
    save_clustered_image(centers, labels, img.copy(), initial_output_path)
    result_data = process_clusters(object_bboxes, object_bboxes_wo_text_bbox, img, labels, keys, output_folder, num_clusters, font_path)
    output_json_path = os.path.join(output_folder, 'clustered_bboxes.json')
    save_result_data(result_data, output_json_path)


def cluster_3d_obb(
    depth_image_path,
    floor_walls_pose_data,
    mask_folder,
    object_bboxes_wo_text_bbox_json_path,
    image_path,
    output_folder,
    font_path,
    pcd_mask_save_path, # New parameter
    min_clusters=4,  # 早期稳定的数值设为6
    max_clusters=8,  # 早期稳定的数值设为12
    visualize_combined_pcd=False
):
    os.makedirs(output_folder, exist_ok=True)

    with open(object_bboxes_wo_text_bbox_json_path, 'r') as f:
        object_bboxes = json.load(f)
    
    with open(object_bboxes_wo_text_bbox_json_path.replace('_wo_text_bbox',''), 'r') as f:
        object_bboxes_with_text = json.load(f)

    img = Image.open(image_path)
    keys = list(object_bboxes.keys())
    
    # --- OBB Feature Calculation ---
    # This logic is moved from the old load_3d_data function to here.
    item_mask_path_list = [os.path.join(mask_folder, f'{name}_mask.png') for name in keys]

    pcb_center_xyz_list, _, _, _ = cal_and_visualize_scene_obj_bbox_fitting(
        depth_image_path=depth_image_path,
        obj_mask_path_list=item_mask_path_list,
        color_image_path=image_path,
        wall_floor_pose=floor_walls_pose_data,
        save_path=pcd_mask_save_path, # Pass the new path here
        generate_pcd_mask_plane=['xoy', 'yoz'],
        visualize_combined_pcd=visualize_combined_pcd
    )
    
    obb_feat = []
    for center in pcb_center_xyz_list:
        # Use y and z for clustering as they represent the vertical and depth axes
        xy = [center[1], center[2]]
        obb_feat.append(np.array(xy))
    # --- End of OBB Feature Calculation ---

    if len(obb_feat) <= min_clusters:
        num_clusters = len(obb_feat)
    else:
        num_clusters = find_optimal_clusters(obb_feat, min_clusters, max_clusters) if min_clusters != max_clusters else max_clusters
    
    print(f'Finding optimal number of clusters: {num_clusters}')

    labels = perform_kmeans_clustering(np.array(obb_feat), num_clusters)
    
    result_data = process_clusters(
        object_bboxes_with_text, object_bboxes, img, labels, keys, output_folder, num_clusters, font_path
    )
    output_json_path = os.path.join(output_folder, 'clustered_bboxes.json')
    save_result_data(result_data, output_json_path)


def cluster_mask_ditance(object_bboxes_json_path, object_bboxes_wo_text_bbox_json_path, image_path, output_folder, font_path, min_clusters=8, max_clusters=12):
    prepare_output_folder(output_folder)
    initial_output_path = os.path.join(output_folder, 'initial_clustering.png')

    object_bboxes, object_bboxes_wo_text_bbox, img = load_data(object_bboxes_json_path, object_bboxes_wo_text_bbox_json_path, image_path)
    centers, areas, keys = extract_centers_and_areas(object_bboxes)

    # 获取mask
    path = Path(output_folder)
    # 去除最后一个目录
    mask_folder = str(path.parent)
    mask_all_list = None
    with open(os.path.join(mask_folder, 'masks.pkl'), "rb") as f:
        mask_all_list = pickle.load(f)
    mask_json = None
    with open(os.path.join(mask_folder, 'result.json'), "rb") as f:
        mask_json = json.load(f)

    #filter wall
    masks_list = []
    for idx,key in enumerate(mask_json["categorys"]):
        if re.match(r'(ground|wall|ceiling|floor)_\d+', key):
            continue
        else:
            mask_ = np.array(mask_all_list[idx])[np.newaxis, :]
            # 二值化
            binary_mask = (mask_ > 0).astype(np.uint8)
            masks_list.append(binary_mask)

    mask_np = np.vstack(masks_list)

    # Step 1:  计算接邻矩阵
    def adjacent_distance_matrix(mask, padding=5):
        """
        计算两个堆叠掩码的 Dice 距离矩阵。

        参数:
            mask (np.ndarray): 第一个掩码，形状为 (N, H, W)。
            padding (int): 膨胀操作的半径，默认为 0。

        返回:
            np.ndarray: Dice 距离矩阵，形状为 (N, M)。
        """
        # 如果 padding 不为 0，对掩码进行膨胀操作
        if padding != 0:
            mask_list = []
            for m in mask:
                # 定义结构元素（核）
                kernel = np.ones((padding, padding), dtype=np.uint8)
                # 使用 OpenCV 进行膨胀
                dilated_image = cv2.dilate(m, kernel, iterations=1)
                mask_list.append(dilated_image)
            mask = np.array(mask_list)
        # 展平掩码
        mask1_flat = mask.reshape(mask.shape[0], -1)  # 形状: (N, H*W)
        mask2_flat = mask.reshape(mask.shape[0], -1)  # 形状: (M, H*W)

        # 初始化 Dice 距离矩阵
        N = mask1_flat.shape[0]
        M = mask2_flat.shape[0]
        adjacent_matrix = np.zeros((N, M))

        # 计算每对掩码的 Dice 距离
        for i in range(N):
            for j in range(M):
                # 计算交集和并集
                intersection = np.sum(mask1_flat[i] * mask2_flat[j])
                if intersection > 0:
                    adjacent_matrix[i, j] = 1
                else:
                    adjacent_matrix[i, j] = 0
                if i == j:
                    adjacent_matrix[i, j] = 0
                # sum_masks = np.sum(mask1_flat[i]) + np.sum(mask2_flat[j])

                # # 计算 Dice 系数
                # dice_coeff = (2 * intersection) / (sum_masks + 1e-8)  # 添加小值避免除零

                # # 计算 Dice 距离
                # dice_matrix[i, j] = 1 - dice_coeff

        return adjacent_matrix
    
    adjacent_matrix = adjacent_distance_matrix(mask_np)

    # Step2: 寻找最佳聚类数量
    adjacent_num = np.sum(adjacent_matrix, axis=1)
    
    #找到大于80%的idx
    adjacent_num_idx = np.where(adjacent_num > 3)[0]
    optimal_clusters = len(adjacent_num_idx)
    labels = [0] * len(centers)

    for id,adjacent_num_idx in enumerate(adjacent_num_idx):
        id_list  = np.where(adjacent_matrix[adjacent_num_idx]> 0)[0]
        for idx in id_list:
            if idx == adjacent_num_idx:
                continue
            labels[idx] = id+1

    id_max = max(labels)
    for idx,label in enumerate(labels):
        if label == 0:
            id_max += 1
            labels[idx] = id_max

    save_clustered_image(centers, labels, img.copy(), initial_output_path)
    result_data = process_clusters(object_bboxes, object_bboxes_wo_text_bbox, img, labels, keys, output_folder, optimal_clusters, font_path)
    output_json_path = os.path.join(output_folder, 'clustered_bboxes.json')
    save_result_data(result_data, output_json_path)

def _process_single_obj_crop(key, object_bboxes, object_bboxes_wo_text_bbox, image_path, mask_list, mask_json, mask_diation_list, top_down_masks_dialation, wall_mask_dict, wall_color_name, output_folder, font_path, pcd_mask_path, mask_folder, debug=False):
    img = Image.open(image_path)
    mask_index = mask_json["categorys"].index(key)
    mask_dilation = np.array(mask_diation_list[mask_index])
    mask = np.array(mask_list[mask_index])
    
    all_paint_keys = [key]
    all_bboxes_wo_text_bbox = [object_bboxes_wo_text_bbox[key]]
    all_mask = [mask]
    near_obj_name = []
    near_wall_color_dict = {}
    avoid_colors = []


    #depth near check
    top_down_obj_mask_dialation = top_down_masks_dialation.get(key)
    if top_down_obj_mask_dialation is None:
            print(f"Warning: Could not read mask for {key} from {os.path.join(pcd_mask_path, f'{key}_xoy.png')}. Skipping depth checks for this object.")

    # find other objects near the object
    for other_key in object_bboxes.keys():
        if key == other_key:
            continue
        
        #image near check
        other_mask_index = mask_json["categorys"].index(other_key)
        other_mask = np.array(mask_list[other_mask_index])
        other_mask_dilation = np.array(mask_diation_list[other_mask_index])

        if not np.any(other_mask_dilation&mask_dilation):
            continue
        
        intersection_count = 0
        if top_down_obj_mask_dialation is not None:
            #depth near check
            top_down_other_key_mask_dialation = top_down_masks_dialation.get(other_key)
            
            if top_down_other_key_mask_dialation is None:
                continue # Skip if the other mask can't be read

            intersection = top_down_obj_mask_dialation&top_down_other_key_mask_dialation
            # 计算交集区域的像素个数
            intersection_count = np.count_nonzero(intersection)
        
        if intersection_count<=0:
            continue
        
        all_paint_keys.append(other_key)
        all_bboxes_wo_text_bbox.append(object_bboxes_wo_text_bbox[other_key])
        all_mask.append(other_mask)
        near_obj_name.append(other_key)


    #finde the wall color near the object
    for wall_key, wall_mask in wall_mask_dict.items():
        if np.any(wall_mask&mask_dilation):
            near_wall_color_dict[wall_key] = wall_color_name[wall_key][0]
            avoid_colors.append(wall_color_name[wall_key][1])
    max_paint_box = object_bboxes_wo_text_bbox[key]
    
    
    for mask in all_mask:
        
        # 计算bounding rect
        x_min, y_min, w, h = cv2.boundingRect(mask)

        # 计算x_max, y_max
        x_max = x_min + w
        y_max = y_min + h

        max_paint_box[0] = min(max_paint_box[0], x_min)
        max_paint_box[1] = min(max_paint_box[1], y_min)
        max_paint_box[2] = max(max_paint_box[2], x_max)
        max_paint_box[3] = max(max_paint_box[3], y_max)
        
    #paddding

    expand_x = img.width * 0.1
    expand_y = img.height * 0.1
    max_paint_box = (
        max(0, max_paint_box[0] - expand_x),
        max(0, max_paint_box[1] - expand_y),
        min(img.width, max_paint_box[2] + expand_x),
        min(img.height, max_paint_box[3] + expand_y)
    )
    
    #check and adjust the box
    max_paint_box = (
        int(max(0, max_paint_box[0])),
        int(max(0, max_paint_box[1])),
        int(min(img.width, max_paint_box[2])),
        int(min(img.height, max_paint_box[3]))
    )
    
    labels_positions = []
    cropped_output_path = os.path.join(output_folder, f'{key}.png')
    
    if debug:
        cropped_img = img.crop(max_paint_box)
        draw = ImageDraw.Draw(cropped_img)
        
        font_size = int(min(cropped_img.size) / 70)  # 可以调整这个比例
        font_size = max(15, font_size)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
        
        # colors = generate_distinct_colors(len(object_bboxes),avoid_colors=avoid_colors)
        # color_map = {key: colors[i] for i, key in enumerate(object_bboxes.keys())}
        color_map = {key: (57, 255, 20) for i, key in enumerate(object_bboxes.keys())}
        
        color_map[key] = (255, 0, 0)#main object color red
        
        # Pass mask_list and mask_json to avoid reloading
        draw_contur(draw, all_paint_keys, all_bboxes_wo_text_bbox, color_map, max_paint_box, mask_folder, mask_list=mask_list, mask_json=mask_json)

        drawn_text_positions = []
        labels_positions = draw_labels_mask(draw, all_paint_keys, all_mask, font, drawn_text_positions, color_map, max_paint_box[0], max_paint_box[1], cropped_img.width, cropped_img.height)
        cropped_img.save(cropped_output_path)
    else:
        # 计算 labels_positions，即使不绘图也需要返回 bbox 信息
        # 这里 labels_positions 是 list of (key, text_x, text_y)
        # text_x, text_y 是相对于 crop 的坐标
        # 如果不绘图，我们只需要 key，text_x/y 其实无所谓，但是下面的 return 语句用到了
        # 我们模拟一下 labels_positions
        # draw_labels_mask 实际上做了位置调整。如果不调用它，我们就得不到调整后的位置。
        # 但是 result_entry['bbox'] 用到了 labels_positions 中的 key。
        # result_entry['bbox'] = [(key, object_bboxes_wo_text_bbox[key]) for key, _, _ in labels_positions]
        # 这里实际上就是 all_paint_keys 里的 key。
        # 所以我们可以直接构造
        labels_positions = [(k, 0, 0) for k in all_paint_keys]

    
    labels_positions.sort(key=lambda x: (x[2], x[1]))
    result_entry = {
        'image_path':cropped_output_path,
        'near_obj_name':near_obj_name,
        'bbox':[(key, object_bboxes_wo_text_bbox[key]) for key, _, _ in labels_positions],
        'near_wall_color_dict':near_wall_color_dict
    }
    return key, result_entry

def obj_bbox_crop_and_save(object_bboxes_json_path, object_bboxes_wo_text_bbox_json_path, image_path, output_folder, font_path, pcd_mask_path, debug=False):
    prepare_output_folder(output_folder)
    object_bboxes, object_bboxes_wo_text_bbox, img = load_data(object_bboxes_json_path, object_bboxes_wo_text_bbox_json_path, image_path)
    # We don't need img here, but we need image_path for the worker. img is opened in load_data but we can just use image_path.
    
    result_data = {}
    
    # 获取mask
    path = Path(output_folder)
    # 去除最后一个目录
    mask_folder = str(path.parent)
    
    mask_list = None
    with open(os.path.join(mask_folder, 'masks.pkl'), "rb") as f:
        mask_list = pickle.load(f)
    mask_json = None
    with open(os.path.join(mask_folder, 'result.json'), "rb") as f:
        mask_json = json.load(f)
    with open(os.path.join(mask_folder, 'wall_color_name.json'), "rb") as f:
        wall_color_name = json.load(f)
    
    
    #获取wall的mask
    wall_mask_dict = {}
    for key in wall_color_name.keys():
        wall_mask_dict[key] = np.array(mask_list[mask_json["categorys"].index(key)])
        
    mask_dilation_radius = 10
    kernel = np.ones((mask_dilation_radius, mask_dilation_radius), np.uint8)  
    mask_diation_list = [cv2.dilate(np.array(mask), kernel, iterations=1) for mask in mask_list]
    
    # Pre-load and dilate top-down masks to avoid repeated disk I/O
    top_down_masks_dialation = {}
    for k in object_bboxes.keys():
        mask_path = os.path.join(pcd_mask_path, f'{k}_xoy.png')
        if os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                top_down_masks_dialation[k] = cv2.dilate(m, kernel, iterations=1)
            else:
                print(f"Warning: Could not read mask for {k} from {mask_path}")
        else:
             pass # File doesn't exist, handle implicitly by absence in dict

    # Use multiprocessing
    num_processes = min(4, len(object_bboxes))
    keys = list(object_bboxes.keys())
    
    process_func = partial(
        _process_single_obj_crop,
        object_bboxes=object_bboxes,
        object_bboxes_wo_text_bbox=object_bboxes_wo_text_bbox,
        image_path=image_path,
        mask_list=mask_list,
        mask_json=mask_json,
        mask_diation_list=mask_diation_list,
        top_down_masks_dialation=top_down_masks_dialation,
        wall_mask_dict=wall_mask_dict,
        wall_color_name=wall_color_name,
        output_folder=output_folder,
        font_path=font_path,
        pcd_mask_path=pcd_mask_path,
        mask_folder=mask_folder,
        debug=debug
    )

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, keys)
    
    for key, entry in results:
        result_data[key] = entry

    output_json_path = os.path.join(output_folder, 'crop_masks.json')
    save_result_data(result_data, output_json_path)

        
def is_adjacent(bbox1, bbox2):
    """
    判断bbox1和bbox2是否相邻
    """
    # 检查两个bbox在水平方向或垂直方向上的接触
    horizontal_adjacent = (bbox1[1] < bbox2[3] and bbox1[3] > bbox2[1]) and (bbox1[2] == bbox2[0] or bbox1[0] == bbox2[2])
    vertical_adjacent = (bbox1[0] < bbox2[2] and bbox1[2] > bbox2[0]) and (bbox1[3] == bbox2[1] or bbox1[1] == bbox2[3])
    
    return horizontal_adjacent or vertical_adjacent

def is_nested(bbox1, bbox2):
    """
    判断bbox1是否被bbox2嵌套
    """
    return bbox2[0] <= bbox1[0] and bbox2[1] <= bbox1[1] and bbox2[2] >= bbox1[2] and bbox2[3] >= bbox1[3]


def is_adjacent_masks(mask1, mask2, radius):
    """
    判断两个膨胀后的掩码是否有交集
    
    mask1, mask2: 输入的二值掩码
    radius: 膨胀的半径
    
    返回：如果有交集则返回True，否者返回False
    """
    # 膨胀掩码
    dilated_mask1 = dilation(mask1, disk(radius))  # 使用圆盘形状进行膨胀
    dilated_mask2 = dilation(mask2, disk(radius))
    
    # 判断两个膨胀掩码是否有交集
    intersection = dilated_mask1 & dilated_mask2
    
    # 如果交集不为空，则说明有交集
    return np.any(intersection)