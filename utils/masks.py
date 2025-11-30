import numpy as np
import pickle
import json
import os
from PIL import Image
import cv2
import copy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.partition import generate_distinct_colors
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_missing_mask_area(mask, min_kernel_size=3, max_kernel_size=21):
    # mask = cv2.imread(mask_path, 0)
    # 获取图像的高度和宽度
    height, width = mask.shape[:2]
    
    # 计算图像的对角线长度
    diagonal = np.sqrt(height**2 + width**2)
    
    # 根据对角线长度计算核大小，确保是奇数
    kernel_size = int(diagonal * 0.01)  # 使用图像对角线的1%作为核大小
    kernel_size = max(min_kernel_size, min(kernel_size, max_kernel_size))
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    
    # 创建核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 执行闭运算
    open_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # open_mask = Image.fromarray(open_mask)
    # open_mask.save("./processed_inverted_combined_mask.png")
    
    return open_mask

def downsample_points(points, target_count):
    """下采样点集到目标数量"""
    if len(points) <= target_count:
        return points
    
    indices = np.random.choice(len(points), target_count, replace=False)
    return points[indices]

def find_optimal_clusters(points, clusters_section, target_sample_count=1000):
    # 下采样
    sampled_points = downsample_points(points, target_sample_count)
    
    silhouette_scores = []
    for n_clusters in range(clusters_section[0], min(clusters_section[1] + 1, len(sampled_points))):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(sampled_points)
        score = silhouette_score(sampled_points, kmeans.labels_)
        silhouette_scores.append((n_clusters, score))
    
    # 选择轮廓系数最高的聚类数
    optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    return optimal_clusters

def expand_rectangle(rect, image_shape):
    x_min, y_min, x_max, y_max = rect
    width = x_max - x_min
    height = y_max - y_min
    
    # 计算扩展量，每个方向扩展原尺寸的1/4
    x_expand = width // 4
    y_expand = height // 4
    
    # 扩展矩形
    new_x_min = max(0, x_min - x_expand)
    new_y_min = max(0, y_min - y_expand)
    new_x_max = min(image_shape[1] - 1, x_max + x_expand)
    new_y_max = min(image_shape[0] - 1, y_max + y_expand)

    # 如果扩展后的尺寸小于128x128，则调整到128x128
    if new_x_max - new_x_min < 128:
        center_x = (new_x_min + new_x_max) // 2
        new_x_min = max(0, center_x - 64)
        new_x_max = min(image_shape[1] - 1, center_x + 64)
    
    if new_y_max - new_y_min < 128:
        center_y = (new_y_min + new_y_max) // 2
        new_y_min = max(0, center_y - 64)
        new_y_max = min(image_shape[0] - 1, center_y + 64)
        
    # 计算新的宽度和高度
    new_width = new_x_max - new_x_min
    new_height = new_y_max - new_y_min
    
    # 检查长宽比
    aspect_ratio = max(new_width, new_height) / min(new_width, new_height)
    if aspect_ratio > 3:
        if new_width > new_height:
            # 宽度大于高度，增加高度
            target_height = new_width // 3
            height_diff = target_height - new_height
            new_y_min = max(0, new_y_min - height_diff // 2)
            new_y_max = min(image_shape[0] - 1, new_y_max + height_diff // 2)
        else:
            # 高度大于宽度，增加宽度
            target_width = new_height // 3
            width_diff = target_width - new_width
            new_x_min = max(0, new_x_min - width_diff // 2)
            new_x_max = min(image_shape[1] - 1, new_x_max + width_diff // 2)
    
    return (new_x_min, new_y_min, new_x_max, new_y_max)

def find_connected_components(mask, label):
    # 找到特定标签的所有点
    y, x = np.where(mask == label)
    binary_mask = np.zeros_like(mask, dtype=np.uint8)
    binary_mask[y, x] = 255

    # 找到连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # 获取每个连通域的最小包围框
    bounding_boxes = []
    for i in range(1, num_labels):  # 跳过背景（标签0）
        x, y, w, h, area = stats[i]
        bounding_boxes.append((x, y, x+w, y+h))

    return bounding_boxes

def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=10):
    x1, y1 = pt1
    x2, y2 = pt2
    
    # 绘制水平线
    for x in range(x1, x2, dash_length * 2):
        start = (x, y1)
        end = (min(x + dash_length, x2), y1)
        cv2.line(img, start, end, color, thickness)
        start = (x, y2)
        end = (min(x + dash_length, x2), y2)
        cv2.line(img, start, end, color, thickness)
    
    # 绘制垂直线
    for y in range(y1, y2, dash_length * 2):
        start = (x1, y)
        end = (x1, min(y + dash_length, y2))
        cv2.line(img, start, end, color, thickness)
        start = (x2, y)
        end = (x2, min(y + dash_length, y2))
        cv2.line(img, start, end, color, thickness)
        
def find_and_visualize_rectangles(ori_img_path, open_mask, cropped_region_save_folder, clusters_section=[2,4]):
    # 读取原始图片
    ori_img = cv2.imread(ori_img_path)

    # 找到所有值为255的像素点
    y, x = np.where(open_mask == 255)
    points = np.column_stack((x, y))

    if len(points) == 0:
        print("No white pixels found in the mask.")
        return None, None, None, None, None

    # 找到最佳聚类数量
    # clusters_section = [4,8]    # 最稳定的区间
    # clusters_section = [2,8]
    optimal_clusters = find_optimal_clusters(points, clusters_section)
    print(f"Optimal number of clusters: {optimal_clusters}")

    # 使用最佳聚类数量进行KMeans聚类
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0, n_init=10).fit(points)
    labels = kmeans.labels_

    # 创建一个彩色图像用于可视化
    result = cv2.cvtColor(open_mask, cv2.COLOR_GRAY2BGR)

    # 定义不同的颜色用于绘制矩形
    colors = generate_distinct_colors(optimal_clusters)

    json_output = {}
    cropped_and_marked_img_list = []
    cropped_img_list = []
    ori_img_unmarked = copy.deepcopy(ori_img)
    for i in range(optimal_clusters):
        cluster_points = points[labels == i]
        if len(cluster_points) > 0:
            x_min, y_min = np.min(cluster_points, axis=0)
            x_max, y_max = np.max(cluster_points, axis=0)
            # 使用虚线绘制主矩形
            draw_dashed_rectangle(result, (x_min, y_min), (x_max, y_max), colors[i], thickness=2, dash_length=10)
            
            # 扩展矩形
            x_min, y_min, x_max, y_max = expand_rectangle((x_min, y_min, x_max, y_max), open_mask.shape)
            
            # 绘制主矩形
            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), colors[i], 2)
        
            # 找到该区域内的连通域
            cluster_mask = np.zeros_like(open_mask, dtype=np.uint8)
            cluster_mask[y[labels == i], x[labels == i]] = 255
            connected_components = find_connected_components(cluster_mask, 255)

            # 将信息添加到JSON输出
            region_info = {
                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "connected_components": []
            }
            
            # 裁剪未标记的图像
            cropped_img_unmarked = ori_img_unmarked[y_min:y_max, x_min:x_max].copy()
            cropped_img_list.append(cropped_img_unmarked)
            
            # 绘制连通域的边缘并添加到JSON
            for cc in connected_components:
                cv2.rectangle(result, (cc[0], cc[1]), (cc[2], cc[3]), colors[i], 2)
                # 创建单个连通域的掩码
                component_mask = np.zeros_like(cluster_mask, dtype=np.uint8)
                component_mask[cc[1]:cc[3], cc[0]:cc[2]] = cluster_mask[cc[1]:cc[3], cc[0]:cc[2]]

                # 获取连通域的边缘
                edges = cv2.Canny(component_mask, 100, 200)

                # 扩展边缘线条
                dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

                ori_img[dilated_edges == 255] = (0, 0, 255)  # 用红色绘制边缘

                region_info["connected_components"].append([int(cc[0]), int(cc[1]), int(cc[2]), int(cc[3])])
                
            # 裁剪并保存图像
            cropped_img = ori_img[y_min:y_max, x_min:x_max]
            cropped_and_marked_img_list.append(cropped_img)
            cropped_region_save_path = os.path.join(cropped_region_save_folder, f"cropped_region_{i}.png")
            cv2.imwrite(cropped_region_save_path, cropped_img)

            json_output[f"regions_{i}"] = region_info

    return result, optimal_clusters, json_output, cropped_img_list, cropped_and_marked_img_list

def _process_single_mask(args):
    """处理单个mask的辅助函数，用于多线程并行处理"""
    i, mask, category, save_folder = args
    category = category.replace(' ', '_')
    mask_np = np.array(mask)
    mask_processed = mask_np if mask_np.ndim == 2 else mask_np[:, :, -1]  # 选择 A 通道作为二值掩码
    
    # 将mask转换为PIL.Image对象然后保存
    mask_image = Image.fromarray(mask_processed)
    filename = f"{category}_mask.png"
    filepath = os.path.join(save_folder, filename)
    mask_image.save(filepath)
    
    return i, mask_processed

def generate_mask(input_folder):
    mask_file = f'{input_folder}/masks.pkl'
    catagory_json = f'{input_folder}/result.json'
    
    # 读取实例分割掩码数据
    with open(mask_file, "rb") as f:
        masks = pickle.load(f)

    # 读取类别标签数据
    with open(catagory_json, "r") as f:
        result_data = json.load(f)
    categorys = result_data["categorys"]

    # 检查掩码数量和类别标签数量是否一致
    assert len(masks) == len(categorys), "Masks and categories length mismatch."

    save_folder = f'{input_folder}/masks'
    os.makedirs(save_folder, exist_ok=True)

    # 获取第一个mask的尺寸，并创建combined_mask
    first_mask = np.array(masks[0])
    combined_mask = np.zeros((first_mask.shape[0], first_mask.shape[1]), dtype=np.uint8)
    
    # 使用多线程并行处理mask的保存
    # 准备参数列表
    tasks = [(i, mask, category, save_folder) for i, (mask, category) in enumerate(zip(masks, categorys))]
    
    # 存储处理后的mask，用于后续更新combined_mask
    processed_masks = {}
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=min(32, len(tasks))) as executor:
        futures = {executor.submit(_process_single_mask, task): task[0] for task in tasks}
        for future in as_completed(futures):
            try:
                i, mask_processed = future.result()
                processed_masks[i] = mask_processed
            except Exception as e:
                print(f"处理mask {futures[future]} 时出错: {e}")
    
    # 按顺序更新combined_mask（需要串行处理）
    for i in sorted(processed_masks.keys()):
        combined_mask = np.maximum(combined_mask, processed_masks[i])
    
    # # 保存combined_mask
    # combined_mask_image = Image.fromarray(combined_mask)
    # combined_mask_image.save(os.path.join(save_folder, "combined_mask.png"))
    
    save_folder_for_processing = f'{input_folder}/processed_masks'
    os.makedirs(save_folder_for_processing, exist_ok=True)
    # 创建并保存combined_mask的反转版本
    inverted_combined_mask = 255 - combined_mask
    inverted_combined_mask_image = Image.fromarray(inverted_combined_mask)
    inverted_combined_mask_image.save(os.path.join(save_folder_for_processing, "inverted_combined_mask.png"))

def _process_single_mask_with_index(args):
    """处理单个mask的辅助函数（带索引版本），用于多线程并行处理"""
    i, mask, category, save_folder = args
    category = category.replace(' ', '_')
    # 将mask转换为PIL.Image对象然后保存
    mask_image = Image.fromarray(mask)
    filename = f"instance_{i}_{category}_mask.png"
    filepath = os.path.join(save_folder, filename)
    mask_image.save(filepath)
    
    return i, mask

def generate_mask_and_process_missing_part(input_folder):
    mask_file = f'{input_folder}/masks.pkl'
    catagory_json = f'{input_folder}/result.json'
    
    # 读取实例分割掩码数据
    with open(mask_file, "rb") as f:
        masks = pickle.load(f)

    # 读取类别标签数据
    with open(catagory_json, "r") as f:
        result_data = json.load(f)
    categorys = result_data["categorys"]

    # 检查掩码数量和类别标签数量是否一致
    assert len(masks) == len(categorys), "Masks and categories length mismatch."

    save_folder = f'{input_folder}/masks'
    os.makedirs(save_folder, exist_ok=True)

    # 获取第一个mask的尺寸，并创建combined_mask
    first_mask = np.array(masks[0])
    combined_mask = np.zeros((first_mask.shape[0], first_mask.shape[1]), dtype=np.uint8)
    
    # 使用多线程并行处理mask的保存
    # 准备参数列表
    tasks = [(i, mask, category, save_folder) for i, (mask, category) in enumerate(zip(masks, categorys))]
    
    # 存储处理后的mask，用于后续更新combined_mask
    processed_masks = {}
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=min(32, len(tasks))) as executor:
        futures = {executor.submit(_process_single_mask_with_index, task): task[0] for task in tasks}
        for future in as_completed(futures):
            try:
                i, mask_processed = future.result()
                processed_masks[i] = mask_processed
            except Exception as e:
                print(f"处理mask {futures[future]} 时出错: {e}")
    
    # 按顺序更新combined_mask（需要串行处理）
    for i in sorted(processed_masks.keys()):
        combined_mask = np.maximum(combined_mask, processed_masks[i])
    
    # # 保存combined_mask
    # combined_mask_image = Image.fromarray(combined_mask)
    # combined_mask_image.save(os.path.join(save_folder, "combined_mask.png"))
    
    save_folder_for_processing = f'{input_folder}/processed_masks'
    os.makedirs(save_folder_for_processing, exist_ok=True)
    # 创建并保存combined_mask的反转版本
    inverted_combined_mask = 255 - combined_mask
    inverted_combined_mask_image = Image.fromarray(inverted_combined_mask)
    inverted_combined_mask_image.save(os.path.join(save_folder_for_processing, "inverted_combined_mask.png"))

    # 形态学处理
    open_mask = process_missing_mask_area(inverted_combined_mask, min_kernel_size=3, max_kernel_size=21)

    # 连通域聚类，分区画框
    ori_img_path = os.path.join(input_folder, 'ori.png')
    mask_result, optimal_clusters, json_output, cropped_img_list, cropped_and_marked_img_list = find_and_visualize_rectangles(ori_img_path, open_mask, save_folder_for_processing)

    # 保存结果图像
    output_path = os.path.join(save_folder_for_processing, f'mask_with_{optimal_clusters}_expanded_rectangles.png')
    cv2.imwrite(output_path, mask_result)
    print(f"Result image saved as {output_path}")

    # 保存JSON输出
    json_output_path = os.path.join(save_folder_for_processing, 'regions_bboxs_info.json')
    with open(json_output_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON output saved as {json_output_path}")

    return json_output, cropped_img_list, cropped_and_marked_img_list