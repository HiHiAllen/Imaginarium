"""
Legacy functions from S1_scene_parsing_op.py
These functions are migrated as-is to maintain compatibility.
All logic is preserved from the original S1 file.
"""
import argparse
import os
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import json
import pickle
import re
import time
import copy
import multiprocessing
import cv2
import glob
import functools

# ===== 让所有print自动flush，避免输出被缓冲 =====
print = functools.partial(print, flush=True)
from omegaconf import OmegaConf
from tqdm import tqdm
import open3d as o3d
from requests.exceptions import ReadTimeout
from collections import defaultdict
from skimage.measure import find_contours
from utils.masks import generate_mask, generate_mask_and_process_missing_part
from utils.dino_api import dino_api
from utils.partition import obj_bbox_crop_and_save, cluster_3d_obb
from utils.llm_api import GPTApi,parallel_processing_requests,extract_json_with_re,extract_list_with_re
from utils.ransac import estimate_floor_and_walls
from utils.obb import estimate_obj_depth_obb_faster
from utils.logger import Logger
from prompts.used_prompts import *

####  仅需安装 pip install dds-cloudapi-sdk==0.2.1 gradio==4.22.0
logger = None
gpt = None
FONT_TTF_PATH = None  # 从 config.yaml 的 font_ttf_path 中获取，由 parsing.py 的 _init_legacy_globals() 初始化
GROUND_DINO_TOKEN = None # 从 config.yaml 的 ground_dino_token 中获取，由 parsing.py 的 _init_legacy_globals() 初始化

def init_worker(gpt_params, logger_params, token=None):
    """Initializes gpt and logger for worker processes."""
    global gpt, logger, GROUND_DINO_TOKEN
    GROUND_DINO_TOKEN = token
    gpt = GPTApi(**gpt_params)
    if logger_params:
        logger_instance = Logger(**logger_params)
        logger = logger_instance.get_logger()
    
def draw_mask(mask, draw, random_color=True, color=(30, 144, 255, 153)):
    """Draws a mask with a specified color on an image.

    params:
        mask (np.array): Binary mask as a NumPy array.
        draw (ImageDraw.Draw): ImageDraw object to draw on the image.
        random_color (bool): Whether to use a random color for the mask.
        color (tuple): The color to use for the mask if not using a random color.
    """
    if random_color:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            153,
        )
    
    nonzero_coords = np.transpose(np.nonzero(mask))
    
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

def get_image_filenames(img_path_input):
    image_paths = []
    
    # Handle ListConfig from Hydra
    if OmegaConf.is_list(img_path_input):
        img_path_input = OmegaConf.to_container(img_path_input)

    if isinstance(img_path_input, list):
        return img_path_input
        
    if isinstance(img_path_input, str):
        if os.path.isfile(img_path_input):
            if img_path_input.endswith('.txt'):
                with open(img_path_input, 'r') as f:
                    image_paths = [line.strip() for line in f.readlines()]
            else:
                image_paths = [img_path_input]
        elif os.path.isdir(img_path_input):
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
                image_paths.extend(glob.glob(os.path.join(img_path_input, ext)))
    
    return [p for p in image_paths if os.path.isfile(p)]

def adjust_brightness(color, factor=1.5):
    """Adjust the brightness of a color."""
    return tuple(min(int(c * factor), 255) for c in color)

def visualize(image_pil: Image.Image,
              result: dict,
              draw_width: float = 2.5,
              return_mask=True,
              draw_score=True,
              box_threshold: float = 0.1) -> Image:
    # Get the bounding boxes and labels from the target dictionary
    masks = result.get("masks", [])
    valid_indices = [i for i, score in enumerate(result["scores"]) if score >= box_threshold]
    # 支持根据box_threshold过滤来生成final图片输出
    # 创建新的列表，只包含scores大于等于阈值的物体
    boxes = [result["boxes"][i] for i in valid_indices]
    scores = [result["scores"][i] for i in valid_indices]
    categorys = [result["categorys"][i] for i in valid_indices]
    masks = [masks[i] for i in valid_indices]
    
    # Find all unique categories and build a cate2color dictionary
    cate2color = {}
    
    wall_color_dict = {
        (0, 255, 255):'cyan',
        (0, 0, 255):'blue',
        (255, 165, 0):'orange',
        (128, 0, 128):'purple',
        (255, 20, 147):'pink',

    }
    
    wall_color_pop = list(wall_color_dict.keys())
    wall_color_name = {}
    
    unique_categorys = set(categorys)
    for cate in unique_categorys:
        base_color = tuple(np.random.randint(0, 255, size=3).tolist())
        bright_color = adjust_brightness(base_color)
        cate2color[cate] = bright_color
        if re.match(r'(ground|wall|ceiling|floor)_\d+', cate):
            if wall_color_pop:  # 检查 wall_color_pop 是否为空
                cate2color[cate] = wall_color_pop.pop()
                wall_color_name[cate] = (wall_color_dict[cate2color[cate]],cate2color[cate])
    
    # Create a PIL ImageDraw object to draw on the input image
    if isinstance(image_pil, np.ndarray):
        image_pil = Image.fromarray(image_pil)
    image_pil_ori = image_pil.copy()
    draw = ImageDraw.Draw(image_pil)
    
    # Create a new binary mask image with the same size as the input image
    mask = Image.new("L", image_pil.size, 0)
    # Create a PIL ImageDraw object to draw on the mask image
    mask_draw = ImageDraw.Draw(mask)

    # Draw boxes, labels, and masks for each box and label in the target dictionary
    for box, score, category in zip(boxes, scores, categorys):
        # Extract the box coordinates
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        color = cate2color[category]

        # Draw the box outline on the input image
        draw.rectangle([x0, y0, x1, y1], outline=color, width=int(draw_width))

        # Draw the label and score on the input image
        if draw_score:
            text = f"{category} {score:.2f}"
        else:
            text = f"{category}"
        
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), text, font)
        else:
            # Fallback for older Pillow versions
            bbox = draw.textbbox((x0, y0), text, font)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), text, fill="white")

    grounding_result = image_pil.copy()
    # Draw the mask on the input image if masks are provided
    if len(masks) > 0 and return_mask:
        size = image_pil.size
        mask_image = Image.new("RGBA", size, color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            mask_np = np.array(mask)
            mask = mask_np if mask_np.ndim == 2 else mask_np[:, :, -1] 
            draw_mask(mask, mask_draw)

        image_pil = Image.alpha_composite(image_pil.convert("RGBA"), mask_image).convert("RGB")
        
    # 下面这个用于绘制过滤掉特定物体、不显示分数、并且使用更大更粗的字体的图片
    # 创建新的图像用于特定需求的可视化
    filtered_image = image_pil_ori.copy()
    filtered_draw = ImageDraw.Draw(filtered_image)
    
    filtered_image_cont = image_pil_ori.copy()
    filtered_draw_cont = ImageDraw.Draw(filtered_image_cont)

    wall_floor_paint_image = image_pil_ori.copy().convert("RGBA")

    # 计算适合图像大小的字体大小
    font_size = int(min(image_pil_ori.size) / 70)  # 可以调整这个比例
    font_size = max(15, font_size)

    # 使用自定义字体文件，并设置字体大小
    font = ImageFont.truetype(FONT_TTF_PATH, font_size)

    # 记录已绘制文本的位置
    drawn_text_positions = []

    # 记录物体标签和包围框
    object_bboxes = {}
    object_bboxes_wo_text_bbox = {}

    # 先绘制所有边界框

    for box, category in zip(boxes, categorys):
        if not re.match(r'(ground|wall|ceiling|floor)_\d+', category):
            x0, y0, x1, y1 = map(int, box)
            color = cate2color[category]

            # 绘制边界框
            filtered_draw.rectangle([x0, y0, x1, y1], outline=color, width=int(draw_width*1.5))
            
            #绘制contur
            mask_np = np.array(masks[categorys.index(category)])
            mask = mask_np if mask_np.ndim == 2 else mask_np[:, :, -1]

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
                    points.append((x,y))
                filtered_draw_cont.line(points+[points[0]], fill=color, width=2)  # 绘制闭合轮廓

    # 再绘制所有文本标签
    for box, category in zip(boxes, categorys):
        if not re.match(r'(ground|wall|ceiling|floor)_\d+', category):
            x0, y0, x1, y1 = map(int, box)
            color = cate2color[category]

            # 绘制标签，使用更大的字体
            text = category
            
            # 使用 font.getbbox() 获取文本边界框
            left, top, right, bottom = font.getbbox(text)
            text_width = right - left
            text_height = bottom - top
            
            # 初始文本位置
            text_x, text_y = x0, y0

            # 尝试微调文本位置以避免重叠
            max_attempts = 10
            attempt = 0
            while attempt < max_attempts:
                bbox = [text_x, text_y, text_x + text_width, text_y + text_height]
                if not any([overlap(bbox, pos) for pos in drawn_text_positions]):
                    break
                text_x += 5  # 微调位置
                text_y += 5
                attempt += 1

            drawn_text_positions.append(bbox)
            
            # 绘制黑色边缘
            for offset in [(1, 1), (-1, -1), (1, -1), (-1, 1), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                filtered_draw.text((text_x + offset[0], text_y + offset[1]), text, fill="black", font=font)
                filtered_draw_cont.text((text_x + offset[0], text_y + offset[1]), text, fill="black", font=font)

            # 绘制与边界框相同颜色的文本
            filtered_draw.text((text_x, text_y), text, fill=color, font=font)
            filtered_draw_cont.text((text_x, text_y), text, fill=color, font=font)
            
            # 记录物体标签和包围框
            object_bboxes[category] = [
                min(x0, text_x),
                min(y0, text_y),
                max(x1, text_x + text_width),
                max(y1, text_y + text_height)
            ]
            
            object_bboxes_wo_text_bbox[category] = [x0, y0, x1, y1]

    # wall_floor_paint: 只绘制墙壁/地板的mask（不带标注）
    for mask, category, box in zip(masks, categorys, boxes):
        if re.match(r'(ground|wall|ceiling|floor)_\d+', category):
            
            # 获取对应类别的颜色
            fill_color = cate2color[category]

            # 将mask转换为二进制形式，并计算中心点
            mask_np = np.array(mask)
            binary_mask = mask_np > 0  if mask_np.ndim == 2 else mask_np[:, :, -1] > 0
            ys, xs = np.where(binary_mask)
            
            # 将 NumPy 数组转换为 Pillow 图像
            base_image = Image.fromarray(mask_np)
            #绘制透明mask
            # 创建一个透明图层 (RGBA)
            overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))

            # 在透明图层上绘制半透明内容
            overlay_draw = ImageDraw.Draw(overlay)
            
            fill_color = list(fill_color)
            fill_color += [50]  # 设置透明度
            fill_color = tuple(fill_color)
            # 绘制mask
            draw_mask(binary_mask, overlay_draw, random_color=False, color=fill_color)

            # 将透明图层叠加到原始图像上
            wall_floor_paint_image = Image.alpha_composite(wall_floor_paint_image, overlay)

    return image_pil, grounding_result, mask_image, filtered_image, filtered_image_cont, wall_floor_paint_image, object_bboxes, object_bboxes_wo_text_bbox, wall_color_name


def generate_wall_floor_annotation(save_folder: str, detection_results: dict):
    """
    轻量级函数：在 wall_floor_paint.png 基础上添加墙壁/地板标注
    复用已有的 wall_floor_paint.png，只添加文字标注
    """
    wall_floor_paint_path = os.path.join(save_folder, 'wall_floor_paint.png')
    if not os.path.exists(wall_floor_paint_path):
        logger.warning(f"wall_floor_paint.png not found at {wall_floor_paint_path}")
        return
    
    # 读取已有的 wall_floor_paint.png
    wall_floor_paint_image = Image.open(wall_floor_paint_path).convert("RGBA")
    draw = ImageDraw.Draw(wall_floor_paint_image)
    
    # 设置字体
    font_size = int(min(wall_floor_paint_image.size) / 20)
    font = ImageFont.truetype(FONT_TTF_PATH, font_size)
    
    # 构建颜色映射（按顺序分配，确保一致性）
    wall_color_dict = {
        (0, 255, 255): 'cyan',
        (0, 0, 255): 'blue',
        (255, 165, 0): 'orange',
        (128, 0, 128): 'purple',
        (255, 20, 147): 'pink',
    }
    wall_color_list = list(wall_color_dict.keys())
    cate2color = {}
    color_idx = 0
    for cate in detection_results['categorys']:
        if re.match(r'(ground|wall|ceiling|floor)_\d+', cate) and cate not in cate2color:
            if color_idx < len(wall_color_list):
                cate2color[cate] = wall_color_list[color_idx]
                color_idx += 1
    
    masks = detection_results.get('masks', [])
    categorys = detection_results['categorys']
    
    # 添加标注
    for mask, category in zip(masks, categorys):
        if not re.match(r'(ground|wall|ceiling|floor)_\d+', category):
            continue
        if category not in cate2color:
            continue
            
        mask_np = np.array(mask)
        binary_mask = mask_np > 0 if mask_np.ndim == 2 else mask_np[:, :, -1] > 0
        ys, xs = np.where(binary_mask)
        if xs.size == 0 or ys.size == 0:
            continue
        
        center_x, center_y = np.mean(xs), np.mean(ys)
        text = category
        left, top, right, bottom = font.getbbox(text)
        text_width, text_height = right - left, bottom - top
        text_x = center_x - text_width // 2
        text_y = center_y - text_height // 2
        
        # 边界检查
        image_width, image_height = wall_floor_paint_image.size
        text_x = max(0, min(text_x, image_width - text_width))
        text_y = max(0, min(text_y, image_height - text_height))
        
        # 绘制白色背景
        background_margin = 5
        draw.rectangle(
            [text_x - background_margin, text_y - background_margin,
             text_x + text_width + background_margin, text_y + text_height + background_margin],
            fill="white"
        )
        
        # 绘制文本
        fill_color = cate2color[category]
        draw.text((text_x, text_y), text, fill=fill_color, font=font)
    
    # 保存
    wall_floor_paint_image.save(os.path.join(save_folder, 'wall_floor_paint_with_annotation.png'))


def overlap(bbox1, bbox2):
    """Check if two bounding boxes overlap."""
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    return not (x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1)
    
def get_params():
    parser = argparse.ArgumentParser(description="Interactive Inference")
    parser.add_argument(
        "--token",
        type=str,
        help="The token for T-Rex2 API. We are now opening free API access to T-Rex2",
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.3, help="The threshold for box score"
    )
    return parser.parse_args()

def save_result_and_partition(image, results, save_folder, box_threshold, save_masks_pkl=False):
    os.makedirs(save_folder, exist_ok=True)
    if isinstance(image, np.ndarray):
        # Assume the input NumPy array is in RGB format.
        image_pil = Image.fromarray(image)
    elif isinstance(image, str):
        image_pil = Image.open(image).convert("RGB")
    else: # Assumes it's already a PIL Image
        image_pil = image
    image_pil.save(os.path.join(save_folder, 'ori.png'))
    
    if not all(item.split('_')[-1].isdigit() for item in results['categorys']):
        # 给每个实例编号
        category_counts = {}
        output_list = []
        for item in results['categorys']:
            if item not in category_counts:
                category_counts[item] = 0
            else:
                category_counts[item] += 1
            output_list.append(f"{item}_{category_counts[item]}")

        results['categorys']=output_list
    
    # 注意box_threshold仅在输出图片上体现，所有数据均保存至json和pkl文件
    image_pil, grounding_result, mask_image, filtered_image, filtered_image_cont, wall_floor_paint, object_bboxes, object_bboxes_wo_text_bbox, wall_color_name = visualize(image_pil, results, draw_width=2.5, box_threshold=box_threshold)
    grounding_result.save(os.path.join(save_folder, 'bbox_result.png'))
    mask_image.save(os.path.join(save_folder, 'mask_result.png'))
    image_pil.save(os.path.join(save_folder, 'bbox_mask_result.png'))
    filtered_image.save(os.path.join(save_folder, 'filtered_bbox_result.png'))
    filtered_image_cont.save(os.path.join(save_folder, 'filtered_bbox_cont_result.png'))
    wall_floor_paint.save(os.path.join(save_folder, 'wall_floor_paint.png'))
    with open(os.path.join(save_folder, 'wall_color_name.json'), 'w') as f:
        json.dump(wall_color_name, f, indent=2)
    if save_masks_pkl:
        with open(os.path.join(save_folder, 'masks.pkl'), "wb") as f:
            pickle.dump(results['masks'], f)
    
    # 创建不包含 masks 的副本
    results_copy = copy.deepcopy(results)
    del results_copy['masks']

    # 保存结果到 JSON
    with open(os.path.join(save_folder, 'result.json'), 'w') as f:
        json.dump(results_copy, f, indent=2)

    # 保存其他 JSON 文件
    with open(os.path.join(save_folder, 'object_bboxes_json.json'), 'w') as f:
        json.dump(object_bboxes, f, indent=2)

    with open(os.path.join(save_folder, 'object_bboxes_wo_text_bbox_json.json'), 'w') as f:
        json.dump(object_bboxes_wo_text_bbox, f, indent=2)
                    
def format_object_name(name):
    """格式化物体名称，用于统一处理大小写和特殊字符"""
    return name.replace('-', '_').replace(' ', '_').lower()

def is_object_in_list(obj, obj_list):
    """检查格式化后的物体名称是否在给定列表中"""
    return format_object_name(obj) in obj_list

def hierarchical_traversal_of_scene_objs(image, class_en_list, task_id=None):
    # 预处理class_en_list，格式化并去重
    formatted_class_en_list = set(format_object_name(item) for item in class_en_list)

    predefined_eng_categories_list = list(formatted_class_en_list)
    prompt = SCENE_HIERARCHICAL_TRAVERSAL_PROMPT.format(predefined_eng_categories_list=predefined_eng_categories_list)
    
    scene_graph_dict_result = None
    for _ in range(3):
        response = gpt.get_response(prompt, image, temperature=0.7)
        logger.info(f'图片解析json内容为:\n{response}')
        scene_graph_dict_result = extract_json_with_re(response)
        if scene_graph_dict_result:
            break
        else:
            logger.info(f'进程{task_id}无法解析出dict: {response}')

    if not scene_graph_dict_result:
        return None, []

    # 提取所有物体清单，按照gdino的输入格式，并确保它们在class_en_list中
    list_result = {format_object_name(obj) for area_data in scene_graph_dict_result.values() for obj in area_data.keys()}
    list_result.update(format_object_name(child) for area_data in scene_graph_dict_result.values() for child_objects in area_data.values() for child in child_objects)
    list_result = {'floor', 'wall', 'ceiling', 'carpet'}.union(list_result)  # 添加默认物体
    
    # 过滤 list_result 中不在 formatted_class_en_list 和 ['floor', 'wall', 'ceiling'] 中的元素
    filtered_list_result = {
        item for item in list_result 
        if item in formatted_class_en_list or item in ['floor', 'wall', 'ceiling']
    }

    # 更新 list_result
    list_result = filtered_list_result
        
    list_result = list(list_result.intersection(formatted_class_en_list))  # 确保结果仅包含class_en_list中的元素

    return scene_graph_dict_result, list_result


def cut_and_draw_bbox(image_path, region_info):
    # 读取原始图片
    raw_image = Image.open(image_path)
    
    # 获取区域的 bbox
    x1, y1, x2, y2 = region_info['bbox']
    
    # 裁剪区域图片
    region_pic = raw_image.crop((x1, y1, x2, y2))
    
    # 创建一个可以在region_pic上绘图的对象
    draw = ImageDraw.Draw(region_pic)
    
    # 在region_pic上绘制connected_components的bbox
    for bbox in region_info['connected_components']:
        # 由于我们裁剪了图片，需要调整bbox坐标
        adjusted_bbox = [
            max(0, bbox[0] - x1 - 10),
            max(0, bbox[1] - y1 - 10),
            min(region_pic.width, bbox[2] - x1 + 10),
            min(region_pic.height, bbox[3] - y1 + 10)
        ]
        # 绘制矩形
        draw.rectangle(adjusted_bbox, outline="red", width=3)
    
    return region_pic

def process_single_request_of_gpt_and_gd(params):
    region_bbox, prompt, image_list, cropped_img_np, class_en_list = params
    res = gpt.get_response(prompt, image=image_list)
    add_list = extract_list_with_re(res)
    inspected_detect_items_list_result = list(set(add_list))
    
    # 确保所有label都在formatted_class_en_list里面
    if class_en_list:
        formatted_class_en_list = set(format_object_name(item) for item in class_en_list) 
        for label in inspected_detect_items_list_result:
            formatted_label = format_object_name(label)  # 格式化label以匹配格式化的class_en_list
            if formatted_label not in formatted_class_en_list and formatted_label not in ['floor', 'wall', 'ceiling']:
                # 这里选择抛出异常，但你也可以选择记录警告
                raise ValueError(f"在查漏补缺阶段, Label '{label}' not found in class_en_list.")
    
    detect_items = [item.replace(' ', '_') for item in inspected_detect_items_list_result]
    inspected_detect_items_GD_prompt = ".".join(detect_items)
    
    if not inspected_detect_items_list_result: return (region_bbox, None, None)
    
    prompts = dict(image=cropped_img_np, prompt=inspected_detect_items_GD_prompt)
    max_retries = 20
    for attempt in range(max_retries):
        try:
            #TODO: replace the gd to local gd
            # results = gdino.inference(prompts, return_mask=True)
            # #convert mask to 2D
            
            results = dino_api(prompts, token=GROUND_DINO_TOKEN)
            # The mask orientation is now corrected in dino_api, so the correction logic here is removed.
            
            # Convert RGBA masks to 2D binary masks by extracting the alpha channel.
            results['masks'] = [np.array(mask)[:, :, 3] for mask in results['masks']]
            break
        except ReadTimeout:
            if attempt < max_retries - 1:
                logger.warning(f"ReadTimeout occurred. Retrying {attempt + 1}/{max_retries}...")
                time.sleep(2)  # 等待2秒后重试
            else:
                logger.error("Max retries reached. Raising exception.")
                return (region_bbox, None, None)

    return (region_bbox, inspected_detect_items_list_result, results)

def parallel_processing_gpt_and_gd_requests(region_bbox_list, all_image_list, all_prompt_list, cropped_img_np_list, save_folder, gpt_params, predefined_eng_categories_list=None, num_processes=8):
    all_results = []

    # 创建参数列表
    params_list = [(region_bbox, prompt, image_list, cropped_img_np, predefined_eng_categories_list) for region_bbox, prompt, image_list, cropped_img_np in zip(region_bbox_list, all_prompt_list, all_image_list, cropped_img_np_list)]

    # 使用进程池并行处理
    logger_params = {'name': 'S1_Worker', 'log_file': os.path.join(save_folder, "S1_scene_parsing_op_worker.log"), 'level': "INFO"}
    with multiprocessing.Pool(processes=min(len(params_list), num_processes), initializer=init_worker, initargs=(gpt_params, logger_params)) as pool:
        all_results = pool.map(process_single_request_of_gpt_and_gd, params_list)
    return all_results

def secondary_inspection_of_scene_objs_detection(init_results, area_bbox_json_output, cropped_img_list, cropped_and_marked_img_list, class_en_list, save_folder, gpt_params):
    '''
    area_bbox_json_output: {'regions_0': {'bbox': [723, 666, 1493, 1009], 'connected_components': [[1281, 723, 1366, 767], [1289, 890, 1345, 953]]}, 'regions_1': ...}
    detect_items_list_result:  ['wall', 'ground', 'ceiling', 'window', 'sign']
    '''
    predefined_eng_categories_list = set(format_object_name(item) for item in class_en_list)
    
    assert len(area_bbox_json_output) == len(cropped_and_marked_img_list)
    all_image_list= []
    all_prompt_list = []
    region_bbox_list = []
    cropped_img_np_list = []
    for (region_index, content), cropped_img_np, cropped_and_marked_img_np in tqdm(zip(area_bbox_json_output.items(), cropped_img_list, cropped_and_marked_img_list)):
        # image_list = [image, cropped_and_marked_img_np]    # 原图地址+分割后的图片
        image_list = [cropped_and_marked_img_np]    # 分割后的图片
        secondary_inspection_prompt = SECONDARY_GET_ITEM_LIST_PROMPT.format(predefined_eng_categories_list=predefined_eng_categories_list)
        all_image_list.append(image_list)
        all_prompt_list.append(secondary_inspection_prompt)
        region_bbox_list.append(content['bbox'])
        cropped_img_np_list.append(cropped_img_np)
    logger.info(f'开始并发处理区域对话')
    
    # 使用进程池并行处理   返回的是一个list， 每个元素是一个tuple  (region_bbox, detect_items, results)   results指的是region_bbox经过gd得到的结果
    sec_results = parallel_processing_gpt_and_gd_requests(region_bbox_list, all_image_list, all_prompt_list, cropped_img_np_list, save_folder, gpt_params, predefined_eng_categories_list, num_processes=min(len(region_bbox_list), 8))
    
    # 把新检测的结果增量的加到第一次检测的结果之上
    updated_result, added_labels = update_segmentation(init_results, sec_results)
    return sec_results, updated_result, added_labels

def secondary_inspection_of_scene_objs(image, area_bbox_json_output, detect_items_list_result, gpt_params):
    '''
    area_bbox_json_output: {'regions_0': {'bbox': [723, 666, 1493, 1009], 'connected_components': [[1281, 723, 1366, 767], [1289, 890, 1345, 953]]}, 'regions_1': ...}
    detect_items_list_result:  ['wall', 'ground', 'ceiling', 'window', 'sign']
    '''
    # 按照区域分多次对话，将原图+区域图一起输入gpt4v，查漏补缺
    all_image_list= []
    all_prompt_list = []
    for region_index, content in tqdm(area_bbox_json_output.items()):
        region_pic = cut_and_draw_bbox(image, content)
        image_list = [image, region_pic]    # 原图地址+分割后的图片
        secondary_inspection_prompt = SECONDARY_INSPECTION_PROMPT.format(detect_items_list_result=detect_items_list_result)
        all_image_list.append(image_list)
        all_prompt_list.append(secondary_inspection_prompt)
    logger.info(f'开始并发处理区域对话')
    
    # 使用进程池并行处理
    results = parallel_processing_requests(gpt_params, all_image_list, all_prompt_list, return_list=True, return_json=False, return_dict=False, num_processes=8)
    for i, result in enumerate(results):
        logger.info(f"Result for request {i}: {result}")
        if result: detect_items_list_result.extend(result) 
    inspected_detect_items_list_result = list(set(detect_items_list_result))
    
    detect_items = [item.replace(' ', '_') for item in inspected_detect_items_list_result]
    inspected_detect_items_GD_prompt = ".".join(detect_items)
    return inspected_detect_items_GD_prompt, inspected_detect_items_list_result

def update_segmentation(init_results, sec_results):
    # 提取已经存在的类别名称
    existing_categories = set(init_results["categorys"])
    
    # 将所有的初始掩码合并为一个整体
    original_combined_mask = np.zeros_like(np.array(init_results["masks"][0]))
    for mask in init_results["masks"]:
        mask_np = np.array(mask) 
        original_combined_mask = np.maximum(original_combined_mask, mask_np)
    
    combined_mask = original_combined_mask.copy()
    
    # 创建一个列表来存储新添加的标签
    added_labels = []
    
    # 遍历 sec_results
    for region_bbox, _, results in sec_results:
        if not results: continue
        x1, y1, x2, y2 = region_bbox
        for i, mask in enumerate(results["masks"]):
            # 将 mask 转换为 numpy 数组
            mask_array = np.array(mask)
            
            # 创建一个与原始图像大小相同的空白掩码
            full_mask = np.zeros_like(combined_mask)
            # 将裁剪区域的掩码放回原始图像坐标中
            full_mask[y1:y2, x1:x2] = mask_array
            
            # 计算在 original_combined_mask 中的空白区域
            blank_region = (original_combined_mask == 0)
            # 计算重叠区域
            overlap = full_mask & blank_region
            overlap_ratio = np.sum(overlap) / (np.sum(mask_array) / 255)
            
            # 计算当前 region_bbox 的空白区域
            current_blank_region = (combined_mask[y1:y2, x1:x2] == 0)
            new_blank_region = current_blank_region & (full_mask[y1:y2, x1:x2] == 0)
            
            # 检查是否可以替换
            if overlap_ratio >= 0.7 and np.sum(new_blank_region) < np.sum(current_blank_region):
                # 生成新标签
                base_category = results["categorys"][i]
                index = 0
                new_label = base_category + f'_{index}'
                while new_label in existing_categories:
                    index += 1
                    new_label = '_'.join(new_label.split('_')[:-1]) + f'_{index}'
                
                # 更新类别和掩码
                existing_categories.add(new_label)
                combined_mask = np.where(overlap, 1, combined_mask)
                init_results["scores"].append(results["scores"][i])
                init_results["categorys"].append(new_label)
                
                # 将新标签添加到新标签列表
                added_labels.append(new_label)
                
                # 计算新的边界框
                non_zero_indices = np.argwhere(overlap)
                if non_zero_indices.size > 0:
                    min_y, min_x = non_zero_indices.min(axis=0)
                    max_y, max_x = non_zero_indices.max(axis=0)
                    new_box = (int(min_x), int(min_y), int(max_x), int(max_y))
                else:
                    new_box = tuple(map(int, region_bbox))
                init_results["boxes"].append(new_box)
                init_results["masks"].append(Image.fromarray((overlap * 255).astype(np.uint8)))

    return init_results, added_labels

def generate_scene_graph_geometry(region_partition_result_folder, filtered_bbox_image, all_items_list, gpt_params):
    """
    优化版场景图生成：数据在内存中迭代处理，减少中间文件保存
    Optimized scene graph generation: process data in memory, minimize intermediate file saves
    """
    with open(os.path.join(region_partition_result_folder, 'clustered_bboxes.json'), 'r') as f:
        region_partition_result_dict = json.load(f)
    
    parent_path = region_partition_result_folder.replace("clustered_result",'')
    with open(os.path.join(parent_path, 'wall_color_name.json'), 'r') as f:
        wall_color_name_dict = json.load(f)
        
    obj_result_folder = region_partition_result_folder.replace('clustered_result','obj_partition_result')
    with open(os.path.join(obj_result_folder, 'crop_masks.json'), 'r') as f:
        obj_crop_result_dict = json.load(f)
    
    wall_color_name = ''
    for wall_name, color in wall_color_name_dict.items():
        wall_color_name += f"{wall_name} is {color[0]}, "

    # 生成初始场景图
    all_image_list= []
    all_prompt_list = []
    for cluster_index, content in tqdm(region_partition_result_dict.items()):
        items_in_region = []
        assert os.path.exists(content['image_path'])
        image_list = [content['image_path'], filtered_bbox_image]    
        obj_near_str = ''
        for item in content['objects']:
            key = item['key']
            items_in_region.append(key)
            near_obj_name = obj_crop_result_dict[key]['near_obj_name']
            
            if len(near_obj_name) > 0:
                obj_near_str += f"{key}:{near_obj_name}\n"
            else:
                near_obj_name = list(wall_color_name_dict.keys())
                obj_near_str += f"{key}:{near_obj_name}\n"

        generate_scene_graph_prompt = GENERATE_SCENE_GRAPH_PROMPT_FLOOR_WALL.format(
            all_items_list=all_items_list, obj_near_str=obj_near_str, 
            items_in_region=items_in_region, wall_color_name=wall_color_name
        )
        all_image_list.append(image_list)
        all_prompt_list.append(generate_scene_graph_prompt)
    
    logger.info(f'开始并发处理区域scene graph (共{len(all_image_list)}个区域)')
    scene_graph_result = {}
    results = parallel_processing_requests(gpt_params, all_image_list, all_prompt_list, 
                                          return_list=False, return_json=True, return_dict=False, num_processes=8)
    for i, result in enumerate(results):
        logger.info(f"Result for request {i}: {result}")
        if result:
            scene_graph_result.update(result)
    logger.info(f'scene_graph_result: {scene_graph_result}')

    # === 在内存中迭代更新场景图 ===
    
    # Step 1: 添加支撑关系 (supported)
    for key, content in scene_graph_result.items():
        if content['isHangingFromCeiling']:
            scene_graph_result[key]['supported'] = 'ceiling_0'
            continue
        if content['isOnFloor']:
            scene_graph_result[key]['supported'] = 'floor_0'
            continue
        if content['isHangingOnWall']:
            scene_graph_result[key]['supported'] = 'wall'
            continue
        if re.match(r'(curtain|window)_\d+', key):
            scene_graph_result[key]['supported'] = 'wall'
            continue
        
        supported_item = supported_generate(key, parent_path)
        if supported_item is not None:
            scene_graph_result[key]['supported'] = supported_item
        else:
            scene_graph_result[key]['supported'] = 'wall'

    # Step 2: 添加靠墙关系 (againstWall)
    for key, content in scene_graph_result.items():
        if content['isAgainstWall']:
            wall_keys, most_like_wall = against_wall_generate_top_down(key, parent_path)
            scene_graph_result[key]['againstWall'] = wall_keys
            scene_graph_result[key]['most_like_wall'] = most_like_wall
            
            if len(wall_keys) == 0:
                scene_graph_result[key]['againstWall'] = None
                scene_graph_result[key]['most_like_wall'] = most_like_wall

    # Step 3: 修正支撑关系
    for key, content in scene_graph_result.items():
        if content['supported'] == 'wall':
            if 'againstWall' not in content.keys():
                scene_graph_result[key]['supported'] = 'floor_0'
                continue
            if content['againstWall'] is not None:
                scene_graph_result[key]['supported'] = scene_graph_result[key]['most_like_wall']
            else:
                scene_graph_result[key]['supported'] = 'floor_0'

    return scene_graph_result

def create_grids_for_floor_verification(objects_data, background_image, masks_folder, save_dir, grid_size=4):
    """
    创建NxN grid图像用于验证父物体是否为地面
    每个grid cell包含一个物体的裁剪图像及其bbox
    
    Args:
        objects_data: 物体数据字典
        background_image: 背景图像
        masks_folder: mask文件夹路径
        save_dir: 保存目录
        grid_size: grid的大小，默认为4表示4x4 grid (可选择3表示3x3)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    grid_images_info = {}
    object_items = list(objects_data.items())
    
    # 定义grid规格 - 可配置grid_size (3x3 或 4x4)
    cell_size = 256
    margin = 10
    grid_dim = grid_size * cell_size + (grid_size + 1) * margin
    font = ImageFont.truetype(FONT_TTF_PATH, 20)

    for i in range(0, len(object_items), grid_size * grid_size):
        chunk = dict(object_items[i : i + grid_size * grid_size])
        grid_image = Image.new('RGB', (grid_dim, grid_dim), (0, 0, 0))  # 黑色背景
        
        object_names_in_grid = []
        for j, (obj_name, data) in enumerate(chunk.items()):
            mask_path = os.path.join(masks_folder, f'{obj_name}_mask.png')
            if not os.path.exists(mask_path):
                continue

            mask_img = Image.open(mask_path).convert('L')
            bbox = mask_img.getbbox()
            
            if not bbox:
                continue

            # 1. 创建扩展的场景裁剪及bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # 调整扩展以使裁剪更接近正方形，并增加扩展比例以便模型看到更多周围上下文
            if width < 100 and height < 100:
                # 小物体：增加固定扩展到100px，以便看清楚周围的支撑关系
                expand_x = 100
                expand_y = 100
            else:
                # 大物体：从10%增加到40%，让模型看到更多周围环境来判断支撑关系
                base_padding_ratio = 0.4
                expand_x = int(width * base_padding_ratio)
                expand_y = int(height * base_padding_ratio)

            # 计算填充后的尺寸并添加更多填充到较小的一侧以使其正方形
            padded_width = width + 2 * expand_x
            padded_height = height + 2 * expand_y
            if padded_width > padded_height:
                expand_y += (padded_width - padded_height) // 2
            elif padded_height > padded_width:
                expand_x += (padded_height - padded_width) // 2

            img_width, img_height = background_image.size
            expanded_bbox = [
                max(0, bbox[0] - expand_x),
                max(0, bbox[1] - expand_y),
                min(img_width, bbox[2] + expand_x),
                min(img_height, bbox[3] + expand_y)
            ]
            scene_crop_raw = background_image.crop(expanded_bbox)

            draw_crop = ImageDraw.Draw(scene_crop_raw)
            original_bbox_in_crop = [
                bbox[0] - expanded_bbox[0],
                bbox[1] - expanded_bbox[1],
                bbox[2] - expanded_bbox[0],
                bbox[3] - expanded_bbox[1]
            ]
            
            # 动态边框宽度
            box_line_width = max(2, min(5, int(min(width, height) * 0.04)))
            draw_crop.rectangle(original_bbox_in_crop, outline="red", width=box_line_width)
            
            # 2. 调整大小并填充以适应cell
            ratio = min(cell_size / scene_crop_raw.width, cell_size / scene_crop_raw.height)
            new_size = (int(scene_crop_raw.width * ratio), int(scene_crop_raw.height * ratio))
            resized_img = scene_crop_raw.resize(new_size, Image.Resampling.LANCZOS)
            
            cell_img = Image.new('RGB', (cell_size, cell_size), (0, 0, 0))  # 黑色cell背景
            paste_pos = ((cell_size - new_size[0]) // 2, (cell_size - new_size[1]) // 2)
            cell_img.paste(resized_img, paste_pos)
            
            # 3. 在cell图像上绘制物体名称
            cell_draw = ImageDraw.Draw(cell_img)
            cell_draw.text((10, 225), obj_name, font=font, fill=(0, 255, 0))

            # 4. 将cell粘贴到主grid中
            grid_row, grid_col = j // grid_size, j % grid_size
            paste_x = margin + grid_col * (cell_size + margin)
            paste_y = margin + grid_row * (cell_size + margin)
            grid_image.paste(cell_img, (paste_x, paste_y))
            
            object_names_in_grid.append(obj_name)
        
        grid_save_path = os.path.join(save_dir, f'floor_verification_grid_{i // (grid_size*grid_size)}.png')
        grid_image.save(grid_save_path)
        grid_images_info[grid_save_path] = object_names_in_grid
        
    return grid_images_info


def verify_floor_parent_with_vlm(save_folder, scene_graph_result, gpt_params, grid_size=3):
    """
    使用VLM验证父物体为地面的物体是否真的在地面上
    如果GPT认为不是地面，则删除该物体
    
    Args:
        save_folder: 保存文件夹路径
        scene_graph_result: 场景图结果
        gpt_params: GPT参数
        grid_size: grid大小，3表示3x3，4表示4x4，默认为3
        
    Returns:
        scene_graph_result: 更新后的场景图（在内存中）
        objects_to_delete: 被删除的异常物体列表
    """
    logger.info(f"开始使用VLM进行地面父物体异常检测... (使用 {grid_size}x{grid_size} grid)")
    start_time = time.time()
    
    masks_folder = os.path.join(save_folder, 'masks')
    ori_scene_img_path = os.path.join(save_folder, 'ori.png')
    
    # 1. 找到所有父物体为地面的物体
    floor_objects = {}
    for obj, props in scene_graph_result.items():
        if props.get('supported') == 'floor_0':
            floor_objects[obj] = props
    
    if not floor_objects:
        logger.info("没有找到父物体为地面的物体，跳过地面验证步骤。")
        return scene_graph_result, []
    
    logger.info(f"找到 {len(floor_objects)} 个父物体为地面的物体: {list(floor_objects.keys())}")
    
    # 2. 为VLM准备数据
    ori_scene_img = Image.open(ori_scene_img_path)
    grid_save_dir = os.path.join(save_folder, 'vlm_floor_verification_grids')
    grid_images_info = create_grids_for_floor_verification(floor_objects, ori_scene_img, masks_folder, grid_save_dir, grid_size=grid_size)
    
    all_image_list = []
    all_prompt_list = []
    
    for grid_path, object_names in grid_images_info.items():
        object_names_str = "\n".join([f"- {name}" for name in object_names])
        prompt = FLOOR_VERIFICATION_PROMPT.format(object_names=object_names_str)
        all_image_list.append(grid_path)
        all_prompt_list.append(prompt)
    
    logger.info(f"向VLM发送 {len(all_prompt_list)} 个请求进行地面验证...")
    
    # 3. 调用VLM
    results = parallel_processing_requests(gpt_params, all_image_list, all_prompt_list, return_list=False, return_json=True, return_dict=False, num_processes=8)

    # 4. 解析结果并在内存中删除异常物体
    objects_to_delete = []
    for result_dict in results:
        if result_dict and isinstance(result_dict, dict):
            for obj_name, result_data in result_dict.items():
                is_on_floor = result_data['is_floor_supported']
                if obj_name in scene_graph_result:
                    if not is_on_floor:  # GPT认为不在地面上
                        objects_to_delete.append(obj_name)
                        logger.info(f"检测到异常物体 '{obj_name}': GPT认为该物体不在地面上，将被删除；reason: {result_data['reason']}")
    
    # 5. 从scene_graph_result中删除异常物体
    for obj_name in objects_to_delete:
        if obj_name in scene_graph_result:
            del scene_graph_result[obj_name]
            logger.info(f"已从scene graph中删除异常物体: {obj_name}")
    
    logger.info(f"地面父物体验证完成。共删除 {len(objects_to_delete)} 个异常物体。耗时: {time.time() - start_time:.2f}s.")

    return scene_graph_result, objects_to_delete

def draw_item(draw,center_x,center_y,radius,color="orange"):
        # 计算圆形的边界框
    left_up_point = (center_x - radius, center_y - radius)
    right_down_point = (center_x + radius, center_y + radius)
    bounding_box = [left_up_point, right_down_point]

    # 绘制橙色填充、白色边框的圆形
    draw.ellipse(bounding_box, fill=color, outline="white", width=5)
        
def vis_scene_graph(scene_graph_path,scene_graph_result,id=0):
    
    with open(os.path.join(scene_graph_path, 'result.json'), 'r') as f:
        detect_results = json.load(f)
        
    #get obj-bbox
    bbox_dict = {}
    for key,bbox in zip(detect_results['categorys'],detect_results['boxes']):
        bbox_dict[key] = bbox

    img = Image.open(f"{scene_graph_path}/ori.png")
    draw = ImageDraw.Draw(img)
    for key,edge_target in scene_graph_result.items():
        key_center = (bbox_dict[key][0]+bbox_dict[key][2])//2, (bbox_dict[key][1]+bbox_dict[key][3])//2
        draw.text(key_center, key, fill="black",size=20)
        
        if 'supported' in edge_target:
            edge_target_parent = edge_target['supported'] 
            
            if edge_target_parent not in bbox_dict: 
                draw.text((key_center[0]-10, key_center[1]+10), edge_target_parent, fill="orange",size=20)
            else:
                if not re.match(r'(ground|wall|ceiling|floor)_\d+', edge_target_parent):
                    parent_center = (bbox_dict[edge_target_parent][0]+bbox_dict[edge_target_parent][2])//2, (bbox_dict[edge_target_parent][1]+bbox_dict[edge_target_parent][3])//2
                    draw.text(parent_center, edge_target_parent, fill="orange")
                    draw.line([key_center, parent_center], fill='black', width=3)
                    if 'SpatialRel' in edge_target:
                        edge_target_SpatialRel = edge_target['SpatialRel']
                        spation_rel_center = ((key_center[0]+parent_center[0])//2, (key_center[1]+parent_center[1])//2)
                        draw.text(spation_rel_center, edge_target_SpatialRel, fill="green")
                else:
                    draw.text((key_center[0]-10, key_center[1]+10), edge_target_parent, fill="orange",size=20)



        if 'againstWall' in edge_target:
            if edge_target['againstWall'] is not None:
                edge_target_againstWall = edge_target['againstWall'][0]
                if edge_target_againstWall:
                    draw.text((key_center[0]-15, key_center[1]-15), edge_target_againstWall, fill="blue",size=20)
        else:
            if 'isAgainstWall'  in edge_target:
                is_againstWall = edge_target['isAgainstWall']
                if is_againstWall:
                    draw.text((key_center[0]-10, key_center[1]-10), "YES-isAgainstWall", fill="blue",size=20)

        if 'isOnFloor' in edge_target:
            isOnFloor = edge_target['isOnFloor']
            if isOnFloor:
                draw.text((key_center[0]-10, key_center[1]-30), "YES-isOnFloor", fill="blue",size=20)
        if 'isHangingFromCeiling' in edge_target:
            isHangingFromCeiling = edge_target['isHangingFromCeiling']
            if isHangingFromCeiling:
                draw.text((key_center[0]-10, key_center[1]-50), "YES-isHangingFromCeiling", fill="blue",size=20)

    img.save(f"{scene_graph_path}/scene_graph_{id}.png")


def supported_generate(key,parent_path):

    mask_folder = os.path.join(parent_path, 'pcd_mask')

    obj_result_folder = os.path.join(parent_path, 'obj_partition_result')

    # Use a corrected path for crop_masks.json by removing the extra subdirectory.
    crop_masks_path = os.path.join(obj_result_folder, 'crop_masks.json')
    if not os.path.exists(crop_masks_path):
        logger.error(f"Error: crop_masks.json not found at {crop_masks_path}")
        return None
    with open(crop_masks_path, 'r') as f:
        obj_crop_result_dict = json.load(f)

    near_obj_name = obj_crop_result_dict[key]['near_obj_name']
    mask_path = os.path.join(mask_folder, f'{key}_yoz.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning(f"Warning: Could not read mask for {key} from {mask_path}. Cannot determine support.")
        return None
    
    supported_item = None
    other_mask_list = []

    for name in near_obj_name:
        other_mask = cv2.imread(os.path.join(mask_folder,f'{name}_yoz.png'),cv2.IMREAD_GRAYSCALE)
        other_mask_list.append(other_mask)

    # 取mask底部的像素
    bottom_mask = extract_bottom_percent(mask,percent=0.05)
    # logger.info(f"get:{key}")

    contact_ratios, first_contact_mask_id = find_first_contact_mask(bottom_mask,other_mask_list)

    if first_contact_mask_id !=None:
        supported_item = near_obj_name[first_contact_mask_id]

    # if first_contact_mask_id!=None:
    #     logger.info([f"{x}:{y}" for x,y in zip(near_obj_name,list(contact_ratios))])

    #floor check
    floor_mask_path = os.path.join(mask_folder, 'full_yoz.png')
    floor_mask = cv2.imread(floor_mask_path, cv2.IMREAD_GRAYSCALE)
    if floor_mask is not None:
        floor_mask = extract_bottom_percent(floor_mask,percent=0.08)
        if np.any(floor_mask&bottom_mask):
            supported_item = 'floor_0'
    else:
        logger.warning(f"Warning: Could not read floor mask from {floor_mask_path}.")
    
    #celling check
    celling_mask_path = os.path.join(mask_folder, 'full_yoz.png')
    celling_mask = cv2.imread(celling_mask_path, cv2.IMREAD_GRAYSCALE)
    if celling_mask is not None:
        celling_mask = extract_upper_percent(celling_mask,percent=0.03)
        if np.any(celling_mask&bottom_mask):
            supported_item = 'ceiling_0'
    else:
        logger.warning(f"Warning: Could not read ceiling mask from {celling_mask_path}.")

    return supported_item

def extract_upper_percent(mask,percent=0.1):
    # 找到所有有效像素的坐标
    valid_coords = np.argwhere(mask != 0)

    # 如果没有有效像素，返回全零掩码
    if valid_coords.size == 0:
        return np.zeros_like(mask)

    # 按行（y坐标）从大到小排序，即从底部向上
    sorted_coords = valid_coords[np.argsort(-valid_coords[:, 0])]

    # 计算需要提取的有效像素数量（向上取整）
    num_valid_pixels = len(sorted_coords)
    num_pixels_to_extract = int(np.ceil(num_valid_pixels * percent))
    
    # 获取顶部10%有效像素的坐标
    upper_coords = sorted_coords[-num_pixels_to_extract:]
    
    upper_mask = np.zeros_like(mask)
    
    for coord in upper_coords:
        upper_mask[coord[0], coord[1]] = 255
    
    return upper_mask
    
def extract_bottom_percent(mask,percent=0.1):
    # 找到所有有效像素的坐标
    valid_coords = np.argwhere(mask != 0)

    # 如果没有有效像素，返回全零掩码
    if valid_coords.size == 0:
        return np.zeros_like(mask)

    # 按行（y坐标）从大到小排序，即从底部向上
    sorted_coords = valid_coords[np.argsort(-valid_coords[:, 0])]

    # 计算需要提取的有效像素数量（向上取整）
    num_valid_pixels = len(sorted_coords)
    num_pixels_to_extract = int(np.ceil(num_valid_pixels * percent))

    # 获取底部10%有效像素的坐标
    bottom_coords = sorted_coords[:num_pixels_to_extract]

    # 创建一个全零掩码
    bottom_mask = np.zeros_like(mask)

    # 将底部10%有效像素的位置设置为1
    for coord in bottom_coords:
        bottom_mask[coord[0], coord[1]] = 255

    return bottom_mask

def is_mask1_inside_mask2(mask1, mask2):
    """
    判断 mask1 是否被 mask2 完全包围。

    参数：
    - mask1: 二值掩码，2D NumPy 数组，值为 0 或 1。
    - mask2: 二值掩码，2D NumPy 数组，值为 0 或 1。

    返回：
    - True 如果 mask1 被 mask2 完全包围，否则 False。
    """
    # 确保掩码为二值类型
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)

    # 查找 mask2 的轮廓
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有找到轮廓，mask2 为空
    if len(contours) == 0:
        return False

    # 创建一个空白图像用于绘制轮廓
    mask2_contour = np.zeros_like(mask2)

    # 绘制 mask2 的轮廓
    cv2.drawContours(mask2_contour, contours, -1, (1), thickness=cv2.FILLED)

    # 检查 mask1 的所有有效像素是否都在 mask2 的轮廓内
    inside = np.all(mask2_contour[mask1 != 0] == 1)

    return inside

def find_first_contact_mask(base_mask, other_masks):
    """
    计算每个其他掩码被base_mask向下投影首先接触的比例，并确定最先接触的掩码ID。

    参数：
    - base_mask: 二进制掩码，2D NumPy数组，值为0或1。
    - other_masks: 其他掩码的列表，每个都是与base_mask形状相同的2D NumPy数组。

    返回：
    - contact_ratios: 每个掩码被首先接触的比例列表。
    - first_contact_mask_id: 最先接触的掩码ID（从0开始）。
    """
    if len(other_masks)==0:
        return None,None
    
    contact_counts = []
    
    #合并所有的mask
    all_mask = np.zeros(base_mask.shape,dtype=np.int8)
    for id,mask in enumerate(other_masks):
        all_mask[mask!=0] = id+1
        contact_counts.append(0)

    #统计所有非0的列
    vis_mask = base_mask.copy()
    first_non_zero_indices = (base_mask != 0).argmax(axis=0)
    for col,height in enumerate(first_non_zero_indices):
        
        if base_mask[height,col] == 0:
            continue

        sub_array = all_mask[height:,col]
        if np.any(sub_array)>0:
            # 查找第一个大于 0 的元素的索引
            first_non_zero_index = np.argmax(sub_array > 0)
            mask_id = int(sub_array[first_non_zero_index])-1
            contact_counts[mask_id] += 1
            vis_mask[first_non_zero_index,col] = 255

    total_contacts = np.sum(contact_counts)
    contact_counts = np.array(contact_counts)

    contact_ratios = contact_counts / total_contacts if total_contacts > 0 else contact_counts
    first_contact_mask_id = np.argmax(contact_counts) if total_contacts > 0 else None
    return contact_ratios, first_contact_mask_id

def against_wall_generate(key, parent_path):
    
    #获取所有的mask
    with open(os.path.join(parent_path, 'masks.pkl'), "rb") as f:
        masks = pickle.load(f)
    with open(os.path.join(parent_path, 'result.json'), 'r') as f:
        mask_json = json.load(f)
        
    #获取key的mask
    key_mask = masks[mask_json["categorys"].index(key)]
    key_mask = np.array(key_mask)
    key_mask = cv2.dilate(key_mask, np.ones((40,40),np.uint8), iterations=1)            

    wall_mask = {}
    #获取wall的mask
    for id,mask in enumerate(masks):
        key = mask_json["categorys"][id]
        
        if re.match(r'(wall)_\d+', key):
            #扩充mask
            mask = np.array(mask)
            # 找到轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 初始化一个空的掩码来填充凸包
            hull_mask = np.zeros_like(mask)

            # 计算每个轮廓的凸包并填充
            for contour in contours:
                hull = cv2.convexHull(contour)
                cv2.drawContours(hull_mask, [hull], 0, 255, thickness=cv2.FILLED)
                
            wall_mask[key] = hull_mask
            
    wall_keys = [] 
    max_inter = 0
    most_like_wall = ''
    for wall_key, wall_mask in wall_mask.items():
        inter = np.sum(wall_mask & key_mask)
        if inter > 0:
            wall_keys.append(wall_key)
        if inter > max_inter:
            max_inter = inter
            most_like_wall = wall_key
            
    return wall_keys,most_like_wall

def against_wall_generate_top_down(key, parent_path):
    
    input_dir = os.path.join(parent_path, 'pcd_mask')

    #获取key的mask
    key_mask_path = os.path.join(input_dir, f'{key}_xoy.png')
    key_mask = cv2.imread(key_mask_path, cv2.IMREAD_GRAYSCALE)
    if key_mask is None:
        logger.warning(f"Warning: Could not read mask for {key} from {key_mask_path}. Skipping.")
        return [], ''
        
    key_mask = cv2.dilate(key_mask, np.ones((40,40),np.uint8), iterations=1)

    wall_mask = {}
    #获取wall的mask
    files = os.listdir(input_dir)
    for file in files:
        name_without_extension = file.split('.')[0].replace('_xoy','')
        if re.match(r'(wall)_\d+', name_without_extension):
            mask_path = os.path.join(input_dir, f'{name_without_extension}_xoy.png')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Warning: Could not read mask for {name_without_extension} from {mask_path}. Skipping.")
                continue
            mask = cv2.dilate(mask, np.ones((20,20),np.uint8), iterations=1)
            wall_mask[name_without_extension] = mask
    wall_keys = [] 
    max_inter = 0
    most_like_wall = ''
    for wall_key, wall_mask in wall_mask.items():
        inter = np.sum(wall_mask & key_mask)
        if inter > 0:
            wall_keys.append(wall_key)
        if inter > max_inter:
            max_inter = inter
            most_like_wall = wall_key
            
    return wall_keys,most_like_wall


def find_frequent_parents(scene_graph_results, threshold=2):
    parent_count = defaultdict(int)
    child_map = defaultdict(list)

    # 遍历 scene_graph_results，统计每个 parent 的出现次数，并记录其子物体
    for child, attributes in scene_graph_results.items():
        parent = attributes['parent']
        parent_count[parent] += 1
        child_map[parent].append(child)

    # 创建结果字典，只包含出现次数超过阈值且不匹配特定模式的 parent
    result = {parent: children for parent, children in child_map.items() 
              if parent_count[parent] >= threshold and not re.match(r'(ground|wall|ceiling|floor|carpet)_\d+', parent)}

    return result

def process_stage1_inference(params_list):
    task_id, image, class_en_list, _, _ = params_list
    logger.info(f"正在执行任务 #{task_id}")
    
    # 执行场景对象的层次遍历
    scene_graph_dict_result, detect_items_list_result = hierarchical_traversal_of_scene_objs(image, class_en_list, task_id)
    logger.info(f'进程{task_id}的scene_graph_dict_result: {scene_graph_dict_result}')

    # 返回 detect_items_list_result
    return detect_items_list_result
            
def parallel_process_stage1_inference(params_list, save_folder, gpt_params, num_processes=8):
    logger_params = {'name': 'S1_Worker', 'log_file': os.path.join(save_folder, "S1_scene_parsing_op_worker.log"), 'level': "INFO"}
    with multiprocessing.Pool(min(len(params_list), num_processes), initializer=init_worker, initargs=(gpt_params, logger_params)) as pool:
        results = pool.map(process_stage1_inference, params_list)

    # 合并并去重所有进程的 detect_items_GD_prompt
    combined_detect_items_list_result = set()
    for result in results:
        combined_detect_items_list_result.update(result)
    
    # 用gpt4进行语义去重
    logger.info(f'标签语义去重前：{combined_detect_items_list_result}')
    semantic_dedup_prompt = SCENE_DEDUPLICATION_PROMPT.format(object_list = combined_detect_items_list_result)
    semantic_dedup_list_result = gpt.get_response(semantic_dedup_prompt)
    semantic_dedup_list = extract_list_with_re(semantic_dedup_list_result)
    
    # 确保所有label都在class_en_list里面
    class_en_list = params_list[0][2]
    formatted_class_en_list = set(format_object_name(item) for item in class_en_list)  # 格式化并去重
    for label in semantic_dedup_list:
        formatted_label = format_object_name(label)  # 格式化label以匹配格式化的class_en_list
        if formatted_label not in formatted_class_en_list and formatted_label not in ['floor', 'wall', 'ceiling']:
            semantic_dedup_list.remove(formatted_label)
            logger.warning((f"Label '{label}' not found in class_en_list. 此处将其删除"))
    
    detect_items = [item.replace(' ', '_') for item in semantic_dedup_list]
    detect_items_GD_prompt = ".".join(detect_items)  # demo举例: pillow.chandelier.tall_bookshelf.curtain.vase_plant.wall_mounted_picture_frame.carpet.backrest_chair.open_book.tv_remote_control.wall_mounted_lcd_tv.floor_lamp.tv_cabinet.magazine.single_sofa_chair.coffee_table.floor.wall.ceiling

    logger.info(f'标签语义去重后的 detect_items_GD_prompt: {detect_items_GD_prompt}')

    # 进行一次检测分割
    image_np = params_list[0][1]
    prompts = dict(image=image_np, prompt=detect_items_GD_prompt)  # 假设使用第一个图像
    
    # 之前的单次逻辑
    save_folder_name = params_list[0][3]  # 假设使用第一个保存文件夹

    max_retries = 5
    for attempt in range(max_retries):
        try:
            results = dino_api(prompts, token=GROUND_DINO_TOKEN)
            results['masks'] = [np.array(mask)[:, :, 3] for mask in results['masks']]
            break
        except ReadTimeout:
            if attempt < max_retries - 1:
                logger.warning(f"ReadTimeout occurred. Retrying {attempt + 1}/{max_retries}...")
                time.sleep(2)
            else:
                logger.error("Max retries reached. Raising exception.")
                raise
    
    # 保存结果
    os.makedirs(save_folder_name, exist_ok=True)
    save_result_and_partition(params_list[0][1], results, save_folder_name, params_list[0][4])
    return results

def restore_deleted_object_mask(items_to_del, s3_updated_result):
    final_detect_result = {
        "scores": [],
        "categorys": [],
        "boxes": [],
        "masks": []
    }

    for i in range(len(s3_updated_result["categorys"])):
        label = s3_updated_result["categorys"][i]
        mask = s3_updated_result["masks"][i]

        if label in items_to_del:
            closest_index = find_most_overlapping_object(i, s3_updated_result["masks"])
            if closest_index is not None:
                # 将 PIL Image 转换为 numpy 数组
                mask_array = np.array(mask)
                closest_mask_array = np.array(s3_updated_result["masks"][closest_index])
                    
                # 合并 RGBA 掩码
                merged_mask_array = np.maximum(closest_mask_array, mask_array)

                # 将合并后的数组转换回 PIL Image
                s3_updated_result["masks"][closest_index] = Image.fromarray(merged_mask_array)

            # 删除当前物体的掩码
            if isinstance(mask, np.ndarray):
                s3_updated_result["masks"][i] = Image.new("RGBA", mask.shape)
            else:
                s3_updated_result["masks"][i] = Image.new("RGBA", mask.size)
            

    # 处理未被删除的物体
    for i in range(len(s3_updated_result["categorys"])):
        if s3_updated_result["categorys"][i] not in items_to_del:
            final_detect_result["scores"].append(s3_updated_result["scores"][i])
            final_detect_result["categorys"].append(s3_updated_result["categorys"][i])
            final_detect_result["boxes"].append(s3_updated_result["boxes"][i])
            final_detect_result["masks"].append(s3_updated_result["masks"][i])

    return final_detect_result

def find_most_overlapping_object(index, masks):
    target_mask = np.array(masks[index])  
    target_mask = target_mask if target_mask.ndim==2 else target_mask[:, :, 3]  # 使用 alpha 通道

    max_overlap = 0
    closest_index = None

    target_contours = find_contours(target_mask, 0.5)
    if not target_contours:
        return None

    edge_mask = np.zeros_like(target_mask)
    for contour in target_contours[0]:
        x, y = int(contour[0]), int(contour[1])
        edge_mask[max(0, x-1):min(edge_mask.shape[0], x+2), max(0, y-1):min(edge_mask.shape[1], y+2)] = 1

    for i, mask in enumerate(masks):
        if i == index:
            continue
        other_mask = np.array(mask)
        other_mask = other_mask if other_mask.ndim==2 else other_mask[:, :, 3]  # 使用 alpha 通道
        overlap = np.sum(edge_mask & other_mask)
        if overlap > max_overlap:
            max_overlap = overlap
            closest_index = i

    return closest_index

def obb_intersects(obb1, obb2):
    # 实现分离轴定理
    axes1 = np.array(obb1.R).T
    axes2 = np.array(obb2.R).T
    axes_to_test = np.vstack((axes1, axes2))
    
    for axis in axes_to_test:
        proj1 = project_obb(obb1, axis)
        proj2 = project_obb(obb2, axis)
        if proj1[1] < proj2[0] or proj2[1] < proj1[0]:
            return False
    return True

def project_obb(obb, axis):
    vertices = np.asarray(obb.get_box_points())
    projections = np.dot(vertices, axis)
    return np.min(projections), np.max(projections)

def get_intersection_volume(obb1, obb2, num_samples=100000):
    # 使用蒙特卡洛方法估算交集体积
    bbox1 = obb1.get_axis_aligned_bounding_box()
    bbox2 = obb2.get_axis_aligned_bounding_box()
    
    min_bound = np.maximum(bbox1.min_bound, bbox2.min_bound)
    max_bound = np.minimum(bbox1.max_bound, bbox2.max_bound)
    
    samples = np.random.uniform(min_bound, max_bound, (num_samples, 3))
    
    in_obb1 = obb1.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(samples))
    in_obb2 = obb2.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(samples))
    
    intersection_points = np.intersect1d(in_obb1, in_obb2)
    intersection_volume = len(intersection_points) / num_samples * np.prod(max_bound - min_bound)
    
    return intersection_volume

def find_anomaly_detects(logger, updated_result, depth_image_path, category_to_retrieval_category_dict, added_labels=None):
    """
    检测并识别异常或重复的物体检测结果。

    该函数通过分析物体的掩码、边界框和定向包围盒（OBB）来识别可能的异常或重复检测。
    它主要关注两种情况：
    1. 掩码有显著重叠的物体
    2. 同类物体中有一个是后续检测添加的，且它们的OBB有显著重叠

    函数使用多个标准来判断异常，包括掩码重叠率、边界框IOU和OBB体积重叠率。
    为了提高效率，OBB仅在必要时计算，并使用缓存避免重复计算。

    参数:
    updated_result (dict): 包含检测结果的字典，应包含以下键：
        - 'scores': 检测置信度列表
        - 'categorys': 检测类别列表
        - 'boxes': 边界框列表，每个元素为 (x1, y1, x2, y2)
        - 'masks': 掩码列表，每个元素为二维numpy数组或PIL Image对象
    depth_image_path (str): 深度图像的文件路径
    category_to_retrieval_category_dict (dict): 将检测类别映射到检索类别的字典
    added_labels (list, optional): 后续检测中新添加的标签列表。默认为None。

    返回:
    set: 需要删除的物体标签集合
    """

    items_to_del = set()
    n = len(updated_result['categorys'])
    if n < 2:
        return []
    
    # 预处理 ----------------------------------------------------------------
    # 读取深度图数据（仅读取一次）
    depth_mm = np.array(Image.open(depth_image_path)).astype(np.float32)
    
    # 预计算mask面积
    mask_areas = [np.sum(mask)/255 if np.max(mask)==255 else np.sum(mask) for mask in updated_result['masks']]
    
    # 创建OBB缓存字典 {index: obb}
    obb_cache = {}
    
    # 遍历所有物体对 --------------------------------------------------------
    for i in range(n):
        for j in range(i+1, n):
            label_i = updated_result['categorys'][i]
            label_j = updated_result['categorys'][j]
            
            # 获取retrieval category (为两个条件共用，故提前)
            category_i = (re.sub(r'_\\d+$', '', label_i)).replace('-', '_').replace(' ', '_').lower()
            retrieval_category_i = category_to_retrieval_category_dict.get(
                category_i, 
                category_i
            )
            category_j = (re.sub(r'_\\d+$', '', label_j)).replace('-', '_').replace(' ', '_').lower()
            retrieval_category_j = category_to_retrieval_category_dict.get(
                category_j,
                category_j
            )

            # 条件1: mask交集检查
            mask_i = updated_result['masks'][i]
            mask_j = updated_result['masks'][j]
            
            # 计算交集面积
            intersection = np.logical_and(mask_i, mask_j)
            area_intersect = np.sum(intersection)
            min_area = min(mask_areas[i], mask_areas[j])
            
            if min_area > 0 and (area_intersect / min_area) >= 0.5:
                # 新增逻辑：检查是否为不同类别的小物体附着在大的物体上
                # 如果两者不属于同一retrieval_category，并且小物体面积不到大物体面积的15%，则保留
                should_keep = False
                if retrieval_category_i != retrieval_category_j:
                    larger_area = max(mask_areas[i], mask_areas[j])
                    if larger_area > 0 and (min_area / larger_area) < 0.15:
                        should_keep = True
                
                # 结构性元素保护逻辑：wall/floor/ceiling 等结构性元素不应因和非结构性元素重叠而被删除
                # 检查是否为结构性元素（wall, floor, ceiling, ground）
                is_structural_i = bool(re.match(r'^(wall|floor|ceiling|ground)_\d+$', label_i))
                is_structural_j = bool(re.match(r'^(wall|floor|ceiling|ground)_\d+$', label_j))
                
                # 如果其中一个是结构性元素，另一个不是，则保留结构性元素
                if is_structural_i and not is_structural_j:
                    should_keep = True
                    logger.info(f"保护结构性元素 {label_i}，虽然它与非结构性元素 {label_j} 重叠")
                elif is_structural_j and not is_structural_i:
                    should_keep = True
                    logger.info(f"保护结构性元素 {label_j}，虽然它与非结构性元素 {label_i} 重叠")

                if not should_keep:
                    # 原逻辑: 记录需要删除的较小物体
                    if mask_areas[i] < mask_areas[j]:
                        items_to_del.add(label_i)
                    else:
                        items_to_del.add(label_j)
                
                continue  # 只要mask重叠率高，就跳过后续的OBB判断

            # 条件2: 同类物体的OBB检查, 且其中有物体是二次查漏补缺加进来的
            # (retrieval category已在前面获取)
            
            # 同类, 且其中有物体是二次查漏补缺加进来的
            if not (retrieval_category_i == retrieval_category_j and (label_i in added_labels or label_j in added_labels)):
                continue
            
            # BBox重叠检查
            box_i = updated_result['boxes'][i]
            box_j = updated_result['boxes'][j]
            
            # 计算IOU
            inter_x1 = max(box_i[0], box_j[0])
            inter_y1 = max(box_i[1], box_j[1])
            inter_x2 = min(box_i[2], box_j[2])
            inter_y2 = min(box_i[3], box_j[3])
            
            # 无效交集区域检查
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue
                
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area_i = (box_i[2]-box_i[0])*(box_i[3]-box_i[1])
            area_j = (box_j[2]-box_j[0])*(box_j[3]-box_j[1])
            min_box_area = min(area_i, area_j)
            
            if min_box_area == 0 or (inter_area / min_box_area) < 0.8:
                continue
                
            # 按需生成OBB ---------------------------------------------------
            def get_or_create_obb(index):
                """获取或创建OBB的辅助函数，增加了重试机制"""
                if index not in obb_cache:
                    mask = np.array(updated_result['masks'][index])
                    if mask.ndim == 3:
                        mask = mask[:, :, -1]  # 转换为二维掩码
                    
                    obb = None
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # 尝试计算OBB
                            _, obb, _, _ = estimate_obj_depth_obb_faster(depth_mm, mask)
                            # 如果成功生成了有效的OBB，则跳出循环
                            if obb is not None:
                                break
                        except Exception as e:
                            print(f"Warning: OBB estimation for index {index} failed on attempt {attempt + 1}/{max_retries}. Error: {e}")
                            # 如果是最后一次尝试，则打印错误并放弃
                            if attempt + 1 == max_retries:
                                print(f"Error: OBB estimation failed for index {index} after {max_retries} attempts. Skipping this object.")
                    
                    obb_cache[index] = obb
                return obb_cache[index]
            
            # 获取两个物体的OBB
            obb_i = get_or_create_obb(i)
            obb_j = get_or_create_obb(j)
            
            # 有效性检查
            if obb_i is None or obb_j is None:
                continue
                
            # OBB相交性检查
            if not obb_intersects(obb_i, obb_j):
                continue
                
            # 计算交集体积
            inter_vol = get_intersection_volume(obb_i, obb_j)
            vol_i = obb_i.volume()
            vol_j = obb_j.volume()
            min_vol = min(vol_i, vol_j)
            
            if min_vol > 0 and (inter_vol / min_vol) >= 0.7:
                if vol_i < vol_j:
                    items_to_del.add(label_i)
                else:
                    items_to_del.add(label_j)

    return items_to_del

def perform_3d_clustering(save_folder, depth_image, floor_walls_pose, visualize_combined_pcd=False):
    """
    Performs 3D OBB clustering and generates projection masks.
    """
    clustered_result_folder = os.path.join(save_folder, 'clustered_result')
    mask_folder = os.path.join(save_folder, 'masks') 
    
    # Correct the pcd_mask_path to point to the S1 results directory
    pcd_mask_path = os.path.join(save_folder, 'pcd_mask') # Define path for projection masks
    os.makedirs(pcd_mask_path, exist_ok=True)

    cluster_3d_obb(
        depth_image_path=depth_image,
        floor_walls_pose_data=floor_walls_pose,
        mask_folder=mask_folder,
        object_bboxes_wo_text_bbox_json_path=os.path.join(save_folder, 'object_bboxes_wo_text_bbox_json.json'),
        image_path=os.path.join(save_folder, 'wall_floor_paint.png'),
        output_folder=clustered_result_folder,
        font_path=FONT_TTF_PATH,
        pcd_mask_save_path=pcd_mask_path, # Pass the new path
        visualize_combined_pcd=visualize_combined_pcd
    )
    return clustered_result_folder, pcd_mask_path

def visualize_objects_by_support(save_folder, scene_graph_result):
    """
    Generates three separate visualizations of objects based on their support relationship
    (on floor, on wall/ceiling, on other objects), filtering out foundational elements.
    """
    # Load necessary data
    try:
        with open(os.path.join(save_folder, 'result.json'), 'r') as f:
            detect_results = json.load(f)
        ori_image = Image.open(os.path.join(save_folder, 'ori.png'))
    except FileNotFoundError as e:
        logger.error(f"Error loading required files for visualization: {e}")
        return {}

    # Create a mapping from category name to its bounding box
    bbox_dict = {cat: box for cat, box in zip(detect_results['categorys'], detect_results['boxes'])}

    # Categorize objects based on their support
    categorized_objects = defaultdict(list)
    for obj, attributes in scene_graph_result.items():
        # Filter out foundational objects
        if re.match(r'(ground|wall|ceiling|floor|carpet)_\d+', obj):
            continue

        supported_by = attributes.get('supported')
        if supported_by:
            if 'floor' in supported_by:
                categorized_objects['floor'].append(obj)
            elif 'wall' in supported_by or 'ceiling' in supported_by:
                categorized_objects['wall_ceiling'].append(obj)
            else:
                categorized_objects['others'].append(obj)

    output_data = {}
    
    # Define common drawing parameters
    draw_width = 2.5
    font_size = int(min(ori_image.size) / 50)
    font_size = max(15, font_size)
    try:
        font = ImageFont.truetype(FONT_TTF_PATH, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Generate an image for each category
    for category, objects in categorized_objects.items():
        if not objects:
            continue

        img_copy = ori_image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Unique color for each object to avoid confusion
        cate2color = {obj: tuple(np.random.randint(0, 255, size=3).tolist()) for obj in objects}

        # First, draw all bounding boxes
        for obj_name in objects:
            if obj_name in bbox_dict:
                box = bbox_dict[obj_name]
                x0, y0, x1, y1 = map(int, box)
                color = cate2color[obj_name]

                # Draw bounding box with increased width
                draw.rectangle([x0, y0, x1, y1], outline=color, width=int(draw_width * 1.5))

        # Then, draw all text labels with overlap avoidance
        drawn_text_positions = []
        for obj_name in objects:
            if obj_name in bbox_dict:
                box = bbox_dict[obj_name]
                x0, y0, x1, y1 = map(int, box)
                color = cate2color[obj_name]

                # Prepare text
                text = obj_name
                
                if hasattr(font, "getbbox"):
                    text_bbox = font.getbbox(text)
                else:
                    # Fallback for older Pillow versions
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                
                left, top, right, bottom = text_bbox
                text_width = right - left
                text_height = bottom - top
                
                # Initial text position
                text_x, text_y = x0, y0

                # Attempt to adjust text position to avoid overlap
                max_attempts = 10
                attempt = 0
                while attempt < max_attempts:
                    bbox = [text_x, text_y, text_x + text_width, text_y + text_height]
                    if not any(overlap(bbox, pos) for pos in drawn_text_positions):
                        break
                    text_x += 5  # Micro-adjust position
                    text_y += 5
                    attempt += 1
                
                drawn_text_positions.append(bbox)

                # Draw black outline for the text for better visibility
                for offset in [(1, 1), (-1, -1), (1, -1), (-1, 1), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                    draw.text((text_x + offset[0], text_y + offset[1]), text, fill="black", font=font)

                # Draw the actual text
                draw.text((text_x, text_y), text, fill=color, font=font)

        # Save the image and record its path and object list
        output_path = os.path.join(save_folder, f'group_{category}.png')
        img_copy.save(output_path)
        output_data[category] = {
            "image_path": output_path,
            "objects": objects
        }
        logger.info(f"Generated visualization for '{category}' support group at: {output_path}")

    return output_data

def analyze_groups_and_facing_relations(save_folder, scene_graph_result, gpt_params):
    """
    Analyzes object groups and facing relations using GPT-4V on categorized visualizations.
    """
    # Step 1: Generate categorized visualizations
    support_group_data = visualize_objects_by_support(save_folder, scene_graph_result)

    if not support_group_data:
        logger.warning("No support groups were generated for analysis. Skipping semantic relationship analysis.")
        return scene_graph_result

    # Step 2: Prepare parameters for parallel GPT processing
    all_image_list = []
    all_prompt_list = []
    for category, data in support_group_data.items():
        if data["objects"]:
            all_image_list.append(data["image_path"])
            prompt = GENERATE_SEMANTIC_RELATIONSHIPS_PROMPT.format(object_list=data["objects"])
            # print(prompt)
            all_prompt_list.append(prompt)

    if not all_image_list:
        logger.info("No objects found in support groups to analyze.")
        return scene_graph_result

    # Step 3: Run parallel processing
    logger.info(f"Analyzing {len(all_image_list)} support group images for semantic relationships...")
    results = parallel_processing_requests(
        gpt_params, 
        all_image_list, 
        all_prompt_list, 
        return_list=False, 
        return_json=False, 
        return_dict=True, 
        num_processes=min(len(all_image_list), 8)
    )

    # Step 4: Merge results and update the scene graph
    group_id_counter = 0
    
    for res in results:
        if not res:
            continue
            
        # Handle groups
        if "groups" in res:
            groups_data = res["groups"]
            
            # Handle both formats: dict with group keys or list of lists
            if isinstance(groups_data, dict):
                # Format: {"group_0": ["obj1", "obj2"], "group_1": ["obj3", "obj4"]}
                for group_key, group_objects in groups_data.items():
                    if len(group_objects) > 1:
                        for obj in group_objects:
                            if obj in scene_graph_result:
                                scene_graph_result[obj]["group"] = str(group_id_counter)
                        logger.info(f"Group {group_id_counter}: {group_objects}")
                        group_id_counter += 1
                        
            elif isinstance(groups_data, list):
                # Format: [["obj1", "obj2"], ["obj3", "obj4"]]
                for group_objects in groups_data:
                    if len(group_objects) > 1:
                        for obj in group_objects:
                            if obj in scene_graph_result:
                                scene_graph_result[obj]["group"] = str(group_id_counter)
                        logger.info(f"Group {group_id_counter}: {group_objects}")
                        group_id_counter += 1

        # Handle facing relationships
        if "facing_relationships" in res:
            facing_data = res["facing_relationships"]
            
            for key, facing_target in facing_data.items():
                # Handle group-based facing (e.g., "group_0": "dining_table_0")
                if key.startswith("group_") and "groups" in res:
                    groups_data = res["groups"]
                    if isinstance(groups_data, dict) and key in groups_data:
                        # Apply facing relationship to all objects in the group
                        for obj in groups_data[key]:
                            if obj in scene_graph_result:
                                scene_graph_result[obj]["directlyFacing"] = facing_target
                        logger.info(f"Applied facing relationship: {groups_data[key]} -> {facing_target}")
                
                # Handle direct object facing (e.g., "chair_1": "table_1")
                elif key in scene_graph_result:
                    scene_graph_result[key]["directlyFacing"] = facing_target
                    logger.info(f"Applied facing relationship: {key} -> {facing_target}")
                else:
                    logger.warning(f"Could not map facing relationship key '{key}' to scene objects.")

    logger.info("Successfully merged semantic relationship results into the scene graph.")
    return scene_graph_result

def run_scene_parsing_pipeline(logger, image_np, depth_image, df, save_folder, params, gpt_params):
    """
    优化版场景解析流水线：数据在内存中流动，减少不必要的文件I/O
    Optimized scene parsing pipeline: data flows in memory, minimizing unnecessary file I/O
    """
    start_time = time.time()
    os.makedirs(save_folder, exist_ok=True)

    # --- Setup ---
    class_en_list = list(set(df['class_en'].tolist()))
    equivalence_retrieval = df.groupby('class_en')['retrieval_class_en'].unique().reset_index()
    equivalence_retrieval['class_en'] = equivalence_retrieval['class_en'].apply(lambda x: format_object_name(x))
    category_to_retrieval_category_dict = equivalence_retrieval.set_index('class_en')['retrieval_class_en'].to_dict()
    box_threshold = params.box_threshold

    # =================================================================================
    #  LOGIC STEP 1: Object Detection, Optional Inspection, and Anomaly Filtering
    # =================================================================================
    step1_start_time = time.time()
    logger.info("Pipeline Step 1: Starting Object Detection...")
    
    # 检查检查点
    preprocessed_results_path = os.path.join(save_folder, 'preprocessed_results.pkl')
    if os.path.exists(preprocessed_results_path):
        logger.info("Found existing preprocessed results, loading from checkpoint...")
        with open(preprocessed_results_path, 'rb') as f:
            preprocessed_results = pickle.load(f)
    else:
        # Part 1.1: Initial Object Detection
        part1_1_start_time = time.time()
        initial_detection_folder = os.path.join(save_folder, '01_initial_detection')
        os.makedirs(initial_detection_folder, exist_ok=True)
        
        params_list = [(i, image_np, class_en_list, initial_detection_folder, box_threshold) for i in range(params.hierarchical_traversal_times)]
        detection_results = parallel_process_stage1_inference(params_list, save_folder, gpt_params)
        logger.info(f"Part 1.1: Initial Object Detection finished. Time: {time.time() - part1_1_start_time:.2f}s.")

        # Part 1.2: Optional Secondary Inspection
        added_labels = []
        if params.enable_secondary_inspection:
            part1_2_start_time = time.time()
            logger.info("Starting secondary inspection to find missing objects...")
            area_bbox_json_output, cropped_img_list, cropped_and_marked_img_list = generate_mask_and_process_missing_part(initial_detection_folder)
            _, detection_results, added_labels = secondary_inspection_of_scene_objs_detection(
                detection_results, area_bbox_json_output, cropped_img_list, cropped_and_marked_img_list, class_en_list, save_folder, gpt_params
            )
            logger.info(f"Part 1.2: Secondary Inspection finished. Time: {time.time() - part1_2_start_time:.2f}s.")
        else:
            logger.info("Secondary inspection is disabled.")

        # Part 1.3: Anomaly Filtering
        part1_3_start_time = time.time()
        logger.info("Starting anomaly detection to remove duplicate objects...")
        items_to_del = find_anomaly_detects(logger, detection_results, depth_image, category_to_retrieval_category_dict, added_labels)
        if items_to_del:
            logger.info(f'Found and will remove anomalous/duplicate objects: {items_to_del}')
            preprocessed_results = restore_deleted_object_mask(items_to_del, detection_results)
        else:
            logger.info('No anomalous objects found.')
            preprocessed_results = detection_results
        logger.info(f"Part 1.3: Anomaly Filtering finished. Time: {time.time() - part1_3_start_time:.2f}s.")
        
        # 保存检查点
        with open(preprocessed_results_path, 'wb') as f:
            pickle.dump(preprocessed_results, f)
        save_result_and_partition(image_np, preprocessed_results, save_folder, box_threshold, save_masks_pkl=True)
    
    logger.info(f"Pipeline Step 1 finished. Time: {time.time() - step1_start_time:.2f}s.")

    # =================================================================================
    #  LOGIC STEP 2: Floor and Wall Segmentation
    # =================================================================================
    step2_start_time = time.time()
    logger.info("Pipeline Step 2: Starting Floor & Wall Segmentation...")

    floor_walls_pose_path = os.path.join(save_folder, 'floor_walls_pose.json')
    final_detect_items_path = os.path.join(save_folder, 'final_detect_items.pkl')

    # 检查检查点
    if os.path.exists(floor_walls_pose_path) and os.path.exists(final_detect_items_path):
        logger.info("Found existing floor and wall segmentation, loading from checkpoint...")
        with open(floor_walls_pose_path, 'r') as f:
            floor_walls_pose = json.load(f)
        with open(final_detect_items_path, 'rb') as f:
            final_detection_results = pickle.load(f)
    else:
        # 直接使用内存中的 preprocessed_results，无需重新读取
        part2_1_start_time = time.time()
        final_detection_results, floor_walls_pose, unfilled_wall_and_ground_masks = estimate_floor_and_walls(
            image_np, preprocessed_results, depth_image, save_folder, max_wall_num=3
        )
        logger.info(f"Part 2.1: estimate_floor_and_walls finished. Time: {time.time() - part2_1_start_time:.2f}s.")
        
        # 保存检查点
        with open(final_detect_items_path, 'wb') as f:
            pickle.dump(final_detection_results, f)
        with open(floor_walls_pose_path, 'w') as f:
            json.dump(floor_walls_pose, f, indent=2)
        
        # 更新 masks.pkl、result.json 和 wall_color_name.json，确保包含更新后的 wall 和 floor mask
        masks_pkl_path = os.path.join(save_folder, 'masks.pkl')
        result_json_path = os.path.join(save_folder, 'result.json')
        wall_color_name_json_path = os.path.join(save_folder, 'wall_color_name.json')
        
        logger.info("Updating masks.pkl, result.json and wall_color_name.json with refined wall and floor masks...")
        with open(masks_pkl_path, 'wb') as f:
            pickle.dump(final_detection_results['masks'], f)
        
        # 更新 result.json
        result_data = {
            'boxes': final_detection_results['boxes'],
            'categorys': final_detection_results['categorys'],
            'scores': final_detection_results['scores']
        }
        with open(result_json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        # 更新 wall_color_name.json，确保所有 wall/floor/ceiling 都有颜色映射
        # 读取旧的 wall_color_name
        old_wall_color_name = {}
        if os.path.exists(wall_color_name_json_path):
            with open(wall_color_name_json_path, 'r') as f:
                old_wall_color_name = json.load(f)
        
        # 定义可用的颜色池
        wall_color_dict = {
            (0, 255, 255): 'cyan',
            (0, 0, 255): 'blue',
            (255, 165, 0): 'orange',
            (128, 0, 128): 'purple',
            (255, 20, 147): 'pink',
        }
        wall_color_list = list(wall_color_dict.items())
        
        # 创建新的 wall_color_name，确保所有实际存在的 wall/floor/ceiling 都有颜色
        new_wall_color_name = {}
        used_colors = set()
        
        for category in final_detection_results['categorys']:
            if re.match(r'(ground|wall|ceiling|floor)_\d+', category):
                # 如果旧的映射中有这个类别且颜色未被使用，保留它
                if category in old_wall_color_name:
                    color_name, color_rgb = old_wall_color_name[category]
                    if color_name not in used_colors:
                        new_wall_color_name[category] = [color_name, color_rgb]
                        used_colors.add(color_name)
                        continue
                
                # 否则，为新类别分配一个未使用的颜色
                for color_rgb, color_name in wall_color_list:
                    if color_name not in used_colors:
                        new_wall_color_name[category] = [color_name, list(color_rgb)]
                        used_colors.add(color_name)
                        break
        
        # 保存更新后的 wall_color_name.json
        with open(wall_color_name_json_path, 'w') as f:
            json.dump(new_wall_color_name, f, indent=2)
        
        logger.info(f"Updated wall_color_name.json with {len(new_wall_color_name)} entries: {list(new_wall_color_name.keys())}")
        logger.info("masks.pkl, result.json and wall_color_name.json updated successfully.")
        
        # Generate individual mask files needed for clustering
        part2_2_start_time = time.time()
        generate_mask(save_folder)
        logger.info(f"Part 2.2: generate_mask finished. Time: {time.time() - part2_2_start_time:.2f}s.")
        
        # 使用更新后的 final_detection_results 重新生成 wall_floor_paint_with_annotation.png
        generate_wall_floor_annotation(save_folder, final_detection_results)
        logger.info("wall_floor_paint_with_annotation.png regenerated with refined masks.")
    
    logger.info(f"Pipeline Step 2 finished. Time: {time.time() - step2_start_time:.2f}s.")

    # =================================================================================
    #  LOGIC STEP 3: Scene Graph Generation
    # =================================================================================
    step3_start_time = time.time()
    logger.info("Pipeline Step 3: Starting Scene Graph Generation...")

    scene_graph_final_path = os.path.join(save_folder, 'scene_graph_result.json')
    
    # 检查最终结果
    if os.path.exists(scene_graph_final_path):
        logger.info(f"Found existing scene graph, skipping Step 3. All results are saved in: {save_folder}")
        with open(scene_graph_final_path, 'r') as f:
            scene_graph_result = json.load(f)
    else:
        # 直接使用内存中的 final_detection_results 和 floor_walls_pose，无需重新读取
        
        # Part 3.1: Perform 3D clustering and generate projection masks
        part3_1_start_time = time.time()
        
        debug_mode = getattr(params, 'debug', False)
        
        clustered_result_folder, pcd_mask_path = perform_3d_clustering(
            save_folder, 
            depth_image, 
            floor_walls_pose, 
            visualize_combined_pcd=debug_mode
        )
        logger.info(f"Part 3.1: 3D clustering finished. Time: {time.time() - part3_1_start_time:.2f}s.")

        # Part 3.2: Generate cropped images and metadata for each object
        part3_2_start_time = time.time()
        obj_partition_folder = os.path.join(save_folder, 'obj_partition_result')
        obj_bbox_crop_and_save(
            object_bboxes_json_path=os.path.join(save_folder, 'object_bboxes_json.json'),
            object_bboxes_wo_text_bbox_json_path=os.path.join(save_folder, 'object_bboxes_wo_text_bbox_json.json'),
            image_path=os.path.join(save_folder, 'wall_floor_paint.png'),
            output_folder=obj_partition_folder,
            font_path=FONT_TTF_PATH,
            pcd_mask_path=pcd_mask_path,
            debug=debug_mode
        )
        logger.info(f"Part 3.2: obj_bbox_crop_and_save finished. Time: {time.time() - part3_2_start_time:.2f}s.")
        
        # Part 3.3: Generate initial scene graph (在内存中迭代，不保存中间版本)
        part3_3_start_time = time.time()
        filtered_bbox_image_path = os.path.join(save_folder, 'filtered_bbox_result.png')
        all_items_list = final_detection_results['categorys']
        
        scene_graph_result = generate_scene_graph_geometry(
            clustered_result_folder, filtered_bbox_image_path, all_items_list, gpt_params
        )
        logger.info(f"Part 3.3: Scene graph generation finished. Time: {time.time() - part3_3_start_time:.2f}s.")

        # Part 3.4: Floor parent verification (在内存中更新)
        part3_4_start_time = time.time()
        scene_graph_result, deleted_objects = verify_floor_parent_with_vlm(
            save_folder, scene_graph_result, gpt_params, grid_size=3
        )
        logger.info(f"Part 3.4: Floor parent verification finished. Deleted {len(deleted_objects)} abnormal objects. Time: {time.time() - part3_4_start_time:.2f}s.")

        # Part 3.5: Semantic Group and Facing Analysis (在内存中更新)
        part3_5_start_time = time.time()
        logger.info("Starting Part 3.5: Semantic Group and Facing Analysis...")
        scene_graph_result = analyze_groups_and_facing_relations(save_folder, scene_graph_result, gpt_params)
        logger.info(f"Part 3.5: Semantic analysis finished. Time: {time.time() - part3_5_start_time:.2f}s.")

        # 保存最终的 Scene Graph 结果
        with open(scene_graph_final_path, 'w') as f:
            json.dump(scene_graph_result, f, indent=2)
        
        # # 可选：保存调试可视化（如果需要）
        # if params.get('save_debug_visualizations', True):
        vis_scene_graph(save_folder, scene_graph_result, id='final')

        logger.info(f"Pipeline Step 3 finished. Time: {time.time() - step3_start_time:.2f}s.")

    logger.info(f"Full pipeline completed. Total time: {time.time() - start_time:.2f}s.")
    logger.info(f"Final results are saved in: {save_folder}")
    
    return scene_graph_result

