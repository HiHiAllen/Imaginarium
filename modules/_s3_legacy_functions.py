"""
Legacy functions from S3_pose_inference_op.py
"""
import os
import torch
import cv2
import json
import re
import time
import pickle
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import open3d as o3d
from torchvision import transforms
import functools

# ===== 让所有print自动flush，避免输出被缓冲 =====
print = functools.partial(print, flush=True)
import sys
from prompts.used_prompts import *

from utils.view_matching import find_view_best_match_obb, convert_camera_pose_of_render_view_to_obj_pose_to_blender_coordinates, rotation_matrix_to_angle_diff, convert_obb_pose_to_blender_coordinates
from utils.obb import estimate_obj_depth_obb_faster
from models.ae_net import tensor_collection as tc
from models.ae_net.crop import CropResizePad
from models.ae_net.ae_net import AENet
from models.ae_net.matching import LocalSimilarity
from models.ae_net.vis_numpy import plot_keypoints
from models.ae_net.vis_torch import convert_tensor_to_image, save_tensor_to_image

class Transforms:
    def __init__(self):
        self.normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.crop_transform = CropResizePad(target_size=224)

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
def load_json(json_path):
    if json_path.endswith('json'):
        with open(json_path, 'r') as f:
            all_data = json.load(f)
    elif json_path.endswith('jsonl'):
        with open(json_path, 'r') as f:
            all_data=[json.loads(line) for line in f.readlines()]
    else:
        raise
    return all_data

def save_mask(mask, filename):
    if isinstance(mask, dict):
        # 如果mask是字典,需要决定使用哪个键的值
        # 例如,假设我们要使用名为 'tensor' 的键
        mask = mask['tensor']
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().squeeze().numpy()
    elif isinstance(mask, np.ndarray):
        mask = mask.squeeze()
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)}")
    
    # 将 mask 转换为 PIL Image
    mask_image = Image.fromarray((mask * 255).astype('uint8'))
    
    # 保存图像
    mask_image.save(filename)
    print(f"Image saved as {filename}")

def crop_and_resize_mask(mask, target_bbox_size):
    B, C, H, W = mask.shape
    assert C == 1, "Mask should have only one channel"
    assert target_bbox_size.shape == (B, 4), "target_bbox_size should be of shape [B, 4]"
    
    result = []
    for b in range(B):
        # 获取单个 mask 和对应的目标大小
        single_mask = mask[b, 0]
        tar_x1, tar_y1, tar_x2, tar_y2 = target_bbox_size[b]
        tar_W = tar_x2 - tar_x1
        tar_H = tar_y2 - tar_y1
        
        # 找到非零元素的索引
        y_indices, x_indices = torch.where(single_mask > 0)
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            # 如果 mask 是空的,返回全零 mask
            result.append(torch.zeros(1, tar_H, tar_W))
            continue
        
        # 计算边界框
        top, left = y_indices.min(), x_indices.min()
        bottom, right = y_indices.max(), x_indices.max()
        
        # 裁剪 mask
        cropped = single_mask[top:bottom+1, left:right+1]
        
        # 调整大小
        resized = torch.nn.functional.interpolate(cropped.unsqueeze(0).unsqueeze(0).float(), 
                                size=(tar_H, tar_W), 
                                mode='nearest')
        
        # 二值化
        resized = (resized > 0.5).float()
        
        result.append(resized.squeeze(0))
    
    return torch.stack(result)

def resize_and_intersect(cropped_mask, normalized_rgb):
    # 获取 normalized_rgb 的大小
    B, C, H, W = normalized_rgb.shape
    
    # 将 cropped_mask 调整到与 normalized_rgb 相同的大小,保持通道数为 1
    resized_mask = torch.nn.functional.interpolate(cropped_mask.float(), size=(H, W), mode='nearest')
    
    # 确保 mask 是二值的
    resized_mask = (resized_mask > 0.5).float()
    
    # 取交集（将 mask 应用到 normalized_rgb 上）
    intersected = normalized_rgb * resized_mask
    
    return intersected, resized_mask

def save_tensor_as_image(tensor, filename):
    import torchvision.transforms as T
    # 确保张量在 CPU 上
    tensor = tensor.cpu()
    
    # 如果张量是 4D 的 (批次, 通道, 高度, 宽度),我们取第一个图像
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # 将张量从 [0, 1] 范围转换到 [0, 255]
    tensor = tensor * 255
    
    # 将张量转换为 PIL 图像
    image = T.ToPILImage()(tensor.byte())
    
    # 保存图像
    image.save(filename)
    print(f"Image saved as {filename}")
    
def load_local_model(model_name, ori_dino_weights_path):
    # local_dinov2_path = os.path.join(os.environ['PYTHONPATH'], 'dinov2')
    local_dinov2_path = 'src/dinov2'
    if os.path.exists(local_dinov2_path):
        model = torch.hub.load(local_dinov2_path, model_name, source='local', pretrained=False)
    else:
        print(f"Local path {local_dinov2_path} not found, loading from facebookresearch/dinov2", flush=True)
        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=False)
    
    model.load_state_dict(torch.load(ori_dino_weights_path))
    return model

def process_real(target_img_path, target_mask_path, bbox, transforms):
    rgb = T.ToTensor()(Image.open(target_img_path).convert('RGB'))
    
    # 将 bbox 转换为 PyTorch 张量
    bbox_tensor = torch.tensor(bbox, dtype=torch.long).reshape(1, 4)

    if 'bed_' in target_mask_path.split('/')[-1]:
        # 直接使用 bbox 截取
        cropped_data = transforms.crop_transform(bbox_tensor, images=rgb.unsqueeze(0))
        cropped_image = cropped_data['images'][0].to('cuda')
        
        # 创建全亮的 mask
        cropped_mask = torch.ones_like(cropped_image[0]).to('cuda')
    else:
        mask = T.ToTensor()(Image.open(target_mask_path).convert('L'))

        # 确保 mask 的维度与 rgb 匹配
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # 添加通道维度

        # 扩展 mask 的维度以匹配 rgb
        mask_expanded = mask.expand_as(rgb)

        m_rgb = rgb * mask_expanded
        m_rgba = torch.cat([m_rgb, mask], dim=0)
        cropped_data = transforms.crop_transform(bbox_tensor, images=m_rgba.unsqueeze(0))

        cropped_image = cropped_data['images'][0, :3].to('cuda')
        cropped_mask = cropped_data['images'][0, -1].to('cuda')

    normalized_image = transforms.normalize(cropped_image)

    return normalized_image, cropped_mask

def load_ae_net(ae_net_weights_path, ori_dino_weights_path):
    model_name = 'dinov2_vitl14'
    # weights_path = os.path.join(os.environ['PYTHONPATH'], f'{model_name}.pth') # 确保这个路径是正确的
    dinov2_model = load_local_model(model_name, ori_dino_weights_path)

    ae_net = AENet(
        model_name=model_name,
        dinov2_model=dinov2_model,
        max_batch_size=64,
        descriptor_size=1024
    )
    # 加载预训练权重
    state_dict = torch.load(ae_net_weights_path, map_location='cpu')
    ae_net.load_state_dict(state_dict)
    ae_net.to("cuda")
    
    return ae_net


def save_features(features, file_path):
    torch.save(features, file_path)

def load_features(file_path):
    return torch.load(file_path)

def procrustes_no_rotation(X, Y):
    # 平移：将两组点的质心对齐
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    
    # 缩放：按比例缩放两组点
    norm_X = np.linalg.norm(X_centered)
    norm_Y = np.linalg.norm(Y_centered)
    X_scaled = X_centered / norm_X
    Y_scaled = Y_centered / norm_Y
    
    # 计算Procrustes距离
    distance = np.linalg.norm(X_scaled - Y_scaled)
    
    return X_scaled, Y_scaled, distance

def cal_procrustes(src_pts_batch, tar_pts_batch):
    batch = src_pts_batch.shape[0]
    distance_list_batch = []
    sorted_indices_batch = []

    for idx in range(batch):
        distance_list = []
        for src_pts, tar_pts in zip(src_pts_batch[idx], tar_pts_batch[idx]):
            mask = tar_pts[:, 0] != -1
            X, Y = src_pts[mask], tar_pts[mask]

            # 检查有效点对的数量
            if X.shape[0] < 2 or Y.shape[0] < 2:
                # 如果点对少于2个,无法计算有效距离
                distance = np.inf if X.shape[0] == 0 else 0
            else:
                X_aligned, Y_aligned, distance = procrustes_no_rotation(X, Y)

            distance_list.append(distance)

        distance_array = np.array(distance_list)
        sorted_indices = np.argsort(distance_array)

        distance_list_batch.append(distance_array)
        sorted_indices_batch.append(sorted_indices)

    return distance_list_batch, sorted_indices_batch

def mp_preprocess_template_rgb_and_mask(template_dir, transforms, device, return_images=False):
    cache_file_masks = os.path.join(template_dir, 'processed_masks.pt')

    # Check if cached files exist
    if not return_images and os.path.exists(cache_file_masks):
        print('已提前保存预处理结果, 直接加载...', flush=True)
        cropped_masks = torch.load(cache_file_masks, map_location=device)
        return None, cropped_masks
    else:
        print('未提前保存预处理结果, 或需要重新读取图片, 即时处理...', flush=True)
        def load_image_and_bbox(i):
            try:
                img_path = os.path.join(template_dir, f'{i:06d}.png')
                image = Image.open(img_path).convert('RGBA')
                image_tensor = T.ToTensor()(image)
                alpha = image_tensor[3]
                non_zero = torch.nonzero(alpha, as_tuple=True)
                if non_zero[0].size(0) > 0:
                    y_min, y_max = torch.min(non_zero[0]).item(), torch.max(non_zero[0]).item()
                    x_min, x_max = torch.min(non_zero[1]).item(), torch.max(non_zero[1]).item()
                    # 确保边界框至少有1像素的高度和宽度
                    if y_max == y_min:
                        y_max = min(y_min + 1, alpha.shape[0] - 1)
                    if x_max == x_min:
                        x_max = min(x_min + 1, alpha.shape[1] - 1)
                else:
                    y_min, x_min = 0, 0
                    y_max, x_max = alpha.shape[-2] - 1, alpha.shape[-1] - 1
                
                bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]
                return image_tensor, bbox
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                return None, None

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(load_image_and_bbox, range(162)))

        # Filter out None results
        valid_results = [r for r in results if r[0] is not None and r[1] is not None]
        
        if not valid_results:
            raise ValueError("No valid images found in the template directory.")

        images, bboxes = zip(*valid_results)
        images_tensor = torch.stack(images)
        bboxes_tensor = torch.tensor(bboxes)

        def validate_bboxes(bboxes_tensor):
            valid_bboxes = []
            for bbox in bboxes_tensor:
                x_min, y_min, x_max, y_max = bbox
                if isinstance(x_min, torch.Tensor):
                    x_min = x_min.item()
                if isinstance(y_min, torch.Tensor):
                    y_min = y_min.item()
                if isinstance(x_max, torch.Tensor):
                    x_max = x_max.item()
                if isinstance(y_max, torch.Tensor):
                    y_max = y_max.item()
                
                # 将值转换为整数
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                
                if x_max > x_min and y_max > y_min:
                    valid_bboxes.append([x_min, y_min, x_max, y_max])
                else:
                    # 如果边界框无效，使用整个图像
                    valid_bboxes.append([0, 0, int(images_tensor.shape[-1] - 1), int(images_tensor.shape[-2] - 1)])
            return torch.tensor(valid_bboxes, dtype=torch.long)  # 使用 long 类型

        valid_bboxes_tensor = validate_bboxes(bboxes_tensor)
        
        try:
            # Apply batch cropping
            cropped_data = transforms.crop_transform(valid_bboxes_tensor, images_tensor)

            cropped_images = cropped_data['images'][:, :3]
            cropped_masks = cropped_data['images'][:, 3]

            cropped_images = cropped_images.to(device)
            cropped_masks = cropped_masks.to(device)
            

            # Save processed data to disk (Only masks)
            if not os.path.exists(cache_file_masks):
                torch.save(cropped_masks, cache_file_masks)
                print(f'processed_masks.pt 保存在{template_dir}', flush=True)
            
        except Exception as e:
            print(f"Error during image processing: {str(e)}")
            raise
    
    if return_images:
        normalized_image = transforms.normalize(cropped_images)
        return normalized_image, cropped_masks
    else:
        return None, cropped_masks

def parallel_process_template_rgb_and_mask(template_dir, total_items_info_dict, transforms, device, return_images=False):
    def process_template(top1_template_name):
        _template_dir = os.path.join(template_dir, top1_template_name)
        return mp_preprocess_template_rgb_and_mask(_template_dir, transforms, device, return_images=return_images)

    with ThreadPoolExecutor(max_workers = 8) as executor:
        results = list(executor.map(process_template, total_items_info_dict.values()))

    return results

def plot_keypoints_tar_2_nSrc(
    tar_img,
    tar_pts,
    src_imgs,
    src_pts,
    unnormalize=True,
    patch_size=14,
    num_samples=16,
    concate_input_in_pred=True,
    write_num_matches=True,
):
    '''
    tar_img: torch.Size([1, 3, 224, 224]) 
    tar_pts: torch.Size([1, 5, 256, 2])
    src_imgs: torch.Size([1, 5, 3, 224, 224])
    src_pts: torch.Size([1, 5, 256, 2])
    '''
    batch_size = src_imgs.shape[0]
    template_num = src_imgs.shape[1]

    # convert tensor to numpy
    tar_img = convert_tensor_to_image(tar_img, unnormalize=unnormalize) # # (1, 224, 224, 3)
    src_imgs_np = [convert_tensor_to_image(src_imgs[:, i]) for i in range(src_imgs.shape[1])] # len 10   每个元素是
    src_imgs = np.stack(src_imgs_np, axis=1)    # (43, 10, 224, 224, 3)
    
    src_pts = src_pts.cpu().numpy()
    tar_pts = tar_pts.cpu().numpy()

    matching_imgs_batch_list = []
    for idx in range(batch_size):
        matching_imgs = []
        for template_idx in range(template_num):
            mask = src_pts[idx, template_idx, :, 0] != -1
            border_color = None  # [255, 0, 0]
            concate_input = concate_input_in_pred
            keypoint_img = plot_keypoints(
                src_img=tar_img[idx],
                src_pts=tar_pts[idx][template_idx][mask],
                tar_img=src_imgs[idx][template_idx],
                tar_pts=src_pts[idx][template_idx][mask],
                border_color=border_color,
                concate_input=concate_input,
                write_num_matches=write_num_matches,
                patch_size=patch_size,
            )
            matching_imgs.append(torch.from_numpy(keypoint_img / 255.0).permute(2, 0, 1))
        matching_imgs = torch.stack(matching_imgs)
        matching_imgs_batch_list.append(matching_imgs)
    return matching_imgs_batch_list

def crop_and_concat_images(target_img_path, bbox, best_template_img_path, save_path, target_height=500, rotation=0, save_img=True):
    # 打开并裁剪目标图像
    target_img = Image.open(target_img_path)
    cropped_target = target_img.crop(bbox)
    
    # 打开最佳模板图像
    best_template_img = Image.open(best_template_img_path)
    # 如果有旋转角度，对最佳模板图像进行旋转
    if rotation != 0:
        best_template_img = best_template_img.rotate(rotation, expand=True)
        
    # 计算调整后的宽度,保持原始宽高比
    target_aspect_ratio = cropped_target.width / cropped_target.height
    template_aspect_ratio = best_template_img.width / best_template_img.height
    
    target_new_width = int(target_height * target_aspect_ratio)
    template_new_width = int(target_height * template_aspect_ratio)
    
    # 调整图像大小
    cropped_target_resized = cropped_target.resize((target_new_width, target_height), Image.Resampling.LANCZOS)
    best_template_resized = best_template_img.resize((template_new_width, target_height), Image.Resampling.LANCZOS)
    
    # 确定两部分的最大宽度，确保左右两边宽度一致
    half_width = max(target_new_width, template_new_width)
    new_width = half_width * 2
    
    # 创建新图像
    new_img = Image.new('RGB', (new_width, target_height), (0, 0, 0))  # 黑色背景
    
    # 粘贴调整大小后的图像（居中粘贴）
    # 左侧部分
    x_offset1 = (half_width - target_new_width) // 2
    new_img.paste(cropped_target_resized, (x_offset1, 0))
    
    # 右侧部分
    x_offset2 = half_width + (half_width - template_new_width) // 2
    new_img.paste(best_template_resized, (x_offset2, 0))
    
    # 保存图像
    if save_img:
        new_img.save(save_path)
    return new_img

def cal_obj_xyz(target_img_path, ground_mask_path, depth_image_path):
    estimated_depth, center, obb, pcd, hull_pcd = estimate_obj_depth_obb_faster(depth_image_path, ground_mask_path, target_img_path)
    return center.tolist()

def find_ground_ceiling_and_walls_mask_files(input_folder):
    input_folder = os.path.join(input_folder, 'masks')
    ground_mask_path = None
    wall_mask_paths_list = []
    ceiling_mask_path = None

    # 编译正则表达式
    ground_pattern = re.compile(r'_ground_(\d+)_mask\.png$')
    wall_pattern = re.compile(r'_wall_(\d+)_mask\.png$')
    ceiling_pattern = re.compile(r'_ceiling_(\d+)_mask\.png$')

    # 遍历input_folder下的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            full_path = os.path.join(input_folder, filename)
            # 检查是否匹配ground模式
            if ground_mask_path is None and ground_pattern.search(filename):
                ground_mask_path = full_path
            if ceiling_mask_path is None and ceiling_pattern.search(filename):
                ceiling_mask_path = full_path
            # 检查是否匹配wall模式
            if wall_pattern.search(filename):
                wall_mask_paths_list.append(full_path)

            return ground_mask_path, ceiling_mask_path, wall_mask_paths_list

def parallel_process_target_img_and_mask(target_img_path, total_items_info_dict, grounding_result_dict, input_folder, transforms):
    def process_item(name):
        bbox = grounding_result_dict[name]
        bbox = [int(m) for m in bbox]
        target_mask_path = os.path.join(input_folder, f'masks/{name}_mask.png')
        return process_real(target_img_path, target_mask_path, bbox, transforms)

    with ThreadPoolExecutor(max_workers = 30) as executor:
        results = list(executor.map(process_item, total_items_info_dict.keys()))

    return results


def extract_features_in_batches(target_img_data, ae_net, batch_size=16):
    # 创建数据集和数据加载器
    if isinstance(target_img_data, torch.Tensor):
        dataset = TensorDataset(target_img_data)
    elif isinstance(target_img_data, (list, tuple)):
        dataset = TensorDataset(torch.stack(target_img_data))
    else:
        raise TypeError("Input must be a Tensor or a list/tuple of Tensors.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features = []
    # 批量处理
    with torch.no_grad():
        for batch_images in dataloader:
            batch_images = batch_images[0].to('cuda')  # 确保数据在 GPU 上
            features = ae_net(batch_images)
            all_features.append(features)

    # 将所有特征拼接在一起
    all_features = torch.cat(all_features, dim=0)
    
    return all_features

def split_dict_by_feature_count(d, max_unique_features):
    sub_dicts = []
    if not d:
        return sub_dicts

    items = list(d.items())
    current_chunk = {}
    current_features = set()

    for key, value in items:
        # Check if adding the new item's feature would exceed the limit
        if value not in current_features and len(current_features) >= max_unique_features:
            # If so, finalize the current chunk and start a new one
            if current_chunk:
                sub_dicts.append(current_chunk)
            current_chunk = {key: value}
            current_features = {value}
        else:
            # Otherwise, add the item to the current chunk
            current_chunk[key] = value
            current_features.add(value)
    
    # Don't forget to add the last chunk
    if current_chunk:
        sub_dicts.append(current_chunk)
        
    return sub_dicts


def add_text_to_image(image, text, position, color=(0, 255, 0), font_scale=1, thickness=2):
    text=' '
    """在图像上添加文本。"""
    # 添加文本
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_rectangle_on_image(image, region_index, image_width, image_height, color=(0, 255, 0), thickness=3):
    """在图像上绘制矩形框。"""
    # 计算每个区域的宽高
    region_width = image_width // 4
    region_height = image_height // 3
    
    # 计算行列索引
    row = (region_index - 1) // 4
    col = (region_index - 1) % 4
    
    # 计算左上角和右下角位置
    top_left = (col * region_width, row * region_height)
    bottom_right = ((col + 1) * region_width, (row + 1) * region_height)
    
    # 绘制矩形框
    cv2.rectangle(image, top_left, bottom_right, color, thickness)

def get_position_for_region(region_index, image_width, image_height):
    """根据区域索引计算文本位置放在左上角。"""
    # 计算每个区域的宽高
    region_width = image_width // 4
    region_height = image_height // 3
    
    # 计算行列索引
    row = (region_index - 1) // 4
    col = (region_index - 1) % 4
    
    # 计算左上角位置,留一些边距
    x = col * region_width + 10
    y = row * region_height + 40
    
    return (x, y)

def process_and_mark_saved_images(sorted_indices_batch, final_indices, name, sample_path):
    print(sorted_indices_batch, final_indices, name)
    # 读取图像一次
    image = cv2.imread(sample_path)
    image_height, image_width = image.shape[:2]

    final_indices_global = [sorted_indices_batch[i] for i in final_indices if i!=-1]
    
    for j in range(len(sorted_indices_batch)):  # 遍历每个子图
        original_index = j + 1
        procrustes_index = sorted_indices_batch.index(j) + 1
        
        if j in final_indices_global:
            final_index = final_indices_global.index(j) + 1
            view_order = f"{original_index} -> {procrustes_index} -> {final_index}"
        else:
            view_order = f"{original_index} -> {procrustes_index}"

        # 获取文本位置
        position = get_position_for_region(original_index, image_width, image_height)

        # 在图像上标记排序信息
        add_text_to_image(image, view_order, position)

    # 绘制 top1 的绿色框
    if len(final_indices_global) > 0:
        top1_index = final_indices_global[0] + 1
        draw_rectangle_on_image(image, top1_index, image_width, image_height)

    # 保存覆盖原图
    cv2.imwrite(sample_path, image)

def compute_homography(X, Y):
    # H, status = cv2.findHomography(X, Y, method=cv2.RANSAC)
    H, status = cv2.findHomography(X, Y, method=0)
    return H

def extract_rotation(H):
    try:
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R
    except np.linalg.LinAlgError:
        print("SVD did not converge for matrix:", H)
        return None

def rotation_difference(R):
    if R is None:
        return np.inf
    angle = np.arccos((np.trace(R) - 1) / 2)
    return np.degrees(angle)

def homography_rotation_distance(X, Y):
    H = compute_homography(X, Y)
    if H is not None:
        R = extract_rotation(H)
        angle_difference = rotation_difference(R)
    else:
        angle_difference = np.inf
    return angle_difference

def preprocess_of_homo_pred(X, Y):
    # 平移：将两组点的质心对齐
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    
    # 缩放：按比例缩放两组点
    norm_X = np.linalg.norm(X_centered)
    norm_Y = np.linalg.norm(Y_centered)
    X_scaled = X_centered / norm_X
    Y_scaled = Y_centered / norm_Y
    
    return X_scaled, Y_scaled


def cal_homography_rotation_difference(src_pts_batch, tar_pts_batch):
    batch = src_pts_batch.shape[0]
    distance_list_batch = []
    sorted_indices_batch = []

    for idx in range(batch):
        distance_list = []
        for src_pts, tar_pts in zip(src_pts_batch[idx], tar_pts_batch[idx]):
            mask = tar_pts[:, 0] != -1
            # X, Y = src_pts[mask], tar_pts[mask]
            X, Y = preprocess_of_homo_pred(src_pts[mask], tar_pts[mask])
            
            if X.shape[0] < 4 or Y.shape[0] < 4:
                distance = np.inf
            else:
                distance = homography_rotation_distance(X, Y)

            distance_list.append(distance)

        distance_array = np.array(distance_list)
        sorted_indices = np.argsort(distance_array)

        distance_list_batch.append(distance_array)
        sorted_indices_batch.append(sorted_indices)

    return distance_list_batch, sorted_indices_batch


def create_point_cloud(depth_image, fx, fy, cx, cy):
    height, width = depth_image.shape
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    
    z = depth_image / 1000.0
    
    x_3d = (x_grid - cx) * z / fx
    y_3d = (y_grid - cy) * z / fy
    
    points = np.stack((x_3d, y_3d, z), axis=-1)
    points = points.reshape(-1, 3)
    
    return points


def visualize_prediction_result(color_image_path, depth_image_path, save_dir):
    scene_name = (save_dir.split('/')[-1]).split('_save_results')[0]
    predictions_pose_result = load_json(os.path.join(save_dir, f'{scene_name}_id_prediction.json'))
    
    # 构造点云
    depth_image = np.array(Image.open(depth_image_path)).astype(np.float32)
    # depth_filtered = ndimage.median_filter(depth_image, size=3)
    
    # 设置相机参数
    height, width = depth_image.shape
    estimated_focal_length = int(30/36*width)
    # estimated_focal_length = int(50/36*width)
    fx = fy = estimated_focal_length
    cx, cy = width / 2, height / 2
    
    # 创建点云
    points = create_point_cloud(depth_image, fx, fy, cx, cy)
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 为点云添加颜色
    if os.path.exists(color_image_path):
        color_image = np.array(Image.open(color_image_path))
        
        # 如果是RGBA图像,只取RGB通道
        if color_image.shape[2] == 4:
            color_image = color_image[:, :, :3]
        
        colors = color_image.astype(np.float32) / 255.0  # 归一化颜色值
        colors = colors.reshape(-1, 3)
        
        # 确保点云和颜色数据的数量匹配
        if len(points) == len(colors):
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            print(f"Warning: Number of points ({len(points)}) does not match number of colors ({len(colors)}). Skipping color assignment.")
            
    # 创建坐标轴几何体列表
    coord_frames = []

    # 为每个物体创建坐标轴
    for obj in predictions_pose_result:
        for instance, poses in obj.items():
            # 只取第一个姿态
            first_pose_key = list(poses.keys())[0]
            pose_matrix = np.array(poses[first_pose_key])
            
            # 创建坐标轴
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            coord_frame.transform(pose_matrix)
            
            # 将坐标轴转换为点云
            coord_frame_pcd = o3d.geometry.PointCloud()
            coord_frame_pcd.points = coord_frame.vertices
            coord_frame_pcd.colors = coord_frame.vertex_colors
            
            coord_frames.append(coord_frame_pcd)

    # 将所有几何体合并到一个点云中
    combined_geometry = pcd
    for frame in coord_frames:
        combined_geometry += frame

    # 保存为PLY文件
    output_path = os.path.join(save_dir, f'{scene_name}_with_poses.ply')
    o3d.io.write_point_cloud(output_path, combined_geometry)

    print(f"Point cloud with object poses saved to: {output_path}")

def find_best_reference_obj(predictions_id_result, input_folder):
    # 1.先找到哪堵墙正对着场景
    x_values = []
    wall_keys = []
    pattern = r'^wall_\d+'
    for key, value in predictions_id_result.items():
        if re.match(pattern, key):
            wall_keys.append(key)
            # 假设每个墙的信息都是字典中的第一个键
            first_key = next(iter(value))
            x_values.append(value[first_key]["estimated_xyz"][0])

    # 计算x值的平均值
    average_x = sum(x_values) / len(x_values)

    # 找到与平均x值最接近的wall_n
    closest_wall = None
    closest_distance = float('inf')
    for key, x in zip(wall_keys, x_values):
        distance = abs(x - average_x)
        if distance < closest_distance:
            closest_distance = distance
            closest_wall = key

    print(f"The wall with the x value closest to the average is: {closest_wall}")
    
    # 2.再找到靠墙的最大的那个物体作为reference obj
    scene_graph_result = load_json(os.path.join(input_folder, 'scene_graph_result_final.json'))
    candidates = []
    for key, value in scene_graph_result.items():
        if value['againstWall'] == closest_wall:
            candidates.append(key)
    
    if not candidates:
        # 没有candidates时的处理逻辑
        brief_caption_and_size_result = load_json(os.path.join(input_folder, 'brief_caption_and_size_result.json'))
        # 计算所有物体的体积并排序
        volumes = [(key, value['size'][0] * value['size'][1] * value['size'][2]) for key, value in brief_caption_and_size_result.items()]
        sorted_volumes = sorted(volumes, key=lambda x: x[1], reverse=True)[:5]  # 取体积最大的前五个

        # 从predictions_id_result中找出x值最居中的物体
        x_values = []
        for key, _ in sorted_volumes:
            first_key = next(iter(predictions_id_result[key]))
            x = predictions_id_result[key][first_key]["estimated_xyz"][0]
            x_values.append((key, x))
        
        # 计算这些x值的平均值,找到最接近平均值的物体
        average_x = sum(x[1] for x in x_values) / len(x_values)
        closest_distance = float('inf')
        closest_candidate = None
        for key, x in x_values:
            distance = abs(x - average_x)
            if distance < closest_distance:
                closest_distance = distance
                closest_candidate = key
        
        print(f"The candidate with the largest size (volume) among the top 5 and closest to the center is: {closest_candidate}")
        return closest_candidate
    else:
        brief_caption_and_size_result = load_json(os.path.join(input_folder, 'brief_caption_and_size_result.json'))
        
        max_volume = 0
        max_volume_candidate = None
        # 遍历所有的candidate来找出体积最大的那个
        for candidate in candidates:
            size = brief_caption_and_size_result[candidate]['size']  # [l, w, h]
            volume = size[0] * size[1] * size[2]
            if volume > max_volume:
                max_volume = volume
                max_volume_candidate = candidate
        
        print(f"The candidate with the largest size (volume) is: {max_volume_candidate}")
        return max_volume_candidate

def combine_scene_objects_pose(predictions_id_result, wall_floor_pose, scene_graph_result, retrieval_dict, truncated_info, save_path):
    floor_name = 'floor_0'  # 地面的名字是固定的 floor_0
    result = {
        'reference_obj': floor_name,
        'obj_info': {}
    }
    
    # 以地面为参考物体计算相对xyz,同时合并scene_graph的内容
    # 先处理墙和地面
    for obj, obj_data in wall_floor_pose.items(): 
        pose_matrix_for_blender = convert_obb_pose_to_blender_coordinates(np.array(obj_data['matrix']))
        result['obj_info'][obj] ={
            "isAgainstWall": None,
            "isOnFloor": None,
            "isHangingFromCeiling": None,
            "isHangingOnWall": None,
            "supported": None,
            "againstWall": None,
            'retrieved_asset': None,
            'choose_obb': True,
            'best_match_vid':None,
            'inplane_rotation_angle_for_object':0,
            'pose_matrix_for_blender':pose_matrix_for_blender.tolist(),
            'pcd_obb_size': None,
            'scale' : None,
            'boxes': None,
            } 
    
    for obj, obj_data in predictions_id_result.items():
        retrieved_asset = retrieval_dict[obj][0][0] if retrieval_dict[obj] else None
        retrieved_asset_info = {
            'retrieved_asset': retrieved_asset
            }
        obj_scene_graph_info = scene_graph_result[obj]

        # 合并scene_graph的内容
        result['obj_info'][obj] = obj_scene_graph_info | retrieved_asset_info | obj_data

    for obj_name, obj_info in result['obj_info'].items():
        obj_info['mask_is_truncated'] = truncated_info.get(obj_name, False)
    
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'已保存至 {save_path}')

def split_dict(d, max_items_per_chunk):
    # 将字典项转换为列表
    items = list(d.items())
    # 分割列表为多个子列表,每个子列表最多包含 max_items_per_chunk 项
    return [dict(items[i:i + max_items_per_chunk]) for i in range(0, len(items), max_items_per_chunk)]

def parallel_load_template_features(template_dir, sub_dict):
    unique_template_names = sorted(list(set(sub_dict.values())))

    def load_single_feature(top1_template_name):
        _template_dir = os.path.join(template_dir, top1_template_name)
        template_imgs_ae_features_path = os.path.join(_template_dir, 'template_imgs_ae_features.pt')
        return top1_template_name, load_features(template_imgs_ae_features_path)

    with ThreadPoolExecutor(max_workers=30) as executor:
        # Load features for unique names only
        loaded_features_map = dict(executor.map(load_single_feature, unique_template_names))

    # Reconstruct the feature list in the original order of sub_dict.values()
    src_ae_features_list = [loaded_features_map[name] for name in sub_dict.values()]

    # Stack the features along a new dimension
    stacked_src_ae_features = torch.stack(src_ae_features_list)  # Shape: [N, 162, 1024, 16, 16]
    return stacked_src_ae_features

def parallel_extract_template_ae_features(src_imgs_list, ae_net, template_dir, sub_dict, batch_size=64):
    def load_single_feature(params):
        i, top1_template_name = params
        _template_dir = os.path.join(template_dir, top1_template_name)
        template_imgs_ae_features_path = os.path.join(_template_dir, 'template_imgs_ae_features.pt')
        
        if os.path.exists(template_imgs_ae_features_path):
            return load_features(template_imgs_ae_features_path)  # [162, 1024, 16, 16]
        else:
            # 如果文件不存在，则从 src_imgs_list[i] 提取特征
            return extract_features_in_batches(src_imgs_list[i], ae_net, batch_size=batch_size)

    with ThreadPoolExecutor(max_workers=30) as executor:
        # 使用 enumerate 提供索引 i
        src_ae_features_list = list(executor.map(load_single_feature, enumerate(sub_dict.values())))

    # Stack the features along a new dimension
    stacked_src_ae_features = torch.stack(src_ae_features_list)  # Shape: [N, 162, 1024, 16, 16]
    return stacked_src_ae_features

def rotate_images_and_masks(tar_imgs_list, tar_masks_list, num_rotations=None, angles=None):
    if angles is None:
        if num_rotations is None:
            num_rotations = 4  # 默认值
        angles = [i * 360 / num_rotations for i in range(num_rotations)]

    rotated_imgs = []
    rotated_masks = []

    for img, mask in zip(tar_imgs_list, tar_masks_list):
        # 如果掩码是二维的，添加一个通道维度
        mask_was_2d = False
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
            mask_was_2d = True

        for angle in angles:
            # 旋转图像
            rotated_img = transforms.functional.rotate(img, angle)
            rotated_imgs.append(rotated_img)
            
            # 旋转掩码
            rotated_mask = transforms.functional.rotate(mask, angle)
            
            # 如果掩码原本是二维的，去掉多余的通道维度
            if mask_was_2d:
                rotated_mask = rotated_mask.squeeze(0)
            
            rotated_masks.append(rotated_mask)

    return rotated_imgs, rotated_masks

def align_inplane_rotation(best_pose_matrix_for_object, rotation_angle):
    # 进行反向旋转,取负值
    rotation_radians = np.radians(-rotation_angle)
    
    # 创建绕Y轴反向旋转的旋转矩阵
    rotation_matrix = np.array([
        [np.cos(rotation_radians), 0, np.sin(rotation_radians), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation_radians), 0, np.cos(rotation_radians), 0],
        [0, 0, 0, 1]
    ])
    
    # 将best_pose_matrix_for_object与rotation_matrix相乘
    pose_matrix_for_blender = np.dot(best_pose_matrix_for_object, rotation_matrix)
    
    return pose_matrix_for_blender

def detect_truncated_objects(input_folder):
    mask_file = f'{input_folder}/masks.pkl'
    category_json = f'{input_folder}/result.json'
    
    # 读取实例分割掩码数据
    with open(mask_file, "rb") as f:
        masks = pickle.load(f)

    # 读取类别标签数据
    with open(category_json, "r") as f:
        result_data = json.load(f)
    categories = result_data["categorys"]

    # 检查掩码数量和类别标签数量是否一致
    assert len(masks) == len(categories), "Masks and categories length mismatch."

    truncated_info = {}

    # 遍历每个实例的掩码和类别标签
    for i, (mask, category) in enumerate(zip(masks, categories)):
        mask_np = np.array(mask)
        mask = mask_np if mask_np.ndim == 2 else mask_np[:, :, -1]  # 选择 A 通道作为二值掩码
        
        # 检查掩码是否触及图像边缘
        if (mask[0, :].any() or  # 上边缘
            mask[-1, :].any() or  # 下边缘
            mask[:, 0].any() or  # 左边缘
            mask[:, -1].any()):  # 右边缘
            truncated_info[category]=True
        else:
            truncated_info[category]=False

    return truncated_info

def inference_obj_pose(input_folder, template_dir, depth_image_path, retrieval_dict, loaded_obb_data, ae_net_weights_path, ori_dino_weights_path, save_dir, save_pts_match_imgs=False, use_homography=False, save_comparison_imgs=False):
    start_time = time.time()
    
    # 设置双重输出
    os.makedirs(save_dir, exist_ok=True)
    inference_log_path = os.path.join(save_dir, 'inference_log.txt')
    sys.stdout = Logger(inference_log_path)
    
    assert os.path.exists(depth_image_path)
    
    ae_net = load_ae_net(ae_net_weights_path, ori_dino_weights_path)
    # 初始化 LocalSimilarity
    testing_metric = LocalSimilarity(k=10, sim_threshold=0.5, patch_threshold=3)

    scene_name = (save_dir.split('/')[-1]).split('_save_results')[0]
    inference_obj_pose_result_path = os.path.join(save_dir, f'{scene_name}_id_prediction.json')
    already_inferenced_obj_dict = {}
    if os.path.exists(inference_obj_pose_result_path):
        already_inferenced_obj_dict = load_json(inference_obj_pose_result_path)
    
    grounding_result_json_path = os.path.join(input_folder, 'result.json')
    with open(grounding_result_json_path, 'r') as f:
        grounding_result = json.load(f)
    
    grounding_result_dict = dict(zip(grounding_result['categorys'], grounding_result['boxes']))
    
    target_img_path = os.path.join(input_folder, 'ori.png')
    
    scene_name = (save_dir.split('/')[-1]).split('_save_results')[0]
    save_path = os.path.join(save_dir, f'{scene_name}_id_prediction.json')
    print(f"开始处理物体前的准备工作花费时长为 {time.time() - start_time}s.", flush=True)   # 15s
    
    total_items_info_dict={}
    for key, value in retrieval_dict.items():
        _dict_res = already_inferenced_obj_dict.get(key, None)
        if _dict_res and _dict_res.get('pose_matrix_for_blender', None):
            print(f'{key}的预测结果已生成,跳过!')
            continue

        if value:
            total_items_info_dict[key] = value[0][0]  # 包含top10个结果,每个结果是[name, view_id_rank[0,1,2], similarity]
        else:
            pattern = r'^(ground|wall|ceiling|floor)_\d+$'
            if re.match(pattern, key) is None:
                print(f'{key}没有对应的模型')
                continue
            else:
                # batch_process_wall_ceiling_floor_list.append(key)
                continue  # 以后不处理ground|wall|ceiling|floor这些了
                
    # 初始化存储合并结果的字典
    combined_out_data = {
        "id_src": [],
        "score_src": [],
        "score_pts": [],
        "tar_pts": [],
        "src_pts": [],
        "topk_match_counts": []
    }
    
    assert total_items_info_dict is not None
        
    # 2. 基于dinov2估计每个物体的位姿
    transforms = Transforms()
    device = torch.device('cuda')
    
    max_unique_features_per_batch = 20 # 每个批次最多容纳的不同模版特征数  按经验来看24G显存可设置20, 12G显存则设置为10; A100 80G显存可设置为60
    num_rotations = 2 
    rotation_angle_list = [0, 90]

    # --- 1. 按模板分组排序以优化缓存效率 ---
    sorted_items = sorted(total_items_info_dict.items(), key=lambda item: item[1])
    total_items_info_dict = dict(sorted_items) # 使用排序后的字典
    print("已根据模板名称对处理队列进行排序,以提高缓存命中率。")

    # --- 2. 创建基于特征约束的智能批次 ---
    sub_dicts = split_dict_by_feature_count(total_items_info_dict, max_unique_features_per_batch)
    
    # --- 3. 预分析以实现智能驱逐 ---
    feature_last_use = {}
    all_unique_templates = set(total_items_info_dict.values())
    for template_name in all_unique_templates:
        for i in range(len(sub_dicts) - 1, -1, -1):
            if template_name in sub_dicts[i].values():
                feature_last_use[template_name] = i
                break
    print("已完成特征使用分析,为智能内存驱逐做准备。")

    # --- 4. 带缓存和智能驱逐的执行策略 ---
    feature_cache = {}
    
    def load_single_feature(top1_template_name):
        _template_dir = os.path.join(template_dir, top1_template_name)
        template_imgs_ae_features_path = os.path.join(_template_dir, 'template_imgs_ae_features.pt')
        return top1_template_name, load_features(template_imgs_ae_features_path)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        print(f'**智能分批, 共{len(sub_dicts)}批处理**')
        for id, sub_dict in enumerate(sub_dicts):
            if not sub_dict: continue
            time_1 = time.time()
            tar_img_mask_list = parallel_process_target_img_and_mask(target_img_path, sub_dict, grounding_result_dict, input_folder, transforms) # list 每个元素是一个tuple（target_img, target_mask）
            tar_imgs_list = [item[0] for item in tar_img_mask_list]
            tar_masks_list = [item[1] for item in tar_img_mask_list]
            print(f'下面进行进行面内旋转的配准')
            # 进行面内旋转的配准
            # rotated_tar_imgs_list, rotated_tar_masks_list = rotate_images_and_masks(tar_imgs_list, tar_masks_list, num_rotations)
            rotated_tar_imgs_list, rotated_tar_masks_list = rotate_images_and_masks(tar_imgs_list, tar_masks_list, angles=rotation_angle_list)
            rotated_tar_masks = torch.stack(rotated_tar_masks_list) # torch.Size([43, 224, 224])
            
            # 2. 并行提取所有query image的ae特征
            stacked_tar_ae_features = extract_features_in_batches(rotated_tar_imgs_list, ae_net, batch_size=32)    # torch.Size([N, 1024, 16, 16])
            time_2 = time.time()
            print(f"批次{id} 并行处理query图片rgb与mask并提取特征共花费时长为 {time_2 - time_1}s.", flush=True)  

            # 3. 并行处理模板mask rgb
            src_img_mask_list = parallel_process_template_rgb_and_mask(template_dir, sub_dict, transforms, device, return_images=False) # list 每个元素是一个tuple（template_images, template_masks） torch.Size([162, 3, 224, 224]) torch.Size([162, 224, 224])
            src_imgs_list = [item[0] for item in src_img_mask_list]
            src_mask_list = [item[1] for item in src_img_mask_list]

            src_masks = torch.stack(src_mask_list) # torch.Size([43, 162, 224, 224])
            time_3 = time.time()
            print(f"批次{id} 并行处理模板img mask共花费时长为 {time_3 - time_2}s.", flush=True)   # 15s
            
            # 4a. 按需加载并缓存特征 (复用线程池)
            time_4_start = time.time()
            
            required_names = set(sub_dict.values())
            needed_to_load = list(required_names - set(feature_cache.keys()))

            if needed_to_load:
                print(f"批次{id} 需要从磁盘加载 {len(needed_to_load)} 个新特征...", flush=True)
                newly_loaded = dict(executor.map(load_single_feature, needed_to_load))
                feature_cache.update(newly_loaded)
            
            src_ae_features_list_batch = [feature_cache[name] for name in sub_dict.values()]
            stacked_src_ae_features = torch.stack(src_ae_features_list_batch)
            
            time_4_end = time.time()
            print(f"批次{id} 从内存/磁盘加载特征共花费时长为 {time_4_end - time_4_start}s.", flush=True)

            # 初始化当前批次的结果
            batch_out_data = {
                "id_src": [[] for _ in range(len(sub_dict) * num_rotations)],
                "score_src": [[] for _ in range(len(sub_dict) * num_rotations)],
                "score_pts": [[] for _ in range(len(sub_dict) * num_rotations)],
                "tar_pts": [[] for _ in range(len(sub_dict) * num_rotations)],
                "src_pts": [[] for _ in range(len(sub_dict) * num_rotations)],
                "topk_match_counts": [[] for _ in range(len(sub_dict) * num_rotations)],
            }
        
            for rotation_idx in range(num_rotations):
                tar_feat_batch = stacked_tar_ae_features[rotation_idx::num_rotations]
                tar_mask_batch = rotated_tar_masks[rotation_idx::num_rotations]

                # 并行knn计算
                predictions = testing_metric.test(
                    src_feats=stacked_src_ae_features,  # [N, 162, 1024, 16, 16]
                    tar_feat=tar_feat_batch,            # 当前视角的特征
                    src_masks=src_masks,                # [N, 162, 224, 224]
                    tar_mask=tar_mask_batch,            # 当前视角的掩码
                    max_batch_size=2
                )
                
                # 提取并合并预测结果
                for obj_idx in range(len(sub_dict)):
                    index = obj_idx * num_rotations + rotation_idx
                    batch_out_data["id_src"][index] = predictions.id_src[obj_idx]
                    batch_out_data["score_src"][index] = predictions.score_src[obj_idx]
                    batch_out_data["score_pts"][index] = predictions.score_pts[obj_idx]
                    batch_out_data["tar_pts"][index] = predictions.tar_pts[obj_idx]
                    batch_out_data["src_pts"][index] = predictions.src_pts[obj_idx]
                    batch_out_data["topk_match_counts"][index] = predictions.topk_match_counts[obj_idx]

                # 可视化结果
                if save_pts_match_imgs:
                    mid_save_dir = os.path.join(save_dir, 'intermediate_result')
                    os.makedirs(mid_save_dir, exist_ok=True)
                    print(f"批次{id} 视角{rotation_idx} 保存所有关键点匹配图", flush=True)
                    time_5=time.time()
                    
                    # 重新提取src_imgs_list，因为之前为了节省显存和磁盘空间没有加载
                    src_img_mask_list_vis = parallel_process_template_rgb_and_mask(template_dir, sub_dict, transforms, device, return_images=True)
                    src_imgs_list = [item[0] for item in src_img_mask_list_vis]
                    
                    tar_imgs = torch.stack(rotated_tar_imgs_list[rotation_idx::num_rotations])
                    src_imgs = torch.stack(src_imgs_list)
                    selected_templates = []
                    for i in range(src_imgs.size(0)):
                        batch_images = src_imgs[i, predictions.id_src[i], :, :, :]
                        selected_templates.append(batch_images)

                    selected_templates = torch.stack(selected_templates)
                    
                    vis_pts_list = plot_keypoints_tar_2_nSrc(tar_img=tar_imgs, tar_pts=predictions.tar_pts, src_imgs=selected_templates, src_pts=predictions.src_pts)
                    for i in range(src_imgs.size(0)):
                        name = list(sub_dict.keys())[i]
                        sample_path = f"{mid_save_dir}/{name}_pred_vis_rotation_{rotation_idx}.png"
                        save_tensor_to_image(vis_pts_list[i], sample_path)
                    print(f"批次{id} 视角{rotation_idx} 保存所有关键点匹配图共花费时长为 {time.time() - time_5}s.", flush=True)

                    del tar_imgs
                    del selected_templates
                    del vis_pts_list
                    del src_img_mask_list_vis
                    torch.cuda.empty_cache()

            # 将当前批次的结果合并到全局结果中
            for key in combined_out_data:
                combined_out_data[key].extend(batch_out_data[key])

            # 4c. 智能驱逐: 释放不再需要的特征
            evict_keys = []
            for template_name in feature_cache:
                if feature_last_use.get(template_name) == id:
                    evict_keys.append(template_name)
            
            if evict_keys:
                print(f"批次 {id} 处理完毕, 从缓存中驱逐 {len(evict_keys)} 个不再需要的特征。", flush=True)
                for key in evict_keys:
                    del feature_cache[key]
            
            # 释放gpu缓存
            del src_img_mask_list
            del src_mask_list
            del src_imgs_list
            del stacked_src_ae_features
            del stacked_tar_ae_features
            del src_masks
            del rotated_tar_masks_list
            torch.cuda.empty_cache()

    # 将列表中的张量合并
    # batch_out_data[key] 是 长度156(39*4视角) 每个元素是 torch.Size([10, 256, 2]) 的list
    combined_out_data = {key: torch.stack(value, dim=0) for key, value in combined_out_data.items()}

    # 创建 PandasTensorCollection
    final_predictions = tc.PandasTensorCollection(
        infos=pd.DataFrame(),
        id_src=combined_out_data["id_src"],
        score_src=combined_out_data["score_src"],
        score_pts=combined_out_data["score_pts"],
        tar_pts=combined_out_data["tar_pts"],
        src_pts=combined_out_data["src_pts"],
        topk_match_counts=combined_out_data["topk_match_counts"]
    )
    
    # 假设每个对象有 num_rotations 个旋转视角
    num_objects = final_predictions.tar_pts.shape[0] // num_rotations
    
    # 初始化每个物体的结果存储
    predictions_id_result = {}
    comparison_images = []
    '''
    如果要控制某些物体的num_rotations  rotation_angle_list,
    可以在这里指定,前面的num_rotations  rotation_angle_list只是为了提取特征和对比,
    只要确保他们是下面所有rotation_angle_list的并集就行
    '''
    for obj_index in range(num_objects):
        # Determine rotation settings based on the object name
        name = list(total_items_info_dict.keys())[obj_index]
        if re.search(r'book_\d+', name):
            rotation_angle_list = [0, 90]
        else:
            rotation_angle_list = [0]

        min_angle_diff_for_object = np.inf
        best_obb_match_vid_for_object = None
        best_pose_matrix_for_object = None
        best_rotation_index_for_object = None
        rotation_view_ids_map = {}

        start_idx = obj_index * num_rotations
        end_idx = start_idx + len(rotation_angle_list)

        all_tar_pts = final_predictions.tar_pts[start_idx:end_idx].cpu().numpy()
        all_src_pts = final_predictions.src_pts[start_idx:end_idx].cpu().numpy()
        all_id_src = final_predictions.id_src[start_idx:end_idx]
        all_topk_match_counts = final_predictions.topk_match_counts[start_idx:end_idx]
        
        if use_homography:
            distance_list_batch, sorted_indices_batch = cal_homography_rotation_difference(all_tar_pts, all_src_pts)
            distance_type = 'homography_rotation_angle'
        else:
            distance_list_batch, sorted_indices_batch = cal_procrustes(all_tar_pts, all_src_pts)
            distance_type = 'procrustes_distance'

        # 统计物体的大小, 小物体不适用obb优化
        estimated_size = loaded_obb_data[name]['obb_size']
        obb_size_products = [
            estimated_size[0] * estimated_size[1],
            estimated_size[0] * estimated_size[2],
            estimated_size[1] * estimated_size[2]
        ]
        max_estimated_product = max(obb_size_products)
            
        # 初始化变量
        best_vid = None
        best_view_id = None
        best_rotation_index = None
        min_vid_index = float('inf')
        min_diff = float('inf')
        min_homography_rotation_angle = float('inf')
        previous_sum_match_counts = -float('inf')
        best_rotation_index_for_object = 0
        
        # final_predictions.tar_pts 和 final_predictions.src_pts 的形状为 [batch_size, num_candidates, 256, 2]
        for rotation_index in range(len(rotation_angle_list)):
            current_id_src = all_id_src[rotation_index]
            sorted_indices = torch.tensor(sorted_indices_batch[rotation_index], dtype=torch.long).to(device)
            final_predictions.id_src[start_idx + rotation_index] = current_id_src[sorted_indices]
            
            # print("下面重新排序的物体是:", list(total_items_info_dict.keys())[obj_index], flush=True)
            # print("特征匹配到的点数如下:", all_topk_match_counts[rotation_index], flush=True)
            # print(f"{distance_type}分别为:", distance_list_batch[rotation_index], flush=True)
            # print(f"基于{distance_type}重新排序后的view_id results:", current_id_src, flush=True)
        
            sorted_distances = distance_list_batch[rotation_index][sorted_indices_batch[rotation_index]]
            match_counts = all_topk_match_counts[rotation_index]

            if (match_counts > 80).all():
                # 如果是,设置 distance_threshold 为 inf, 即因为匹配的点都很多,所以不需要删除任何一个候选视角
                distance_threshold = np.inf
            else:
                # 否则,设置 distance_threshold 为 1 或其他默认值
                distance_threshold = 1 if not use_homography else 90    # 如果是use_homography 90代表90度

            top1_template_name = list(total_items_info_dict.values())[obj_index]
            _template_dir = os.path.join(template_dir, top1_template_name)
            pose_numpy_path = os.path.join(_template_dir, 'camera_poses.npy')
            pose_data = np.load(pose_numpy_path)

            valid_indices = sorted_distances < distance_threshold
            # print(f'valid_indices: {valid_indices}', flush=True)
            valid_view_ids = current_id_src[valid_indices].cpu().numpy()

            if len(valid_view_ids) == 0:
                valid_view_ids = current_id_src.cpu().numpy()

            top_3_view_ids = valid_view_ids[:3]
            valid_matrices = [pose_data[vid][:3, :3] for vid in valid_view_ids]

            average_differences = []
            for top_vid in top_3_view_ids:
                top_matrix = pose_data[top_vid][:3, :3]
                differences = [np.linalg.norm(top_matrix - vm, 'fro') for vm in valid_matrices if not np.array_equal(top_matrix, vm)]
                average_difference = np.mean(differences)
                average_differences.append((top_vid, average_difference))

            # print('average_differences:', average_differences)
            
            sorted_top_3_view_ids = [vid for vid, _ in sorted(average_differences, key=lambda x: x[1])]
            sorted_top_10_view_ids = sorted_top_3_view_ids[:10] + [-1] * (10 - len(sorted_top_3_view_ids))
            rotation_view_ids_map[rotation_index] = sorted_top_3_view_ids

            final_indices = []
            for view_id in sorted_top_10_view_ids:
                if view_id == -1:
                    final_indices.append(-1)
                else:
                    index = np.where(current_id_src.cpu().numpy() == view_id)[0][0]
                    final_indices.append(index)
                    
            # print(f"计算top3与其他所有有效视角旋转矩阵的平均Frobenius范数差异,进行排序: {sorted_top_10_view_ids}", flush=True)
            print('init_view_ids_list: ', current_id_src.cpu().numpy())
            print('final_indices: ', final_indices)
            
            if save_pts_match_imgs:
                image_to_be_mark = os.path.join(mid_save_dir, f"{name}_pred_vis_rotation_{rotation_index}.png")
                process_and_mark_saved_images(sorted_indices.cpu().tolist() if len(sorted_indices.cpu().tolist()) > 0 else current_id_src.cpu().numpy().tolist(), final_indices, name, image_to_be_mark)
                # print(f'{name}的重排结果都已标记在图上', flush=True)

            obb_pose_matrix = np.eye(4)
            obb_pose_matrix[:3, :3] = loaded_obb_data[name]['obb_rotation']
            obb_pose_matrix[:3, 3] = loaded_obb_data[name]['center_xyz']
            best_matches = find_view_best_match_obb(pose_data, obb_pose_matrix)

            # 初始化当前 rotation_index 的最小值
            current_min_vid_index = float('inf')
            current_best_vid = None
            current_best_view_id = None
            current_min_diff = float('inf')
            current_best_obb_match_vid_for_object = float('inf')
            current_best_rotation_index_for_object = float('inf')
            current_best_pose_matrix_for_object = float('inf')
            current_min_homography_rotation_angle = float('inf')
            
            # 对于小物体，直接根据关键点匹配质量来确定选取哪个best_rotation_index_for_object，然后取该
            if max_estimated_product < 0.16:
                print(f'{name} 是小物体, 不适用obb的方向优化')
                # 小物体的obb不可信, 直接通过dinov2来判断
                if match_counts.sum() >= previous_sum_match_counts:
                    best_rotation_index_for_object = rotation_index
                    previous_sum_match_counts = match_counts.sum()
                    
            else:
                # 遍历所有视角，寻找当前 rotation_index 中最靠前的满足条件的 vid
                for vid_index, (vid, _) in enumerate(sorted(average_differences, key=lambda x: x[1])):
                    for (view_id, transformed_obb_pose_matrix, obj_pose_matrix, _) in best_matches:
                        # 计算旋转差异
                        pose_vid = convert_camera_pose_of_render_view_to_obj_pose_to_blender_coordinates(pose_data[vid])
                        pose_view_id = convert_camera_pose_of_render_view_to_obj_pose_to_blender_coordinates(pose_data[view_id])
                        _, _diff = rotation_matrix_to_angle_diff(pose_vid[:3, :3], pose_view_id[:3, :3])

                        if _diff < 36:
                            homography_rotation_angle = sorted_distances[vid_index]  # 假设 sorted_distances 存储 homography_rotation_angle
                            # print(f"VID: {vid}, View ID: {view_id}, Difference: {_diff}, " 
                            #     f"Rotation Angle: {homography_rotation_angle}, Rotation Index: {rotation_index}")

                            if vid_index < current_min_vid_index or (vid_index == current_min_vid_index and _diff < current_min_diff):
                                current_min_vid_index = vid_index
                                current_best_vid = vid
                                current_best_view_id = view_id
                                current_min_diff = _diff
                                current_min_homography_rotation_angle = homography_rotation_angle
                                
                                current_best_obb_match_vid_for_object = view_id
                                current_best_rotation_index_for_object = rotation_index
                                current_best_pose_matrix_for_object = transformed_obb_pose_matrix
                                # print(f'view_id: {view_id}, rotation_index: {rotation_index}', flush=True)

            # 更新全局最优解
            if current_best_vid is not None:
                if current_min_vid_index < min_vid_index or (current_min_vid_index == min_vid_index and current_min_homography_rotation_angle < min_homography_rotation_angle):
                    min_vid_index = current_min_vid_index
                    min_diff = current_min_diff
                    min_homography_rotation_angle = current_min_homography_rotation_angle
                    best_vid = current_best_vid
                    best_view_id = current_best_view_id
                    best_obb_match_vid_for_object = current_best_obb_match_vid_for_object
                    best_rotation_index_for_object = current_best_rotation_index_for_object
                    best_pose_matrix_for_object = current_best_pose_matrix_for_object      
        
        # if min_angle_diff_for_object < 36:
        if best_obb_match_vid_for_object is not None:
            best_match_vid = best_obb_match_vid_for_object
            inplane_rotation_angle_for_object = rotation_angle_list[best_rotation_index_for_object]
            pose_matrix_for_blender = align_inplane_rotation(best_pose_matrix_for_object, inplane_rotation_angle_for_object)
            predictions_id_result[name] = {
                'choose_obb': True,
                'best_match_vid': str(best_match_vid),
                'inplane_rotation_angle_for_object': inplane_rotation_angle_for_object,
                'pose_matrix_for_blender': pose_matrix_for_blender.tolist(),
                'pcd_obb_size': loaded_obb_data[name]['obb_size'],
                'boxes': grounding_result_dict[name],
            }
            # print(f'选择了obb方向的其中一个, best_match_vid是{best_match_vid}, inplane_rotation_angle_for_object是{inplane_rotation_angle_for_object}', flush=True)
        else:
            best_match_vid = rotation_view_ids_map[best_rotation_index_for_object][0]
            inplane_rotation_angle_for_object = rotation_angle_list[best_rotation_index_for_object]
            # 注意这里输入convert_camera_pose_of_render_view_to_obj_pose_to_blender_coordinates中的place_position是blender坐标系中的的, obb的坐标要先做转换
            _location = np.array([loaded_obb_data[name]['center_xyz'][0], loaded_obb_data[name]['center_xyz'][2], -loaded_obb_data[name]['center_xyz'][1]])
            best_pose_matrix_for_object = convert_camera_pose_of_render_view_to_obj_pose_to_blender_coordinates(pose_data[best_match_vid], _location)
            pose_matrix_for_blender = align_inplane_rotation(best_pose_matrix_for_object, inplane_rotation_angle_for_object)
            predictions_id_result[name] = {
                'choose_obb': False,
                'best_match_vid': str(best_match_vid),
                'inplane_rotation_angle_for_object': inplane_rotation_angle_for_object,
                'pose_matrix_for_blender': pose_matrix_for_blender.tolist(),
                'pcd_obb_size': loaded_obb_data[name]['obb_size'],
                'boxes': grounding_result_dict[name],
            }
            print(f'选择了162模板中的其中一个, best_match_vid是{best_match_vid}, inplane_rotation_angle_for_object是{inplane_rotation_angle_for_object}', flush=True)

        comparison_img_save_path = os.path.join(save_dir, f'{name}_comparison_{best_match_vid:06d}.png')
        bbox = grounding_result_dict[name]
        bbox = [int(m) for m in bbox]
        top1_template_name = list(total_items_info_dict.values())[obj_index]
        _template_dir = os.path.join(template_dir, top1_template_name)
        best_template_img_path = os.path.join(_template_dir, f'{best_match_vid:06d}.png')
        comparison_img = crop_and_concat_images(target_img_path, tuple(bbox), best_template_img_path, comparison_img_save_path, rotation=inplane_rotation_angle_for_object, save_img=save_comparison_imgs)
        comparison_images.append(comparison_img)

    print(f"****确定物体最终位姿一共花费时间为 {time.time() - start_time}s.****", flush=True)
    scene_name = (save_dir.split('/')[-2]).split('_result')[0]
    save_path = os.path.join(save_dir, f'{scene_name}_id_prediction.json')
    with open(save_path, 'w') as f:
        json.dump(predictions_id_result, f, indent=2)
    return predictions_id_result, comparison_images