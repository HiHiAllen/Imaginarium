"""
Legacy functions from S2_3d_retrieval_op.py
"""
import json
import re
import ast
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import functools

# ===== 让所有print自动flush，避免输出被缓冲 =====
print = functools.partial(print, flush=True)
from collections import defaultdict
import copy
import os
from einops import rearrange
import torch.nn.functional as F

from utils.llm_api import GPTApi,parallel_processing_requests
from models.ae_net.matching import LocalSimilarity
from utils.obb import cal_and_visualize_scene_obj_bbox_fitting
from utils.logger import Logger
from prompts.used_prompts import VLM_SIZE_CORRECTION_PROMPT
logger = None

def get_largest_object_in_group(group_objects, masks_folder):
    """
    Finds the object with the largest mask area from a list of objects in the same group.
    """
    largest_object = None
    max_area = -1

    for obj_name in group_objects:
        mask_path = os.path.join(masks_folder, f'{obj_name}_mask.png')
        if not os.path.exists(mask_path):
            continue
        
        mask_img = Image.open(mask_path).convert('L')
        mask_array = np.array(mask_img)
        area = np.sum(mask_array > 0) # Count non-black pixels
        
        if area > max_area:
            max_area = area
            largest_object = obj_name
            
    return largest_object

def create_grids_for_size_estimation(objects_data, background_image, masks_folder, save_dir):
    """
    Creates 3x3 grid images for VLM size estimation, with improved aesthetics.
    Each grid cell contains a cropped image of an object with its bbox drawn.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    grid_images_info = {}
    object_items = list(objects_data.items())
    
    # Define grid specs
    grid_size = 3
    cell_size = 256
    margin = 10
    grid_dim = grid_size * cell_size + (grid_size + 1) * margin
    font_path = "src/pose_estimation/DejaVuSans-Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        logger.warning(f"Font file not found at {font_path}. Using default font.")
        font = ImageFont.load_default()

    for i in range(0, len(object_items), grid_size * grid_size):
        chunk = dict(object_items[i : i + grid_size * grid_size])
        # grid_image = Image.new('RGB', (grid_dim, grid_dim), (0, 255, 0)) # Green background for margins
        grid_image = Image.new('RGB', (grid_dim, grid_dim), (0, 0, 0)) # black background for margins
        
        object_names_in_grid = []
        for j, (obj_name, data) in enumerate(chunk.items()):
            mask_path = os.path.join(masks_folder, f'{obj_name}_mask.png')
            if not os.path.exists(mask_path):
                continue

            mask_img = Image.open(mask_path).convert('L')
            bbox = mask_img.getbbox()
            
            if not bbox:
                continue

            # 1. Create expanded scene crop with bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # Adjust expansion to make the crop squarer
            if width < 100 and height < 100:
                # For small objects, start with a larger fixed expansion
                expand_x = 50
                expand_y = 50
            else:
                # For larger objects, start with a proportional expansion
                base_padding_ratio = 0.1  # 10% base padding
                expand_x = int(width * base_padding_ratio)
                expand_y = int(height * base_padding_ratio)

            # Calculate padded dimensions and add more padding to the smaller side to make it square
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
            
            # Dynamic line width for bounding box
            box_line_width = max(2, min(5, int(min(width, height) * 0.04)))
            draw_crop.rectangle(original_bbox_in_crop, outline="red", width=box_line_width)
            
            # 2. Resize with padding to fit cell with black background
            ratio = min(cell_size / scene_crop_raw.width, cell_size / scene_crop_raw.height)
            new_size = (int(scene_crop_raw.width * ratio), int(scene_crop_raw.height * ratio))
            resized_img = scene_crop_raw.resize(new_size, Image.Resampling.LANCZOS)
            
            cell_img = Image.new('RGB', (cell_size, cell_size), (0, 0, 0)) # Black cell background
            paste_pos = ((cell_size - new_size[0]) // 2, (cell_size - new_size[1]) // 2)
            cell_img.paste(resized_img, paste_pos)
            
            # 3. Draw object name on the cell image
            cell_draw = ImageDraw.Draw(cell_img)
            cell_draw.text((10, 225), obj_name, font=font, fill=(0, 255, 0))

            # 4. Paste the cell into the main grid
            grid_row, grid_col = j // grid_size, j % grid_size
            paste_x = margin + grid_col * (cell_size + margin)
            paste_y = margin + grid_row * (cell_size + margin)
            grid_image.paste(cell_img, (paste_x, paste_y))
            
            object_names_in_grid.append(obj_name)
        
        grid_save_path = os.path.join(save_dir, f'size_estimation_grid_{i // (grid_size*grid_size)}.png')
        grid_image.save(grid_save_path)
        grid_images_info[grid_save_path] = object_names_in_grid
        
    return grid_images_info

def refine_dimensions_with_vlm(input_folder, save_folder, gpt_params):
    """
    Uses VLM to refine the length and width of floor objects.
    """
    logger.info("Starting dimension refinement with VLM...")
    start_time = time.time()

    scene_graph_path = os.path.join(input_folder, 'scene_graph_result.json')
    pcd_obb_data_path = os.path.join(save_folder, 'pcd_obb_data.json')
    masks_folder = os.path.join(input_folder, 'masks')
    ori_scene_img_path = os.path.join(input_folder, 'ori.png')

    with open(scene_graph_path, 'r') as f:
        scene_graph = json.load(f)
    with open(pcd_obb_data_path, 'r') as f:
        pcd_obb_data = json.load(f)

    # 1. Filter for floor objects and handle groups
    floor_objects = {}
    groups = defaultdict(list)
    for obj, props in scene_graph.items():
        if props.get('supported') == 'floor_0':
            floor_objects[obj] = props
            if props.get('group'):
                groups[props['group']].append(obj)

    # For grouped objects, only process the largest one
    representative_objects = {}
    processed_in_group = set()
    for group_id, group_members in groups.items():
        if len(group_members) > 1:
            largest_obj = get_largest_object_in_group(group_members, masks_folder)
            if largest_obj:
                representative_objects[largest_obj] = {'group_members': group_members}
                processed_in_group.update(group_members)
    
    # Add non-grouped floor objects
    for obj in floor_objects:
        if obj not in processed_in_group:
            representative_objects[obj] = {'group_members': [obj]}

    if not representative_objects:
        logger.info("No representative floor objects found to refine. Skipping VLM step.")
        return

    # 2. Prepare for VLM
    ori_scene_img = Image.open(ori_scene_img_path)
    grid_save_dir = os.path.join(save_folder, 'vlm_size_estimation_grids')
    grid_images_info = create_grids_for_size_estimation(representative_objects, ori_scene_img, masks_folder, grid_save_dir)

    all_image_list = []
    all_prompt_list = []
    initial_dimensions_map = {}
    for grid_path, object_names in grid_images_info.items():
        initial_dims_str = ""
        for name in object_names:
            dims_m = pcd_obb_data[name]['obb_size']
            dims_cm = [round(d * 100, 2) for d in dims_m]
            initial_dimensions_map[name] = dims_cm
            initial_dims_str += f"- {name}: [{dims_cm[0]}, {dims_cm[1]}, {dims_cm[2]}]\n"
        
        prompt = VLM_SIZE_CORRECTION_PROMPT.format(initial_dimensions=initial_dims_str)
        all_image_list.append(grid_path)
        all_prompt_list.append(prompt)

    print(all_prompt_list)
    
    # 3. Call VLM
    logger.info(f"Sending {len(all_prompt_list)} requests to VLM for size refinement...")
    results = parallel_processing_requests(gpt_params, all_image_list, all_prompt_list, return_list=False, return_json=True, return_dict=False, num_processes=8)
    print(results)

    # 4. Parse results and update obb data
    updates_count = 0
    for result_dict in results:
        if result_dict and isinstance(result_dict, dict):
            for obj_name, corrected_dims_cm in result_dict.items():
                if obj_name in pcd_obb_data:
                    initial_dims_cm = initial_dimensions_map.get(obj_name)
                    if initial_dims_cm and isinstance(corrected_dims_cm, list) and len(corrected_dims_cm) == 3:
                        initial_length, initial_width, _ = initial_dims_cm
                        corrected_length, corrected_width, corrected_height = corrected_dims_cm
                        
                        if initial_length > initial_width:
                            if corrected_length < corrected_width:
                                corrected_dims_cm = [corrected_width, corrected_length, corrected_height]
                                logger.info(f"Swapped dimensions for '{obj_name}' to maintain L > W ratio. New: {corrected_dims_cm}")
                        elif initial_length < initial_width:
                            if corrected_length > corrected_width:
                                corrected_dims_cm = [corrected_width, corrected_length, corrected_height]
                                logger.info(f"Swapped dimensions for '{obj_name}' to maintain L < W ratio. New: {corrected_dims_cm}")

                    # Convert back to meters
                    corrected_dims_m = [round(d / 100, 4) for d in corrected_dims_cm]
                    
                    # Find the group this object represents
                    group_to_update = []
                    for rep_obj, data in representative_objects.items():
                        if rep_obj == obj_name:
                            group_to_update = data['group_members']
                            break
                    
                    # Update all members of the group
                    for member_name in group_to_update:
                        if member_name in pcd_obb_data:
                            logger.info(f"Updating dimensions for '{member_name}': from {pcd_obb_data[member_name]['obb_size']} to {corrected_dims_m}")
                            pcd_obb_data[member_name]['obb_size'] = corrected_dims_m
                            updates_count += 1
    
    if updates_count > 0:
        with open(pcd_obb_data_path, 'w') as f:
            json.dump(pcd_obb_data, f, indent=2)
        logger.info(f"Successfully updated dimensions for {updates_count} objects and saved to {pcd_obb_data_path}")
    else:
        logger.warning("VLM processing completed, but no object dimensions were updated.")

    logger.info(f"Dimension refinement with VLM took {time.time() - start_time:.2f}s.")


def extract_dict_with_re(output):
    try:
        dict_pattern = r'\{.*\}'
        dict_match = re.search(dict_pattern, output, re.DOTALL)
        if dict_match:
            dict_str = dict_match.group(0)
            try:
                dict_data = ast.literal_eval(dict_str)
                return dict_data
            except Exception as e:
                logger.error(f"Error parsing  dict: {e}")
                return None
        else:
            raise ValueError("No dict found in the output")
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing  dict: {e}")
        return None


def extract_all_detected_item_images_features(detected_item_list, ori_scene_img, masks_folder, processor, embedding_model, mask_query_pixels_ratio=0.0025, batchsize=16):
    images_pil = []
    mask_query_pixels_threshold = ori_scene_img.size[0] * ori_scene_img.size[1] * mask_query_pixels_ratio   # 像素小于mask_query_pixels_threshold的物体用bbox提取特征
    for item in detected_item_list:
        mask_path = os.path.join(masks_folder, f'{item}_mask.png')
        mask_img = Image.open(mask_path).convert('L')  # Load mask image and convert to grayscale

        # Get the bounding box of the mask
        bbox = mask_img.getbbox()
        if bbox:
            # Crop the mask and original image
            cropped_mask = mask_img.crop(bbox)
            cropped_img = ori_scene_img.crop(bbox)

            # Convert mask and image to NumPy arrays
            mask_array = np.array(cropped_mask) / 255.0
            cropped_array = np.array(cropped_img)

            # Ensure the mask array shape is (height, width, 1)
            mask_array = mask_array[:, :, np.newaxis]

            # Calculate the number of effective pixels in the mask
            effective_pixels = np.sum(mask_array > 0)
            # print(f'Item: {item}, Effective Pixels: {effective_pixels}')

            # Apply conditional logic based on the number of effective pixels
            if effective_pixels < mask_query_pixels_threshold or re.match(r'^(bed)_\d+$', item):
                masked_img = Image.fromarray(cropped_array.astype(np.uint8))
            else:
                masked_img = Image.fromarray((cropped_array * mask_array).astype(np.uint8))

            images_pil.append(masked_img)

    # 存储所有批次的特征
    all_image_features = []
    for i in range(0, len(images_pil), batchsize):
        batch_images = images_pil[i:i + batchsize]
        # 批处理输入
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        inputs = processor(images=batch_images, return_tensors="pt").to(device)

        # 提取特征
        with torch.no_grad():
            outputs = embedding_model(**inputs)
            patch_features = outputs.last_hidden_state
            # The [CLS] token is the first token in the sequence.
            # We discard it to keep only the 256 patch features.
            patch_features = patch_features[:, 1:, :]

        # 将当前批次的特征添加到列表中
        all_image_features.append(patch_features)

    # 拼接所有特征
    all_image_features = torch.cat(all_image_features, dim=0)    # torch.Size([N, num_patches, 1024])
    return all_image_features

def _process_and_sort_candidates(candidates_with_scores, estimated_size, fbx_size_dict, alpha, item_class_name, asset_to_class_en_dict):
    """
    Helper function to process a list of candidates by filtering, calculating size difference,
    and sorting based on a combined score.
    """
    processed_candidates = []
    for asset_name, view_idx, feature_score in candidates_with_scores:
        fbx_size = fbx_size_dict.get(asset_name)
        if not fbx_size:
            continue

        # Boost feature_score for assets with the same class_en
        class_bonus = 0.0
        asset_class_name = asset_to_class_en_dict.get(asset_name)
        if asset_class_name == item_class_name:
            class_bonus = 0.1 * feature_score
            
        size_difference = calculate_size_difference(estimated_size, fbx_size)
        
        # Apply filter
        if feature_score > 0.25 and size_difference < 4:
            # Calculate combined score
            if size_difference < 0.5:
                combined_score = feature_score + class_bonus
            else:
                combined_score = feature_score + class_bonus - alpha * size_difference
            
            processed_candidates.append(
                (asset_name, view_idx, float(round(feature_score, 4)), float(round(size_difference, 4)), float(round(class_bonus, 4)), float(round(combined_score, 4)))
            )

    # Sort by combined score in descending order
    processed_candidates.sort(key=lambda x: x[5], reverse=True)
    return processed_candidates

def cal_embedding_sim(detected_item_list, all_detected_item_images_features, item_to_candidate_models_dict, asset_embedding_folder, fbx_size_dict, estimated_size_dict, asset_to_class_en_dict, alpha):
    """
    Efficiently computes similarity for a batch of detected items against their candidate models
    using a hybrid local-first, global-fallback strategy.
    """
    start_time = time.time()
    
    # 1. Init matcher. Increase max_batch_size for batched target items.
    matcher = LocalSimilarity(k=20, sim_threshold=0.25, patch_threshold=3, max_batch_size=8)
    embedding_similarity_results = {}
    device = all_detected_item_images_features.device

    # 2. Pre-load all necessary asset features into an in-memory cache to avoid redundant disk I/O.
    all_candidate_names = set()
    for item_name in detected_item_list:
        if item_name in item_to_candidate_models_dict:
            all_candidate_names.update(item_to_candidate_models_dict[item_name])
    
    asset_features_cache = {}
    logger.info(f"Pre-loading {len(all_candidate_names)} unique asset features into memory...")
    for asset_name in tqdm(list(all_candidate_names)):
        asset_embedding_path = os.path.join(asset_embedding_folder, f"{asset_name}.pt")
        if os.path.exists(asset_embedding_path):
            # Load and move to device immediately
            asset_features_cache[asset_name] = [f.to(device) for f in torch.load(asset_embedding_path, map_location='cpu')]
    logger.info("Asset features loaded.")

    # 3. Group detected items by their base category to process them in efficient batches.
    category_to_items = defaultdict(list)
    item_to_idx_map = {name: i for i, name in enumerate(detected_item_list)}

    for item_name in detected_item_list:
        # Fallback for items not in the dict, though they shouldn't be processed
        if item_name not in item_to_candidate_models_dict:
            continue
        item_class_name = re.sub(r'_\d+$', '', item_name).lower()
        category_to_items[item_class_name].append(item_name)
        
    # 4. Process each category as a single batch.
    for category, items_in_batch in category_to_items.items():
        if not items_in_batch:
            continue
        
        logger.info(f"Processing batch for category '{category}' with {len(items_in_batch)} items.")

        # a. Prepare target features for this batch
        item_indices = [item_to_idx_map[item_name] for item_name in items_in_batch]
        tar_feats_batch = all_detected_item_images_features[item_indices] # (B, num_patches, 1024)
        tar_feats_batch = rearrange(tar_feats_batch, "b (h w) c -> b c h w", h=16, w=16)
        batch_size = len(items_in_batch)

        # b. Prepare source features. They are the same for all items in a category batch.
        candidate_models = item_to_candidate_models_dict.get(items_in_batch[0], [])
        
        all_candidate_features_views = []
        all_candidate_names_views = []

        for asset_name in candidate_models:
            if asset_name in asset_features_cache:
                for view_idx, asset_view_feat in enumerate(asset_features_cache[asset_name]):
                    all_candidate_features_views.append(asset_view_feat)
                    all_candidate_names_views.append((asset_name, view_idx))
        
        if not all_candidate_features_views:
            for item_name in items_in_batch:
                embedding_similarity_results[item_name] = []
            continue

        src_feats_tensor = torch.stack(all_candidate_features_views) # (N_views, num_patches, 1024)
        src_feats_tensor = rearrange(src_feats_tensor, "n (h w) c -> n c h w", h=16, w=16)
        
        # Repeat source features for each target item in the batch
        src_feats_batch = src_feats_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1) # (B, N_views, C, H, W)
        
        # c. Create dummy masks (match all patches)
        tar_masks_batch = torch.ones(batch_size, 16, 16, device=device)
        src_masks_batch = torch.ones(batch_size, src_feats_batch.shape[1], 16, 16, device=device)

        # d. Run matcher on the entire batch to get LOCAL scores
        predictions = matcher.test(
            src_feats=src_feats_batch,
            tar_feat=tar_feats_batch,
            src_masks=src_masks_batch,
            tar_mask=tar_masks_batch,
        )

        # e. Unpack and process results for each item in the batch
        for i, item_name in enumerate(items_in_batch):
            estimated_size = estimated_size_dict[item_name]
            
            # --- Process Local Candidates ---
            top_local_scores = predictions.score_src[i].cpu().numpy()
            top_indices = predictions.id_src[i].cpu().numpy()
            local_candidates_with_scores = []
            for j, candidate_idx in enumerate(top_indices):
                asset_name, view_idx = all_candidate_names_views[candidate_idx]
                local_candidates_with_scores.append((asset_name, view_idx, top_local_scores[j]))
            
            sorted_local_list = _process_and_sort_candidates(local_candidates_with_scores, estimated_size, fbx_size_dict, alpha, category, asset_to_class_en_dict)

            # --- Process Global Candidates ---
            query_patches = all_detected_item_images_features[item_to_idx_map[item_name]]
            query_global_feat = query_patches.mean(dim=0)
            
            global_candidates_with_scores = []
            for candidate_idx, (asset_name, view_idx) in enumerate(all_candidate_names_views):
                candidate_patches = all_candidate_features_views[candidate_idx]
                candidate_global_feat = candidate_patches.mean(dim=0)
                score_global = F.cosine_similarity(query_global_feat.unsqueeze(0), candidate_global_feat.unsqueeze(0)).item()
                global_candidates_with_scores.append((asset_name, view_idx, score_global))
            
            sorted_global_list = _process_and_sort_candidates(global_candidates_with_scores, estimated_size, fbx_size_dict, alpha, category, asset_to_class_en_dict)
            
            # --- Combine and Store Final Result ---
            final_list = sorted_local_list + sorted_global_list
            embedding_similarity_results[item_name] = final_list
            
            # --- Logging as requested ---
            if final_list:
                top_result = final_list[0]
                # If sorted_local_list is not empty, the top result must be from it.
                source = "LOCAL" if sorted_local_list else "GLOBAL"
                logger.info(f"  > [{item_name}] selected '{top_result[0]}' from {source} list with score: {top_result[5]:.4f}")
            else:
                logger.warning(f"  > [{item_name}] No suitable model found after filtering.")
            
    logger.info(f"Batched similarity calculation took {time.time() - start_time:.2f}s.")
    return embedding_similarity_results


def resize_and_center(image, size=(512, 512), background_color=(255, 255, 255)):
    # 获取原始尺寸
    original_width, original_height = image.size
    # 计算缩放比例
    ratio = min(size[0] / original_width, size[1] / original_height)
    # 计算新尺寸
    new_size = (int(original_width * ratio), int(original_height * ratio))
    # 缩放图像
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    # 创建新的背景画布
    canvas = Image.new('RGB', size, background_color)
    # 计算居中位置
    paste_position = ((size[0] - new_size[0]) // 2, (size[1] - new_size[1]) // 2)
    # 将缩放后的图像粘贴到画布上
    canvas.paste(resized_image, paste_position)
    return canvas

def visualize_retrieval_results(retrieval_results, assets_render_result_folder, assets_imgs_order_pkl_path, masks_folder, ori_scene_img, save_folder, specify_display_angle=False, display_all_candidates=False, plot_num=16):
    '''
    输入：
    - retrieval_results: 
        里面是每个物体检索出来的top10相似的模型库图片, 单个list的内容是[资产名称,视角id,相似度], 如 "shelf_1": [["SM_MetalShelf_200b", 0, 0.489642],["SM_MetalShelf_100a", 0, 0.469615],
    - assets_imgs_folder: 
        里面是所有资产的渲染图片, 如"AirConditioning1_back_right_below.png"
    - assets_imgs_order_pkl_path: 
        这个list代表选取了162个位姿渲染图中的哪几个来提取embedding  如: ['000001.png', '000002.png', '000003.png']
    
    作用：
        为retrieval_results中的所有物体, 将top 1相似的那个资产的视角图片取出来, 和物体图片拼在一起。16个物体结果为一组, 按4*4拼成一张图保存;
        
    输出：
        保存检索结果图
    '''
    with open(assets_imgs_order_pkl_path, 'rb') as file:
        assets_order_list = pickle.load(file)
        
    detected_item_list = list(retrieval_results.keys())
    images_pil = []

    for item in detected_item_list:
        mask_path = os.path.join(masks_folder, f'{item}_mask.png')
        
        if not os.path.exists(mask_path):
            logger.info(f'掩码文件不存在：{mask_path}')
            continue

        mask_img = Image.open(mask_path).convert('L')
        bbox = mask_img.getbbox()
        
        if bbox:
            cropped_mask = mask_img.crop(bbox)
            cropped_img = ori_scene_img.crop(bbox)
            mask_array = np.array(cropped_mask) / 255.0
            cropped_array = np.array(cropped_img)
            mask_array = mask_array[:, :, np.newaxis]
            masked_array = cropped_array * mask_array
            masked_img = Image.fromarray(masked_array.astype(np.uint8))

        if not retrieval_results[item]:
            logger.info(f'以下物品未检索到，或无需检索：{item}')
            continue

        top_result = retrieval_results[item][0]
        asset_name, view_id, _, _, _, _ = top_result

        asset_image_name = assets_order_list[1] if specify_display_angle else assets_order_list[view_id]
        assets_imgs_folder = os.path.join(assets_render_result_folder, asset_name)
        asset_image_path = os.path.join(assets_imgs_folder, asset_image_name)
        
        if not os.path.exists(asset_image_path):
            logger.info(f'资产图像不存在：{asset_image_path}')
            continue

        asset_img = Image.open(asset_image_path)

        resized_masked_img = resize_and_center(masked_img)
        resized_asset_img = resize_and_center(asset_img)

        images_pil.append(resized_masked_img)
        images_pil.append(resized_asset_img)

        if display_all_candidates:
            candidates_folder = os.path.join(save_folder, 'retrieval_candidates')
            os.makedirs(candidates_folder, exist_ok=True)
            candidate_images = [resized_masked_img]

            for candidate in retrieval_results[item][:10]:
                candidate_name, candidate_view_id, _, _, _, _ = candidate
                candidate_image_name = assets_order_list[candidate_view_id]
                assets_imgs_folder = os.path.join(assets_render_result_folder, candidate_name)
                candidate_image_path = os.path.join(assets_imgs_folder, candidate_image_name)
                
                if os.path.exists(candidate_image_path):
                    candidate_img = Image.open(candidate_image_path)
                    resized_candidate_img = resize_and_center(candidate_img)
                    candidate_images.append(resized_candidate_img)

            fig, axes = plt.subplots(1, len(candidate_images), figsize=(15, 5))
            for ax, img in zip(axes, candidate_images):
                ax.imshow(img)
                ax.axis('off')

            candidate_save_path = os.path.join(candidates_folder, f'{item}_candidates.png')
            plt.tight_layout()
            plt.savefig(candidate_save_path)
            plt.close()
            logger.info(f'候选图像保存在 {candidate_save_path}')

    num_items = len(images_pil) // 2
    grid_size = 4
    plot_num = grid_size * grid_size // 2

    for i in range(0, num_items, plot_num):
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()

        for j in range(plot_num):
            if i + j < num_items:
                axes[2 * j].imshow(images_pil[2 * (i + j)])
                axes[2 * j].axis('off')
                axes[2 * j + 1].imshow(images_pil[2 * (i + j) + 1])
                axes[2 * j + 1].axis('off')

        plt_save_path = os.path.join(save_folder, f'retrieval_{i // plot_num + 1}.png')
        plt.tight_layout()
        plt.savefig(plt_save_path)
        plt.close()
        logger.info(f'检索可视化图片保存在: {plt_save_path}')

def calculate_size_difference(estimated_size, fbx_size):
    eps = 1e-3
    # Compute all pairwise products for estimated_size
    estimated_products = [
        estimated_size[0] * estimated_size[1],
        estimated_size[0] * estimated_size[2],
        estimated_size[1] * estimated_size[2]
    ]
    # Compute all pairwise products for fbx_size
    fbx_products = [
        fbx_size[0] * fbx_size[1],
        fbx_size[0] * fbx_size[2],
        fbx_size[1] * fbx_size[2]
    ]
    # Find maximum products
    max_estimated_product = max(estimated_products)
    max_fbx_product = max(fbx_products)
  
    # Check if max_product < 0.16
    if max_estimated_product < 0.16 and max_fbx_product < 0.16:
        return 0
    elif max_estimated_product < 0.16  and max_fbx_product > 0.16:
        volume_difference = abs(max_fbx_product - max_estimated_product) / 0.16
        size_difference = volume_difference
        return size_difference
    else:
        # estimated_size[2] and fbx_size[2] are heights
        height_estimated = estimated_size[2]
        height_fbx = fbx_size[2]
        # estimated_size[0:2] and fbx_size[0:2] need to be sorted
        lw_estimated = sorted(estimated_size[0:2])  # [width, length]
        lw_fbx = sorted(fbx_size[0:2])  # [width, length]
        # Now compare their differences
        height_difference = abs(height_estimated - height_fbx) / max(height_estimated, eps)
        # Check if l and w in estimated_size differ by a factor of 10
        ratio = lw_estimated[1] / max(lw_estimated[0], eps)  # length / width
        if ratio >= 10:
            # Only compute length difference
            length_estimated = lw_estimated[1]
            length_fbx = lw_fbx[1]
            length_difference = abs(length_estimated - length_fbx) / max(length_estimated, eps)
            # Combine height and length differences
            size_difference = (height_difference + length_difference) / 2
            return size_difference
        else:
            # Compute area difference for length and width
            area_estimated = lw_estimated[0] * lw_estimated[1]
            area_fbx = lw_fbx[0] * lw_fbx[1]
            area_difference = abs(area_estimated - area_fbx) / max(area_estimated, eps)
            # Combine height and area differences
            size_difference = (height_difference + area_difference) / 2
            return size_difference

def delete_certain_items(scene_graph_result, retrieval_result, portfolio_assets):
    
    scene_graph_result_for_final_layout = {}
    retrieval_result_for_final_layout = {}
    
    for key, value in scene_graph_result.items():
        if not value: continue
        if not retrieval_result.get(value['supported']):  
            scene_graph_result_for_final_layout[key] = value
            continue
        parent_asset_name = retrieval_result.get(value['supported'])[0][0] if value['supported'] in retrieval_result else None
        
        if parent_asset_name not in portfolio_assets:
            scene_graph_result_for_final_layout[key] = value
    
    for key in retrieval_result:
        if key in scene_graph_result_for_final_layout or re.match(r'(ground|wall|ceiling|floor)_\d+', key):
            retrieval_result_for_final_layout[key] = retrieval_result[key]
    
    return scene_graph_result_for_final_layout, retrieval_result_for_final_layout

def reorder_certain_items_for_consistency_based_on_scene_graph(retrieval_result, scene_graph):
    '''
    基于scene graph中已有的group信息来重构物体一致性功能
    对于同一group的物体，使其使用相同的model，确保后续scale也相同
    
    Args:
        retrieval_result: 检索结果字典
        scene_graph: 场景图，包含group信息
    
    Returns:
        tuple: (更新后的检索结果, 更新后的场景图)
    '''

    retrieval_result_for_final_layout = copy.deepcopy(retrieval_result)
    
    # 基于scene graph中的group信息进行分组处理
    # 1. 收集所有具有group信息的物体
    groups_dict = defaultdict(list)  # group_id -> [labels]
    
    for label, properties in scene_graph.items():
        if 'group' in properties and properties['group']:
            group_id = properties['group']
            groups_dict[group_id].append(label)
    
    # 2. 对每个group进行处理
    for group_id, group_labels in groups_dict.items():
        if len(group_labels) < 2:
            continue

        # 统计该组所有物体的模型得分
        model_scores = defaultdict(lambda: [0, 0])  # (总分, 出现次数)
        valid_labels = []
        
        for label in group_labels:
            if label in retrieval_result:
                valid_labels.append(label)
                for entry in retrieval_result[label]:
                    model_id = entry[0]
                    score = entry[5]
                    model_scores[model_id][0] += score
                    model_scores[model_id][1] += 1
        
        if not valid_labels:
            continue
        
        # 选择最佳模型
        best_model = None
        max_score = -1
        for model_id, (total, count) in model_scores.items():
            avg = total / count
            combined_score = avg * np.log(count + 1)  # 平衡平均分和出现次数
            if combined_score > max_score:
                max_score = combined_score
                best_model = model_id
        
        if not best_model:
            continue
        
        # 更新检索结果 - 将最佳模型设置为第一选择
        for label in valid_labels:
            best_entries = [str(best_model), 0, 1.0, 0.0, 0.2, 1.2]
            original = retrieval_result_for_final_layout[label]
            retrieval_result_for_final_layout[label] = [best_entries] + original
        
        logger.info(f'Group {group_id}: {valid_labels}, 统一使用model: {best_model}')
    
    logger.info(f'基于scene graph的group信息完成物体一致性处理，共处理了 {len(groups_dict)} 个组')
            
    return retrieval_result_for_final_layout

def inference(input_folder, fbx_csv_path, asset_embedding_folder, assets_render_result_folder, save_folder, processor, embedding_model, debug_mode=False):
    start_time = time.time()
    os.makedirs(save_folder, exist_ok=True)
    ori_scene_img_path = os.path.join(input_folder, 'ori.png')
    masks_folder = os.path.join(input_folder, 'masks')
    # pcd_obb_data_dict = json.load(open(os.path.join(save_folder,'pcd_obb_data_refine.json'), 'r'))
    pcd_obb_data_dict = json.load(open(os.path.join(save_folder,'pcd_obb_data.json'), 'r'))
    
    df = pd.read_csv(fbx_csv_path, skiprows=0)
    # 过滤掉有问题的asset
    df = df[df['state'] != 'abort']

    # a new dict for asset_name -> class_en
    df['class_en_normalized'] = df['class_en'].apply(lambda x: str(x).replace('-', '_').replace(' ', '_').lower())
    asset_to_class_en_dict = df.set_index('name_en')['class_en_normalized'].to_dict()

    # 统计每个等价类对应的检索等价类
    equivalence_retrieval = df.groupby('class_en')['retrieval_class_en'].first().reset_index()
    equivalence_retrieval['class_en'] = equivalence_retrieval['class_en'].apply(lambda x: str(x).replace('-', '_').replace(' ', '_').lower())
    class_to_retrieval_class_dict = equivalence_retrieval.set_index('class_en')['retrieval_class_en'].to_dict() # dict  每个等价类对应唯一一个检索等价类

    # Ensure retrieval classes are also in the mapping (mapping to themselves)
    # This handles cases where the detected label matches a retrieval class directly (e.g. 'bottle' -> 'Bottle')
    unique_retrieval_classes = df['retrieval_class_en'].dropna().unique()
    for r_class in unique_retrieval_classes:
        normalized_r_class = str(r_class).replace('-', '_').replace(' ', '_').lower()
        if normalized_r_class not in class_to_retrieval_class_dict:
             class_to_retrieval_class_dict[normalized_r_class] = r_class

    # 统计每个检索等价类对应的模型列表
    retrieval_model_list = df.groupby('retrieval_class_en')['name_en'].apply(list).reset_index()
    retrieval_class_to_model_list_dict = retrieval_model_list.set_index('retrieval_class_en')['name_en'].to_dict()    # dict  每个检索等价类对应多个model

    # 统计fbx_size_dict
    fbx_en_name_list = df['name_en'].tolist()
    bbox_size_list = df['bbx'].tolist()
    eps = 1e-3
    fbx_size_dict = {}
    for class_en, bbox_size_string in zip(fbx_en_name_list, bbox_size_list):
        if bbox_size_string:
            sizes = [float(size) for size in bbox_size_string.split(',')]
            # Apply eps to avoid zero dimensions
            sizes = [s if s > eps else eps for s in sizes]
            fbx_size_dict[str(class_en)] = sizes
        else:
            fbx_size_dict[str(class_en)] = None
    
    equivalence_retrieval = df.groupby('class_en')['retrieval_class_en'].unique().reset_index()
    equivalence_retrieval['class_en'] = equivalence_retrieval['class_en'].apply(lambda x: str(x).replace('-', '_').replace(' ', '_').lower())
    # category_to_retrieval_category_dict = equivalence_retrieval.set_index('class_en')['retrieval_class_en'].to_dict() # dict  每个等价类对应唯一一个检索等价类

    portfolio_assets_df = df[df['portfolio_assets'] == 1]
    portfolio_assets = portfolio_assets_df['name_en'].tolist()
    
    time_prepare = time.time()
    logger.info(f"准备工作共花费时长为 {time_prepare - start_time}s.")
    
    # 1. 首先将所有场景中检测的物体所属检索等价类对应的所有models找出来
    detected_item_list = list(pcd_obb_data_dict.keys())
    item_to_candidate_models_dict = {}
    for item_name in detected_item_list:
        item_class_name = (re.sub(r'_\d+$', '', item_name)).replace('-', '_').replace(' ', '_').lower() # 删除item_name末尾的 _{int}
        
        item_retrieval_class = class_to_retrieval_class_dict.get(item_class_name, None)
        if not item_retrieval_class: raise KeyError(f"无法找到物品label的等价类, 检查是否是gpt输出的item名称不在等价类列表中; got item_name: {item_name} item_class_name: {item_class_name}")
        
        item_to_candidate_models_dict[item_name] = retrieval_class_to_model_list_dict[item_retrieval_class]
        logger.info(f'{item_name} 对应的等价类是{item_class_name}, 检索等价类是 {item_retrieval_class}, 该检索等价类对应的model共{len(item_to_candidate_models_dict[item_name])}个, 如: {item_to_candidate_models_dict[item_name][0]} ...')
        
    # 2. 按detected_item_list的顺序找出所有对应物体的分割图,裁剪出最小包围框的图,拼在一起按batch送入dinov2提取query embedding
    ori_scene_img = Image.open(ori_scene_img_path)
    all_detected_item_images_features = extract_all_detected_item_images_features(detected_item_list, ori_scene_img, masks_folder, processor, embedding_model, mask_query_pixels_ratio=0.0025, batchsize=16)   # 假设len(detected_item_list) = 45  那么输出为torch.Size([45, 1024])  这些特征和detected_item_list同样顺序
    logger.info(f'extract_all_detected_item_images_features 完成!')
    time_extract = time.time()
    logger.info(f"场景物体分割图的特征提取共花费时长为 {time_extract - time_prepare}s.")
    
    # 3. 并行检索
    estimated_size_dict = {k: v['obb_size'] for k, v in pcd_obb_data_dict.items()}
    final_results = cal_embedding_sim(detected_item_list, all_detected_item_images_features, item_to_candidate_models_dict, asset_embedding_folder, fbx_size_dict, estimated_size_dict, asset_to_class_en_dict, alpha=0.1)
    time_cal = time.time()
    logger.info(f"{len(detected_item_list)}个物体的检索总共花费时长为 {time_cal - time_extract}s.")

    # 加入墙 地板 天花板的label名字，倒不用给他们分配具体模型, "wall_0": []这样就行
    all_items = json.load(open(os.path.join(input_folder,'result.json'), 'r'))['categorys']
    for item in all_items:
        if item not in final_results and re.match(r'(ground|wall|ceiling|floor)_\d+', item):
            final_results[item] = []
    # 此时retrieval_results中包含了墙 地板 天花板
    retrieval_results_save_path = os.path.join(save_folder, 'retrieval_results.json')
    with open(retrieval_results_save_path, 'w', encoding='utf-8') as json_file:
        json.dump(final_results, json_file, ensure_ascii=False, indent=2)
    
    time_ori_save = time.time()
    logger.info(f"到保存初始文件总共花费时长为 {time_ori_save - time_cal}s.")

    # 5. 删除一些物体  如果某个物体的parent是组合资产  那就直接把他们删除，更新scene graph和retrieval_results的结果
    scene_graph_result = json.load(open(os.path.join(input_folder,'scene_graph_result.json'), 'r'))

    scene_graph_result_after_del, retrieval_result_after_del = delete_certain_items(scene_graph_result, final_results, portfolio_assets)
    
    # 6. 基于scene graph中的group信息，对于同一group的物体使用相同的model，确保后续scale一致
    retrieval_result_for_final_layout = reorder_certain_items_for_consistency_based_on_scene_graph(retrieval_result_after_del, scene_graph_result_after_del)
    
    scene_graph_result_for_final_layout_save_path = os.path.join(input_folder,'scene_graph_result_final.json')
    retrieval_result_for_final_layout_save_path = retrieval_results_save_path.replace('.json', '_final.json')
    with open(scene_graph_result_for_final_layout_save_path, 'w', encoding='utf-8') as json_file:
        json.dump(scene_graph_result_after_del, json_file, ensure_ascii=False, indent=2)
    with open(retrieval_result_for_final_layout_save_path, 'w', encoding='utf-8') as json_file:
        json.dump(retrieval_result_for_final_layout, json_file, ensure_ascii=False, indent=2)
    logger.info(f"scene_graph_result_for_final_layout saved to {scene_graph_result_for_final_layout_save_path}")
    logger.info(f"retrieval_result_for_final_layout saved to {retrieval_result_for_final_layout_save_path}")
    
    time_6 = time.time()
    logger.info(f" time_7 - time_6 耗时{time_6 - time_6}s.")

    # 7. 显示检索匹配结果
    assets_imgs_order_pkl_path = os.path.join(asset_embedding_folder, 'assets_imgs_order.pkl')
    visualize_retrieval_results(retrieval_result_for_final_layout, assets_render_result_folder, assets_imgs_order_pkl_path, masks_folder, ori_scene_img, save_folder, specify_display_angle=True, display_all_candidates=debug_mode)
    time_vis = time.time()
    logger.info(f" 显示检索匹配结果 耗时{time_vis - time_6}s.")
    logger.info(f" Stage_2 资产检索阶段总耗时 {time_vis - start_time}s.")
    
def cal_pcd_obb_data(input_folder, save_folder):
    #step1: point cloud obj bbox fitting
    start_time = time.time()

    if os.path.exists(os.path.join(save_folder, 'pcd_obb_data.json')):
        logger.info(f'{save_folder} 已生成过, 跳过')
        return
    
    #load scene graph 
    with open(os.path.join(input_folder, 'scene_graph_result.json'), 'r') as f:
        scene_graph_result = json.load(f)
    depth_image_path = os.path.join(input_folder.replace('S1_scene_parsing_results','S0_geometry_pred_results'), 'depth.png')
    target_img_path = os.path.join(input_folder, 'ori.png')
    wall_floor_pose = json.load(open(os.path.join(input_folder, 'floor_walls_pose.json'), 'r'))

    obj_key_list = list(scene_graph_result.keys())
    item_mask_path_list= []
    for name in obj_key_list:
        item_mask_path = os.path.join(input_folder, f'masks/{name}_mask.png')
        item_mask_path_list.append(item_mask_path)
    pred_ply_save_path = os.path.join(save_folder, f'pcd_obj_bbox_fitting.ply')
    pcb_center_xyz_list, pcb_obb_rotation_list, pcd_obb_size_list, obb_points_list = cal_and_visualize_scene_obj_bbox_fitting(depth_image_path, \
                                        item_mask_path_list, target_img_path, pred_ply_save_path, wall_floor_pose, scene_graph_result)

    pcb_center_xyz_dict = {a:np.array(b) for a,b in zip(obj_key_list, pcb_center_xyz_list)}
    pcb_obb_rotation_dict = {a:np.array(b) for a,b in zip(obj_key_list, pcb_obb_rotation_list)}
    pcd_obb_size_dict = {a:np.array(b) for a,b in zip(obj_key_list, pcd_obb_size_list)}
    obb_points_dict = {a:np.array(b) for a,b in zip(obj_key_list, obb_points_list)}

    logger.info(f"Stage_1 计算所有物体的obb位姿矩阵花费时间为 {time.time() - start_time}s.") 

    # 假设你的数据已经计算完成
    obb_data = {}
    for item_name in obj_key_list:
        obb_data[item_name] = {
            "center_xyz": pcb_center_xyz_dict[item_name].tolist(),
            "obb_rotation": pcb_obb_rotation_dict[item_name].tolist(),
            "obb_size": pcd_obb_size_dict[item_name].tolist(),
            "obb_points_dict": obb_points_dict[item_name].tolist(),
        }
    json_save_path = os.path.join(save_folder, f'pcd_obb_data.json')
    with open(json_save_path, 'w') as json_file:
        json.dump(obb_data, json_file, indent=2)
    logger.info(f"数据已保存到 {json_save_path}")
