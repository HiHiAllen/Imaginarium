"""
Module: 3D Asset Retrieval (Step 7)
模块：3D 资产检索 (步骤 7)

Fully migrated from S2_3d_retrieval_op.py
"""
import os
import torch
import numpy as np
import pickle
import json
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from glob import glob
import re

from core.context import Context
from utils.io import load_json, save_json

# Import legacy S2 functions
from modules._s2_legacy_functions import (
    cal_pcd_obb_data,
    refine_dimensions_with_vlm,
    inference as s2_inference
)

class TextureRetrieval:
    def __init__(self, texture_embeddings_path, processor, model, device, logger=None):
        self.texture_embeddings_path = texture_embeddings_path
        self.processor = processor
        self.model = model
        self.device = device
        self.logger = logger
        self.embeddings = self._load_embeddings()
        
    def _load_embeddings(self):
        if not os.path.exists(self.texture_embeddings_path):
            raise FileNotFoundError(f"Texture embeddings not found at {self.texture_embeddings_path}")
        
        # 获取文件大小
        file_size_mb = os.path.getsize(self.texture_embeddings_path) / (1024 * 1024)
        if self.logger:
            self.logger.info(f"  正在加载纹理embedding文件 ({file_size_mb:.1f} MB)，请稍候...")
        else:
            print(f"  正在加载纹理embedding文件 ({file_size_mb:.1f} MB)，请稍候...")
        
        import time
        start_time = time.time()
        
        with open(self.texture_embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        elapsed_time = time.time() - start_time
        if self.logger:
            self.logger.info(f"  ✓ 纹理embedding加载完成 (耗时 {elapsed_time:.2f}s)")
        else:
            print(f"  ✓ 纹理embedding加载完成 (耗时 {elapsed_time:.2f}s)")
        
        return embeddings

    def _extract_feature(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use Patch tokens (skip CLS)
            # Shape: [1, 256, 1024]
            feature = outputs.last_hidden_state[:, 1:, :]
        return feature.cpu()

    def _compute_patch_mask(self, image, patch_size=16):
        """
        计算每个patch的mask权重。
        DINOv2将224x224的图片分成16x16=256个patches。
        对于RGBA图片，根据alpha通道计算每个patch的有效像素比例。
        """
        # DINOv2默认输入尺寸
        input_size = 224
        
        if image.mode == 'RGBA':
            # 获取alpha通道并resize到224x224
            alpha = image.split()[3]
            alpha_resized = alpha.resize((input_size, input_size), Image.BILINEAR)
            alpha_arr = np.array(alpha_resized) / 255.0  # 归一化到0-1
            
            # 计算每个patch的平均alpha值作为权重
            # patches: 16x16 grid, each patch is 14x14 pixels (224/16=14)
            patch_dim = input_size // patch_size
            weights = np.zeros((patch_size, patch_size))
            
            for i in range(patch_size):
                for j in range(patch_size):
                    patch_alpha = alpha_arr[i*patch_dim:(i+1)*patch_dim, j*patch_dim:(j+1)*patch_dim]
                    weights[i, j] = patch_alpha.mean()
            
            # 展平为 [256] 的向量
            weights = weights.flatten()
            return torch.from_numpy(weights).float()
        else:
            # 非RGBA图片，所有patch权重为1
            return torch.ones(patch_size * patch_size)

    def _compute_color_hist(self, image):
        try:
            # Handle RGBA to extract alpha before converting to HSV
            mask = None
            if image.mode == 'RGBA':
                alpha = np.array(image.split()[-1])
                mask = alpha > 10
            elif 'A' in image.getbands():
                 alpha = np.array(image.getchannel('A'))
                 mask = alpha > 10
            
            # Convert to HSV
            img_hsv = image.convert('HSV')
            arr = np.array(img_hsv)

            if mask is not None:
                pixels = arr[mask]
            else:
                pixels = arr.reshape(-1, 3)
                
            if len(pixels) == 0:
                return np.zeros(128)

            # Quantize: H(8), S(4), V(4) -> 128 bins
            h = pixels[:, 0]
            s = pixels[:, 1]
            v = pixels[:, 2]
            
            h_idx = np.clip(h // 32, 0, 7)
            s_idx = np.clip(s // 64, 0, 3)
            v_idx = np.clip(v // 64, 0, 3)
            
            idx = h_idx * 16 + s_idx * 4 + v_idx
            
            hist = np.bincount(idx, minlength=128)
            hist = hist / (hist.sum() + 1e-8)
            return hist
        except Exception as e:
            print(f"Error computing color hist: {e}")
            return np.zeros(128)

    def retrieve_texture(self, category, crop_image, debug_dir=None):
        """
        Find the best matching texture for a given category and image crop.
        Uses a Local-Global similarity approach + Color Histogram:
        1. Global: Cosine sim of average patch features (mask-weighted for RGBA).
        2. Local: Average of max patch similarities (MaxSim).
        3. Color: HSV Histogram intersection.
        """
        if category not in self.embeddings:
            return None, -1.0
        
        # ======= DEBUG: 打印embedding对应的贴图图片 =======
        if category == 'wall' and not hasattr(self, '_debug_printed_wall'):
            # print(f"\n  [DEBUG] Wall texture candidates in embedding ({len(self.embeddings[category])} total):")
            # for idx, (filename, data) in enumerate(self.embeddings[category].items()):
            #    print(f"    {idx+1}. {filename} -> {data['path']}")
            self._debug_printed_wall = True

        # 计算patch mask权重
        patch_weights = self._compute_patch_mask(crop_image)  # [256]
        query_patches = self._extract_feature(crop_image) # [1, N_q, D]
        query_patches = F.normalize(query_patches, p=2, dim=2) # Normalize along dim
        
        # Global query feature: Mask-weighted Average Pooling
        # 使用mask权重来加权平均patch特征
        patch_weights_expanded = patch_weights.unsqueeze(0).unsqueeze(2)  # [1, 256, 1]
        weighted_patches = query_patches * patch_weights_expanded  # [1, 256, D]
        query_global = weighted_patches.sum(dim=1) / (patch_weights.sum() + 1e-8)  # [1, D]
        query_global = F.normalize(query_global, p=2, dim=1)
        
        # Color feature
        query_hist = self._compute_color_hist(crop_image)
        
        best_score = -float('inf')
        best_texture_path = None
        
        # Weights
        w_global = 0.3
        w_local = 0.3
        w_color = 0.4
        
        # ======= DEBUG: 收集所有分数用于排序 =======
        all_scores = []
        # ======= END DEBUG =======
        
        for filename, data in self.embeddings[category].items():
            target_patches = data['embedding'] # [1, N_t, D]
            if target_patches.dim() == 2:
                # Handle case where embedding might be saved as [N_t, D] or [1, D] from previous versions
                target_patches = target_patches.unsqueeze(0)
            
            target_patches = F.normalize(target_patches, p=2, dim=2)
            
            # 1. Global Score
            target_global = target_patches.mean(dim=1) # [1, D]
            target_global = F.normalize(target_global, p=2, dim=1)
            global_score = torch.mm(query_global, target_global.transpose(0, 1)).item()
            
            # 2. Local Score (MaxSim) - Enabled for ALL
            # Compute similarity matrix: [1, N_q, N_t]
            sim_matrix = torch.bmm(query_patches, target_patches.transpose(1, 2)) # [1, N_q, N_t]
            max_sim_per_query, _ = sim_matrix.max(dim=2) # [1, N_q]
            local_score = max_sim_per_query.mean().item()
            
            # 3. Color Score
            if 'color_hist' not in data:
                # Lazy load and cache
                if os.path.exists(data['path']):
                    try:
                        tgt_img = Image.open(data['path']).convert('RGB')
                        data['color_hist'] = self._compute_color_hist(tgt_img)
                    except Exception:
                        data['color_hist'] = np.zeros(128)
                else:
                    data['color_hist'] = np.zeros(128)
            
            tgt_hist = data['color_hist']
            color_score = np.sum(np.minimum(query_hist, tgt_hist))
                
            # Combined Score
            final_score = w_global * global_score + w_local * local_score + w_color * color_score
            
            # ======= DEBUG: 收集分数 =======
            if category in ['wall', 'ceiling', 'floor']:
                all_scores.append((filename, data['path'], final_score))
            # ======= END DEBUG =======
            
            if final_score > best_score:
                best_score = final_score
                best_texture_path = data['path']
        
        # ======= DEBUG: 打印Top-5匹配结果 =======
        if category in ['wall', 'ceiling', 'floor'] and all_scores:
            all_scores.sort(key=lambda x: x[2], reverse=True)
            # print(f"\n  [DEBUG] Wall texture matching scores (Top-3):")
            # for i, (fname, fpath, score) in enumerate(all_scores[:3]):
            #    print(f"    {i+1}. {fname}: {score:.4f}")
        # ======= END DEBUG =======
                
        return best_texture_path, best_score

    def _get_masked_crop(self, ori_image, mask_file):
        """
        Helper to crop image based on mask and return the crop and mask area.
        非mask区域设为透明（RGBA格式）。
        """
        try:
            mask = Image.open(mask_file).convert('L')
            bbox = mask.getbbox()
            if not bbox:
                return None, 0
                
            # Crop image and mask
            crop_img = ori_image.crop(bbox)
            crop_mask = mask.crop(bbox)
            
            # 计算mask区域面积
            mask_arr = np.array(crop_mask)
            mask_bool = mask_arr > 128
            area = np.count_nonzero(mask_bool)
            
            # 转换为RGBA，非mask区域设为透明
            img_rgba = crop_img.convert('RGBA')
            img_arr = np.array(img_rgba)
            img_arr[~mask_bool, 3] = 0  # 设置alpha通道为0（透明）
            masked_crop = Image.fromarray(img_arr, 'RGBA')
            
            return masked_crop, area
        except Exception as e:
            print(f"Error processing mask {mask_file}: {e}")
            return None, 0

    def process_scene(self, masks_folder, ori_image_path, output_path):
        """
        Process all wall/floor/ceiling masks in the folder and save retrieval results.
        All wall objects will share the same texture (best match among all walls).
        """
        if not os.path.exists(masks_folder) or not os.path.exists(ori_image_path):
            print(f"Warning: Masks folder or original image not found: {masks_folder}")
            return {}

        ori_image = Image.open(ori_image_path).convert("RGB")
        results = {}
        
        mask_files = glob(os.path.join(masks_folder, "*_mask.png"))
        
        wall_files = []
        other_files = []
        
        # 1. Categorize files
        for mask_file in mask_files:
            filename = os.path.basename(mask_file)
            name_without_ext = os.path.splitext(filename)[0]
            # name format: object_id_mask
            obj_name = name_without_ext.replace('_mask', '')
            
            if re.match(r'(wall)_\d+', obj_name):
                wall_files.append((obj_name, mask_file))
            elif re.match(r'(floor|ceiling)_\d+', obj_name):
                other_files.append((obj_name, mask_file))
        
        # 2. Handle Walls (Individual Retrieval -> Best Score)
        if wall_files:
            print(f"  发现 {len(wall_files)} 个墙面，正在检索最佳匹配纹理...")
            
            best_wall_texture = None
            best_wall_score = -float('inf')
            
            for obj_name, mask_file in tqdm(wall_files, desc="  检索墙面纹理"):
                crop, area = self._get_masked_crop(ori_image, mask_file)
                # Ignore very small crops
                if crop and area > 100:
                    texture_path, score = self.retrieve_texture('wall', crop)
                    if texture_path and score > best_wall_score:
                        best_wall_score = score
                        best_wall_texture = texture_path
            
            if best_wall_texture and best_wall_score >= 0.5:
                print(f"  ✓ 已选择最佳墙面纹理: {os.path.basename(best_wall_texture)} (分数: {best_wall_score:.4f})")
                # Assign to ALL walls
                for obj_name, _ in wall_files:
                    results[obj_name] = best_wall_texture
            else:
                print("  ✗ 墙面纹理检索失败 (无有效裁剪区域?)")
                
        # 3. Handle Others (Individual Texture)
        if other_files:
            print(f"  发现 {len(other_files)} 个地板/天花板，正在检索纹理...")
        
        for obj_name, mask_file in tqdm(other_files, desc="  检索地板/天花板纹理"):
            category = None
            if re.match(r'(floor)_\d+', obj_name):
                category = 'floor'
            elif re.match(r'(ceiling)_\d+', obj_name):
                category = 'ceiling'
                
            if category:
                crop, _ = self._get_masked_crop(ori_image, mask_file)
                if crop:
                    best_texture, score = self.retrieve_texture(category, crop)
                    if best_texture and score >= 0.5:
                        results[obj_name] = best_texture
                
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✓ 纹理检索完成，共检索到 {len(results)} 个对象的纹理")
            
        return results

class RetrievalModule:
    """
    3D Asset Retrieval Module
    3D 资产检索模块
    
    Performs:
    - OBB computation
    - Visual similarity matching (DINO)
    - Size compatibility filtering
    - VLM-based dimension refinement
    """
    def __init__(self, context: Context):
        self.context = context
        self.logger = context.logger
        self.cfg = context.config.get('S2_3d_retrieval', {})
        self.shared_cfg = context.config.shared
        
    def run(self):
        """
        Main execution: Asset retrieval
        """
        self.logger.info(">>> Stage 3: 3D Asset Retrieval")
        
        input_folder = os.path.join(self.context.output_dir, 'S1_scene_parsing_results')
        save_folder = os.path.join(self.context.output_dir, 'S2_3d_retrieval_results')
        os.makedirs(save_folder, exist_ok=True)
        
        # Smart resume: Check if all required S2 files exist
        required_files = [
            'pcd_obb_data.json',
            'retrieval_results_final.json',
            'texture_retrieval_results.json'
        ]
        
        if not self.context.clean_mode:
            all_exist = all(os.path.exists(os.path.join(save_folder, f)) for f in required_files)
            if all_exist:
                self.logger.info(f"✓ S2 已完成：所有必需文件都存在，跳过此阶段")
                for f in required_files:
                    self.logger.info(f"  - {f}: ✓")
                # Load results into context for next stage
                retrieval_res = load_json(os.path.join(save_folder, 'retrieval_results_final.json'))
                self.context.set_data('retrieval_results', retrieval_res)
                self.logger.info("Asset Retrieval Done (Skipped, all required files exist).")
                return
        
        # Set legacy globals
        import modules._s2_legacy_functions as s2_legacy
        s2_legacy.logger = self.logger
        
        # 1. OBB Data (Re-calc)
        self.logger.info("Computing OBB data...")
        cal_pcd_obb_data(input_folder, save_folder)
        
        # 2. VLM Refinement
        self.logger.info("Refining obb dimensions with VLM...")
        refine_dimensions_with_vlm(input_folder, save_folder, self.context.gpt_params)
        
        # 3. Retrieval (with Original DINOv2 Model - compatible with AENet)
        self.logger.info("Running retrieval with original DINOv2 model (will be reused in S3)...")
        processor = self.context.original_dino_processor
        model = self.context.original_dino_model_for_retrieval
        
        s2_inference(
            input_folder=input_folder,
            fbx_csv_path=self.shared_cfg.fbx_csv_path,
            asset_embedding_folder=self.cfg.asset_embedding_folder,
            assets_render_result_folder=self.shared_cfg.assets_render_result_folder,
            save_folder=save_folder,
            processor=processor,
            embedding_model=model,
            debug_mode=self.context.debug_mode
        )
        
        # 4. Texture Retrieval for Wall/Floor/Ceiling
        
        # # Old DINOv2-based retrieval logic (kept for reference)
        # # 下面是基于dinov2的背景贴图检索，效果不好；可能不如用clip做图文检索准
        # self.logger.info("Running texture retrieval for walls, floors, and ceilings...")
        # texture_embeddings_path = self.cfg.get(
        #     'texture_embeddings_path', 
        #     "asset_data/background_texture_dataset/texture_embeddings.pkl"
        # )
        # masks_folder = os.path.join(input_folder, 'masks')
        # ori_image_path = os.path.join(input_folder, 'ori.png')
        # texture_output_path = os.path.join(save_folder, 'texture_retrieval_results.json')
        
        # texture_retriever = TextureRetrieval(
        #     texture_embeddings_path=texture_embeddings_path,
        #     processor=processor,
        #     model=model,
        #     device=model.device,
        #     logger=self.logger
        # )
        # texture_retriever.process_scene(
        #     masks_folder=masks_folder,
        #     ori_image_path=ori_image_path,
        #     output_path=texture_output_path
        # )
        # self.logger.info(f"Texture retrieval results saved to {texture_output_path}")
        # # Store results in Context for Next Step
        # retrieval_res = load_json(os.path.join(save_folder, 'retrieval_results_final.json'))
        # self.context.set_data('retrieval_results', retrieval_res)
        
        # 4b. Texture Retrieval for Wall/Floor/Ceiling (VLM-based)
        texture_results = self.retrieve_textures_with_vlm(input_folder)
        
        # Map generic categories to specific object instances
        retrieval_res_path = os.path.join(save_folder, 'retrieval_results_final.json')
        if os.path.exists(retrieval_res_path):
            retrieval_res = load_json(retrieval_res_path)
        else:
            retrieval_res = {}
        
        # Build mapping from generic categories to specific object instances
        texture_results_mapped = {}
        import re
        for obj_name in retrieval_res.keys():
            category = None
            if re.match(r'^wall_\d+$', obj_name):
                category = 'wall'
            elif re.match(r'^floor_\d+$', obj_name):
                category = 'floor'
            elif re.match(r'^ceiling_\d+$', obj_name):
                category = 'ceiling'
                
            if category and category in texture_results:
                texture_results_mapped[obj_name] = texture_results[category]
        
        # Save texture results to separate file (with object instance names)
        texture_output_path = os.path.join(save_folder, 'texture_retrieval_results.json')
        save_json(texture_results_mapped, texture_output_path)
        self.logger.info(f"Texture retrieval results saved to {texture_output_path}")
        
        # Store results in Context for Next Step
        self.context.set_data('retrieval_results', retrieval_res)
        
        self.logger.info("Asset Retrieval Done.")

    def retrieve_textures_with_vlm(self, input_folder):
        """
        Use VLM to retrieve textures for walls, floors, and ceilings.
        """
        self.logger.info("Running texture retrieval with VLM...")
        
        dataset_root = self.shared_cfg.get('background_texture_dataset_path', "asset_data/background_texture_dataset")
        captions_path = os.path.join(dataset_root, "texture_captions.json")
        
        if not os.path.exists(captions_path):
            self.logger.warning(f"Texture captions not found at {captions_path}. Skipping VLM texture retrieval.")
            return {}
            
        try:
            with open(captions_path, 'r', encoding='utf-8') as f:
                captions = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load texture captions: {e}")
            return {}
            
        ori_image_path = os.path.join(input_folder, 'ori.png')
        if not os.path.exists(ori_image_path):
             self.logger.warning(f"Original image not found at {ori_image_path}")
             return {}
             
        # Resize image to 512x512
        try:
            image = Image.open(ori_image_path).convert("RGB")
            image = image.resize((512, 512))
        except Exception as e:
            self.logger.error(f"Failed to process image: {e}")
            return {}
        
        # Construct Prompt
        captions_text = json.dumps(captions, indent=2, ensure_ascii=False)
        
        prompt = (
            "You are an interior design expert. "
            "I will provide an image of a room and a list of available background textures with their descriptions. "
            "Please select the most suitable texture for the 'ceiling', 'floor', and 'wall'. "
            "For the wall, select one texture that fits all walls in the room. "
            "Consider the color and style of the room. "
            f"Available textures:\n{captions_text}\n\n"
            "Return a JSON object with keys 'ceiling', 'floor', 'wall' and the selected texture filename (key from the provided list) as values. "
            "Example: {\"ceiling\": \"ceiling/texture1.jpg\", \"floor\": \"floor/texture2.jpg\", \"wall\": \"wall/texture3.jpg\"}"
        )
        
        # Call VLM
        try:
            response = self.context.gpt.get_response(prompt, image=image)
            
            # Parse JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result_str = json_match.group()
                # Fix common JSON errors if any (e.g. single quotes)
                try:
                    result = json.loads(result_str)
                except json.JSONDecodeError:
                    # Try eval as fallback for single quotes
                    try:
                        import ast
                        result = ast.literal_eval(result_str)
                    except:
                        self.logger.error(f"Failed to parse VLM response JSON: {result_str}")
                        return {}
                
                self.logger.info(f"VLM Texture Selection: {result}")
                
                final_result = {}
                for cat in ['ceiling', 'floor', 'wall']:
                    if cat in result:
                         tex_path = result[cat]
                         full_path = os.path.join(dataset_root, tex_path)
                         if os.path.exists(full_path):
                             final_result[cat] = full_path
                         else:
                             self.logger.warning(f"Selected texture {tex_path} not found at {full_path}.")
                             
                return final_result
            else:
                self.logger.error("No JSON found in VLM response.")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error in VLM texture retrieval: {e}")
            return {}