"""
Module: Transformations Estimation (Step 8 - Pose Estimation)
模块：变换估计 (步骤 8 - 姿态估计)

Fully migrated from S3_pose_inference_op.py
"""
import os
import json
import torch
import numpy as np

from core.context import Context
from utils.io import load_json, save_json

# Import legacy S3 functions
from modules._s3_legacy_functions import (
    detect_truncated_objects,
    combine_scene_objects_pose,
    inference_obj_pose as s3_inference_obj_pose
)

class PoseModule:
    """
    Pose Estimation Module
    姿态估计模块
    
    Performs:
    - Rotation estimation (view matching)
    - Translation estimation (OBB alignment)
    - Scale estimation (volume optimization)
    """
    def __init__(self, context: Context):
        self.context = context
        self.logger = context.logger
        self.cfg = context.config.get('S3_pose_inference', {})
        self.shared_cfg = context.config.shared
        
    def run(self):
        """
        Main execution: Pose estimation
        """
        self.logger.info(">>> Stage 4: Pose Estimation")
        
        S1_folder = os.path.join(self.context.output_dir, 'S1_scene_parsing_results')
        S2_folder = os.path.join(self.context.output_dir, 'S2_3d_retrieval_results')
        save_dir = os.path.join(self.context.output_dir, 'S3_pose_inference')
        os.makedirs(save_dir, exist_ok=True)
        
        scene_name = self.context.image_name
        
        # Smart resume: Check if S3 placement info file exists
        placement_info_path = os.path.join(save_dir, f'{scene_name}_placement_info.json')
        
        if not self.context.clean_mode and os.path.exists(placement_info_path):
            self.logger.info(f"✓ S3 已完成：所有必需文件都存在，跳过此阶段")
            self.logger.info(f"  - {scene_name}_placement_info.json: ✓")
            # Store path in context for next stage
            self.context.set_data('placement_info_path', placement_info_path)
            self.logger.info("Pose Estimation Done (Skipped, placement info file exists).")
            return
        
        # 1. Load Data
        depth_image_path = os.path.join(self.context.output_dir, 'S0_geometry_pred_results/depth.png')
        retrieval_dict = self.context.get_data('retrieval_results')
        if not retrieval_dict:
             retrieval_dict = load_json(os.path.join(S2_folder, 'retrieval_results_final.json'))
             
        obb_data_path = os.path.join(S2_folder, 'pcd_obb_data.json')
        loaded_obb_data = load_json(obb_data_path)
        
        # 2. Load Shared AENet Model (reuses DINOv2 from S2)
        ae_net = self.context.get_ae_net(self.cfg.ae_net_weights_path)
        
        # 3. Set legacy globals & Patch model loader
        import modules._s3_legacy_functions as s3_legacy
        s3_legacy.logger = self.logger
        
        # Monkey-patch load_ae_net to return our shared model
        original_load_ae_net = s3_legacy.load_ae_net if hasattr(s3_legacy, 'load_ae_net') else None
        s3_legacy.load_ae_net = lambda *args, **kwargs: ae_net
        
        try:
            # 4. Run Inference
            self.logger.info("Running pose inference with shared AE Net...")
            predictions_id_result, comparison_images = s3_inference_obj_pose(
                S1_folder, 
                self.cfg.template_dir, 
                depth_image_path, 
                retrieval_dict, 
                loaded_obb_data, 
                self.cfg.ae_net_weights_path,  # Path (ignored by patch)
                self.cfg.ori_dino_weights_path,  # Path (ignored by patch)
                save_dir,
                use_homography=self.cfg.get('use_homography', True),
                save_pts_match_imgs=self.context.debug_mode,
                save_comparison_imgs=self.context.debug_mode
            )
        finally:
            # Restore original
            if original_load_ae_net:
                s3_legacy.load_ae_net = original_load_ae_net
        
        # 5. Save & Vis
        if comparison_images:
            from utils.image_concat import stitch_images_grid
            stitch_images_grid(save_dir, os.path.join(save_dir, 'pose_prediction_stitched.png'), comparison_images)
            
        # 6. Placement Info
        wall_floor_pose = load_json(os.path.join(S1_folder, 'floor_walls_pose.json'))
        scene_graph_result = load_json(os.path.join(S1_folder, 'scene_graph_result_final.json'))
        truncated_info = detect_truncated_objects(S1_folder)
        
        save_path = os.path.join(save_dir, f'{scene_name}_placement_info.json')
        combine_scene_objects_pose(
            predictions_id_result, 
            wall_floor_pose, 
            scene_graph_result, 
            retrieval_dict, 
            truncated_info, 
            save_path
        )
        
        self.context.set_data('placement_info_path', save_path)
        self.logger.info(f"Pose Estimation Done. Saved to {save_path}")
        
        # 7. Cleanup S3 resources to free VRAM for Blender (S4)
        self.context.release_models()