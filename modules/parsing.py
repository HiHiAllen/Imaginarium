"""
Module: Semantic Parsing (Step 4 - Foreground Objects Semantic Parsing)
模块：语义解析 (步骤 4 - 前景物体语义解析)

Fully migrated from S1_scene_parsing_op.py
All legacy functions are imported from _s1_legacy_functions
"""
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image

from core.context import Context

# Import all legacy S1 functions
from modules._s1_legacy_functions import run_scene_parsing_pipeline


class SemanticParsingModule:
    """
    Semantic Parsing Module
    语义解析模块
    
    Performs:
    - Object detection (Grounding DINO)
    - Segmentation (SAM)
    - Floor/Wall extraction (RANSAC)
    - Scene graph construction
    """
    def __init__(self, context: Context):
        self.context = context
        self.logger = context.logger
        self.cfg = context.config.get('S1_scene_parsing', {})
        self.shared_cfg = context.config.shared
        
        # Load Asset DB
        self.df = pd.read_csv(self.shared_cfg.fbx_csv_path, skiprows=0)
        
        # Set global vars for legacy functions
        self._init_legacy_globals()
        
    def _init_legacy_globals(self):
        """
        Initialize global variables required by legacy S1 functions.
        """
        import modules._s1_legacy_functions as s1_legacy
        s1_legacy.gpt = self.context.gpt
        s1_legacy.FONT_TTF_PATH = self.shared_cfg.font_ttf_path
        s1_legacy.GROUND_DINO_TOKEN = self.shared_cfg.ground_dino_token
        s1_legacy.logger = self.logger
    
    def run(self):
        """
        Main execution: Scene parsing
        """
        self.logger.info(">>> Stage 2: Semantic Parsing")
        
        save_folder = os.path.join(self.context.output_dir, 'S1_scene_parsing_results')
        os.makedirs(save_folder, exist_ok=True)
        
        # Smart resume: Check if all required S1 files exist
        required_files = [
            'scene_graph_result.json',
            'floor_walls_pose.json',
            'final_detect_items.pkl',
            'masks.pkl'
        ]
        
        if not self.context.clean_mode:
            all_exist = all(os.path.exists(os.path.join(save_folder, f)) for f in required_files)
            if all_exist:
                self.logger.info(f"✓ S1 已完成：所有必需文件都存在，跳过此阶段")
                for f in required_files:
                    self.logger.info(f"  - {f}: ✓")
                self.logger.info("Semantic Parsing Done (Skipped, all required files exist).")
                return
        
        # Get data from Context (memory transfer from S0)
        image_np = self.context.get_data('ori_image_numpy')
        depth_image_path = os.path.join(self.context.output_dir, 'S0_geometry_pred_results/depth.png')
        
        if image_np is None:
            # Fallback: load from disk
            image_np = np.array(Image.open(self.context.image_path).convert("RGB"))
        
        # Inject debug flag into cfg (params)
        from omegaconf import OmegaConf, DictConfig
        if isinstance(self.cfg, DictConfig):
            try:
                OmegaConf.set_struct(self.cfg, False)
                self.cfg.debug = self.context.debug_mode
                OmegaConf.set_struct(self.cfg, True)
            except Exception:
                # Fallback if set_struct fails or not supported
                self.cfg.debug = self.context.debug_mode
        else:
            self.cfg['debug'] = self.context.debug_mode

        # Call the main pipeline function (from legacy)
        self.logger.info("Delegating to legacy S1 parsing pipeline...")
        run_scene_parsing_pipeline(
            self.logger,
            image_np,
            depth_image_path,
            self.df,
            save_folder,
            self.cfg,
            self.context.gpt_params
        )
        
        self.logger.info("Semantic Parsing Done.")
