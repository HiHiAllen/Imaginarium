"""
Module: Geometry Content Analysis (Step 5 - Depth Estimation & Point Cloud Generation)
模块：几何内容分析 (步骤 5 - 深度估计和点云生成)
"""
import os
import cv2
import torch
import numpy as np
import open3d as o3d
from core.context import Context

def estimate_focal_length(width):
    """
    Estimate focal length based on image width.
    UE5 camera: 30mm focal length, 36mm sensor size.
    """
    return int(30 / 36 * width)

def create_point_cloud(color_image, depth_image_mm, fx, fy, cx, cy):
    """
    Create an Open3D point cloud from color and depth images.
    
    Args:
        color_image: RGB image (numpy array)
        depth_image_mm: Depth image in millimeters (numpy array)
        fx, fy, cx, cy: Camera intrinsics
    
    Returns:
        o3d.geometry.PointCloud
    """
    rows, cols = depth_image_mm.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    
    # Convert depth from mm to meters
    z = depth_image_mm / 1000.0
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy
    
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color_image.reshape(-1, 3) / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


class GeometryModule:
    """
    Geometry Analysis Module
    几何分析模块
    
    Performs:
    - Depth estimation using Depth Anything V2
    - Point cloud generation
    - Camera intrinsics estimation
    """
    def __init__(self, context: Context):
        self.context = context
        self.logger = context.logger
        self.cfg = context.config.get('S0_geometry_pred', {})
        self.model = None
        
    def _load_model(self):
        """Load Depth Anything V2 model"""
        from models.depth_anything.dpt import DepthAnythingV2
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        encoder = self.cfg.get('encoder', 'vitl')
        max_depth = self.cfg.get('max_depth', 20)
        load_from = self.cfg.get('load_from')
        
        if not load_from or not os.path.exists(load_from):
            raise FileNotFoundError(f"Depth model weights not found: {load_from}")
        
        self.logger.info(f"Loading Depth Anything V2 ({encoder}) from {load_from}")
        
        model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        model.load_state_dict(torch.load(load_from, map_location='cpu'))
        model = model.to(self.context.device).eval()
        
        self.logger.info("Depth model loaded successfully")
        return model
    
    def _cleanup_model(self):
        """清理深度模型的显存占用"""
        if self.model is not None:
            self.logger.info("Cleaning up depth model from GPU memory...")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("Depth model memory cleaned")
    
    def run(self):
        """
        Main execution: Depth estimation and point cloud generation
        """
        self.logger.info(">>> Stage 1: Geometry Analysis (Depth & Point Cloud)")
        
        image_path = self.context.image_path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
        
        # Output folder
        save_folder = os.path.join(self.context.output_dir, 'S0_geometry_pred_results')
        os.makedirs(save_folder, exist_ok=True)
        
        # Check if depth map already exists (skip if not in clean mode)
        depth_save_path = os.path.join(save_folder, "depth.png")
        depth_already_exists = os.path.exists(depth_save_path)
        
        # Smart resume: Skip entire stage if all required files exist
        if not self.context.clean_mode and depth_already_exists:
            self.logger.info(f"✓ S0 已完成：所有必需文件都存在，跳过此阶段")
            self.logger.info(f"  - depth.png: ✓")
            # Load existing data into context for next stages
            depth_mm = cv2.imread(depth_save_path, cv2.IMREAD_UNCHANGED)
            if depth_mm is not None:
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                fx = fy = estimate_focal_length(width)
                cx, cy = width / 2, height / 2
                self.context.set_data('depth_image', depth_mm)
                self.context.set_data('ori_image_numpy', image_rgb)
                self.context.set_data('intrinsics', {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy})
                self.logger.info("Geometry Analysis Done (Skipped, data loaded from existing files).")
                return
            else:
                self.logger.warning("未找到已有 depth 文件，将重新生成...")
                depth_already_exists = False
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Estimate camera intrinsics
        height, width = image.shape[:2]
        fx = fy = estimate_focal_length(width)
        cx, cy = width / 2, height / 2
        
        self.logger.info(f"Image size: {width}x{height}, Estimated focal length: {fx}")
        
        # Skip depth prediction if already exists and not in clean mode
        if depth_already_exists and not self.context.clean_mode:
            self.logger.info(f"✓ Depth map already exists, loading from {depth_save_path}")
            depth_mm = cv2.imread(depth_save_path, cv2.IMREAD_UNCHANGED)
            if depth_mm is None:
                self.logger.warning("Failed to load existing depth map, will regenerate...")
                depth_already_exists = False
        
        # Generate depth if needed
        if not depth_already_exists or self.context.clean_mode:
            # Load model (lazy loading)
            if self.model is None:
                self.model = self._load_model()
            
            # Depth inference
            self.logger.info("Running depth estimation...")
            pred = self.model.infer_image(image, 518)
            depth_mm = (pred * 1000).astype(np.uint16)
            
            # Save depth map
            cv2.imwrite(depth_save_path, depth_mm)
            self.logger.info(f"Saved depth map to {depth_save_path}")
        
        # Store in Context (in-memory transfer)
        # Convert BGR to RGB before saving (cv2.imread returns BGR format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.context.set_data('depth_image', depth_mm)
        self.context.set_data('ori_image_numpy', image_rgb)
        self.context.set_data('intrinsics', {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy})
        
        # Point cloud generation (optional, for visualization or downstream legacy compatibility)
        color_image = image_rgb
        pcd = create_point_cloud(color_image, depth_mm, fx, fy, cx, cy)
        
        # Save point cloud (debug or always for compatibility)
        if self.context.debug_mode:
            pcd_save_path = os.path.join(save_folder, "pcd.ply")
            o3d.io.write_point_cloud(pcd_save_path, pcd)
            self.logger.info(f"[Debug] Saved point cloud to {pcd_save_path}")
        
        # Cleanup depth model to free GPU memory
        self._cleanup_model()
        
        self.logger.info("Geometry Analysis Done.")
