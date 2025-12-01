from __future__ import annotations
import os
import sys
import shutil
from typing import Optional, Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from utils.llm_api import GPTApi

# from transformers import AutoImageProcessor, AutoModel # Moved to lazy import
from utils.logger import Logger
# from utils.llm_api import GPTApi # Moved to lazy import in gpt property
from core.config import Config

class Context:
    """
    Global Pipeline Context & Data Bus.
    全局流水线上下文与数据总线。
    
    Manages:
    1. Shared Resources (Models, Device, Logger)
    2. Runtime Data (Intermediate results in memory)
    3. Control Flags (Debug mode)
    4. Resource Lifecycle
    
    Usage:
        >>> ctx = Context("demo.png", debug=True)
        >>> ctx.set_data("depth", depth_array)
        >>> depth = ctx.get_data("depth")
        >>> ctx.cleanup()  # Free resources
    """
    
    def __init__(self, image_path: str, debug: bool = False, clean: bool = False, config_path: str = "config/config.yaml"):
        """
        Initialize Pipeline Context.
        
        Args:
            image_path: Path to input image
            debug: Enable debug mode for verbose logging and visualization
            clean: Clean output folder and start fresh (default: resume from existing results)
            config_path: Path to configuration YAML file
        """
        self.image_path = image_path
        self.debug_mode = debug
        self.clean_mode = clean
        
        # Config
        self.config = Config(config_path)
        
        # Paths
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_parent = self.config.get('S0_geometry_pred', {}).get('save_parent_folder', 'saved_results')
        self.output_dir = os.path.join(save_parent, f"{self.image_name}_result")
        
        # Clean output folder only if clean mode is enabled
        if clean:
            if os.path.exists(self.output_dir):
                try:
                    print(f"🗑️  正在清空输出文件夹: {self.output_dir}", flush=True)
                    shutil.rmtree(self.output_dir)
                    print(f"✅ 输出文件夹已清空", flush=True)
                except Exception as e:
                    print(f"⚠️  警告: 清空文件夹时出错: {e}", file=sys.stderr, flush=True)
                    # Try to remove files individually if rmtree fails
                    try:
                        for root, dirs, files in os.walk(self.output_dir):
                            for f in files:
                                os.remove(os.path.join(root, f))
                            for d in dirs:
                                os.rmdir(os.path.join(root, d))
                        os.rmdir(self.output_dir)
                        print(f"✅ 输出文件夹已清空 (使用备用方法)", flush=True)
                    except Exception as e2:
                        print(f"❌ 错误: 无法清空文件夹: {e2}", file=sys.stderr, flush=True)
                        raise
            else:
                print(f"ℹ️  输出文件夹不存在，无需清空: {self.output_dir}", flush=True)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Logger with stage logs support
        log_file = os.path.join(self.output_dir, 'pipeline.log')
        stage_log_dir = os.path.join(self.output_dir, 'stage_logs')
        self.logger = Logger(name="Imaginarium", log_file=log_file, level='DEBUG' if debug else 'INFO', stage_log_dir=stage_log_dir)
        
        self.logger.info(f"🚀 Pipeline Context Initialized")
        self.logger.info(f"📁 Output Directory: {self.output_dir}")
        self.logger.info(f"🐛 Debug Mode: {'ON' if debug else 'OFF'}")
        self.logger.info(f"🔄 Mode: {'CLEAN START' if clean else 'RESUME'}")
        
        # Device
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"💻 Device: {self.device}")
        
        # Data Bus (In-Memory Storage)
        self.data: Dict[str, Any] = {}
        
        # Lazy Models
        self._dino_processor = None
        self._dino_model = None  # HuggingFace DINOv2 (for backward compatibility)
        self._original_dino_model = None  # Original DINOv2 (for S2 & S3)
        self._original_dino_processor = None  # Processor for original DINOv2
        self._ae_net = None
        self._gpt = None
        
    # --- Data Bus Methods ---
    def set_data(self, key: str, value: Any) -> None:
        """
        Store intermediate data in memory.
        
        Args:
            key: Data identifier
            value: Data to store
        """
        self.data[key] = value
        self.logger.debug(f"📦 Stored data: {key}")
        
    def get_data(self, key: str, default: Any = None) -> Any:
        """
        Retrieve intermediate data from memory.
        
        Args:
            key: Data identifier
            default: Default value if key not found
            
        Returns:
            Stored data or default value
        """
        value = self.data.get(key, default)
        if value is None and default is None:
            self.logger.debug(f"⚠️  Data not found: {key}")
        return value
    
    def has_data(self, key: str) -> bool:
        """Check if data exists in memory."""
        return key in self.data
    
    def clear_data(self, key: str) -> None:
        """Remove specific data from memory to free space."""
        if key in self.data:
            del self.data[key]
            self.logger.debug(f"🗑️  Cleared data: {key}")
        
    # --- Resource Management ---
    @property
    def dino_processor(self) -> Any:
        """
        Lazy load DINOv2 Processor (shared across S2 & S3).
        
        Returns:
            DINOv2 image processor
        """
        if self._dino_processor is None:
            from transformers import AutoImageProcessor
            self.logger.info("🔄 Loading Shared DINOv2 Processor...")
            self._dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
            self.logger.info("✅ DINOv2 Processor loaded")
        return self._dino_processor
        
    @property
    def dino_model(self) -> torch.nn.Module:
        """
        Lazy load DINOv2 Model (shared across S2 & S3).
        Now uses original DINOv2 architecture for consistency with AENet.
        
        Returns:
            DINOv2 model on device (with HuggingFace-compatible wrapper)
        """
        # Use original DINOv2 model wrapped for compatibility
        return self.original_dino_model_for_retrieval
    
    @property
    def original_dino_model(self) -> torch.nn.Module:
        """
        Lazy load Original DINOv2 Model (torch.hub version, shared across S2 & S3).
        This is the base model that can be reused for AENet.
        
        Returns:
            Original DINOv2 model on device
        """
        if self._original_dino_model is None:
            import torch
            self.logger.info("🔄 Loading Original DINOv2 Model (for S2 & S3)...")
            
            # Get ori_dino_weights_path from shared config (used by both S2 and S3)
            ori_dino_weights_path = self.config.shared.ori_dino_weights_path
            if ori_dino_weights_path is None:
                raise ValueError("ori_dino_weights_path must be configured in shared.ori_dino_weights_path")
            
            if not os.path.exists(ori_dino_weights_path):
                raise FileNotFoundError(f"DINOv2 weights not found at {ori_dino_weights_path}")
            
            model_name = 'dinov2_vitl14'
            
            # Try to load from local hub first, then fallback to facebookresearch/dinov2
            try:
                local_dinov2_path = 'src/dinov2'
                if os.path.exists(local_dinov2_path):
                    dinov2_model = torch.hub.load(local_dinov2_path, model_name, source='local', pretrained=False)
                else:
                    dinov2_model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=False)
                
                # Load weights
                dinov2_state_dict = torch.load(ori_dino_weights_path, map_location='cpu')
                dinov2_model.load_state_dict(dinov2_state_dict)
                dinov2_model = dinov2_model.to(self.device)
                dinov2_model.eval()
                self._original_dino_model = dinov2_model
                self.logger.info("✅ Original DINOv2 model loaded")
            except Exception as e:
                self.logger.error(f"Failed to load original DINOv2 model: {e}")
                raise
                
        return self._original_dino_model
    
    @property
    def original_dino_processor(self):
        """
        Processor for original DINOv2 model.
        Uses HuggingFace processor for image preprocessing (compatible with original DINOv2).
        
        Returns:
            Image processor
        """
        if self._original_dino_processor is None:
            from transformers import AutoImageProcessor
            self.logger.info("🔄 Loading DINOv2 Processor for original model...")
            self._original_dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
            self.logger.info("✅ DINOv2 Processor loaded")
        return self._original_dino_processor
    
    @property
    def original_dino_model_for_retrieval(self) -> torch.nn.Module:
        """
        Wrapper for original DINOv2 model that makes it compatible with HuggingFace API.
        This allows S2 retrieval code to use original DINOv2 without modification.
        
        Returns:
            Wrapped DINOv2 model with HuggingFace-compatible interface
        """
        class DINOv2Wrapper:
            """Wrapper to make original DINOv2 compatible with HuggingFace API"""
            def __init__(self, original_model):
                self.model = original_model
                self.device = original_model.device if hasattr(original_model, 'device') else next(original_model.parameters()).device
            
            def __call__(self, **inputs):
                """Compatible with HuggingFace API: model(**inputs)"""
                # Extract pixel_values from inputs
                pixel_values = inputs.get('pixel_values')
                if pixel_values is None:
                    raise ValueError("Expected 'pixel_values' in inputs")
                
                import torch
                # Original DINOv2 expects images directly (not wrapped in dict)
                with torch.no_grad():
                    features = self.model.forward_features(pixel_values)
                    
                    # Create a HuggingFace-compatible output object
                    class Output:
                        def __init__(self, last_hidden_state):
                            self.last_hidden_state = last_hidden_state
                    
                    # Use normalized features (matching HuggingFace behavior)
                    # Concatenate CLS token and patch tokens
                    cls_token = features['x_norm_clstoken'].unsqueeze(1)  # [B, 1, D]
                    patch_tokens = features['x_norm_patchtokens']  # [B, N, D]
                    last_hidden_state = torch.cat([cls_token, patch_tokens], dim=1)  # [B, N+1, D]
                    
                    return Output(last_hidden_state)
            
            def eval(self):
                self.model.eval()
                return self
            
            def to(self, device):
                self.model = self.model.to(device)
                return self
        
        return DINOv2Wrapper(self.original_dino_model)

    def get_ae_net(self, weights_path: str, ori_dino_weights_path: str = None) -> torch.nn.Module:
        """
        Lazy load AENet Model (reuses original DINOv2 from S2).
        
        Args:
            weights_path: Path to AENet weights
            ori_dino_weights_path: Path to original DINOv2 weights (optional, will reuse from S2 if available)
            
        Returns:
            AENet model on device
        """
        if self._ae_net is None:
            import torch
            self.logger.info("🔄 Loading Shared AENet Model (reusing DINOv2 from S2)...")
            
            # Import here to avoid circular imports
            from models.ae_net.ae_net import AENet
            
            # Reuse original DINOv2 model from S2 (already loaded)
            # This avoids reloading the same model
            dinov2_model = self.original_dino_model
            self.logger.info("✅ Reusing original DINOv2 model from S2")
            
            # Create AENet with reused original DINOv2 model
            model_name = 'dinov2_vitl14'
            self._ae_net = AENet(
                model_name=model_name,
                dinov2_model=dinov2_model,  # Reuse DINOv2 from S2
                max_batch_size=64,
                descriptor_size=1024
            )
            
            # Load AENet weights
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=self.device)
                try:
                    self._ae_net.load_state_dict(state_dict, strict=True)
                    self.logger.info(f"✅ AENet weights loaded from {weights_path}")
                except RuntimeError as e:
                    # If strict loading fails, try with strict=False
                    # The error message already contains missing/unexpected keys info
                    self.logger.warning(f"⚠️  Some keys in AENet weights don't match: {str(e)[:500]}")
                    self.logger.info("Attempting to load with strict=False...")
                    result = self._ae_net.load_state_dict(state_dict, strict=False)
                    self.logger.info(f"✅ AENet weights loaded from {weights_path} (non-strict mode)")
            else:
                self.logger.warning(f"⚠️  AENet weights not found at {weights_path}, using random initialization")
                
            self._ae_net.to(self.device)
            self._ae_net.eval()
            self.logger.info("✅ AENet Model loaded")
            
        return self._ae_net
    
    @property
    def gpt_params(self) -> Dict[str, Any]:
        """
        Get GPT API parameters (for multiprocessing contexts).
        
        Returns:
            Dictionary of GPT API parameters
        """
        return {
            'model': self.config.shared.gpt_model,
            'GPT_KEY': self.config.shared.gpt_key,
            'GPT_ENDPOINT': self.config.shared.gpt_endpoint,
            'use_openai_client': self.config.shared.use_openai_client
        }
    
    @property
    def gpt(self) -> GPTApi:
        """
        Lazy load GPT API client (shared across all modules).
        
        Returns:
            GPT API client instance
        """
        if self._gpt is None:
            from utils.llm_api import GPTApi
            self.logger.info("🔄 Loading Shared GPT API Client...")
            self._gpt = GPTApi(**self.gpt_params)
            self.logger.info("✅ GPT API Client loaded")
        return self._gpt
    
    def cleanup(self) -> None:
        """
        Free GPU memory and clean up resources.
        Call this after pipeline completion or on error.
        """
        import torch
        self.logger.info("🧹 Cleaning up resources...")
        
        # Clear data bus
        self.data.clear()
        
        # Delete models
        if self._dino_model is not None:
            del self._dino_model
            self._dino_model = None
        
        if self._original_dino_model is not None:
            del self._original_dino_model
            self._original_dino_model = None
            
        if self._ae_net is not None:
            del self._ae_net
            self._ae_net = None
            
        if self._dino_processor is not None:
            del self._dino_processor
            self._dino_processor = None
        
        if self._original_dino_processor is not None:
            del self._original_dino_processor
            self._original_dino_processor = None
            
        if self._gpt is not None:
            del self._gpt
            self._gpt = None
        
        # Force GPU memory release
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("✅ Cleanup complete")
        
        # 清理 logger（恢复标准流）
        self.logger.cleanup()
    
    def release_models(self) -> None:
        import torch
        """
        Release loaded models to free GPU memory, but keep data.
        Useful before calling external processes (like Blender).
        """
        self.logger.info("🧹 Releasing models to free GPU memory...")
        
        if self._dino_model is not None:
            del self._dino_model
            self._dino_model = None
        
        if self._original_dino_model is not None:
            del self._original_dino_model
            self._original_dino_model = None
            
        if self._ae_net is not None:
            del self._ae_net
            self._ae_net = None
            
        if self._dino_processor is not None:
            del self._dino_processor
            self._dino_processor = None
        
        if self._original_dino_processor is not None:
            del self._original_dino_processor
            self._original_dino_processor = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("✅ Models released")

    def __enter__(self):
        """Support context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto cleanup on context manager exit."""
        self.cleanup()
        return False
