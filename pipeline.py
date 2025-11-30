"""
Imaginarium Pipeline
主流水线控制器
"""

import time
import traceback
from core.context import Context
# Removed top-level module imports to speed up initial load
# from modules import (
#     GeometryModule,
#     SemanticParsingModule,
#     RetrievalModule,
#     PoseModule,
#     LayoutModule
# )

class ImaginariumPipeline:
    """
    Imaginarium Pipeline Controller.
    Imaginarium 流水线控制器。
    
    Orchestrates the complete 3D scene layout generation pipeline:
    1. Geometry Analysis (Depth & Point Cloud)
    2. Semantic Parsing (Detection & Segmentation)
    3. Asset Retrieval (3D Model Matching)
    4. Pose Estimation (Rotation, Translation, Scale)
    5. Layout Optimization (Physics & Constraints)
    
    Usage:
        >>> pipeline = ImaginariumPipeline("demo.png", debug=True)
        >>> pipeline.run()
        
    Or with context manager:
        >>> with ImaginariumPipeline("demo.png") as pipeline:
        ...     pipeline.run()
    """
    
    def __init__(self, image_path: str, debug: bool = False, clean: bool = False, config_path: str = "config/config.yaml", startup_time: float = None):
        """
        Initialize Pipeline.
        
        Args:
            image_path: Path to input image
            debug: Enable debug mode (verbose logging + save all visualizations)
            clean: Clean output folder and start fresh (default: resume from existing results)
            config_path: Path to configuration file
            startup_time: Time when module loading started
        """
        self.context = Context(image_path, debug=debug, clean=clean, config_path=config_path)
        self.logger = self.context.logger
        self.startup_time = startup_time
        
        # Initialize Modules (lazy instantiation, models loaded on demand)
        # Import here to avoid heavy load time at startup
        print("🔄 Importing modules...", flush=True)
        from modules.geometry import GeometryModule
        from modules.parsing import SemanticParsingModule
        from modules.retrieval import RetrievalModule
        from modules.pose import PoseModule
        from modules.layout import LayoutModule
        
        self.geometry = GeometryModule(self.context)
        self.parsing = SemanticParsingModule(self.context)
        self.retrieval = RetrievalModule(self.context)
        self.pose = PoseModule(self.context)
        self.layout = LayoutModule(self.context)
        print("✅ Modules imported", flush=True)
        
    def run(self) -> bool:
        """
        Execute the complete pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        self.logger.info("=" * 70)
        self.logger.info(f"🎨 IMAGINARIUM PIPELINE STARTED")
        self.logger.info(f"📷 Input Image: {self.context.image_path}")
        self.logger.info("=" * 70)
        
        try:
            # Stage 1: Geometry Analysis
            self.logger.info("\n" + "─" * 70)
            self.logger.start_stage("S0_geometry")
            self.geometry.run()
            self.logger.end_stage()
            
            if self.startup_time:
                elapsed_s0 = time.time() - self.startup_time
                print(f"⏱️  Time from 'Loading modules' to 'S0 Done': {elapsed_s0:.2f}s", flush=True)
            
            print("Geometry Analysis Done", flush=True)
            
            # Stage 2: Semantic Parsing
            self.logger.info("\n" + "─" * 70)
            self.logger.start_stage("S1_parsing")
            self.parsing.run()
            self.logger.end_stage()
            print("Semantic Parsing Done", flush=True)

            # Stage 3: Asset Retrieval
            self.logger.info("\n" + "─" * 70)
            self.logger.start_stage("S2_retrieval")
            self.retrieval.run()
            self.logger.end_stage()
            print("Asset Retrieval Done", flush=True)

            # Stage 4: Pose Estimation
            self.logger.info("\n" + "─" * 70)
            self.logger.start_stage("S3_pose")
            self.pose.run()
            self.logger.end_stage()
            print("Pose Estimation Done", flush=True)

            # Stage 5: Layout Optimization
            self.logger.info("\n" + "─" * 70)
            self.logger.start_stage("S4_layout")
            self.layout.run()
            self.logger.end_stage()
            print("Layout Optimization Done", flush=True)
            
            elapsed = time.time() - start_time
            self.logger.info("\n" + "=" * 70)
            self.logger.info(f"✅ PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"⏱️  Total Time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
            self.logger.info(f"📁 Results: {self.context.output_dir}")
            self.logger.info("=" * 70 + "\n")
            
            return True
            
        except KeyboardInterrupt:
            self.logger.warning("\n⚠️  Pipeline interrupted by user")
            return False
            
        except Exception as e:
            self.logger.error(f"\n❌ PIPELINE FAILED: {str(e)}")
            self.logger.error("\n" + traceback.format_exc())
            return False
            
        finally:
            # Always cleanup resources
            self.context.cleanup()
    
    def __enter__(self):
        """Support context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto cleanup on context manager exit."""
        self.context.cleanup()
        return False
