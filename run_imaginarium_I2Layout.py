#!/usr/bin/env python3
"""
Imaginarium CLI Entry Point
Imaginarium 命令行入口

用法 / Usage:
    python run_imaginarium_I2Layout_I2Layout.py <image_path> [--debug] [--clean]
    
示例 / Example:
    python run_imaginarium_I2Layout_I2Layout.py demo_images/demo_4.png  # 默认从现有结果继续 / Default: resume from existing results
    python run_imaginarium_I2Layout_I2Layout.py demo_images/demo_4.png --debug  # 输出详细的中间结果, 影响速度 / Output detailed intermediate results, slow down the speed
    python run_imaginarium_I2Layout_I2Layout.py demo_images/demo_4.png --clean  # 清空文件夹, 从头开始 / Clean output folder and start fresh
"""

import argparse
import sys
import os
import time

def main():
    parser = argparse.ArgumentParser(
        description="Imaginarium: Vision-guided 3D Scene Layout Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal mode (resume from existing results by default)
  python run_imaginarium_I2Layout_I2Layout.py demo/demo_4.png
  
  # Debug mode (save all visualizations)
  python run_imaginarium_I2Layout_I2Layout.py demo/demo_4.png --debug
  
  # Clean start (clear output folder and regenerate everything)
  python run_imaginarium_I2Layout_I2Layout.py demo/demo_4.png --clean
  
  # Clean start with debug mode
  python run_imaginarium_I2Layout_I2Layout.py demo/demo_4.png --debug --clean
  
For more information, visit: https://github.com/yourrepo/imaginarium
        """
    )
    
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to input 1024*1024 image (PNG/JPG)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (saves detailed intermediate visualizations, but slows down the speed)"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output folder and start fresh (default: resume from existing results)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Input file not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"⚠️  Warning: Config file not found: {args.config}", file=sys.stderr)
        print(f"   Using default configuration...", file=sys.stderr)
    
    # Run pipeline
    try:
        print(f"\n{'='*60}")
        print(f"🎨 Imaginarium Pipeline Starting...")
        print(f"{'='*60}")
        print(f"📷 Input: {args.image_path}")
        print(f"🐛 Debug Mode: {'ON' if args.debug else 'OFF'}")
        print(f"🔄 Mode: {'CLEAN START' if args.clean else 'RESUME'}")
        print(f"⚙️  Config: {args.config}")
        print(f"{'='*60}")
        print(f"🔄 Loading modules (this may take a few seconds)...", flush=True)
        module_load_start = time.time()

        from pipeline import ImaginariumPipeline
        
        print(f"\n🚀 Pipeline Context Initialized")
        
        pipeline = ImaginariumPipeline(args.image_path, debug=args.debug, clean=args.clean, config_path=args.config, startup_time=module_load_start)
        pipeline.run()
        
        print(f"\n{'='*60}")
        print(f"✅ Pipeline Completed Successfully!")
        print(f"📁 Results saved to: {pipeline.context.output_dir}")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Pipeline interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Pipeline Failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
