import bpy
import os
import glob
import sys
import time
import argparse
import gc
import subprocess
import multiprocessing
from pathlib import Path

def convert_single_fbx(fbx_path):
    """
    Convert a single FBX file to .blend using Blender in the current process.
    This function is meant to be called within Blender's Python environment.
    """
    try:
        blend_path = os.path.splitext(fbx_path)[0] + ".blend"
        
        # 1. Reset scene
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        # 2. Import FBX (using optimized parameters)
        # 性能优化参数：不加载动画、不加载自定义属性等
        bpy.ops.import_scene.fbx(
            filepath=fbx_path,
            use_anim=False,
            ignore_leaf_bones=True,
            automatic_bone_orientation=False,
            use_custom_props=False,
            use_custom_props_enum_as_string=False,
        )
        
        # 3. Save as .blend
        # compress=False for faster saving/loading
        bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=False)
        
        # 4. Cleanup memory
        bpy.ops.wm.read_factory_settings(use_empty=True)
        gc.collect()
        
        return True
    except Exception as e:
        print(f"Failed to convert {fbx_path}: {e}")
        return False

def convert_fbx_to_blend(root_dir, parallel=False, num_workers=None):
    """
    Recursively finds all FBX files in root_dir and converts them to .blend files.
    The .blend files are saved in the same directory as the source .fbx files.
    
    Args:
        root_dir: Root directory to search for FBX files
        parallel: If True, use multiprocessing to convert files in parallel
        num_workers: Number of parallel workers (defaults to CPU count)
    """
    # Find all FBX files
    print(f"Searching for FBX files in {root_dir}...")
    fbx_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.fbx'):
                fbx_path = os.path.join(root, file)
                blend_path = os.path.splitext(fbx_path)[0] + ".blend"
                # Only add if .blend doesn't exist
                if not os.path.exists(blend_path):
                    fbx_files.append(fbx_path)
    
    if not fbx_files:
        print(f"No FBX files to convert in {root_dir} (all may already be converted)")
        return

    print(f"Found {len(fbx_files)} FBX files to convert.")
    
    start_time = time.time()
    
    if parallel:
        # 并行处理模式：使用多个Blender进程
        success_count = convert_parallel(fbx_files, num_workers)
    else:
        # 串行处理模式：原有逻辑
        success_count = 0
        for i, fbx_path in enumerate(fbx_files):
            print(f"[{i+1}/{len(fbx_files)}] Converting: {fbx_path}")
            if convert_single_fbx(fbx_path):
                success_count += 1
            
    total_time = time.time() - start_time
    print(f"\nConversion finished in {total_time:.2f} seconds.")
    print(f"Successfully converted {success_count}/{len(fbx_files)} files.")

# Global variable for worker (needed for multiprocessing pickle)
_worker_config = {}

def _parallel_worker(args):
    """
    Worker function that spawns a Blender process to convert one FBX file.
    Must be at module level for multiprocessing pickle.
    """
    fbx_path, file_idx, total_files = args
    blender_exe = _worker_config['blender_exe']
    
    try:
        # Create a temporary script that converts just this one file
        temp_script = f'''
import bpy
import os
import gc

fbx_path = "{fbx_path}"
blend_path = os.path.splitext(fbx_path)[0] + ".blend"

try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(
        filepath=fbx_path,
        use_anim=False,
        ignore_leaf_bones=True,
        automatic_bone_orientation=False,
        use_custom_props=False,
        use_custom_props_enum_as_string=False,
    )
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=False)
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
'''
        
        # Run Blender in background mode with the temporary script
        result = subprocess.run(
            [blender_exe, "--background", "--python-expr", temp_script],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout per file
        )
        
        success = "SUCCESS" in result.stdout
        status = "✓" if success else "✗"
        print(f"[{file_idx}/{total_files}] {status} {os.path.basename(fbx_path)}")
        
        return success
        
    except subprocess.TimeoutExpired:
        print(f"[{file_idx}/{total_files}] ✗ Timeout: {os.path.basename(fbx_path)}")
        return False
    except Exception as e:
        print(f"[{file_idx}/{total_files}] ✗ Error: {os.path.basename(fbx_path)} - {e}")
        return False

def convert_parallel(fbx_files, num_workers=None):
    """
    Convert FBX files in parallel by spawning multiple Blender processes.
    Each process handles one FBX file at a time.
    """
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), len(fbx_files))
    
    print(f"Using {num_workers} parallel workers...")
    
    # Get the Blender executable path and store in global config
    global _worker_config
    _worker_config['blender_exe'] = bpy.app.binary_path
    
    # Prepare arguments for workers: (fbx_path, index, total)
    total_files = len(fbx_files)
    worker_args = [(fbx_path, i+1, total_files) for i, fbx_path in enumerate(fbx_files)]
    
    # Process files in parallel
    from multiprocessing import Pool
    with Pool(processes=num_workers) as pool:
        results = pool.map(_parallel_worker, worker_args)
    
    success_count = sum(results)
    return success_count

if __name__ == "__main__":
    # Parse command line arguments
    # Usage: 
    #   串行模式: blender --background --python scripts/convert_fbx_to_blend.py -- --fbx_dir /path/to/assets
    #   并行模式: blender --background --python scripts/convert_fbx_to_blend.py -- --fbx_dir /path/to/assets --parallel --workers 8
    
    # Filter arguments after "--" if present
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Batch convert FBX files to .blend format for faster loading.")
    parser.add_argument("--fbx_dir", type=str, default="asset_data/imaginarium_assets", 
                        help="Root directory containing FBX files")
    parser.add_argument("--parallel", action="store_true", 
                        help="Enable parallel processing (significantly faster for many files)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    
    args = parser.parse_args(argv)
    
    target_dir = args.fbx_dir
    
    if not os.path.exists(target_dir):
        print(f"Error: Directory not found: {target_dir}")
    else:
        convert_fbx_to_blend(target_dir, parallel=args.parallel, num_workers=args.workers)

'''
blender --background --python scripts/convert_fbx_to_blend.py -- --fbx_dir asset_data/imaginarium_assets --parallel --workers 8
'''