import bpy
import os
import glob
import sys
import time
import argparse
import gc

def convert_fbx_to_blend(root_dir):
    """
    Recursively finds all FBX files in root_dir and converts them to .blend files.
    The .blend files are saved in the same directory as the source .fbx files.
    """
    # Find all FBX files
    print(f"Searching for FBX files in {root_dir}...")
    fbx_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.fbx'):
                fbx_files.append(os.path.join(root, file))
    
    if not fbx_files:
        print(f"No FBX files found in {root_dir}")
        return

    print(f"Found {len(fbx_files)} FBX files.")
    
    start_time = time.time()
    success_count = 0
    
    for i, fbx_path in enumerate(fbx_files):
        try:
            # Check if target .blend already exists
            blend_path = os.path.splitext(fbx_path)[0] + ".blend"
            if os.path.exists(blend_path):
                print(f"[{i+1}/{len(fbx_files)}] Skipping (already exists): {blend_path}")
                continue
                
            print(f"[{i+1}/{len(fbx_files)}] Converting: {fbx_path}")
            
            # 1. Reset scene
            bpy.ops.wm.read_factory_settings(use_empty=True)
            
            # 2. Import FBX (using same parameters as S4)
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
            success_count += 1
            
            # 4. Cleanup memory
            # Reset scene and garbage collect to prevent memory explosion
            bpy.ops.wm.read_factory_settings(use_empty=True)
            gc.collect()
            
        except Exception as e:
            print(f"Failed to convert {fbx_path}: {e}")
            
    total_time = time.time() - start_time
    print(f"\nConversion finished in {total_time:.2f} seconds.")
    print(f"Successfully converted {success_count} files.")

if __name__ == "__main__":
    # Parse command line arguments
    # Usage: blender --background --python scripts/convert_fbx_to_blend.py -- --fbx_dir /path/to/assets
    
    # Filter arguments after "--" if present
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Batch convert FBX files to .blend format for faster loading.")
    parser.add_argument("--fbx_dir", type=str, default="asset_data/imaginarium_assets", help="Root directory containing FBX files")
    
    args = parser.parse_args(argv)
    
    target_dir = args.fbx_dir
    
    if not os.path.exists(target_dir):
        print(f"Error: Directory not found: {target_dir}")
    else:
        convert_fbx_to_blend(target_dir)

