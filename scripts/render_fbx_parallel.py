#!/usr/bin/env python3
"""
多GPU并行渲染FBX文件的主调度脚本
支持断点续传，自动跳过已渲染的文件
"""

import os
import subprocess
import multiprocessing
from pathlib import Path
import time
import argparse
import sys

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="多GPU并行渲染FBX文件调度器")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="FBX文件所在的输入目录"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="渲染结果输出目录"
    )
    
    parser.add_argument(
        "--num_gpus", 
        type=int, 
        default=8,
        help="使用的GPU数量 (默认: 8)"
    )
    
    parser.add_argument(
        "--num_views", 
        type=int, 
        default=162,
        help="每个物体需要渲染的视角数量，用于检查完整性 (默认: 162)"
    )
    
    parser.add_argument(
        "--blender_script", 
        type=str, 
        default="scripts/render_fbx_single_gpu.py",
        help="Blender调用的Python脚本路径 (默认: scripts/render_fbx_single_gpu.py)"
    )

    return parser.parse_args()

def check_rendered(fbx_path, output_base_dir, num_views=162):
    """
    检查FBX文件是否已经完全渲染
    返回True表示已完成，False表示需要渲染
    """
    object_name = Path(fbx_path).stem
    output_dir = os.path.join(output_base_dir, object_name)
    
    if not os.path.exists(output_dir):
        return False
    
    # 检查PNG文件数量
    png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    
    # 检查camera_poses.npy是否存在
    poses_file = os.path.join(output_dir, 'camera_poses.npy')
    
    if len(png_files) >= num_views and os.path.exists(poses_file):
        return True
    
    return False

def collect_fbx_files(input_dir, output_dir, num_views):
    """
    扫描输入目录，收集所有需要渲染的FBX文件
    """
    fbx_files = []
    
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return fbx_files
    
    # 遍历目录获取所有FBX文件
    for file in sorted(os.listdir(input_dir)):
        if file.lower().endswith('.fbx'):
            fbx_path = os.path.join(input_dir, file)
            
            # 检查是否已经渲染
            if check_rendered(fbx_path, output_dir, num_views):
                print(f"跳过已渲染: {file}")
            else:
                fbx_files.append(fbx_path)
                print(f"待渲染: {file}")
    
    return fbx_files

def split_list(lst, n):
    """
    将列表分成n份，尽量平均分配
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def render_worker(gpu_id, fbx_list, output_dir, blender_script_path):
    """
    单个GPU的渲染工作进程
    """
    if not fbx_list:
        print(f"GPU {gpu_id}: 没有分配到任务")
        return
    
    print(f"\n{'='*60}")
    print(f"GPU {gpu_id} 开始工作")
    print(f"分配到 {len(fbx_list)} 个FBX文件")
    print(f"{'='*60}\n")
    
    # 创建临时文件列表
    temp_file = f"fbx_list_gpu_{gpu_id}.txt"
    with open(temp_file, 'w') as f:
        for fbx_path in fbx_list:
            f.write(f"{fbx_path}\n")
    
    # 设置环境变量，指定使用哪个GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 构建Blender命令
    cmd = [
        'blender',
        '--background',
        '--python', blender_script_path,
        '--',
        '--file_path', temp_file,
        '--base_output_dir', output_dir,
        '--gpu_id', str(gpu_id)
    ]
    
    print(f"GPU {gpu_id} 执行命令: {' '.join(cmd)}")
    print(f"GPU {gpu_id} CUDA_VISIBLE_DEVICES={gpu_id}\n")
    
    try:
        # 执行Blender渲染
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✓ GPU {gpu_id} 完成所有渲染任务")
        else:
            print(f"\n✗ GPU {gpu_id} 渲染过程中出现错误:")
            print(result.stderr)
    
    except Exception as e:
        print(f"\n✗ GPU {gpu_id} 发生异常: {e}")
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"GPU {gpu_id} 清理临时文件: {temp_file}")

def main():
    args = parse_args()

    # 验证Blender脚本是否存在
    if not os.path.exists(args.blender_script):
        print(f"错误: 找不到Blender脚本: {args.blender_script}")
        sys.exit(1)

    print("="*60)
    print("FBX多GPU并行渲染系统")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"GPU数量: {args.num_gpus}")
    print(f"Blender脚本: {args.blender_script}")
    print(f"校验视角数: {args.num_views}")
    
    # 收集需要渲染的FBX文件
    print(f"\n正在扫描FBX文件...")
    fbx_files = collect_fbx_files(args.input_dir, args.output_dir, args.num_views)
    
    if not fbx_files:
        print("\n没有需要渲染的FBX文件！")
        return
    
    print(f"\n总共找到 {len(fbx_files)} 个需要渲染的FBX文件")
    
    # 将任务分配给各个GPU
    # 如果文件数少于GPU数，则只使用必要的GPU
    num_active_gpus = min(args.num_gpus, len(fbx_files))
    fbx_chunks = split_list(fbx_files, num_active_gpus)
    
    print("\n任务分配:")
    for i, chunk in enumerate(fbx_chunks):
        print(f"  GPU {i}: {len(chunk)} 个文件")
    
    # 启动多进程渲染
    print(f"\n启动 {num_active_gpus} 个渲染进程...\n")
    
    processes = []
    for gpu_id in range(num_active_gpus):
        p = multiprocessing.Process(
            target=render_worker,
            args=(gpu_id, fbx_chunks[gpu_id], args.output_dir, args.blender_script)
        )
        p.start()
        processes.append(p)
        time.sleep(2)  # 错开启动时间，避免资源竞争
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    print("\n" + "="*60)
    print("所有渲染任务完成！")
    print("="*60)

if __name__ == "__main__":
    main()
