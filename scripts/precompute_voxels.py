"""
体素预计算脚本
用于批量预计算所有FBX文件的体素数据，加速S4阶段的模拟退火优化
"""

import os
import sys
import numpy as np
import trimesh
import pyassimp
import scipy.ndimage
from pathlib import Path
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import traceback
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Processing timed out")



class VoxelPrecomputer:
    """预计算FBX文件的体素数据"""
    
    def __init__(self, standard_pitch=0.03):
        """
        Args:
            standard_pitch: 标准体素尺寸，默认0.03米
        """
        self.standard_pitch = standard_pitch
    
    def fbx2mesh(self, fbx_path):
        """将FBX文件转换为mesh"""
        try:
            with pyassimp.load(str(fbx_path)) as scene:
                if not scene.meshes:
                    print(f"警告: {fbx_path} 没有mesh数据")
                    return None
                mesh = scene.meshes[0]
                vertices = np.array(mesh.vertices)
                faces = np.array(mesh.faces)
            return trimesh.Trimesh(vertices=vertices, faces=faces)
        except TimeoutException:
            raise
        except Exception as e:
            print(f"错误: 无法加载 {fbx_path}: {e}")
            return None
    
    def approximate_as_box_if_thin(self, mesh: trimesh.Trimesh, pitch: float) -> trimesh.Trimesh:
        """如果网格在某个维度极其薄，则其近似为一个长方体"""
        min_corner, max_corner = mesh.bounds
        size = max_corner - min_corner
        
        i_min = np.argmin(size)
        if size[i_min] < pitch:
            center = (max_corner + min_corner) / 2.0
            half_size = size / 2.0
            half_size[i_min] = pitch / 2.0
            
            box = trimesh.creation.box(extents=2.0 * half_size)
            box.apply_translation(center)
            return box
        else:
            return mesh
    
    def voxelize_surface_approximation(self, mesh):
        """通过表面采样近似计算体素（用于处理死锁/失败的复杂网格）"""
        try:
            # 1. 表面采样
            count = 50000  # 采样点数
            points, _ = trimesh.sample.sample_surface(mesh, count)
            
            # 2. 计算体素索引
            # 使用与正常流程相同的对齐方式：以bounds[0]为原点
            grid_origin = mesh.bounds[0]
            voxel_indices = np.floor((points - grid_origin) / self.standard_pitch).astype(np.int32)
            
            # 3. 去重并构建grid
            # 找到grid范围
            max_idx = np.max(voxel_indices, axis=0)
            grid_shape = max_idx + 1 + 2 # +padding
            
            grid = np.zeros(grid_shape, dtype=bool)
            # 偏移索引以留出padding
            valid_indices = voxel_indices + 1
            
            # 填充grid
            # 过滤越界（理论上不应该，但保险起见）
            mask = (valid_indices[:, 0] < grid_shape[0]) & \
                   (valid_indices[:, 1] < grid_shape[1]) & \
                   (valid_indices[:, 2] < grid_shape[2]) & \
                   (valid_indices >= 0).all(axis=1)
            valid_indices = valid_indices[mask]
            
            grid[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = True
            
            # 4. 填充内部 (Fill holes)
            # 膨胀一下连接断点
            grid = scipy.ndimage.binary_dilation(grid, iterations=1)
            # 填充内部
            grid = scipy.ndimage.binary_fill_holes(grid)
            
            return grid, grid_origin
            
        except TimeoutException:
            raise
        except Exception as e:
            print(f"Fallback approximation failed: {e}")
            return None, None

    def precompute_voxels_fallback(self, fbx_path):
        """
        降级处理模式：使用表面采样近似体素
        """
        mesh = self.fbx2mesh(fbx_path)
        if mesh is None:
            return None
        
        # 处理极薄的mesh
        mesh = self.approximate_as_box_if_thin(mesh, self.standard_pitch)
        
        # 使用近似方法
        grid_final, grid_origin = self.voxelize_surface_approximation(mesh)
        
        if grid_final is None:
            return None
            
        # 提取体素坐标索引（稀疏存储）
        voxel_indices = np.argwhere(grid_final).astype(np.int16)
        
        if len(voxel_indices) == 0:
            print(f"警告: {fbx_path} 近似体素化后为空")
            return None
        
        # 返回预计算数据
        result = {
            'voxel_indices': voxel_indices,
            'origin': np.array(grid_origin, dtype=np.float32),
            'pitch': float(self.standard_pitch),
            'bounds': np.array(mesh.bounds, dtype=np.float32),
            'grid_shape': grid_final.shape
        }
        
        return result

    def precompute_voxels(self, fbx_path):
        """
        预计算单个FBX文件的体素数据
        
        Returns:
            dict: {
                'voxel_indices': np.ndarray,  # 体素坐标索引 (N, 3)
                'origin': np.ndarray,          # 体素网格原点 (3,)
                'pitch': float,                # 体素尺寸
                'bounds': np.ndarray           # mesh边界 (2, 3)
            }
        """
        mesh = self.fbx2mesh(fbx_path)
        if mesh is None:
            return None
        
        # 处理极薄的mesh
        mesh = self.approximate_as_box_if_thin(mesh, self.standard_pitch)
        
        # 体素化
        try:
            voxels = mesh.voxelized(pitch=self.standard_pitch, method='subdivide')
        except TimeoutException:
            raise
        except Exception as e:
            print(f"体素化失败 {fbx_path}: {e}")
            return None
        
        # 提取体素网格数据
        if hasattr(voxels, 'matrix'):
            grid_np = voxels.matrix.copy()
        else:
            grid_np = voxels.encoding.dense.copy()
        
        # 获取原点
        if hasattr(voxels, 'origin'):
            grid_origin = voxels.origin
        elif hasattr(voxels, 'translation'):
            grid_origin = voxels.translation
        elif hasattr(voxels, 'transform'):
            grid_origin = voxels.transform[:3, 3]
        else:
            grid_origin = mesh.bounds[0]
        
        # 膨胀、填充、腐蚀处理（与S4中的处理保持一致）
        dilation_iter = 3
        grid_dilated = scipy.ndimage.binary_dilation(grid_np, iterations=dilation_iter)
        grid_filled = scipy.ndimage.binary_fill_holes(grid_dilated)
        erosion_iter = 1
        grid_eroded = scipy.ndimage.binary_erosion(grid_filled, iterations=erosion_iter)
        
        # 防止消失
        if np.sum(grid_eroded) == 0:
            grid_final = grid_filled
        else:
            grid_final = grid_eroded
        
        # 提取体素坐标索引（稀疏存储）
        voxel_indices = np.argwhere(grid_final).astype(np.int16)
        
        if len(voxel_indices) == 0:
            print(f"警告: {fbx_path} 体素化后为空")
            return None
        
        # 返回预计算数据
        result = {
            'voxel_indices': voxel_indices,
            'origin': np.array(grid_origin, dtype=np.float32),
            'pitch': float(self.standard_pitch),
            'bounds': np.array(mesh.bounds, dtype=np.float32),
            'grid_shape': grid_final.shape
        }
        
        return result


def process_single_fbx(args):
    """处理单个FBX文件（用于多进程）"""
    fbx_path, output_dir, standard_pitch = args
    
    # 设置超时处理 (仅在非Windows系统有效)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)  # 2分钟超时
    
    try:
        # 计算输出文件路径
        relative_path = fbx_path.relative_to(fbx_path.parents[0])
        output_path = output_dir / relative_path.with_suffix('.voxel.pkl')
        
        # 如果已存在，跳过
        if output_path.exists():
            return f"跳过（已存在）: {fbx_path.name}"
        
        # 预计算体素
        precomputer = VoxelPrecomputer(standard_pitch=standard_pitch)
        voxel_data = precomputer.precompute_voxels(fbx_path)
        
        if voxel_data is None:
            return f"失败: {fbx_path.name}"
        
        # 保存到文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(voxel_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        voxel_count = len(voxel_data['voxel_indices'])
        file_size_kb = output_path.stat().st_size / 1024
        
        return f"成功: {fbx_path.name} ({voxel_count} 体素, {file_size_kb:.1f}KB)"
        
    except TimeoutException:
        # 尝试降级方案
        try:
            # 重置超时，给降级方案一些时间
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(60)
            
            precomputer = VoxelPrecomputer(standard_pitch=standard_pitch)
            voxel_data = precomputer.precompute_voxels_fallback(fbx_path)
            
            if voxel_data:
                # 保存到文件
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    pickle.dump(voxel_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                voxel_count = len(voxel_data['voxel_indices'])
                file_size_kb = output_path.stat().st_size / 1024
                return f"成功(近似): {fbx_path.name} ({voxel_count} 体素, {file_size_kb:.1f}KB)"
            else:
                 return f"错误: {fbx_path.name} - 降级处理失败"
                 
        except Exception as e2:
            return f"错误: {fbx_path.name} - 处理超时且降级失败: {e2}"
            
    except Exception as e:
        return f"错误: {fbx_path.name} - {str(e)}\n{traceback.format_exc()}"
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description='批量预计算FBX文件的体素数据')
    parser.add_argument('--fbx_dir', type=str, 
                       default='/opt/data/private/allenxmzhu/3dLayOut/IntelliScene2/asset_data/2015_fbx',
                       help='FBX文件目录')
    parser.add_argument('--output_dir', type=str,
                       default='/opt/data/private/allenxmzhu/3dLayOut/IntelliScene2/asset_data/2015_fbx_voxels',
                       help='输出目录')
    parser.add_argument('--pitch', type=float, default=0.03,
                       help='体素尺寸（米）')
    parser.add_argument('--workers', type=int, default=8,
                       help='并行进程数')
    parser.add_argument('--pattern', type=str, default='**/*.fbx',
                       help='FBX文件匹配模式')
    
    args = parser.parse_args()
    
    fbx_dir = Path(args.fbx_dir)
    output_dir = Path(args.output_dir)
    
    if not fbx_dir.exists():
        print(f"错误: FBX目录不存在: {fbx_dir}")
        return
    
    # 查找所有FBX文件
    print(f"扫描FBX文件: {fbx_dir}")
    fbx_files = list(fbx_dir.glob(args.pattern))
    print(f"找到 {len(fbx_files)} 个FBX文件")
    
    if len(fbx_files) == 0:
        print("未找到FBX文件，退出")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备任务参数
    tasks = [(fbx_path, output_dir, args.pitch) for fbx_path in fbx_files]
    
    # 使用多进程处理
    print(f"\n开始处理，使用 {args.workers} 个进程...")
    print("="*80)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_fbx, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="处理进度") as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                
                if result.startswith("成功"):
                    success_count += 1
                elif result.startswith("跳过"):
                    skip_count += 1
                else:
                    fail_count += 1
                    tqdm.write(result)
    
    print("="*80)
    print(f"\n处理完成!")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  失败: {fail_count}")
    print(f"  总计: {len(tasks)}")
    print(f"\n体素数据保存在: {output_dir}")
    
    # 统计总大小
    total_size_mb = sum(f.stat().st_size for f in output_dir.glob('**/*.voxel.pkl')) / (1024 * 1024)
    print(f"总大小: {total_size_mb:.2f} MB")


if __name__ == '__main__':
    main()

'''
# 运行预计算脚本
python precompute_voxels.py \
    --fbx_dir "asset_data/imaginarium_assets" \
    --output_dir "asset_data/imaginarium_assets_voxels"
'''