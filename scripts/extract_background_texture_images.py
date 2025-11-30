import bpy
import os
import shutil
import glob
import re

# ========== 配置 ==========
BASE_OUTPUT_DIR = "D:/background_texture_images/"

# 创建输出文件夹
CEILING_DIR = os.path.join(BASE_OUTPUT_DIR, "ceiling")
FLOOR_DIR = os.path.join(BASE_OUTPUT_DIR, "floor")
WALL_DIR = os.path.join(BASE_OUTPUT_DIR, "wall")

os.makedirs(CEILING_DIR, exist_ok=True)
os.makedirs(FLOOR_DIR, exist_ok=True)
os.makedirs(WALL_DIR, exist_ok=True)

# ========== 工具函数 ==========
def get_object_base_name(obj_name):
    """获取物体的基础名称（去掉各种数字后缀，转小写）"""
    # 先转小写
    name = obj_name.lower()
    # 去掉 .001、.002 等后缀
    name = re.sub(r'\.\d+$', '', name)
    # 去掉 _0、_1、_2 等后缀（有下划线）
    name = re.sub(r'_\d+$', '', name)
    # 去掉 0、1、2 等后缀（无下划线）
    name = re.sub(r'\d+$', '', name)
    return name

def get_target_folder(obj_name):
    """根据物体名称确定目标文件夹"""
    base_name = get_object_base_name(obj_name)
    
    if base_name == "ceiling":
        return CEILING_DIR, "ceiling"
    elif base_name == "floor":
        return FLOOR_DIR, "floor"
    elif base_name == "wall":
        return WALL_DIR, "wall"
    else:
        return None, None

# ========== 开始处理 ==========
blend_filename = bpy.path.basename(bpy.data.filepath)

# 统计信息
stats = {
    'ceiling': {'objects': 0, 'textures': 0},
    'floor': {'objects': 0, 'textures': 0},
    'wall': {'objects': 0, 'textures': 0}
}

exported_images = set()  # 全局已导出的贴图

# ========== 收集所有目标物体 ==========
target_objects = []

# 遍历场景中的所有物体（包括所有层级）
for obj in bpy.context.scene.objects:
    if obj.type != 'MESH':
        continue
    
    target_folder, obj_type = get_target_folder(obj.name)
    if target_folder:
        target_objects.append((obj, obj_type, target_folder))

# ========== 遍历目标物体提取贴图 ==========
for obj, obj_type, target_folder in target_objects:
    stats[obj_type]['objects'] += 1
    
    # 遍历物体的所有材质
    for mat_slot in obj.material_slots:
        if not mat_slot.material:
            continue
        
        mat = mat_slot.material
        
        if not mat.use_nodes:
            continue
        
        # 收集所有图像纹理（包括节点组内部的）
        def get_all_images(node_tree):
            """递归获取节点树中的所有图像"""
            images = []
            for node in node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    images.append(node.image)
                # 检查节点组
                elif node.type == 'GROUP' and node.node_tree:
                    images.extend(get_all_images(node.node_tree))
            return images
        
        images = get_all_images(mat.node_tree)
        
        for img in images:
                # 避免重复导出（全局去重）
                if img.name in exported_images:
                    continue
                
                exported_images.add(img.name)
                
                # ========== 导出贴图 ==========
                original_saved = False
                
                # 方法1: 尝试从原始路径复制
                if img.filepath:
                    src_path = bpy.path.abspath(img.filepath)
                    
                    if os.path.exists(src_path):
                        # 文件存在，直接复制
                        filename = os.path.basename(src_path)
                        dst_path = os.path.join(target_folder, filename)
                        shutil.copy2(src_path, dst_path)
                        original_saved = True
                        stats[obj_type]['textures'] += 1
                    else:
                        # 文件不存在，尝试智能查找
                        pass
                        
                        # 获取目录和文件名模式
                        src_dir = os.path.dirname(src_path)
                        base_name = os.path.splitext(os.path.basename(src_path))[0]
                        
                        # 移除分辨率后缀 (如 _1k, _2k, _8k)
                        base_pattern = re.sub(r'_\d+k$', '', base_name)
                        
                        # 查找所有可能的文件
                        if os.path.exists(src_dir):
                            pattern = os.path.join(src_dir, f"{base_pattern}*.*")
                            matches = glob.glob(pattern)
                            
                            if matches:
                                # 优先选择分辨率最高的
                                matches.sort(reverse=True)
                                found_file = matches[0]
                                filename = os.path.basename(found_file)
                                dst_path = os.path.join(target_folder, filename)
                                shutil.copy2(found_file, dst_path)
                                original_saved = True
                                stats[obj_type]['textures'] += 1
                
                # 方法2: 直接从Blender保存（适用于打包贴图或方法1失败）
                if not original_saved:
                    try:
                        # 确定文件格式
                        ext_map = {
                            'PNG': 'png',
                            'JPEG': 'jpg',
                            'TARGA': 'tga',
                            'TIFF': 'tif',
                            'OPEN_EXR': 'exr',
                            'HDR': 'hdr'
                        }
                        ext = ext_map.get(img.file_format, 'png')
                        
                        # 清理文件名（移除可能的扩展名）
                        clean_name = os.path.splitext(img.name)[0]
                        filename = f"{clean_name}.{ext}"
                        dst_path = os.path.join(target_folder, filename)
                        
                        # 保存当前贴图
                        img.filepath_raw = dst_path
                        img.save()
                        stats[obj_type]['textures'] += 1
                    except Exception as e:
                        pass

# ========== 完成提示 ==========
print(f"Ceiling: {stats['ceiling']['objects']} 个物体, {stats['ceiling']['textures']} 个贴图")
print(f"Floor:   {stats['floor']['objects']} 个物体, {stats['floor']['textures']} 个贴图")
print(f"Wall:    {stats['wall']['objects']} 个物体, {stats['wall']['textures']} 个贴图")
print(f"总计:    {sum(s['textures'] for s in stats.values())} 个贴图")
