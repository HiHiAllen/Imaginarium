import bpy
import numpy as np
from mathutils import Matrix, Vector, Quaternion, Euler
import os
import math
import json
import re
import pandas as pd
import mathutils
import copy
import argparse
import sys
import torch
import trimesh
import scipy
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from bpy_extras.object_utils import world_to_camera_view
import pyassimp
import functools
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到 Python 路径
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent  # 项目根目录（scripts的父目录）
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.drop_sim_script import run_drop_simulation

# ===== 重要：让所有print自动flush，避免输出被缓冲 =====
print = functools.partial(print, flush=True)

# 确保环境干净 (在业务逻辑开始前释放可能残留的显存)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

eps = 1e-3

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log = open(log_file, 'w', buffering=1)  # 行缓冲

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()  # 立即刷新终端输出
        self.log.write(message)
        self.log.flush()  # 立即刷新文件输出

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ========== Texture Application Helpers ==========
def parse_texture_name(filename):
    """
    解析纹理文件名，提取基础名称和分辨率。
    支持格式: Tiles15_COL_VAR1_6K.jpg -> (Tiles15, 6K)
    """
    name_without_ext = os.path.splitext(filename)[0]
    
    # 尝试提取分辨率 (e.g., 1K, 2k, 4K, 8k)
    resolution = ""
    res_match = re.search(r'[_\-\s](\d+[Kk])', name_without_ext)
    if res_match:
        resolution = res_match.group(1)
        
    # 定义可能的纹理类型标识符
    indicators = [
        'COL_VAR1', 'COL', 'diff', 'diffuse', 'albedo', 
        'NRM16', 'NRM', 'nor', 'normal', 
        'GLOSS', 'rough', 'roughness', 
        'DISP16', 'DISP', 'disp', 'displacement', 'BUMP16', 'BUMP',
        'AO', 'ao'
    ]
    
    base_name = name_without_ext
    
    # 尝试通过移除标识符来找到 base_name
    for ind in indicators:
        pattern = re.compile(re.escape(ind), re.IGNORECASE)
        if pattern.search(name_without_ext):
            parts = pattern.split(name_without_ext)
            if len(parts) > 0:
                candidate = parts[0].rstrip('_- ')
                if candidate:
                    base_name = candidate
                    return base_name, resolution

    # 如果没有匹配到标准标识符，尝试使用简单的正则 (fallback)
    match = re.match(r'(.+?)_(diff|rough|nor|disp|ao|metal|COL|GLOSS|NRM|BUMP)', name_without_ext, re.IGNORECASE)
    if match:
        return match.group(1), resolution
        
    return None, None

def find_related_textures(folder, base_name, resolution):
    """
    在文件夹中查找相关的纹理文件。
    """
    textures = {}
    texture_types = {
        'diff': ['COL', 'diff', 'diffuse', 'color', 'albedo', 'base'],
        'rough': ['rough', 'roughness'],
        'gloss': ['GLOSS', 'gloss'],
        'nor': ['NRM', 'nor', 'normal', 'norm'],
        'disp': ['DISP', 'disp', 'displacement', 'height', 'BUMP', 'bump'],
        'ao': ['AO', 'ao', 'ambient', 'occlusion'],
        'metal': ['metal', 'metallic', 'metalness']
    }
    
    if not os.path.exists(folder):
        return textures

    try:
        files = os.listdir(folder)
    except Exception as e:
        print(f"Error listing directory {folder}: {e}")
        return textures

    for filename in files:
        # 检查文件名是否包含 base_name
        if not filename.startswith(base_name):
            continue
            
        # 如果指定了分辨率，检查分辨率匹配
        if resolution and resolution.lower() not in filename.lower():
            continue
            
        filename_lower = filename.lower()
        
        # 检查每种纹理类型
        for tex_type, keywords in texture_types.items():
            if tex_type in textures: # 已经找到该类型的纹理
                continue
                
            for kw in keywords:
                # 简单的包含检查，区分大小写通常不需要，因为 filename_lower 是小写
                # 增加一些边界检查以避免误匹配 (例如 'color' 匹配 'discolor' - 不太可能但在代码中要注意)
                if kw.lower() in filename_lower:
                    textures[tex_type] = os.path.join(folder, filename)
                    break
    
    return textures

def apply_textures_to_object(obj, textures, texture_size=1.0):
    if not obj or obj.type != 'MESH':
        print(f"❌ Object {obj.name if obj else 'None'} is not a mesh!")
        return
    
    # ⭐ 核心：读取物体尺寸，计算UV缩放
    dimensions = obj.dimensions
    # If dimensions are 0 (e.g. empty mesh), avoid div by zero
    if dimensions.x == 0 or dimensions.y == 0:
         uv_scale = (1.0, 1.0, 1.0)
    else:
        uv_scale = (
            dimensions.x / texture_size,
            dimensions.y / texture_size,
            dimensions.z / texture_size
        )
    
    print(f"📏 物体尺寸: {dimensions.x:.2f}m × {dimensions.y:.2f}m × {dimensions.z:.2f}m")
    print(f"📐 贴图尺寸: {texture_size}m × {texture_size}m")
    print(f"🔢 UV缩放: ({uv_scale[0]:.2f}, {uv_scale[1]:.2f}, {uv_scale[2]:.2f})")
    
    # 创建材质
    base_name = os.path.splitext(os.path.basename(textures.get('diff', 'Material')))[0]
    mat_name = base_name.replace('_diff', '').replace('_8k', '').replace('_4k', '')
    
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # 纹理坐标系统
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-900, 300)
    
    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.location = (-700, 300)
    mapping.inputs['Scale'].default_value = uv_scale  # ⭐ 应用计算的缩放
    
    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    
    # Principled BSDF
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (300, 300)
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (600, 300)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    y_offset = 500
    
    # 漫反射
    if 'diff' in textures:
        try:
            diff_tex = nodes.new(type='ShaderNodeTexImage')
            diff_tex.image = bpy.data.images.load(textures['diff'])
            diff_tex.location = (-300, y_offset)
            links.new(mapping.outputs['Vector'], diff_tex.inputs['Vector'])
            links.new(diff_tex.outputs['Color'], bsdf.inputs['Base Color'])
            print(f"✓ 漫反射: {os.path.basename(textures['diff'])}")
            y_offset -= 300
        except Exception as e:
            print(f"Failed to load diff texture: {e}")
    
    # 粗糙度 / 光泽度
    # 优先使用 Roughness，如果没有则使用 Glossiness 并反转
    rough_path = textures.get('rough')
    gloss_path = textures.get('gloss')
    
    if rough_path:
        try:
            rough_tex = nodes.new(type='ShaderNodeTexImage')
            rough_tex.image = bpy.data.images.load(rough_path)
            rough_tex.image.colorspace_settings.name = 'Non-Color'
            rough_tex.location = (-300, y_offset)
            links.new(mapping.outputs['Vector'], rough_tex.inputs['Vector'])
            links.new(rough_tex.outputs['Color'], bsdf.inputs['Roughness'])
            print(f"✓ 粗糙度: {os.path.basename(rough_path)}")
            y_offset -= 300
        except Exception as e:
            print(f"Failed to load rough texture: {e}")
    elif gloss_path:
        try:
            gloss_tex = nodes.new(type='ShaderNodeTexImage')
            gloss_tex.image = bpy.data.images.load(gloss_path)
            gloss_tex.image.colorspace_settings.name = 'Non-Color'
            gloss_tex.location = (-600, y_offset)
            
            invert_node = nodes.new(type='ShaderNodeInvert')
            invert_node.location = (-300, y_offset)
            invert_node.inputs['Fac'].default_value = 1.0
            
            links.new(mapping.outputs['Vector'], gloss_tex.inputs['Vector'])
            links.new(gloss_tex.outputs['Color'], invert_node.inputs['Color'])
            links.new(invert_node.outputs['Color'], bsdf.inputs['Roughness'])
            print(f"✓ 光泽度 (Inverted to Roughness): {os.path.basename(gloss_path)}")
            y_offset -= 300
        except Exception as e:
            print(f"Failed to load gloss texture: {e}")

    # 法线
    if 'nor' in textures:
        try:
            nor_tex = nodes.new(type='ShaderNodeTexImage')
            nor_tex.image = bpy.data.images.load(textures['nor'])
            nor_tex.image.colorspace_settings.name = 'Non-Color'
            nor_tex.location = (-600, y_offset)
            
            normal_map = nodes.new(type='ShaderNodeNormalMap')
            normal_map.location = (-300, y_offset)
            
            links.new(mapping.outputs['Vector'], nor_tex.inputs['Vector'])
            links.new(nor_tex.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
            print(f"✓ 法线: {os.path.basename(textures['nor'])}")
            y_offset -= 300
        except Exception as e:
            print(f"Failed to load nor texture: {e}")
    
    # 置换
    if 'disp' in textures:
        try:
            disp_tex = nodes.new(type='ShaderNodeTexImage')
            disp_tex.image = bpy.data.images.load(textures['disp'])
            disp_tex.image.colorspace_settings.name = 'Non-Color'
            disp_tex.location = (-600, y_offset)
            
            disp_node = nodes.new(type='ShaderNodeDisplacement')
            disp_node.location = (300, 0)
            disp_node.inputs['Scale'].default_value = 0.1
            
            links.new(mapping.outputs['Vector'], disp_tex.inputs['Vector'])
            links.new(disp_tex.outputs['Color'], disp_node.inputs['Height'])
            links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
            print(f"✓ 置换: {os.path.basename(textures['disp'])}")
        except Exception as e:
            print(f"Failed to load disp texture: {e}")
    
    # 应用材质
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    print(f"\n✅ 材质已应用到 '{obj.name}'")

def apply_texture_from_path(obj, diff_texture_path):
    if not os.path.exists(diff_texture_path):
        print(f"Texture file not found: {diff_texture_path}")
        return
        
    folder = os.path.dirname(diff_texture_path)
    filename = os.path.basename(diff_texture_path)
    
    base_name, resolution = parse_texture_name(filename)
    
    if not base_name:
        print(f"Could not parse texture name: {filename}")
        textures = {'diff': diff_texture_path}
    else:
        textures = find_related_textures(folder, base_name, resolution)
        # Make sure we at least have the diff texture provided
        if 'diff' not in textures:
             textures['diff'] = diff_texture_path
        
    apply_textures_to_object(obj, textures, texture_size=1.0)
                        
class BlenderManager:
    """
    用于管理Blender场景中的物体操作:
      - 导入/导出FBX模型
      - 设置物体变换
      - 处理物体之间的空间关系
      - 更新和保存场景信息
    """
    def __init__(self, obj_list=None, obj_dimensions=None, tree_sons=None, processed_matrix=None, carpet=None):
        self.obj_list = obj_list if obj_list is not None else {}
        self.obj_dimensions = obj_dimensions if obj_dimensions is not None else {}
        self.tree_sons = tree_sons if tree_sons is not None else {}
        self.processed_matrix = processed_matrix if processed_matrix is not None else {}
        self.CARPET = carpet if carpet is not None else ["carpet_0", "rug_0"]
        self._loaded_assets = {}  # 缓存已加载的FBX对象
        
    def import_fbx(self, filepath):
        """导入FBX模型（优化版 + 缓存机制）"""
        
        # 检查缓存
        if filepath in self._loaded_assets:
            source_obj = self._loaded_assets[filepath]
            # 确保源对象仍然存在
            try:
                if source_obj.name in bpy.data.objects:
                    # 复制对象
                    new_obj = source_obj.copy()
                    new_obj.data = source_obj.data.copy()  # 深度复制Mesh，确保独立性
                    
                    # 链接到当前集合
                    bpy.context.collection.objects.link(new_obj)
                    
                    # 选中新对象
                    bpy.ops.object.select_all(action='DESELECT')
                    new_obj.select_set(True)
                    bpy.context.view_layer.objects.active = new_obj
                    
                    # 重置变换，确保状态干净
                    new_obj.location = (0, 0, 0)
                    new_obj.rotation_euler = (0, 0, 0)
                    new_obj.scale = (1, 1, 1)
                    
                    return new_obj
                else:
                    # 对象不存在，移除缓存
                    del self._loaded_assets[filepath]
            except Exception as e:
                print(f"Error reusing cached asset: {e}")
                pass

        # ⚡ 性能优化：优先尝试加载同名 .blend 文件
        blend_path = os.path.splitext(filepath)[0] + ".blend"
        if os.path.exists(blend_path):
            try:
                # 使用 Append 方式加载 .blend 中的所有 Object
                with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
                    data_to.objects = data_from.objects
                
                imported_objects = []
                for obj in data_to.objects:
                    if obj is not None:
                        bpy.context.collection.objects.link(obj)
                        imported_objects.append(obj)
                
                # 选中导入的物体
                bpy.ops.object.select_all(action='DESELECT')
                for obj in imported_objects:
                    obj.select_set(True)
                
                if imported_objects:
                    # 尝试找到根物体（没有父物体的物体），如果没有则返回第一个
                    root_obj = next((obj for obj in imported_objects if obj.parent is None), imported_objects[0])
                    bpy.context.view_layer.objects.active = root_obj
                    
                    # 存入缓存
                    self._loaded_assets[filepath] = root_obj
                    return root_obj
            except Exception as e:
                print(f"Error loading blend file {blend_path}: {e}, falling back to FBX.")

        # ⚡ 性能优化：跳过不必要的导入处理以加快加载速度
        bpy.ops.import_scene.fbx(
            filepath=filepath,
            use_anim=False,  # 跳过动画数据
            ignore_leaf_bones=True,  # 跳过叶子骨骼
            automatic_bone_orientation=False,  # 跳过骨骼方向计算
            use_custom_props=False,  # 跳过自定义属性
            use_custom_props_enum_as_string=False,  # 跳过枚举属性
        )
        
        obj = bpy.context.selected_objects[0]
        self._loaded_assets[filepath] = obj  # 存入缓存
        return obj
    
    def clear_scene(self,):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        # 删除所有集合
        for collection in bpy.data.collections:
            bpy.data.collections.remove(collection)
        # 删除所有孤立的数据块
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    
    def set_object_transform(self, obj, transform_matrix):
        """设置物体的变换矩阵"""
        blender_matrix = Matrix([
            transform_matrix[0], 
            transform_matrix[1], 
            transform_matrix[2], 
            transform_matrix[3]
        ])
        obj.matrix_world = blender_matrix

    def align_object_z_to_world_z(self, obj):
        """将物体的Z轴对齐到世界坐标系Z轴"""
        # 获取物体的世界变换矩阵
        world_matrix = obj.matrix_world

        # 提取物体的本地 z 轴在世界坐标系中的方向
        local_z_axis = world_matrix.to_3x3() @ Vector((0, 0, 1))
        local_z_axis.normalize()

        # 计算旋转轴和角度
        rotation_axis = local_z_axis.cross(Vector((0, 0, 1)))
        rotation_angle = local_z_axis.angle(Vector((0, 0, 1)))

        # 如果旋转轴非常小，说明已经对齐或需要 180 度旋转
        if rotation_axis.length < 1e-6:
            if local_z_axis.z > 0:
                return  # 已经对齐，无需操作
            else:
                #  180 度旋转，选择 x 轴
                rotation_axis = Vector((1, 0, 0))
                rotation_angle = 3.14159  # pi

        # 创建旋转四元数
        rotation_quat = Quaternion(rotation_axis, rotation_angle)

        # 创建新的旋转矩阵
        new_rotation = rotation_quat.to_matrix().to_4x4()

        # 保持原始位置
        new_matrix = Matrix.Translation(world_matrix.translation) @ new_rotation @ world_matrix.to_3x3().to_4x4()

        # 应用新的变换矩阵
        obj.matrix_world = new_matrix

    def extract_transform_components(self, matrix):
        """从4x4矩阵中提取变换分量"""
        # 从4x4矩阵中提取位置、旋转和缩放
        
        # 提取缩放
        # 使用矩阵的列向量长度来获取缩放值
        scale_x = matrix.col[0].xyz.length
        scale_y = matrix.col[1].xyz.length
        scale_z = matrix.col[2].xyz.length
        
        return Vector((scale_x, scale_y, scale_z))

    def get_matrix_world(self, obj):
        """获取物体的世界变换矩阵"""
        # 从当前的 matrix_world 中提取变换组件
        scale = self.extract_transform_components(obj.matrix_world)
        
        # Create translation matrix
        translation_matrix = Matrix.Translation(obj.location)
        
        # Create rotation matrix
        rotation_matrix = obj.rotation_euler.to_matrix().to_4x4()
        
        # Create scale matrix
        scale_matrix = Matrix.Scale(scale.x, 4, (1, 0, 0)) @ \
                      Matrix.Scale(scale.y, 4, (0, 1, 0)) @ \
                      Matrix.Scale(scale.z, 4, (0, 0, 1))
        
        # Combine translation, rotation and scale
        combined_matrix = translation_matrix @ rotation_matrix @ scale_matrix
        
        # Apply to object's matrix_world
        return combined_matrix

    def setup_camera(self, name):
        """设置场景相机"""
        bpy.ops.object.camera_add(location=(0, 0, 0))
        camera = bpy.context.object
        camera.name = name
        camera.data.lens = 30  # 焦距 (mm)  应该和深度估计时假设的相机参数一致
        camera.data.sensor_width = 36  # 传感器宽度 (mm)  应该和深度估计时假设的相机参数一致
        camera.data.clip_start = 0.1
        camera.data.clip_end = 100
        return camera

    def set_scene_world_render(self, enable=False):
        try:
            for window in bpy.context.window_manager.windows:
                screen = window.screen
                for area in screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                space.shading.use_scene_world_render = enable
                                return True
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False

    def ensure_object_visible(self, obj):
        if obj.type != 'MESH':
            print(f"Warning: {obj.name} is not a mesh object. Skipping material assignment.")
            return
        
        # 确保物体有材质
        if not obj.data.materials:
            mat = bpy.data.materials.new(name="Default_Material")
            obj.data.materials.append(mat)
        
        # 设置材质为不透明
        for mat in obj.data.materials:
            mat.use_nodes = True
            principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if principled_bsdf:
                principled_bsdf.inputs['Alpha'].default_value = 1.0
                
    #def render_scene(output_path, resolution_x=1920, resolution_y=1080):
    def render_scene(self, output_path, resolution_x=1024, resolution_y=1024, samples=32):
        """
        渲染场景
        
        Args:
            output_path: 输出路径
            resolution_x: 分辨率宽度
            resolution_y: 分辨率高度
            samples: 采样数（仅当 use_cycles=True 时有效）
            use_cycles: 是否使用 Cycles 渲染器（False 则使用 EEVEE，速度更快）
        """
        # 删除已有的太阳光源和相机补光
        for obj in list(bpy.data.objects):
            if obj.type == 'LIGHT':
                if obj.data.type == 'SUN' or obj.name == "Camera_Fill_Light":
                    bpy.data.objects.remove(obj, do_unlink=True)
            
        # 1. 设置世界环境光 (World Background) - 确保没有死黑
        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")
        
        world = bpy.context.scene.world
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get('Background')
        if not bg_node:
            bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
            world.node_tree.links.new(bg_node.outputs['Background'], world.node_tree.nodes['World Output'].inputs['Surface'])
        
        # 设置环境光颜色和强度
        bg_node.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0) # 浅灰色，防止死黑
        bg_node.inputs['Strength'].default_value = 1.0 

        # 2. 添加主光源 (Sun) - 模拟方向光
        bpy.ops.object.light_add(type='SUN')
        sun = bpy.context.object
        sun.location = (0, 0, 10)
        sun.data.energy = 3.0  # 稍微降低太阳强度，让环境光发挥作用
        sun.data.angle = 0.2   # 增加一点角度，使阴影边缘柔和 (弧度)

        # 3. 添加相机方向的面光 (Area Light) - 模拟闪光灯/补光，提亮主体
        scene_camera = bpy.context.scene.camera
        if scene_camera:
            bpy.ops.object.light_add(type='AREA', location=scene_camera.location, rotation=scene_camera.rotation_euler)
            area_light = bpy.context.object
            area_light.name = "Camera_Fill_Light"
            # 如果场景很大，3W 可能太暗。通常 Area Light 在 Cycles 中需要较高的 Watt 值
            # 假设场景是室内尺度 (几米范围)，尝试几百瓦
            area_light.data.energy = 50.0  
            area_light.data.size = 2.0       # 柔和补光
            
            # 将补光稍微移到相机后上方，避免产生奇怪的高光
            # (这里简单起见还是保持在相机位置，或者稍微偏置)

        print(f"使用 Cycles 渲染引擎，采样数: {samples}")
        bpy.context.scene.render.engine = 'CYCLES'
        
        # 设置渲染采样数
        bpy.context.scene.cycles.samples = samples
        
        # 设置渲染滤波阈值
        bpy.context.scene.cycles.denoising_threshold = 0.1
        
        # 配置 Cycles 设置
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA'  # 或 'OPTIX' 如果使用 NVIDIA RTX 卡
        cycles_prefs.get_devices()  # 刷新设备列表
        
        # 设置场景使用 GPU 计算
        bpy.context.scene.cycles.device = 'GPU'
        
        # 启用所有可用的 GPU 设备
        for device in cycles_prefs.devices:
            if device.type == 'CUDA':  # 或 'OPTIX'
                device.use = True
        
        # 设置渲染分辨率
        bpy.context.scene.render.resolution_x = resolution_x
        bpy.context.scene.render.resolution_y = resolution_y
        bpy.context.scene.render.pixel_aspect_x = 1.0
        bpy.context.scene.render.pixel_aspect_y = 1.0
        bpy.context.scene.render.resolution_percentage = 100
        
        # 设置输出路径和格式
        bpy.context.scene.render.filepath = output_path
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        
        # 设置活动相机
        scene_camera = bpy.context.scene.camera
        if not scene_camera:
            raise Exception("No active camera in the scene!")
        
        # 取消选择所有物体
        bpy.ops.object.select_all(action='DESELECT')
        
        # #### 下面的代码会导致相机位姿发生变化, 由于scale有部分计算是基于相机渲染的, 相机内外参需要严格与深度估计一致，所以此处不能用下面的代码
        # # 选择不匹配特定模式的物体,放入视野
        # for obj in bpy.context.scene.objects:
        #     if not re.match(r'^(wall|floor)_\d+', obj.name):
        #         obj.select_set(True)
        # # 将相机对准选中的物体
        # bpy.ops.view3d.camera_to_view_selected()
        
        self.set_scene_world_render(False)
            
        # 渲染图片
        bpy.ops.render.render(write_still=True)
    
    def get_world_bound_box(self, obj):
        """获取物体的世界坐标系包围盒"""
        world_matrix = self.get_matrix_world(obj)
        bbox = [world_matrix @ Vector(corner) for corner in obj.bound_box]
        return bbox

    def process_z(self, ground_name, obj_list, tree_sons, ground_height=0):
        """处理物体在Z轴方向上的位置关系，从地面开始"""
        ground_bbox = self.get_world_bound_box(obj_list[ground_name])
        ground_max_z = max(point.z for point in ground_bbox)

        # 首先处理直接放在地面上的物体及其后代
        for son in tree_sons.get(ground_name, []):
            if son not in obj_list:
                continue
            son_bbox = self.get_world_bound_box(obj_list[son])
            son_min_z = min(point.z for point in son_bbox)
            
            # 计算需要的位移
            delta_z = ground_max_z - son_min_z + ground_height
            
            # 调整子物体位置
            obj_list[son].location.z += delta_z
            
            # 递归调整该子物体的所有后代
            self.adjust_descendants(son, obj_list, tree_sons, delta_z)

        # 然后处理其他物体的位置关系
        self.process_other_objects(ground_name, obj_list, tree_sons, ground_max_z + ground_height)

    def adjust_descendants(self, obj_id, obj_list, tree_sons, delta_z):
        """递归调整物体及其所有后代的z位置"""
        if obj_id in tree_sons:
            for son in tree_sons[obj_id]:
                if son in obj_list:
                    obj_list[son].location.z += delta_z
                    self.adjust_descendants(son, obj_list, tree_sons, delta_z)

    def process_other_objects(self, parent_id, obj_list, tree_sons, parent_height):
        """处理非地面直接子物体的位置关系"""
        if parent_id in tree_sons:
            for son in tree_sons[parent_id]:
                if son not in obj_list:  # 是相机或需要内部摆放的物体
                    continue
                son_bbox = self.get_world_bound_box(obj_list[son])
                son_min_z = min(point.z for point in son_bbox)
                son_max_z = max(point.z for point in son_bbox)
                
                parent_bbox = self.get_world_bound_box(obj_list[parent_id])
                parent_max_z = max(point.z for point in parent_bbox)
                
                # 如果子物体高于父物体且差异超过20cm，调整子物体位置
                if son_min_z > parent_max_z and son_min_z - parent_max_z > 0.2:
                    obj_list[son].location.z -= (son_min_z - parent_max_z - 0.2)
                # 如果子物体的下表面在父物体的上表面下方，调整子物体位置
                elif son_min_z < parent_max_z:
                    obj_list[son].location.z += (parent_max_z - son_min_z)
                
                # 递归处理子物体
                self.process_other_objects(son, obj_list, tree_sons, son_max_z - son_min_z + parent_height)

    @staticmethod
    def process_rotation_against_wall(obj_name, obj_info, wall_name):
        """处理物体的旋转，使其 X 轴或 Y 轴的正负方向与墙壁对齐"""
        obj = bpy.data.objects[obj_name]
        wall = bpy.data.objects[wall_name]
        

        align_closest_axis_to_world_z(obj)
        
        wall_rotation = wall.rotation_euler.to_matrix()
        wall_normal = wall_rotation @ Vector((0, 0, 1))
        wall_normal.z = 0  # 投影到XY平面
        wall_normal.normalize()

        # 对于父物体是墙的物体，并且 alignToWallNormal==1，应该调整他们的位姿，让他们的正方向和墙的法向一致
        parent = obj_info.get('supported')
        should_align = obj_info.get('alignToWallNormal', 0) == 1
        if parent == wall_name and should_align:
            # 计算目标角度
            target_angle = math.atan2(wall_normal.y, wall_normal.x)
            
            # 设置物体旋转 (Y+ align with Wall Normal)
            obj.rotation_euler[2] = target_angle + math.pi / 2
            
            print(f"Force aligned {obj_name} positive direction (Y+) to wall {wall_name}")
            bpy.context.view_layer.objects.active = obj
            bpy.context.view_layer.update()
            return

        # 获取物体的局部 X 轴和 Y 轴在世界坐标系中的方向
        obj_x = obj.matrix_world.to_3x3() @ Vector((1, 0, 0))
        obj_y = obj.matrix_world.to_3x3() @ Vector((0, 1, 0))
        obj_x.z = obj_y.z = 0  # 投影到XY平面
        obj_x.normalize()
        obj_y.normalize()
        
        # 计算各轴与墙体法线的夹角
        angles = [
            (abs(obj_x.dot(wall_normal)), obj_x, "X+"),
            (abs((-obj_x).dot(wall_normal)), -obj_x, "X-"),
            (abs(obj_y.dot(wall_normal)), obj_y, "Y+"),
            (abs((-obj_y).dot(wall_normal)), -obj_y, "Y-")
        ]
        
        # 选择夹角最接近 1 (0°或180°) 的轴
        best_angle, best_axis, axis_name = max(angles, key=lambda x: x[0])
        
        # 计算需要旋转的角度
        dot_product = best_axis.dot(wall_normal)
        rotation_angle = math.acos(max(min(dot_product, 1), -1))
        
        # 确定旋转方向
        cross_product = best_axis.cross(wall_normal)
        rotation_direction = 1 if cross_product.z > 0 else -1
        
        # 如果夹角接近180度，选择较小的旋转角度
        if rotation_angle > math.pi/2:
            rotation_angle = math.pi - rotation_angle
            rotation_direction *= -1
        
        # 应用旋转
        obj.rotation_euler[2] += rotation_angle * rotation_direction
        
        print(f"Aligned {axis_name} axis. Rotation angle: {math.degrees(rotation_angle):.2f} degrees")
        
        bpy.context.view_layer.objects.active = obj
        bpy.context.view_layer.update()
    
    def process_rotation_against_wall_hierarchical(self, obj_info, obj_list, tree_sons):
        """按层级顺序处理靠墙物体的旋转（支持多级）"""
        # 生成层级处理顺序：从父到子
        hierarchy_levels = {}
        
        # def get_level(obj_name):
        #     if obj_name in hierarchy_levels:
        #         return hierarchy_levels[obj_name]
        #     parent = obj_info[obj_name].get('supported')
        #     if parent and parent in obj_list:
        #         hierarchy_levels[obj_name] = get_level(parent) + 1
        #     else:
        #         hierarchy_levels[obj_name] = 0  # 顶层物体层级为0
        #     return hierarchy_levels[obj_name]
        def get_level(obj_name, visited=None):
            if visited is None:
                visited = set()
            
            if obj_name in hierarchy_levels:
                return hierarchy_levels[obj_name]
            
            level = 0
            current = obj_name
            path = []
            
            while True:
                if current in visited:
                    # 检测到循环依赖，中断循环
                    print(f"警告：检测到对象的循环依赖: {current}")
                    break
                
                visited.add(current)
                path.append(current)
                
                parent = obj_info[current].get('supported')
                if parent and parent in obj_list:
                    if parent in hierarchy_levels:
                        level = hierarchy_levels[parent] + 1
                        break
                    current = parent
                    level += 1
                else:
                    break
            
            # 为路径中的所有对象分配层级
            for i, obj in enumerate(reversed(path)):
                hierarchy_levels[obj] = level - i
            
            return hierarchy_levels[obj_name]

        # 只处理需要靠墙的物体
        target_objects = [obj for obj in obj_info if obj_info[obj].get("againstWall")]
        for obj in target_objects:
            get_level(obj)

        # 按层级从浅到深排序（父级在前）
        sorted_objects = sorted(target_objects, key=lambda x: hierarchy_levels[x])

        # 按层级处理
        for obj_name in sorted_objects:
            if not obj_info[obj_name].get("againstWall"):
                continue

            # 创建针对当前物体子树的pose manager
            current_tree = {obj_name: tree_sons.get(obj_name, [])}
            pose_manager = RelativePoseManager(
                obj_list={k: v for k,v in obj_list.items() if k in [obj_name]+current_tree[obj_name]},  # 只包含当前子树
                tree_sons=current_tree,
                output_data_s2=obj_info
            )
            
            # 记录当前物体的子物体相对位姿
            pose_manager.record_relative_poses(obj_list[obj_name], current_tree[obj_name])
            
            # 处理当前物体旋转
            wall_name = obj_info[obj_name]["most_like_wall"]
            self.process_rotation_against_wall(obj_name, obj_info[obj_name], wall_name)
            
            # 恢复子物体相对位姿
            pose_manager.restore_relative_poses(obj_list[obj_name], current_tree[obj_name])
            bpy.context.view_layer.update()
            
            print(f"Processed {obj_name} (level {hierarchy_levels[obj_name]}) with {len(current_tree[obj_name])} children")

    @staticmethod
    def process_translation_against_wall(obj_info, obj_list):
        """处理靠墙物体的位置"""
        for instance_id, obj in obj_list.items():
            info = obj_info[instance_id]
            if info["againstWall"]:
                # 将 againstWall 转换为列表
                wall_ids = [info["againstWall"]] if isinstance(info["againstWall"], str) else info["againstWall"]
                    
                for wall_id in wall_ids:
                    wall = obj_list[wall_id]
                    
                    # 获取墙的旋转矩阵和法向量
                    wall_rotation = wall.rotation_euler.to_matrix()
                    normal_vector = wall_rotation @ Vector((0, 0, 1))
                    normal_vector.z = 0  # 投影到XY平面
                    normal_vector.normalize()

                    # 计算墙到场景中心的向量
                    wall_to_center = Vector((0, 0, 0)) - wall.location

                    # 判断墙的法向是否指向场景中心
                    normal_points_to_center = normal_vector.dot(wall_to_center) > 0

                    # 计算物体的中心点
                    obj_center = obj.location

                    # 计算物体中心到墙的距离
                    center_distance = (obj_center - wall.location).dot(normal_vector)

                    # 判断物体是否在墙内（基于中心点）
                    is_inside = (center_distance > 0) if normal_points_to_center else (center_distance < 0)
                    
                    # 计算物体的包围盒
                    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

                    # 找到最靠近墙和最远离墙的点
                    closest_point = min(bbox_corners, key=lambda p: (p - wall.location).dot(normal_vector))
                    farthest_point = max(bbox_corners, key=lambda p: (p - wall.location).dot(normal_vector))

                    # 计算最近点和最远点到墙的距离
                    closest_distance = (closest_point - wall.location).dot(normal_vector)
                    farthest_distance = (farthest_point - wall.location).dot(normal_vector)
                    
                    # 计算墙的厚度
                    wall_thickness = wall.dimensions[2]

                    # 根据is_inside和点的位置计算移动距离
                    if is_inside:
                        # 物体中心在墙内，移动到完全进入墙内
                        move_direction_tag = farthest_distance + wall_thickness / 2
                    else:
                        # 物体中心在墙外，移动到完全进入墙内
                        move_direction_tag = closest_distance - wall_thickness / 2

                    # 计算移动方向（始终朝向墙内移动）
                    move_direction = -normal_vector if move_direction_tag > 0 else normal_vector

                    # 计算具体的移动距离
                    if is_inside:
                        if closest_distance*farthest_distance >0:
                            distance = min(abs(closest_distance), abs(farthest_distance)) - wall_thickness / 2
                            distance = distance * move_direction
                        elif closest_distance*farthest_distance <0:
                            distance = min(abs(closest_distance), abs(farthest_distance)) + wall_thickness / 2
                            distance = -distance * move_direction
                        else:
                            distance = wall_thickness / 2 * move_direction
                    else:
                        distance = max(abs(closest_distance), abs(farthest_distance)) + wall_thickness / 2
                        distance = distance * move_direction

                    # 计算新的位置
                    new_location = obj.location + distance

                    # 保持原来的z坐标
                    new_location.z = obj.location.z

                    # 更新物体位置
                    obj.location = new_location

                    # 更新场景
                    bpy.context.view_layer.update()
                        
    def process_wall(self, wall_id, obj_list, ground_name):
        """处理墙体位置"""
        min_penetration = float('0')
        wall = obj_list[wall_id] 
        ground = obj_list[ground_name]
        ground_bbox = self.get_world_bound_box(ground)
        ground_max_z = max(point.z for point in ground_bbox)
        wall_rotation = wall.rotation_euler.to_matrix()
        normal_vector = wall_rotation @ Vector((0, 0, 1))
        
        normal_vector.z = 0  # Project to XY plane
        normal_vector.normalize()
        wall_location = wall.location + ground_max_z * normal_vector
        # print("wall_id",wall.location,wall_location)
        
        for instance_id, obj in obj_list.items():
            if re.match(r"wall_\d+", instance_id):
                continue
            if instance_id == ground_name:
                continue
            
            obj_bbox = self.get_world_bound_box(obj)
            # Calculate penetration for each bbox point
            projections = []
            for point in obj_bbox:
                diff_vector = point - wall_location
                projection = diff_vector.dot(normal_vector)
                projections.append(projection)
            
            min_projection = min(projections)
            min_penetration = min(min_penetration, min_projection)
        
        print(wall_id, min_penetration)
        wall.location += normal_vector * min_penetration

    @staticmethod
    def process_directly_facing(all_obj_info, fbx_scaling_strategy):
        for obj_name, obj_info in all_obj_info.items():
            if obj_info.get('directlyFacing', None) is None:
                continue
            
            facing_item = obj_info['directlyFacing']
            retrieved_asset = all_obj_info[facing_item]["retrieved_asset"]
            scaling_strategy = fbx_scaling_strategy[retrieved_asset]
            
            obj = bpy.data.objects[obj_name]
            facing_obj = bpy.data.objects[facing_item]
            if scaling_strategy == 'RADIAL':
                # 旋转obj, 使其正方向指向facing_obj的中心
                direction = facing_obj.location - obj.location
                angle = math.atan2(direction.y, direction.x)
                obj.rotation_euler[2] = angle + math.pi / 2
            else:
                ''''
                计算obj中心在facing_obj的局部坐标系x轴上的矢量投影长度(dis_x)，假设facing_obj在x轴上的dimension投影是(-dimension_x/2, dimension_x/2)
                计算obj中心在facing_obj的局部坐标系y轴上的矢量投影长度(dis_y)，假设facing_obj在y轴上的dimension投影是(-dimension_y/2, dimension_y/2) 
                计算min(abs(dis_x-dimension_x/2）, abs(dis_x-(-dimension_x/2)))，如果更小的是前者，那x方向的候选对齐轴就是facing_obj的x负半轴（反一下） 
                计算min(abs(dis_y-dimension_y/2）, abs(dis_y-(-dimension_y/2)))，如果更小的是前者，那y方向的候选对齐轴就是facing_obj的y负半轴（反一下） 
                接下来就是确认obj的正方向应该与facing_obj的x负半轴旋转至一致还是与facing_obj的y负半轴旋转至一致。。。
                '''
                # 计算obj中心在facing_obj局部坐标系中的位置
                local_pos = facing_obj.matrix_world.inverted() @ obj.location
                # 还要考虑facing_obj的scale带来的影响
                local_pos = local_pos*facing_obj.scale
                
                # 获取facing_obj的尺寸
                dimension_x, dimension_y, _ = facing_obj.dimensions

                # 计算在x和y轴上的投影距离
                dis_x = local_pos.x
                dis_y = local_pos.y

                # 确定x方向的候选对齐轴
                x_align = -1 if abs(dis_x - dimension_x/2) < abs(dis_x + dimension_x/2) else 1

                # 确定y方向的候选对齐轴
                y_align = -1 if abs(dis_y - dimension_y/2) < abs(dis_y + dimension_y/2) else 1

                # 确定最终的对齐方向
                if abs(dis_x) > dimension_x/2 and abs(dis_y) > dimension_y/2:
                    direction = facing_obj.location - obj.location
                    direction.normalize()
                    align_axis = facing_obj.matrix_world.to_3x3().inverted() @ direction
                elif abs(dis_x) > dimension_x/2:
                    align_axis = mathutils.Vector((x_align, 0, 0))
                elif abs(dis_y) > dimension_y/2:
                    align_axis = mathutils.Vector((0, y_align, 0))
                else:
                    # 确定最终的对齐方向
                    if min(abs(dis_x + x_align*dimension_x/2), abs(dis_y + y_align*dimension_y/2)) == abs(dis_x + x_align*dimension_x/2):
                        align_axis = mathutils.Vector((x_align, 0, 0))
                    else:
                        align_axis = mathutils.Vector((0, y_align, 0))

                # 将align_axis转换到世界坐标系
                world_align_axis = facing_obj.matrix_world.to_3x3() @ align_axis

                # 计算旋转
                rot_quat = world_align_axis.to_track_quat('-Y', 'Z')
                obj.rotation_euler = rot_quat.to_euler()

            # 更新场景
            bpy.context.view_layer.update()

class Obj:
    """
    用于存储单个物体在优化过程中的关键信息:
      - instance_id: 物体在场景中的唯一标识
      - original_pos: 原始 (x, y) 位置
      - current_pos: 当前 (x, y) 位置 (随优化更新)
      - parent_id: 父物体 ID
      - bounding_box: 包含 min, max, length, theta 等信息的字典
      - is_against_wall: 是否靠墙 (如果有对应的墙ID)
      - relation: 物体与父物体的空间关系: "inside" / "on" / "None" 等
      - pose_3d: 原始或最新的 3D 位姿矩阵 (4x4)
    """
    def __init__(self, instance_id, info, base_fbx_path):
        self.instance_id = instance_id
        self.parent_id = info.get('supported', None)
        self.is_against_wall = info.get("againstWall", None)
        self.relation = info.get("SpatialRel", None)
        self.pose_3d = info.get("pose_matrix_for_blender", None)
        fbx_name = info['retrieved_asset']
        self.fbx_path = f"{base_fbx_path}/{fbx_name}.fbx"
        # 通过 bbox 中心来初始化原始位置和当前位置
        bbox_points = np.array(info["bbox"])
        min_corner = np.min(bbox_points, axis=0)
        max_corner = np.max(bbox_points, axis=0)
        center_x = (min_corner[0] + max_corner[0]) / 2.0
        center_y = (min_corner[1] + max_corner[1]) / 2.0

        self.original_pos = (center_x, center_y)
        self.current_pos = [center_x, center_y]

        # 物体的 bounding_box 信息: 包括 min, max, length, theta 等
        length = max_corner - min_corner
        theta = 0.0
        if self.pose_3d is not None:
            # 根据 4x4 矩阵, 提取 z 轴旋转角度
            theta = math.atan2(self.pose_3d[1][0], self.pose_3d[0][0])  
        self.bounding_box = {
            "length": [length[0], length[1]],  # 只使用 x, y 长度
            "min": [float(min_corner[0]), float(min_corner[1]), float(min_corner[2])],
            "max": [float(max_corner[0]), float(max_corner[1]), float(max_corner[2])],
            "theta": float(theta),
            # 记录初始中心位置 (x, y)，用于后续计算移动距离
            "x": float(center_x),
            "y": float(center_y),
        }


# ==================== VoxelManager类 ====================
class VoxelManager:
    """用于管理场景中体素化和碰撞检测"""
    
    def __init__(self, resolution=(128, 128, 128), precomputed_voxel_dir=None):
        self.resolution = resolution
        self.voxel_grids = {}
        self.mesh_cache = {}
        self.scene_bounds = {
            'min': [float('inf'), float('inf'), float('inf')],
            'max': [-float('inf'), -float('inf'), -float('inf')]
        }
        self.voxel_size = None
        self.scene_initialized = False
        
        # 预计算体素数据目录
        self.precomputed_voxel_dir = Path(precomputed_voxel_dir) if precomputed_voxel_dir else None
        self.precomputed_cache = {}  # 缓存加载的预计算数据
        self.voxel_load_stats = {'precomputed': 0, 'realtime': 0, 'failed': 0}
        
    def initialize_scene_bounds(self, obj_dict, wall_dict):
        """预先计算整个场景的边界"""
        if self.scene_initialized:
            return
        
        standard_directions = {
            'left': np.array([1, 0, 0]),
            'right': np.array([-1, 0, 0]),
            'front': np.array([0, -1, 0]),
            'back': np.array([0, 1, 0])
        }
        
        self.wall_constraints = {}
        
        for wall_id, wall_info in wall_dict.items():
            if not wall_id.startswith('wall'):
                continue
            
            wall_pose = np.array(wall_info['pose_matrix_for_blender'])
            wall_normal = wall_pose[:3, :3] @ np.array([0, 0, 1])
            wall_normal = wall_normal / np.linalg.norm(wall_normal)
            wall_point = wall_pose[:3, 3]
            
            wall_type = max(standard_directions.items(), 
                          key=lambda x: np.dot(wall_normal, x[1]))[0]
            
            if wall_type == 'left':
                self.wall_constraints['left'] = wall_point[0]
            elif wall_type == 'right':
                self.wall_constraints['right'] = wall_point[0]
            elif wall_type == 'front':
                self.wall_constraints['front'] = wall_point[1]
            elif wall_type == 'back':
                self.wall_constraints['back'] = wall_point[1]
            
            print(f"Wall {wall_id} classified as {wall_type} with constraint value {wall_point}")
        
        print("Final wall constraints:", self.wall_constraints)
        
        for inst_id, obj in obj_dict.items():
            mesh = self.load_mesh(obj.fbx_path)
            if obj.pose_3d is not None:
                mesh = mesh.copy()
                mesh = mesh.apply_transform(obj.pose_3d)
            
            bounds = mesh.bounds
            self.scene_bounds['min'] = np.minimum(self.scene_bounds['min'], bounds[0])
            self.scene_bounds['max'] = np.maximum(self.scene_bounds['max'], bounds[1])

        SCENE_MARGIN_FACTOR = 0.25
        original_scene_size = (np.array(self.scene_bounds['max']) - np.array(self.scene_bounds['min']))
        
        if 'left' in self.wall_constraints:
            self.scene_bounds['min'][0] = min(self.wall_constraints['left'], self.scene_bounds['min'][0])
        else:
            self.scene_bounds['min'][0] -= SCENE_MARGIN_FACTOR * original_scene_size[0]

        if 'right' in self.wall_constraints:
            self.scene_bounds['max'][0] = max(self.wall_constraints['right'], self.scene_bounds['max'][0])
        else:
            self.scene_bounds['max'][0] += SCENE_MARGIN_FACTOR * original_scene_size[0]

        if 'back' in self.wall_constraints:
            self.scene_bounds['min'][1] = min(self.wall_constraints['back'], self.scene_bounds['min'][1])
        else:
            self.scene_bounds['min'][1] -= SCENE_MARGIN_FACTOR * original_scene_size[1]

        if 'front' in self.wall_constraints:
            self.scene_bounds['max'][1] = max(self.wall_constraints['front'], self.scene_bounds['max'][1])
        else:
            self.scene_bounds['max'][1] += SCENE_MARGIN_FACTOR * original_scene_size[1]
        
        scene_size = (np.array(self.scene_bounds['max']) - np.array(self.scene_bounds['min']))
        self.voxel_size = scene_size / np.array(self.resolution)
        self.voxel_size = np.array([min(self.voxel_size)] * 3)
        self.resolution = (int(round(scene_size[0] / self.voxel_size[0])),
                           int(round(scene_size[1] / self.voxel_size[1])),
                           int(round(scene_size[2] / self.voxel_size[2])))
        self.scene_initialized = True
        
        print("Scene bounds:", self.scene_bounds)
        print("Voxel size:", self.voxel_size)

    def fbx2mesh(self, fbx_path):
        with pyassimp.load(str(fbx_path)) as scene:
            mesh = scene.meshes[0]
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def load_mesh(self, fbx_path):
        """加载mesh"""
        mesh = self.fbx2mesh(fbx_path)
        return mesh

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
    
    def load_precomputed_voxels(self, mesh_path):
        """
        加载预计算的体素数据
        
        Returns:
            dict or None: 预计算的体素数据，如果不存在则返回None
        """
        if self.precomputed_voxel_dir is None:
            return None
        
        # 计算预计算文件路径
        mesh_path = Path(mesh_path)
        relative_path = mesh_path.relative_to(mesh_path.parents[0])
        voxel_file = self.precomputed_voxel_dir / relative_path.with_suffix('.voxel.pkl')
        
        # 检查缓存
        cache_key = str(mesh_path)
        if cache_key in self.precomputed_cache:
            return self.precomputed_cache[cache_key]
        
        # 尝试加载文件
        if not voxel_file.exists():
            return None
        
        try:
            with open(voxel_file, 'rb') as f:
                voxel_data = pickle.load(f)
            self.precomputed_cache[cache_key] = voxel_data
            return voxel_data
        except Exception as e:
            print(f"警告: 无法加载预计算体素 {voxel_file}: {e}")
            return None
    
    def voxelize_from_precomputed(self, voxel_data, instance_id, pose, scale=None):
        """
        从预计算的体素数据创建场景体素网格
        
        Args:
            voxel_data: 预计算的体素数据
            instance_id: 实例ID
            pose: 位姿矩阵
            scale: 缩放因子
        
        Returns:
            torch.Tensor: 体素网格
        """
        if not self.scene_initialized:
            raise RuntimeError("Scene bounds not initialized. Call initialize_scene_bounds first.")
        
        # 提取预计算数据
        voxel_indices = voxel_data['voxel_indices']  # (N, 3) int16
        origin = voxel_data['origin']  # (3,) float32
        pitch = voxel_data['pitch']  # float
        
        # 转换到世界坐标
        voxel_points_local = origin + voxel_indices * pitch
        
        # 应用缩放
        if scale is not None:
            scale_array = np.array(scale)
            voxel_points_local = voxel_points_local * scale_array
        
        # 应用位姿变换
        transform = np.array(pose)
        voxel_points_homogeneous = np.hstack([voxel_points_local, np.ones((len(voxel_points_local), 1))])
        voxel_points_world = (transform @ voxel_points_homogeneous.T).T[:, :3]
        
        # 转为Tensor放入GPU
        voxel_points_tensor = torch.from_numpy(voxel_points_world.astype(np.float32)).cuda()
        
        # 映射到整个场景的Grid系统中
        scene_min = torch.tensor(self.scene_bounds['min'], device='cuda', dtype=torch.float32)
        voxel_size_tensor = torch.tensor(self.voxel_size, device='cuda', dtype=torch.float32)
        
        relative_pos = voxel_points_tensor - scene_min
        voxel_coords = (relative_pos / voxel_size_tensor).long()
        
        # 创建全场景Grid
        grid = torch.zeros(self.resolution, dtype=torch.bool, device='cuda', requires_grad=False)
        
        # 过滤越界体素
        valid_mask = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < self.resolution[0]) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < self.resolution[1]) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < self.resolution[2])
        )
        voxel_coords = voxel_coords[valid_mask]
        
        if len(voxel_coords) == 0:
            print(f"警告: {instance_id} 的所有体素都超出场景边界")
            self.voxel_grids[instance_id] = grid
            return grid
        
        # 填充Grid
        grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = True
        
        self.voxel_grids[instance_id] = grid
        return grid

    def voxelize_object(self, mesh_path, instance_id, pose, scale=None):
        """
        将物体mesh转换为体素网格 (修复空心 + 防消失保护)
        优先尝试加载预计算的体素数据，如果不存在则实时计算
        """
        if not self.scene_initialized:
            raise RuntimeError("Scene bounds not initialized. Call initialize_scene_bounds first.")
        
        # 尝试加载预计算的体素数据
        precomputed_data = self.load_precomputed_voxels(mesh_path)
        if precomputed_data is not None:
            try:
                grid = self.voxelize_from_precomputed(precomputed_data, instance_id, pose, scale)
                self.voxel_load_stats['precomputed'] += 1
                return grid
            except Exception as e:
                print(f"警告: 使用预计算体素失败 ({instance_id}): {e}，回退到实时计算")
                self.voxel_load_stats['failed'] += 1
        
        # 回退到原始的实时体素化流程
        self.voxel_load_stats['realtime'] += 1
        
        mesh = self.load_mesh(mesh_path)
        if scale is not None:
            mesh.apply_scale(scale)

        transform = np.array(pose)
        mesh = mesh.apply_transform(transform)

        pitch = float(min(self.voxel_size))
        mesh = self.approximate_as_box_if_thin(mesh, pitch)

        # 1. 基础体素化
        voxels = mesh.voxelized(pitch=pitch, method='subdivide')
        
        if hasattr(voxels, 'matrix'):
            grid_np = voxels.matrix.copy()
        else:
            grid_np = voxels.encoding.dense.copy()

        # 获取 Origin (兼容性处理)
        if hasattr(voxels, 'origin'):
            grid_origin = voxels.origin
        elif hasattr(voxels, 'translation'):
            grid_origin = voxels.translation
        elif hasattr(voxels, 'transform'):
            grid_origin = voxels.transform[:3, 3]
        else:
            grid_origin = mesh.bounds[0]

        # ==================== 核心修改：带保护的实心化流程 ====================
        
        # A. 膨胀 (Dilation) - 封堵缝隙
        # 建议设置为 2 或 3，足以封住大部分家具底部的洞
        dilation_iter = 2
        grid_dilated = scipy.ndimage.binary_dilation(grid_np, iterations=dilation_iter)
        
        # B. 填充孔洞 (Fill Holes) - 实心化
        grid_filled = scipy.ndimage.binary_fill_holes(grid_dilated)
        
        # C. 安全腐蚀 (Safe Erosion) - 还原尺寸，但防止消失
        # 尝试腐蚀回去，次数通常比膨胀少 1 次，或者相等
        erosion_iter = 1 # 如果膨胀是2，腐蚀1比较安全；如果膨胀3，腐蚀2
        
        grid_eroded = scipy.ndimage.binary_erosion(grid_filled, iterations=erosion_iter)
        
        # --- 关键判断 ---
        if np.sum(grid_eroded) == 0:
            # 如果腐蚀把物体搞没了（针对画框等薄物体），就放弃腐蚀，使用填充后的版本
            # print(f"Notice: {instance_id} is too thin for erosion, keeping filled volume.")
            grid_final = grid_filled
        else:
            # 如果还有东西，就使用腐蚀后的版本（尺寸更准）
            grid_final = grid_eroded
            
        # ===================================================================

        # 计算体素在场景中的位置
        voxel_points_indices = np.argwhere(grid_final) # 使用 grid_final
        
        if len(voxel_points_indices) == 0:
            # 如果连膨胀后都是空的（极少见），做个兜底
            print(f"Warning: Object {instance_id} vanished completely!")
            grid = torch.zeros(self.resolution, dtype=torch.bool, device='cuda')
            self.voxel_grids[instance_id] = grid
            return grid

        # 将局部索引转为世界坐标
        voxel_points_world = grid_origin + voxel_points_indices * pitch
        
        # 转为 Tensor 放入 GPU
        voxel_points_tensor = torch.from_numpy(voxel_points_world).float().cuda()
        
        # 映射到整个场景的 Grid 系统中
        relative_pos = voxel_points_tensor - torch.tensor(self.scene_bounds['min'], device='cuda')
        voxel_coords = (relative_pos / torch.tensor(self.voxel_size, device='cuda')).long()
        
        # 创建全场景 Grid
        grid = torch.zeros(self.resolution, dtype=torch.bool, device='cuda', requires_grad=False)
        
        voxel_coords = torch.clamp(
            voxel_coords,
            torch.tensor(0, device='cuda'),
            torch.tensor(self.resolution, device='cuda') - 1
        )

        grid.index_put_(
            (voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]),
            torch.ones(len(voxel_coords), dtype=torch.bool, device='cuda'),
            accumulate=True
        )

        self.voxel_grids[instance_id] = grid
        return grid
        
    def move_grid(self, instance_id, offset):
        """移动体素网格"""
        dx, dy, dz = [int(round(o)) for o in offset]
        grid = self.voxel_grids[instance_id]
        
        if (abs(dx) >= grid.shape[0] or abs(dy) >= grid.shape[1] or abs(dz) >= grid.shape[2]):
            return False
        
        if dx > 0:
            if torch.any(grid[grid.shape[0]-dx:]):
                return False
        elif dx < 0:
            if torch.any(grid[:abs(dx)]):
                return False
            
        if dy > 0:
            if torch.any(grid[:, grid.shape[1]-dy:]):
                return False
        elif dy < 0:
            if torch.any(grid[:, :abs(dy)]):
                return False
            
        if dz > 0:
            if torch.any(grid[:, :, grid.shape[2]-dz:]):
                return False
        elif dz < 0:
            if torch.any(grid[:, :, :abs(dz)]):
                return False
        
        if dx > 0:
            grid = torch.cat([torch.zeros_like(grid[:dx]), grid[:-dx]], dim=0)
        elif dx < 0:
            grid = torch.cat([grid[-dx:], torch.zeros_like(grid[:-dx])], dim=0)
        
        if dy > 0:
            grid = torch.cat([torch.zeros_like(grid[:, :dy]), grid[:, :-dy]], dim=1)
        elif dy < 0:
            grid = torch.cat([grid[:, -dy:], torch.zeros_like(grid[:, :-dy])], dim=1)
        
        if dz > 0:
            grid = torch.cat([torch.zeros_like(grid[:, :, :dz]), grid[:, :, :-dz]], dim=2)
        elif dz < 0:
            grid = torch.cat([grid[:, :, -dz:], torch.zeros_like(grid[:, :, :-dz])], dim=2)
        
        self.voxel_grids[instance_id] = grid
        return True

    def world_to_voxel_offset(self, world_offset):
        """将世界坐标系的偏移转换为体素坐标系的偏移"""
        return world_offset / self.voxel_size

    def visualize_voxels(self, instance_ids=None, show_all=False):
        """
        可视化体素网格
        Args:
            instance_ids: 指定要可视化的物体ID列表，如果为None且show_all=True则显示所有物体
            show_all: 是否显示所有物体的体素
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if show_all:
            # 为不同物体随机分配颜色
            colors = plt.cm.rainbow(np.linspace(0, 1, len(self.voxel_grids)))
            for idx, (obj_id, grid) in enumerate(self.voxel_grids.items()):
                occupied = grid.cpu().numpy()
                x, y, z = np.where(occupied)
                ax.scatter(x, y, z, c=[colors[idx]], alpha=0.6, label=obj_id)
        elif instance_ids is not None:
            # 确保 instance_ids 是列表
            if isinstance(instance_ids, str):
                instance_ids = [instance_ids]
            
            # 指定的物体分配颜色
            colors = plt.cm.rainbow(np.linspace(0, 1, len(instance_ids)))
            for idx, obj_id in enumerate(instance_ids):
                if obj_id in self.voxel_grids:
                    occupied = self.voxel_grids[obj_id].cpu().numpy()
                    x, y, z = np.where(occupied)
                    ax.scatter(x, y, z, c=[colors[idx]], alpha=0.6, label=obj_id)
                else:
                    print(f"Warning: {obj_id} not found in voxel grids")
        else:
            print("No valid instance_ids provided and show_all=False")
            return
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if show_all or (instance_ids and len(instance_ids) > 1):
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title('Voxelized objects')
        plt.tight_layout()
        plt.savefig(f"voxel_visualization_{instance_ids}.png")


def save_voxel_debug_img_plt(obj_manager, output_path):
    """
    [加速版] 使用 matplotlib 渲染体素网格
    通过降采样 (Downsampling) 极大提高渲染速度
    """
    print("\n" + "="*60)
    print("生成体素调试图 (Matplotlib - Fast Mode)...")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: 未找到 matplotlib")
        return

    vm = obj_manager.voxel_manager
    
    # --- 优化核心: 降采样步长 ---
    # step = 1: 原分辨率 (128^3 -> 200万点, 极慢)
    # step = 2: 1/8 数据量 (64^3, 较快)
    # step = 4: 1/64 数据量 (32^3, 极快)
    # 自动根据分辨率选择步长，保持渲染在 grid < 64 左右
    raw_res = vm.resolution
    max_dim = max(raw_res)
    
    if max_dim >= 64:
        step = 2 # 中等压缩
    else:
        step = 1
        
    print(f"  原始分辨率: {raw_res}, 降采样步长: {step} (加速渲染)")

    # 计算新的分辨率
    # 使用切片 [::step] 后的形状
    temp_slice = np.zeros(raw_res)[::step, ::step, ::step]
    res = temp_slice.shape
    
    # 准备绘图数据
    voxels = np.zeros(res, dtype=bool)
    colors = np.zeros(res + (4,), dtype=np.float32)
    
    palette = [
        (1, 0, 0, 0.6), (0, 1, 0, 0.6), (0, 0, 1, 0.6), (1, 1, 0, 0.6),
        (1, 0, 1, 0.6), (0, 1, 1, 0.6), (1, 0.5, 0, 0.6), (0.5, 0, 1, 0.6)
    ]
    
    idx_counter = 0
    has_data = False
    
    print("  合并并降采样体素数据...")
    for inst_id, grid_tensor in vm.voxel_grids.items():
        if torch.is_tensor(grid_tensor):
            # 这里不用 cpu().numpy() 整个数组，太慢。
            # 先在 GPU 上切片，再转 CPU，大幅减少数据传输
            # 注意：PyTorch 的切片是 view 操作，很快
            sliced_tensor = grid_tensor[::step, ::step, ::step]
            obj_grid = sliced_tensor.detach().cpu().numpy()
        else:
            obj_grid = grid_tensor[::step, ::step, ::step]
            
        if np.sum(obj_grid) == 0:
            continue
            
        voxels |= obj_grid
        color = palette[idx_counter % len(palette)]
        colors[obj_grid] = color # 这一步利用了 numpy 的 boolean indexing
        idx_counter += 1
        has_data = True

    if not has_data:
        print("  警告: 场景为空，未生成图片")
        return

    print("  开始渲染 Plot (加速中)...")
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 关键优化: edgecolor=None 去掉网格线，渲染速度快一倍
    ax.voxels(voxels, facecolors=colors, edgecolor=None, shade=True)
    
    # 设置 Box Aspect 保持比例
    ax.set_box_aspect(res)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"S4 Voxel Debug (Step={step}) - Objs: {idx_counter}")
    
    # 鸟瞰视角
    ax.view_init(elev=30, azim=-45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100) # dpi 稍微调低一点也有助于保存速度
    plt.close(fig)
    
    print(f"  体素图已保存: {output_path}")
    print("="*60 + "\n")


class ObjManager:
    """
    用于管理场景中所有物体以及执行优化程:
     - 维护 Obj 列表并提供碰撞检测、重叠面积计算、移动距离计算、模拟退火等功能
    """
    def __init__(self, precomputed_voxel_dir=None):
        self.obj_dict = {}          # instance_id -> Obj
        self.wall_dict = {}         # wall_id -> Obj
        self.overlap_list = []      # 用于预先存储物体间可能发生重叠的对
        self.initial_state = False  # 记录 overlap_list 是否初始化
        self.total_size = 0         # overlap_list 的大小
        self.carpet_list = ["carpet_0", "rug_0"]
        self.n = 0                  # 物体总数
        self.obj_info = {}          # 存储从 placement_info_new.json 中读取的 obj_info
        self.ground_name = None     # reference_obj
        self.voxel_manager = VoxelManager(precomputed_voxel_dir=precomputed_voxel_dir)
        
        # 修改：使用 defaultdict(float) 自动初始化为 0.0，彻底解决 KeyError
        self.per_obj_loss = defaultdict(float)
        self.temp_per_obj_loss = defaultdict(float)


    def load_data(self, base_dir):
        """
        从 placement_info_new.json 中加载数据，创建 Obj 实例并存储,
        ���初始化 voxel_manager
        """
        with open(f"{base_dir}/placement_info_new.json", 'r') as f:
            placement_info = json.load(f)

        self.obj_info = placement_info["obj_info"]
        self.ground_name = placement_info["reference_obj"]

        # 加正则表达式模式
        skip_pattern = re.compile(r'^(floor_\d+|wall_\d+|scene_camera)')
        
        for instance_id, info in self.obj_info.items():
            # 如果物体名称匹配模式则跳过
            if skip_pattern.match(instance_id):
                print(f"Skipping {instance_id}")
                self.wall_dict[instance_id] = {"pose_matrix_for_blender": info["pose_matrix_for_blender"]}
                continue
                
            obj = Obj(instance_id, info)
            self.obj_dict[instance_id] = obj

        self.voxel_manager.initialize_scene_bounds(self.obj_dict,self.wall_dict)
        for instance_id, obj in self.obj_dict.items():
            print(obj.fbx_path,instance_id)
            mesh_path = Path(obj.fbx_path)
            pose = obj.pose_3d
            self.voxel_manager.voxelize_object(mesh_path, instance_id,pose,scale=[1.1,1.1,1.0])

    def build_bbox_items(self):
        """
        构建 bbox_items 列表，用于后续初始化 overlap list
        [(instance_id, bounding_box), ...]
        """
        bbox_items = []
        for inst_id, obj in self.obj_dict.items():
            bbox_items.append((inst_id, obj.bounding_box))
        return bbox_items

    def init_overlap(self):
        """
        初始化 overlap_list，用于加速频繁的重叠检测:
          - 使用 bbox 快速预筛选能发生碰撞的物体对
          - 如果两个物体的 bbox  z 轴或 x,y 平面上不可能重叠，就不放入 overlap_list
        """
        if self.initial_state:
            return

        bbox_items = self.build_bbox_items()
        self.n = len(bbox_items)
        self.overlap_list = []
        self.total_size = 0

        for i in range(self.n):
            instance_id_i, bbox_i = bbox_items[i]
            overlap_i = []
            
            # 跳过地毯等特殊物体
            if instance_id_i in self.carpet_list:
                self.overlap_list.append([])
                continue

            for j in range(i + 1, self.n):
                instance_id_j, bbox_j = bbox_items[j]
                
                # 跳过地毯
                if instance_id_j in self.carpet_list:
                    continue

                # 跳过父子关系的体对
                if (self.obj_dict[instance_id_i].parent_id == instance_id_j or 
                    self.obj_dict[instance_id_j].parent_id == instance_id_i):
                    continue

                # 检查 z 轴方向是否重叠
                if bbox_i["min"][2] >= bbox_j["max"][2] - eps or bbox_j["min"][2] >= bbox_i["max"][2] - eps:
                    continue

                # 检查 x,y 平面上的距离
                # 考虑到物体可能动，在原始 bbox 基础上增加一定余量
                margin_x = (bbox_j["length"][0] + bbox_i["length"][0]) * 0.5
                margin_y = (bbox_j["length"][1] + bbox_i["length"][1]) * 0.5
                
                if (bbox_i["min"][0] >= bbox_j["max"][0] + margin_x or 
                    bbox_j["min"][0] >= bbox_i["max"][0] + margin_x):
                    continue
                    
                if (bbox_i["min"][1] >= bbox_j["max"][1] + margin_y or 
                    bbox_j["min"][1] >= bbox_i["max"][1] + margin_y):
                    continue

                # 将可能发生碰撞的物体对添加到列表
                overlap_i.append(j)

            self.overlap_list.append(overlap_i)
            self.total_size += len(overlap_i)
            
        # 修改：初始化完成后，立即重置一次 loss 字典，确保外部调用 calc 不报错
        self.reset_temp_loss()
        # 同步 per_obj_loss 以便第一次随机采样有数据
        self.per_obj_loss = self.temp_per_obj_loss.copy()

        self.initial_state = True

    def get_obj_index(self, inst_id, bbox_items):
        """
        返回在 bbox_items 列表中的索引, 用于在 overlap_list 中找到相目
        """
        for idx, (iid, _) in enumerate(bbox_items):
            if iid == inst_id:
                return idx
        return -1

    def reset_temp_loss(self):
        """重置临时 Loss 记录"""
        # 重新生成一个 defaultdict，确保之前的累积清零
        self.temp_per_obj_loss = defaultdict(float)
        # 显式把所有 key 置为 0，防止有些物体没有任何 loss 导致 key 不存在
        for k in self.obj_dict.keys():
            self.temp_per_obj_loss[k] = 0.0

    def calc_overlap_area(self, debug_mode=False, batch_size=8):
        """使用体素化方法批量计算重叠，只检查预筛选的物体对"""
        total_overlap = 0
        bbox_items = self.build_bbox_items()
        
        # 收集所有需要检查的物体对
        pairs_to_check = []
        for i, overlap_indices in enumerate(self.overlap_list):
            if not overlap_indices:
                continue
            id_i = bbox_items[i][0]
            for j in overlap_indices:
                id_j = bbox_items[j][0]
                pairs_to_check.append((id_i, id_j))
        
        # 批量处理物体对
        for start_idx in range(0, len(pairs_to_check), batch_size):
            batch_pairs = pairs_to_check[start_idx:start_idx + batch_size]
            
            # 准备这个批次的网格
            grids_1 = []
            grids_2 = []
            for id_1, id_2 in batch_pairs:
                grids_1.append(self.voxel_manager.voxel_grids[id_1])
                grids_2.append(self.voxel_manager.voxel_grids[id_2])
            
            # 将网格堆叠成批次
            batch_grids_1 = torch.stack(grids_1)  # [batch_size, *grid_shape]
            batch_grids_2 = torch.stack(grids_2)  # [batch_size, *grid_shape]
            
            # 批量计算重叠
            batch_overlap = torch.logical_and(batch_grids_1, batch_grids_2).sum(dim=(1,2,3))
            total_overlap += batch_overlap.sum().item()
            
            # --- 修改：安全的 Loss 累加 ---
            overlap_vals = batch_overlap.tolist()
            for idx, val in enumerate(overlap_vals):
                if val > 0:
                    id_1, id_2 = batch_pairs[idx]
                    # defaultdict 不需要检查 key 是否存在
                    self.temp_per_obj_loss[id_1] += val
                    self.temp_per_obj_loss[id_2] += val
            # ---------------------------
            
            if debug_mode:
                # 输出这个批次中有重叠的物体对
                for idx, (id_1, id_2) in enumerate(batch_pairs):
                    overlap = overlap_vals[idx]
                    if overlap > 0:
                        print(f"Overlap between {id_1} and {id_2}: {overlap}")
        
        return total_overlap

    def calc_movement(self):
        """
        计算所有物体的移动距离(平方和)
        """
        total_move = 0
        for inst_id, obj in self.obj_dict.items():
            ox, oy = obj.original_pos
            cx, cy = obj.current_pos
            total_move += (cx - ox)**2 + (cy - oy)**2
        return total_move

    def calc_constraints(self):
        """
        计算物体相对于其父物体的越界程度, 若出某个范围则产生罚分
        """
        k = 2
        cost = 0
        bbox_items = self.build_bbox_items()

        for inst_id, obj in self.obj_dict.items():
            parent_id = obj.parent_id
            if parent_id is None or parent_id not in self.obj_dict:
                continue
            if obj.relation == "inside":
                # 内部关系暂时忽略
                continue

            fa_obj = self.obj_dict[parent_id]
            fa_bbox = fa_obj.bounding_box  # 父物体的 bbox

            # 当前物体位置
            cx, cy = obj.current_pos
            length_x, length_y = obj.bounding_box["length"][0], obj.bounding_box["length"][1]

            # 检查是否在父物体 bbox 的某个范围内(这里是简单示例, ��据需要微调)
            if (cx - length_x/k >= fa_bbox["min"][0] and
                cx + length_x/k <= fa_bbox["max"][0] and
                cy - length_y/k >= fa_bbox["min"][1] and
                cy + length_y/k <= fa_bbox["max"][1]):
                continue

            # 计算与父物体 bbox 中心的距离并累计
            this_cost = (cx - fa_bbox["x"])**2 + (cy - fa_bbox["y"])**2
            cost += this_cost
            
            # --- 修改：安全累加 ---
            self.temp_per_obj_loss[inst_id] += this_cost * 100
            # --------------------

        return cost

    def try_perturb_random_obj(self, iteration, max_iterations):
        """
        随机扰动一个物体 (修改版: 加权采样 + 正态分布步长)
        """
        inst_ids = list(self.obj_dict.keys())
        
        # --- 策略1：物体选择 - 加权随机采样 (Weighted Sampling) ---
        # 优先选择 Loss 大的物体
        epsilon = 1.0 
        weights = []
        for inst_id in inst_ids:
            # defaultdict 保证了即使没有记录也是 0.0
            w = self.per_obj_loss[inst_id]
            weights.append(w + epsilon)
        
        weights = np.array(weights)
        obj_probs = weights / np.sum(weights)
        
        # 按概率选择物体
        chosen_id = np.random.choice(inst_ids, p=obj_probs)
        chosen_obj = self.obj_dict[chosen_id]

        old_x, old_y = chosen_obj.current_pos

        # --- 策略2：步长选择 - 动态正态分布衰减 (Gaussian Decay) ---
        # 随着 iteration 增加，最大步长从 20 线性衰减到 1
        progress = iteration / max_iterations
        
        # 计算当前允许的最大步长 (20 -> 1)
        current_max_scale = max(1, int(20 * (1.0 - progress)))
        
        if current_max_scale == 1:
            scale = 1
        else:
            # 生成候选步长 [1, 2, ..., current_max_scale]
            candidates = np.arange(1, current_max_scale + 1)
            
            # 使用"半正态分布"逻辑生成概率权重
            # 我们希望 1 的概率最大，current_max_scale 的概率最小
            # mu = 1 (分布中心在最左侧)
            # sigma 控制衰减速度。设为 current_max_scale / 2.5 可以保证平滑的拖尾
            sigma = max(1.0, current_max_scale / 2.5)
            
            # 高斯公式: exp(- (x - mu)^2 / (2 * sigma^2))
            # x 为步长，mu 为 1
            scale_weights = np.exp(-((candidates - 1)**2) / (2 * sigma**2))
            
            # 归一化，使其和为 1
            scale_probs = scale_weights / np.sum(scale_weights)
            
            # 按概率分布采样步长
            scale = np.random.choice(candidates, p=scale_probs)
        
        # 生成扰动方向
        move_x = np.random.choice([True, False])
        move_positive = np.random.choice([True, False])
        voxel_size = self.voxel_manager.voxel_size
        perturbation = np.zeros(2)

        if move_x:
            perturbation[0] = scale * voxel_size[0] if move_positive else -scale * voxel_size[0]
        else:
            perturbation[1] = scale * voxel_size[1] if move_positive else -scale * voxel_size[1]

        # 墙壁约束处理
        wall_id = chosen_obj.is_against_wall
        if wall_id is not None:
            wall_pose = self.wall_dict[wall_id[0]]["pose_matrix_for_blender"]
            wall_np = np.array(wall_pose)
            normal_3d = wall_np[:3, :3] @ np.array([0, 0, 1])
            normal_2d = normal_3d[:2]
            normal_len = np.linalg.norm(normal_2d)
            if normal_len > 1e-9:
                normal_2d = normal_2d / normal_len
                dot_val = np.dot(perturbation, normal_2d)
                perturbation = perturbation - dot_val * normal_2d
        
        # 特殊处理 floor_lamp
        if chosen_obj.instance_id == "floor_lamp_0":
            perturbation[1] *= 10

        voxel_perturbation = np.array([int(round(perturbation[0] / voxel_size[0])), 
                                       int(round(perturbation[1] / voxel_size[1])), 0])
        
        move_success = self.voxel_manager.move_grid(chosen_obj.instance_id, voxel_perturbation)
        if not move_success:
            return lambda: None
            
        chosen_obj.current_pos[0] += perturbation[0]
        chosen_obj.current_pos[1] += perturbation[1]

        def revert():
            chosen_obj.current_pos[0] = old_x
            chosen_obj.current_pos[1] = old_y
            self.voxel_manager.move_grid(chosen_obj.instance_id, -voxel_perturbation)

        return revert

    def simulated_annealing(self, initial_temp, alpha, max_iterations, penalty_factor):
        """模拟退火优化 (修改：管理 Loss 状态)"""
        M = 100
        log_time = 400
        
        # 初始化状态
        self.reset_temp_loss()
        # 计算初始能量的同时，会填充 self.temp_per_obj_loss
        current_energy = ( M*( self.calc_overlap_area() + self.calc_constraints() ) 
                           + self.calc_movement() )
        
        # 初始状态被接受，同步 Loss 记录
        self.per_obj_loss = self.temp_per_obj_loss.copy()
        
        temperature = initial_temp

        for iteration in range(max_iterations):
            # 传入 iteration 和 max_iterations 用于计算动态步长
            revert_callback = self.try_perturb_random_obj(iteration, max_iterations)

            # 准备计算新状态能量，重置临时 Loss
            self.reset_temp_loss()
            
            new_energy = ( M*( self.calc_overlap_area() + self.calc_constraints() ) 
                           + self.calc_movement() )

            if iteration % log_time == 0:
                print(f"Iteration {iteration}, Energy: {new_energy:.2f}, Temp: {temperature:.4f}")

            delta_energy = new_energy - current_energy
            
            # Metropolis 准则
            if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
                # 接受新状态
                current_energy = new_energy
                # 关键：更新用于下一次采样的权重分布
                self.per_obj_loss = self.temp_per_obj_loss.copy()
            else:
                # 拒绝新状态，回退
                revert_callback()
                # per_obj_loss 保持不变（还是上一次成功状态的 Loss）

            temperature *= alpha

            if temperature < 1e-3 and current_energy == 0:
                print("Converged early.")
                break

        return current_energy

    def save_to_json(self, file_path, data):
        """
        将数据存储到 JSON 文件
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def main(self, base_dir):
        """
        执行流程:
          1. 读取 placement_info_new.json
          2. 预处理并初始化 overlap
          3. 运行模拟退火
          4. 将结果写回 JSON
        """
        self.load_data(base_dir)
        self.init_overlap()  # 准备 overlap_list1
        initial_temp = 100.0
        alpha = 0.99
        max_iterations = 10000
        penalty_factor = 1000.0
        # 添加可视化调用
        print("Visualizing initial voxel grids...")
        self.voxel_manager.visualize_voxels(show_all=True)

        final_energy = self.simulated_annealing(initial_temp, alpha, max_iterations, penalty_factor)

        # 将最终位置写进 final_pos.json
        final_position = {}
        for inst_id, obj in self.obj_dict.items():
            final_position[inst_id] = {
                "x": float(obj.current_pos[0]),
                "y": float(obj.current_pos[1])
            }
            moved_dist = math.sqrt(
                (obj.current_pos[0] - obj.original_pos[0])**2 + 
                (obj.current_pos[1] - obj.original_pos[1])**2
            )
            print(inst_id, "移动距离:", moved_dist)

        # 添加可视化调用
        print("Visualizing initial voxel grids...")
        self.voxel_manager.visualize_voxels(show_all=True)
        print("Final Overlap:", self.calc_overlap_area(debug_mode=True))
        print("Final Constraints:", self.calc_constraints())
        print("Final Energy:", final_energy)

        self.save_to_json(f"{base_dir}/final_pos.json", final_position)


class RelativePoseManager:
    def __init__(self, obj_list, tree_sons, output_data_s2):
        self.obj_list = obj_list
        self.tree_sons = tree_sons
        self.output_data_s2 = output_data_s2
        self.relative_poses = {}

    def record_relative_poses(self, parent_obj, sons_list):
        for son_name in sons_list:
            son_obj = bpy.data.objects[son_name]
            
            # 计算相对变换矩阵
            relative_matrix = parent_obj.matrix_world.inverted() @ son_obj.matrix_world
            self.relative_poses[son_name] = relative_matrix
            
            # 递归处理当前子对象的子对象
            if son_name in self.tree_sons:
                self.record_relative_poses(son_obj, self.tree_sons[son_name])

    def record_all(self):
        for obj_name, obj in self.obj_list.items():
            if obj_name not in self.relative_poses and obj_name in self.tree_sons and self.output_data_s2["obj_info"][obj_name]["againstWall"] is not None:
                self.record_relative_poses(obj, self.tree_sons[obj_name])

    def restore_relative_poses(self, parent_obj, sons_list):
        for son_name in sons_list:
            son_obj = bpy.data.objects[son_name]
            
            if son_name in self.relative_poses:
                # 使用存储的相对矩阵恢复子对象的矩阵
                son_obj.matrix_world = parent_obj.matrix_world @ self.relative_poses[son_name]
            
            # 递归恢复子对象
            if son_name in self.tree_sons:
                self.restore_relative_poses(son_obj, self.tree_sons[son_name])

    def restore_all(self):
        stack = [(obj_name, obj) for obj_name, obj in self.obj_list.items() if obj_name in self.tree_sons]
        processed = set()  # 用于跟踪已处理的对象
        
        while stack:
            parent_name, parent_obj = stack.pop()
            if parent_name in processed:
                continue  # 如果这个对象已经处理过，跳过它
            processed.add(parent_name)
            
            sons_list = self.tree_sons[parent_name]
            
            for son_name in sons_list:
                son_obj = bpy.data.objects[son_name]
                
                if son_name in self.relative_poses:
                    son_obj.matrix_world = parent_obj.matrix_world @ self.relative_poses[son_name]
                
                if son_name in self.tree_sons and son_name not in processed:
                    stack.append((son_name, son_obj))



'''
下面开始是内部摆放的算法函数
'''

def get_closest_subspace(obj_name, parent_name, subspaces_info):
    obj = bpy.data.objects[obj_name]
    parent_obj = bpy.data.objects[parent_name]

    original_center = obj.matrix_world.translation.copy()
    parent_center = parent_obj.matrix_world.translation.copy()

    # 找到最靠近世界坐标系 z 轴的那个轴
    identity_axes = [Vector(axis) for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]
    axes = [parent_obj.matrix_world.to_3x3() @ axis for axis in identity_axes]
    z_axis = Vector((0, 0, 1))
    closest_axis = max(axes, key=lambda axis: abs(axis.dot(z_axis)))

    # 计算物体中心到该轴的投影距离
    center_to_axis_projection = (original_center - parent_center).dot(closest_axis)

    # 计算 parent_obj 在该轴上的投影长度
    bounds = [parent_obj.matrix_world @ Vector(b) for b in parent_obj.bound_box]
    axis_projections = [(b - parent_center).dot(closest_axis) for b in bounds]
    parent_axis_length = max(axis_projections) - min(axis_projections)

    # 计算到表面的距离
    distance_to_surface = abs(abs(center_to_axis_projection) - parent_axis_length / 2)

    # 找到最近的子空间
    min_distance = float('inf')
    closest_subspace_info = None

    for subspace in subspaces_info:
        subspace_center = bpy.data.objects[subspace['name']].matrix_world.translation
        subspace_projection = (subspace_center - parent_center).dot(closest_axis)
        distance = abs(center_to_axis_projection - subspace_projection)
        if distance < min_distance:
            min_distance = distance
            closest_subspace_info = subspace

    # 判断是 on 还是 inside
    if distance_to_surface < min_distance:
        return "on", None
    else:
        return "inside", closest_subspace_info
    
def resolve_collisions_in_subspace(objects, subspace_obj, max_attempts=100):
    items_failed_and_del = []
    center = sum((obj.location for obj in objects), Vector()) / len(objects)
    directions = [obj.location - center for obj in objects]
    average_direction = sum(directions, Vector()).normalized()
    local_average_direction = subspace_obj.matrix_world.inverted() @ average_direction
    # Define local axes excluding the Z axis
    local_axes = [Vector((1, 0, 0)), Vector((0, 1, 0))]
    axis_scores = [local_average_direction.dot(axis) for axis in local_axes]
    main_axis = local_axes[np.argmax(axis_scores)]
    world_main_axis = subspace_obj.matrix_world.to_3x3() @ main_axis

    for _ in range(max_attempts):
        collision_found = False
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                if check_collision(obj1, obj2):
                    collision_found = True
                    # Try to resolve collision for obj2
                    if not resolve_single_collision(obj2, objects, subspace_obj, world_main_axis, main_axis):
                        print(f'failed, delete {obj2.name}')
                        items_failed_and_del.append(obj2.name)
                        # If unable to resolve, remove obj2
                        objects.remove(obj2)
                        bpy.data.objects.remove(obj2)
                    break  # Exit loop after handling the first collision
            if collision_found:
                break  # Exit outer loop if any collision was found and handled
        if not collision_found:
            break  # Exit if no collisions were found
    return items_failed_and_del

def resolve_single_collision(obj, objects, subspace_obj, world_main_axis, main_axis, max_adjustments=1000):
    """
    Attempt to adjust the position of a single object to resolve collisions
    along the main axis.
    """
    move_distance = 0.001  # Adjust move distance as needed

    # Transform main axis to world space
    initial_location = obj.location.copy()

    # Try moving in the positive direction
    for _ in range(max_adjustments):
        obj.location += world_main_axis * move_distance
        bpy.context.view_layer.update()

        # Check if no collisions and still within subspace
        if not any(check_collision(obj, other) for other in objects if other != obj) and check_within_subspace_direction(obj, subspace_obj, main_axis):
            return True  # Successfully resolved collision

        # Stop if out of subspace
        if not check_within_subspace_direction(obj, subspace_obj, main_axis):
            break

    # Reset to initial location
    obj.location = initial_location
    bpy.context.view_layer.update()

    # Try moving in the negative direction
    for _ in range(max_adjustments):
        obj.location -= world_main_axis * move_distance
        bpy.context.view_layer.update()

        # Check if no collisions and still within subspace
        if not any(check_collision(obj, other) for other in objects if other != obj) and check_within_subspace_direction(obj, subspace_obj, main_axis):
            return True  # Successfully resolved collision

        # Stop if out of subspace
        if not check_within_subspace_direction(obj, subspace_obj, main_axis):
            break

    # Reset to initial location if unsuccessful
    obj.location = initial_location
    bpy.context.view_layer.update()

    return False  # Could not resolve collision

def check_collision(obj1, obj2):
    """
    Check if two objects are colliding.
    """
    obj1_bounds = [obj1.matrix_world @ Vector(corner) for corner in obj1.bound_box]
    obj2_bounds = [obj2.matrix_world @ Vector(corner) for corner in obj2.bound_box]

    obj1_min = Vector((min(v[i] for v in obj1_bounds) for i in range(3)))
    obj1_max = Vector((max(v[i] for v in obj1_bounds) for i in range(3)))
    obj2_min = Vector((min(v[i] for v in obj2_bounds) for i in range(3)))
    obj2_max = Vector((max(v[i] for v in obj2_bounds) for i in range(3)))

    return all(obj1_max[i] > obj2_min[i] and obj1_min[i] < obj2_max[i] for i in range(3))

def check_within_subspace_direction(obj, subspace_obj, main_axis):
    """
    Check if the object is entirely within the subspace boundaries along a specific direction.
    """
    # Normalize the direction
    direction = main_axis.normalized()
    
    # Get local coordinates of the object's bounding box in the subspace's local space
    obj_bounds_local = [subspace_obj.matrix_world.inverted() @ (obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
    
    # Get local coordinates of the subspace's bounding box
    subspace_bounds_local = [Vector(corner) for corner in subspace_obj.bound_box]
    
    # Project the bounding box corners onto the direction vector
    obj_projections = [corner.dot(direction) for corner in obj_bounds_local]
    subspace_projections = [corner.dot(direction) for corner in subspace_bounds_local]
    
    # Calculate min and max projections for the object and subspace
    obj_min_proj = min(obj_projections)
    obj_max_proj = max(obj_projections)
    subspace_min_proj = min(subspace_projections)
    subspace_max_proj = max(subspace_projections)
    
    # Check if the object is within the subspace boundaries along the direction
    tolerance = 1e-5
    if obj_min_proj < subspace_min_proj - tolerance or obj_max_proj > subspace_max_proj + tolerance:
        return False

    return True


def find_closest_subspace(vase_name, subspaces):
    vase = bpy.data.objects[vase_name]
    vase_center = vase.matrix_world.translation
    min_distance = float('inf')
    closest_subspace = None
    
    for subspace in subspaces:
        subspace_center = bpy.data.objects[subspace['name']].matrix_world.translation
        distance = (vase_center - subspace_center).length
        if distance < min_distance:
            min_distance = distance
            closest_subspace = subspace
    
    return closest_subspace

def align_obj_to_closest_subspace(obj_name, closest_subspace_info):
    obj = bpy.data.objects[obj_name]
    original_center = obj.matrix_world.translation.copy()

    subspace_obj = bpy.data.objects[closest_subspace_info['name']]
    subspace_matrix = subspace_obj.matrix_world

    # 获取子空间的z轴
    subspace_z = subspace_matrix.to_3x3().col[2]

    # 获取物体的三个轴
    obj_axes = [obj.matrix_world.to_3x3().col[i] for i in range(3)]

    # 找到与子空间z轴最接近的物体轴
    closest_axis = max(obj_axes, key=lambda axis: abs(axis.dot(subspace_z)))

    # 确保方向正确
    if closest_axis.dot(subspace_z) < 0:
        closest_axis = -closest_axis

    # 计算旋转矩阵
    rotation_axis = closest_axis.cross(subspace_z)
    angle = closest_axis.angle(subspace_z)

    if rotation_axis.length > 0.0001:
        # 创建旋转矩阵
        rotation_matrix = Matrix.Rotation(angle, 4, rotation_axis)

        # 保存当前的变换矩阵
        original_matrix = obj.matrix_world.copy()

        # 平移到原点，旋转，然后平移回去
        obj.matrix_world.translation -= original_center
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        obj.matrix_world.translation += original_center

        # 应用原始的平移
        obj.matrix_world.translation = original_matrix.translation

    # 恢复缩放
    bpy.context.view_layer.update()

    # 将物体的边界转换到子空间坐标系
    obj_bounds_local = [subspace_obj.matrix_world.inverted() @ (obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]

    # 计算物体在子空间坐标系中的最小和最大坐标
    obj_min_local = Vector((min(v[i] for v in obj_bounds_local) for i in range(3)))
    obj_max_local = Vector((max(v[i] for v in obj_bounds_local) for i in range(3)))

    # 使用子空间的尺寸计算边界
    subspace_min_local = -0.5 * subspace_obj.dimensions
    subspace_max_local = 0.5 * subspace_obj.dimensions
    # 计算平移量，使物体进入子空间
    translation_offset_local_min = Vector((0, 0, 0))
    translation_offset_local_max = Vector((0, 0, 0))
    translation_offset_local = Vector((0, 0, 0))
    for i in range(3):
        if obj_min_local[i] < subspace_min_local[i] or obj_max_local[i] > subspace_max_local[i]:
            translation_offset_local_min[i] = subspace_min_local[i] - obj_min_local[i]
            translation_offset_local_max[i] = subspace_max_local[i] - obj_max_local[i]
            translation_offset_local[i] = (translation_offset_local_max[i] + translation_offset_local_min[i])/2

    # 将平移量从子空间坐标系转换回世界坐标系
    translation_offset_world = subspace_obj.matrix_world.to_3x3() @ translation_offset_local

    # 应用平移
    obj.matrix_world.translation += translation_offset_world
    bpy.context.view_layer.update()

    # 检查并缩放物体以适应子空间
    scale_obj_to_fit_subspace(obj, subspace_obj)
    bpy.context.view_layer.update()

    # [Fix] Apply scale so dimensions are baked into mesh, avoiding offset issues in alignment
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # 计算需要的平移量，使物体底部与子空间底部对齐
    # 下面这个代码有问题
    # 此处先把所有物体缩放应用到自己本身，这样他们的scale都是111  就不用操心move_obj_along_closest_axis_to_z中物体和subspace_obj的缩放问题了
    move_obj_along_closest_axis_to_z(obj, subspace_obj)
    bpy.context.view_layer.update()
    return

def scale_obj_to_fit_subspace(obj, subspace_obj):
    # 计算物体和子空间的边界
    obj_bounds = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    subspace_bounds = [subspace_obj.matrix_world @ Vector(corner) for corner in subspace_obj.bound_box]

    # 计算物体的最小和最大坐标
    obj_min = Vector((min(v[i] for v in obj_bounds) for i in range(3)))
    obj_max = Vector((max(v[i] for v in obj_bounds) for i in range(3)))

    # 计算子空间的最小和最大坐标
    subspace_min = Vector((min(v[i] for v in subspace_bounds) for i in range(3)))
    subspace_max = Vector((max(v[i] for v in subspace_bounds) for i in range(3)))

    # 检查物体是否在子空间内
    scale_factor = 1.0
    for i in range(3):
        obj_size = obj_max[i] - obj_min[i]
        subspace_size = subspace_max[i] - subspace_min[i]

        if obj_size > subspace_size:
            scale_factor = min(scale_factor, subspace_size / obj_size)

    # 如果需要缩放
    if scale_factor < 1.0:
        obj.scale *= scale_factor
        obj.scale *= 0.8 #不能太大
        bpy.context.view_layer.objects.active = obj
        bpy.context.view_layer.update()

    return scale_factor

def calculate_centered_translation(obj_min, obj_max, subspace_min, subspace_max, translation_offset):
    # 计算物体新的中心位置
    new_obj_center = (obj_min + obj_max) * 0.5 + translation_offset
    # 计算子空间中心位置
    subspace_center = (subspace_min + subspace_max) * 0.5

    # 计算中心偏移
    center_offset = subspace_center - new_obj_center

    # 只沿着进入方向平移
    for i in range(3):
        if translation_offset[i] != 0:
            translation_offset[i] += center_offset[i]

    return translation_offset

def find_closest_axis_to_world_z(obj):
    matrix_world = obj.matrix_world
    rotation_matrix = matrix_world.to_3x3().normalized()

    world_z = Vector((0, 0, 1))
    min_angle = float('inf')
    min_index = -1

    for i in range(3):
        axis = rotation_matrix.col[i]
        angle = world_z.angle(axis)

        if angle < min_angle:
            min_angle = angle
            min_index = i

    return rotation_matrix.col[min_index], min_index

def align_closest_axis_to_world_z(obj):
    closest_axis, axis_index = find_closest_axis_to_world_z(obj)

    # Calculate the rotation needed to align the closest axis to the world z-axis
    world_z = Vector((0, 0, 1))
    rotation_axis = closest_axis.cross(world_z)
    angle = closest_axis.angle(world_z)

    if rotation_axis.length > 0:
        rotation_axis.normalize()
        # Create a rotation matrix from the axis and angle
        rotation_matrix = Matrix.Rotation(angle, 4, rotation_axis)
        
        # Extract the translation part of the matrix
        translation = obj.matrix_world.to_translation()

        # Apply the rotation to the object's local rotation matrix
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        obj.matrix_world.translation = translation
        bpy.context.view_layer.update()
        
def move_obj_along_closest_axis_to_z(obj, target_obj):
    # 找到 target_obj 的最近轴
    closest_axis, axis_index = find_closest_axis_to_world_z(target_obj)

    # 计算物体和目标对象沿着该轴的最低点
    obj_low_point = min(
        [obj.matrix_world @ Vector(corner) for corner in obj.bound_box],
        key=lambda x: x.dot(closest_axis)
    )
    target_low_point = min(
        [target_obj.matrix_world @ Vector(corner) for corner in target_obj.bound_box],
        key=lambda x: x.dot(closest_axis)
    )

    # 计算沿该轴的移动距离
    move_distance = target_low_point.dot(closest_axis) - obj_low_point.dot(closest_axis)

    # 创建移动向量
    move_vector = closest_axis * move_distance

    # 应用移动
    obj.location += move_vector
    
    bpy.context.view_layer.objects.active = obj
    bpy.context.view_layer.update()
    

def create_subspace(name, parent_name, transform_matrix, scale_ratio):
    parent = bpy.data.objects[parent_name]
    parent_size = parent.dimensions
    parent_matrix = parent.matrix_world
    
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    cube = bpy.context.active_object
    cube.name = name
    
    # 计算缩放
    scale = [parent_size[i] * scale_ratio[i] for i in range(3)]
    
    # 应用变换矩阵
    local_matrix = Matrix(transform_matrix)
    
    # 先应用局部变换，再设置缩放
    cube.matrix_world = parent_matrix @ local_matrix
    cube.scale = scale
    
    bpy.context.view_layer.objects.active = cube
    bpy.context.view_layer.update()
    return

# 计算物体的几何中心
def calculate_geometric_center(obj):
    local_center = 0.125 * sum((Vector(b) for b in obj.bound_box), Vector())
    return obj.matrix_world @ local_center

'''
下面是对墙和地面的位姿的纠正
'''
def align_wall_to_axes(walls, ground):
    for wall in walls:
        geom_center = calculate_geometric_center(wall)
        
        # 将墙体的原点设置为几何中心
        bpy.context.view_layer.objects.active = wall
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        
        ## 离世界坐标系z轴最近的那个轴对齐z轴正或负轴
        # 获取墙体的局部坐标系轴
        local_axes = [wall.matrix_world.to_3x3() @ Vector(axis) for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]
        
        # 找到最接近z轴的局部轴
        world_z_axis = Vector((0, 0, 1))
        closest_to_z = max(local_axes, key=lambda axis: abs(axis.dot(world_z_axis)))
        
        # 计算与z轴正方向和负方向的角度
        angle_to_positive_z = closest_to_z.angle(world_z_axis)
        angle_to_negative_z = closest_to_z.angle(-world_z_axis)
        
        # 选择最小的旋转角度
        if angle_to_positive_z < angle_to_negative_z:
            target_z_axis = world_z_axis
            z_angle = angle_to_positive_z
        else:
            target_z_axis = -world_z_axis
            z_angle = angle_to_negative_z
        
        # 计算旋转轴
        z_rotation_axis = closest_to_z.cross(target_z_axis)
        if z_rotation_axis.length > 0:
            z_rotation_axis.normalize()
            z_rotation_matrix = Matrix.Rotation(z_angle, 4, z_rotation_axis)
            wall.matrix_world = Matrix.Translation(geom_center) @ z_rotation_matrix @ Matrix.Translation(-geom_center) @ wall.matrix_world
            
        bpy.context.view_layer.objects.active = wall
        bpy.context.view_layer.update()
        
    wall_nums = len(walls)
    if wall_nums == 3:
        for wall in walls:
            ## 离世界坐标系x轴最近的那个轴对齐x轴正或负轴
            # 更新局部轴
            geom_center = calculate_geometric_center(wall)
            local_axes = [wall.matrix_world.to_3x3() @ Vector(axis) for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]
            
            # 找到最接近x轴的局部轴
            world_x_axis = Vector((1, 0, 0))
            closest_to_x = max(local_axes, key=lambda axis: abs(axis.dot(world_x_axis)))
            
            # 计算与x轴正方向和负方向的角度
            angle_to_positive_x = closest_to_x.angle(world_x_axis)
            angle_to_negative_x = closest_to_x.angle(-world_x_axis)
            
            # 选择最小的旋转角度
            if angle_to_positive_x < angle_to_negative_x:
                target_x_axis = world_x_axis
                x_angle = angle_to_positive_x
            else:
                target_x_axis = -world_x_axis
                x_angle = angle_to_negative_x
            
            # 计算旋转轴
            x_rotation_axis = closest_to_x.cross(target_x_axis)
            if x_rotation_axis.length > 0:
                x_rotation_axis.normalize()
                x_rotation_matrix = Matrix.Rotation(x_angle, 4, x_rotation_axis)
                wall.matrix_world = Matrix.Translation(geom_center) @ x_rotation_matrix @ Matrix.Translation(-geom_center) @ wall.matrix_world
            
            bpy.context.view_layer.objects.active = wall
            bpy.context.view_layer.update()
    bpy.context.view_layer.update()


def align_wall_to_world_axes(wall_x, wall_y, ground):
    # 获取 wall_x 的局部 Z 轴向量
    local_z_x = wall_x.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1))
    
    # 计算 wall_x 的 Z 轴与世界 Y 轴负方向的夹角
    target_vector = mathutils.Vector((0, -1, 0))
    angle_to_y_neg = local_z_x.angle(target_vector)
    
    # 确定旋转方向
    cross_prod_x = local_z_x.cross(target_vector)
    if cross_prod_x.z < 0:
        angle_to_y_neg = -angle_to_y_neg
    
    # 计算旋转矩阵
    rotation_matrix = mathutils.Matrix.Rotation(angle_to_y_neg, 4, 'Z')
    
    # 将所有对象绕 ground 的原点旋转
    for obj in bpy.context.scene.objects:
        obj_matrix = obj.matrix_world
        obj_location = obj.location - ground.location
        obj.matrix_world = rotation_matrix @ obj_matrix
        obj.location = rotation_matrix @ obj_location + ground.location
    
    # 使 wall_y 的 Z 轴与 wall_x 的 Z 轴垂直
    local_z_x = wall_x.matrix_world.to_3x3() @ Vector((0, 0, 1))
    local_z_y = wall_y.matrix_world.to_3x3() @ Vector((0, 0, 1))
    
    # 计算 wall_y 的 Z 轴与 wall_x 的 Z 轴的夹角
    angle_to_perpendicular = local_z_y.angle(local_z_x) - (math.pi / 2)
    
    # 确定旋转方向
    cross_prod_y = local_z_y.cross(local_z_x)
    if cross_prod_y.z > 0:
        angle_to_perpendicular = -angle_to_perpendicular
    
    # 将 wall_y 绕自身原点绕世界 Z 轴旋转
    bpy.context.view_layer.objects.active = wall_y
    bpy.ops.object.select_all(action='DESELECT')
    wall_y.select_set(True)
    bpy.ops.transform.rotate(value=angle_to_perpendicular, orient_axis='Z', orient_type='GLOBAL')
    bpy.context.view_layer.update()

'''
仿真功能函数
'''
def add_rigid_body(obj,dynamic=True):
    # 添加刚体物理
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    # 设置刚体类型
    if dynamic:
        obj.rigid_body.type = 'ACTIVE'  # 静态物体

    else:
        obj.rigid_body.type = 'PASSIVE'  # 动态物体
    
    # 配置刚体属性
    obj.rigid_body.mass = 10  # 质量
    obj.rigid_body.friction = 10  # 摩擦力
    #弹性设置
    # 设置弹性为 0
    obj.rigid_body.restitution = 0.0
     # 设置线性和角度阻尼
    obj.rigid_body.linear_damping = 1  # 线性阻尼
    obj.rigid_body.angular_damping = 0.1  # 角度阻尼
        
    #设置碰撞
    num_faces = len(obj.data.polygons)
    print(f"num_faces: {num_faces}")
    print(obj.name)
    if num_faces>2000:

        # # 获取选中的物体
        # obj = bpy.context.object
        # # 添加 Solidify 修饰器
        # modifier = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
        # # 设置厚度，膨胀方向
        # modifier.thickness = 0.01
        # # 应用修饰器
        # bpy.ops.object.modifier_apply(modifier="Solidify")
        # # 获取选中的物体
        # obj = bpy.context.object

        # # 添加 Decimate 修饰器
        modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
        # 设置为 Planar 模式
        modifier.decimate_type = 'DISSOLVE'

        # 设置角度阈值（例如：15度）
        modifier.angle_limit = 15/180*3.1415926
        # # 设置简化比例（减少 50% 顶点）
        # modifier.ratio = 0.5
        # 应用修饰器
        bpy.ops.object.modifier_apply(modifier="Decimate")
        
        # 设置碰撞形状为 Mesh
        obj.rigid_body.collision_shape = 'MESH'

        # 可选：启用碰撞边距
        obj.rigid_body.use_margin = True
        obj.rigid_body.collision_margin = 0.001  # 碰撞边距值
        obj.rigid_body.use_deform = True  # 启用变形碰撞源
    else:
        obj.rigid_body.collision_shape = 'CONVEX_HULL'
        
        obj.rigid_body.use_margin = True
        obj.rigid_body.collision_margin = 0.001
                
# def process_rotation_against_wall(obj_name, obj_info, wall_name):
#     """处理物体的旋转以对齐墙壁, 这里是强迫模型的正朝向背靠墙面，这个是不合理的"""
#     # 获取墙壁的旋转矩阵并计算法向量
#     obj = bpy.data.objects[obj_name]
#     wall = bpy.data.objects[wall_name]
    
#     if not obj_info.get("natural_pose",False):
#         obj.rotation_euler[0] = 0
#         obj.rotation_euler[1] = 0
        
#     wall_rotation = wall.rotation_euler.to_matrix()
#     normal_vector = wall_rotation @ Vector((0, 0, 1))
#     normal_vector.z = 0  # 投影到XY平面
#     normal_vector.normalize()
    
#     # 计算物体的旋转角度
#     angle = math.atan2(normal_vector.y, normal_vector.x)
#     obj.rotation_euler[2] = angle + math.pi / 2


def cal_scale_refer_bbox(obj_name, scene_camera_name, bbox_size):
    """
    计算物体在 X 和 Y 方向上的缩放因子，使其在相机视图中的投影宽度和高度
    分别与指定的像素宽度和高度对齐。

    参数：
    - obj_name (str): 物体的名称。
    - scene_camera_name (str): 相机的名称。
    - bbox_width (float): 期望的物体投影的宽度（像素）。
    - bbox_height (float): 期望的物体投影的高度（像素）。

    返回值：
    - tuple: (scale_x, scale_y)，物体在 X 和 Y 方向上的缩放因子。
    """
    bbox_width, bbox_height = bbox_size
    # 获取物体和相机对象
    obj = bpy.data.objects[obj_name]
    camera = bpy.data.objects[scene_camera_name]

    scene = bpy.context.scene
    render = scene.render

    # 确保使用指定的相机作为活动相机
    scene.camera = camera

    # 获取渲染分辨率
    resolution_x = render.resolution_x * render.resolution_percentage / 100.0
    resolution_y = render.resolution_y * render.resolution_percentage / 100.0

    # 获取物体包围盒在世界坐标系中的顶点
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

    # 将包围盒的每个顶点投影到相机视图（归一化设备坐标，NDC）
    coords_ndc = [world_to_camera_view(scene, camera, corner) for corner in bbox_corners]

    # 检查物体是否在相机视野内
    if not coords_ndc:
        print("警告：无法获取物体的投影坐标。")
        return 1, 1

    # 获取包围盒在相机视图中的最小和最大X、Y值
    min_x_ndc = min(c.x for c in coords_ndc)
    max_x_ndc = max(c.x for c in coords_ndc)
    min_y_ndc = min(c.y for c in coords_ndc)
    max_y_ndc = max(c.y for c in coords_ndc)

    # 将NDC坐标转换为像素坐标
    min_x_pixel = min_x_ndc * resolution_x
    max_x_pixel = max_x_ndc * resolution_x
    min_y_pixel = (1 - max_y_ndc) * resolution_y
    max_y_pixel = (1 - min_y_ndc) * resolution_y

    # 计算物体当前投影的宽度和高度（像素）
    current_width = abs(max_x_pixel - min_x_pixel)
    current_height = abs(max_y_pixel - min_y_pixel)

    # 检查当前宽度和高度是否为零，避免除以零
    if current_width == 0 or current_height == 0:
        print("错误：物体的当前投影宽度或高度为零，无法计算缩放因子。")
        return 1, 1

    # 计算在宽度和高度方向上需要的缩放因子
    scale_axis_x = bbox_width / current_width
    scale_axis_z = bbox_height / current_height

    return scale_axis_x, scale_axis_z
    
def estimate_scale_factors(obj_name, model_init_size, obb_size, scaling_strategy, mask_is_truncated, bbox_size=None, scene_camera_name=None):
    '''
    根据模型原始尺寸 model_init_size, obb_size 和缩放策略 scaling_strategy，计算模型最终摆放时在长、宽、高上的缩放参数。
    
    缩放策略 (Scaling Strategy) 命名体系：
    基于"约束维度"的命名，直接反映算法在计算 scale 时锁定了哪些轴，或者是否依赖 Pose。
    
    - ISOTROPIC (等比缩放)
        含义：各向同性。XYZ 三轴缩放比例必须完全一致。
        适用于：球体、艺术品、枪支等不能变形的物体。
        
    - RADIAL (径向约束)
        含义：圆柱状逻辑。X和Y轴（径向）必须保持比例一致（锁定），但Z轴（轴向）可以自由缩放。
        适用于：轮胎、桶、圆桌等圆柱形物体。
        
    - ALIGNED_ANISOTROPIC (对齐的各向异性)
        含义：XYZ 三轴完全独立缩放，且严格按照当前 Pose 的 X 对 X，Y 对 Y。
        适用于：箱子、墙画，或者你非常信任 Pose 估计准确性的情况。
        
    - SORTED_ANISOTROPIC (排序的各向异性)
        含义：XYZ 三轴独立缩放，但不信任 Pose 的方向，而是通过将长宽高排序（长对长、短对短）来计算缩放。
        适用于：长方体家具（墙画、书、显示器等），防止因为 Pose 预测偏了90度导致物体被压扁。
    
    缩放策略详细逻辑:
    
    [小物体情况: max(obb_size_products) <= 0.25]
    - ISOTROPIC: 
        计算当前model在相机视角中的像素height，然后求解scale_axis_z，返回[scale_axis_z, scale_axis_z, scale_axis_z]
    - SORTED_ANISOTROPIC 或 ALIGNED_ANISOTROPIC:
        计算当前model在相机视角中的像素height和width，然后求解scale_axis_x和scale_axis_z，返回[scale_axis_x, min(scale_axis_x, scale_axis_z), scale_axis_z]
    - RADIAL:
        计算当前model在相机视角中的像素height和width，然后求解scale_axis_x和scale_axis_z，返回[scale_axis_x, scale_axis_x, scale_axis_z]
        
    [大物体情况: max(obb_size_products) > 0.25]
    - ISOTROPIC: 
        scale_h = obb_size[2] / model_init_size[2]，返回(scale_h, scale_h, scale_h)
        
    - SORTED_ANISOTROPIC: 
        - 如果 model_init_size 的最长边/最短边比例 <= 5 (如桌子柜子)：
            把 model_init_size 和 obb_size 都 sort 一下，将它们作为对应，计算三条边的 scale。
        - 如果比例 > 5 (薄片，如画和屏幕)：
            把 model_init_size 和 obb_size 都 sort 一下，将它们作为对应，计算最长的两条边的 scale，第三条边按原先模型尺寸比例计算一个适中比例。
            
    - ALIGNED_ANISOTROPIC: 
        - 如果 model_init_size 的最长边/最短边比例 <= 5 (如桌子柜子)：
            model_init_size 和 obb_size 一一对应，直接计算相应的 xyz 的 scale，这种方式考虑到了 pose
        - 如果比例 > 5 (薄片，如画和屏幕)：
            model_init_size 和 obb_size 一一对应，只计算最长的两条边的 scale，第三条边按原先模型尺寸比例计算一个适中比例。防止画或者地毯太厚
            
    - RADIAL: 
        圆柱状的物体（如桶、轮胎和圆桌）；将模型的高按照 obb 的高进行缩放，长宽按照 obb 的尺寸里最大值（长、宽）进行统一缩放。
        
    另外，对于物体mask边缘被图片边缘截断的物体(mask_is_truncated=True)，需要进行更保守的scale计算，这部分处理待完善
    '''
    # Sanitize model_init_size to avoid division by zero
    model_init_size = [max(x, eps) for x in model_init_size]

    SCALE_THRESHOLD = [0.1, 5]
    scale_factors = [1,1,1]
    def apply_threshold(scale_factors, threshold):
        return [max(min(s, threshold[1]), threshold[0]) for s in scale_factors]

    # 计算 obb_size 的乘积
    obb_size_products = [
        obb_size[0] * obb_size[1],
        obb_size[0] * obb_size[2],
        obb_size[1] * obb_size[2]
    ]
    
    # 计算 model_init_size 的乘积
    model_init_size_products = [
        model_init_size[0] * model_init_size[1],
        model_init_size[0] * model_init_size[2],
        model_init_size[1] * model_init_size[2]
    ]

    # 处理小物体的情况
    if max(obb_size_products) <= 0.25:
        scale_axis_x, scale_axis_z = cal_scale_refer_bbox(obj_name, scene_camera_name, bbox_size)
        # 加个异常判断，如果scale_axis_z远大于scale_axis_x，可能说明物体的位姿估计错了，类似偏了90°
        if max(scale_axis_x,scale_axis_z) / min(scale_axis_x,scale_axis_z)> 5:
            return [1,1,1]
            
        if scaling_strategy == 'ISOTROPIC':
            scale_factors = [scale_axis_z, scale_axis_z, scale_axis_z]
        elif scaling_strategy in ['SORTED_ANISOTROPIC', 'ALIGNED_ANISOTROPIC']:
            # scale_factors = [scale_axis_x, (scale_axis_x + scale_axis_z)/2, scale_axis_z]
            scale_factors = [scale_axis_x, min(scale_axis_x, scale_axis_z), scale_axis_z]
        elif scaling_strategy == 'RADIAL':
            scale_factors = [scale_axis_x, scale_axis_x, scale_axis_z]
                
        if max(model_init_size_products) <= 0.15:
            return apply_threshold(scale_factors, [1,5])
        else:
            return apply_threshold(scale_factors, SCALE_THRESHOLD)

    # 处理大物体的情况
    else:
        if scaling_strategy == 'ISOTROPIC':
            scale_h = obb_size[2] / model_init_size[2]
            scale_factors = [scale_h, scale_h, scale_h]
        elif scaling_strategy == 'SORTED_ANISOTROPIC':
            model_ratio = max(model_init_size) / min(model_init_size)
            sorted_obb = sorted(obb_size, reverse=True)
            sorted_model = sorted(model_init_size, reverse=True)
            if model_ratio <= 5:
                scales = [o / m for o, m in zip(sorted_obb, sorted_model)]
            else:
                scales = [sorted_obb[i] / sorted_model[i] for i in range(2)]
                scales.append((scales[0] + scales[1]) / 2)
            
            # 创建从原始尺寸到排序后索引的映射
            size_to_index = {size: i for i, size in enumerate(sorted_model)}
            # 使用映射来创建scale_factors
            scale_factors = [scales[size_to_index[size]] for size in model_init_size]
                
        elif scaling_strategy == 'ALIGNED_ANISOTROPIC':
            model_ratio = max(model_init_size) / min(model_init_size)
            if model_ratio <= 5:
                # 将模型的 3 条边对应缩放到与 obb 尺寸一致
                scale_w = obb_size[0] / model_init_size[0]
                scale_h = obb_size[1] / model_init_size[1]
                scale_l = obb_size[2] / model_init_size[2]
                scale_factors = [scale_w, scale_h, scale_l]
            else:
                sorted_sizes = sorted(enumerate(model_init_size), key=lambda x: x[1], reverse=True)
                longest_two_indices = [idx for idx, _ in sorted_sizes[:2]]
                shortest_index = sorted_sizes[-1][0]
                # 计算最长两条边的缩放比例
                scale_factors = [1.0, 1.0, 1.0]
                for i in longest_two_indices:
                    scale_factors[i] = obb_size[i] / model_init_size[i]
                # 对最短边使用适中的缩放比例
                # scale_factors[shortest_index] = (scale_factors[longest_two_indices[0]] + scale_factors[longest_two_indices[1]]) / 2
                scale_factors[shortest_index] = min(scale_factors[longest_two_indices[0]] , scale_factors[longest_two_indices[1]])
                    
        elif scaling_strategy == 'RADIAL':
            scale_h = obb_size[2] / model_init_size[2]
            max_wl_target = max(obb_size[0], obb_size[1])
            max_wl_model = max(model_init_size[0], model_init_size[1])
            scale_wl = max_wl_target / max_wl_model
            scale_factors = [scale_wl, scale_wl, scale_h]
        return apply_threshold(scale_factors, SCALE_THRESHOLD)

def format_list(lst):
    return [f"{x:.3f}" for x in lst]

def estimate_scale_factors_for_object(obj_name, pcd_obb_size, pose, retrieved_asset_bbox_size, bbox_size, scene_camera_name, scaling_strategy, mask_is_truncated):
    """
    估计物体的缩放因子。

    参数：
    - obj_name: string，物体名称
    - pcd_obb_size: ndarray，形状为 (3,)，相机坐标系下观察到的包围盒尺寸 [dx, dy, dz]
    - pose: ndarray，形状为 (4, 4)，物体的姿态矩阵（从物体坐标系到世界坐标系的变换）
    - retrieved_asset_bbox_size: ndarray，形状为 (3,)，检索到的资产的包围盒尺寸 [dx, dy, dz]
    - bbox_size: tuple，物体在图像中的像素包围盒尺寸 (width, height)
    - scene_camera_name: string，场景相机名称
    - scaling_strategy: string，缩放策略，可选值：
        - 'ISOTROPIC': 等比缩放，XYZ 三轴缩放比例完全一致
        - 'RADIAL': 径向约束，XY 轴保持比例一致，Z 轴独立
        - 'ALIGNED_ANISOTROPIC': 对齐的各向异性，XYZ 按 Pose 一一对应缩放
        - 'SORTED_ANISOTROPIC': 排序的各向异性，通过长宽高排序匹配缩放
    - mask_is_truncated: Bool，该物体的 mask 是否被图片边界截断 False 或 True

    返回：
    - scale_factor: list，长度为 3，物体在 x, y, z 方向上的缩放因子
    """
    # 提取物体的旋转矩阵
    pose = np.array(pose)
    pcd_obb_size = np.array(pcd_obb_size)
    retrieved_asset_bbox_size = np.array(retrieved_asset_bbox_size)
    rotation_matrix = pose[:3, :3]
    
    # 定义物体局部坐标轴
    local_axes = np.eye(3)
    
    # 计算物体局部坐标轴在世界坐标系中的方向
    world_axes_vectors = rotation_matrix @ local_axes
    
    # 找出物体在世界坐标系中主要对齐的轴
    alignment = np.argmax(np.abs(world_axes_vectors), axis=1)
    
    # 根据对齐情况重新排列retrieved_asset_bbox_size
    reordered_retrieved_asset_bbox_size = retrieved_asset_bbox_size[alignment]
    
    # 估计缩放因子
    reordered_scale_factor = estimate_scale_factors(
        obj_name, 
        reordered_retrieved_asset_bbox_size,
        pcd_obb_size,
        scaling_strategy,
        mask_is_truncated,
        bbox_size,
        scene_camera_name
    )
    
    # 将缩放因子映射回原始坐标系
    original_scale_factor = np.zeros(3)
    for i, axis in enumerate(alignment):
        original_scale_factor[axis] = reordered_scale_factor[i]
    original_scale_factor=original_scale_factor.tolist()
    
    print('\nobj_name:', obj_name)
    print('retrieved_asset_bbox_size:', format_list(retrieved_asset_bbox_size), 
        'pcd_obb_size:', format_list(pcd_obb_size), 
        'reordered_retrieved_asset_bbox_size:', format_list(reordered_retrieved_asset_bbox_size), 
        'reordered_scale_factor:', format_list(reordered_scale_factor),
        'original_scale_factor:', format_list(original_scale_factor))
    print('scaling_strategy:', scaling_strategy, '\n')

    return original_scale_factor

def simplify_placement(obj_placement_info):
    base_level_pattern = r'^(wall|floor|ceiling|carpet|rug)_\d+'
    first_level_obj_list = []
    second_level_obj_list = []

    # 获取一级摆放物体列表
    for obj_name, obj_info in obj_placement_info['obj_info'].items():
        parent = obj_info['supported']
        if parent and re.match(base_level_pattern, parent) and re.match(r'^(carpet|rug)_\d+', obj_name):
            obj_info['supported'] = 'floor_0'  # 地毯不作为一级摆放物体
        elif parent and not re.match(r'^(wall|floor|ceiling)_\d+', obj_name):
            first_level_obj_list.append(obj_name)

    # 获取二级摆放物体列表
    for obj_name, obj_info in obj_placement_info['obj_info'].items():
        parent = obj_info['supported']
        if parent in first_level_obj_list:
            second_level_obj_list.append(obj_name)

    # 合并三级及以上摆放物体的支持关系
    for obj_name, obj_info in obj_placement_info['obj_info'].items():
        parent = obj_info['supported']
        if parent in second_level_obj_list:
            obj_info['supported'] = obj_placement_info['obj_info'][parent]['supported']

    return obj_placement_info

# ==================== 辅助函数 ====================
def create_cuboid(name, dimensions, color=(0.5, 0.5, 0.5, 1)):
    bpy.ops.mesh.primitive_cube_add(size=1)
    cuboid = bpy.context.active_object
    cuboid.name = name
    cuboid.scale = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    material = bpy.data.materials.new(name=f"{name}_material")
    material.use_nodes = True
    node_tree = material.node_tree

    node_tree.nodes.clear()

    principled_bsdf = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    material_output = node_tree.nodes.new(type='ShaderNodeOutputMaterial')

    principled_bsdf.inputs['Base Color'].default_value = color

    node_tree.links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    if cuboid.data.materials:
        cuboid.data.materials[0] = material
    else:
        cuboid.data.materials.append(material)

    return cuboid

def process_circular_dependencies(tree_sons, obj_info):
    def dfs(node, path):
        if node in path:
            # 检测到循环依赖
            cycle = path[path.index(node):]
            print(f"警告：检测到循环依赖: {' -> '.join(cycle + [node])}")
            return cycle
        
        path.append(node)
        for child in tree_sons.get(node, [])[:]:  # 创建一个副本以便我们可以在迭代时修改
            cycle = dfs(child, path)
            if cycle:
                if node in cycle:
                    # 当前节点在循环中，移除其父子关系
                    parent = obj_info[node].get('supported')
                    if parent:
                        obj_info[node]['supported'] = None
                        print(f"移除 {node} 的父对象 {parent}")
                    # 从tree_sons中移除这个子对象
                    tree_sons[node].remove(child)
                    print(f"从tree_sons中移除 {node} 的子对象 {child}")
                    return cycle[1:] if cycle[0] == node else cycle
                else:
                    # 当前节点不在循环中，继续向上传播
                    return cycle
        path.pop()
        return None

    for root in list(tree_sons.keys()):  # 使用keys的列表，因为我们可能会在迭代过程中修改tree_sons
        dfs(root, [])

    return tree_sons, obj_info

def get_bbox_info(obj):
    """获取物体的包围盒信息（世界坐标系）"""
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    min_corner = Vector((min(v.x for v in bbox_corners), 
                        min(v.y for v in bbox_corners), 
                        min(v.z for v in bbox_corners)))
    max_corner = Vector((max(v.x for v in bbox_corners), 
                        max(v.y for v in bbox_corners), 
                        max(v.z for v in bbox_corners)))
    
    length = max_corner - min_corner
    
    return {
        "min": min_corner,
        "max": max_corner,
        "length": length
    }

def check_bbox_overlap_fast(bbox1, bbox2):
    """快速检查两个bbox是否可能重叠（复用init_overlap的逻辑）"""
    # 检查 z 轴方向是否重叠
    if bbox1["min"][2] >= bbox2["max"][2] - eps or bbox2["min"][2] >= bbox1["max"][2] - eps:
        return False
    
    # 检查 x 轴方向
    if bbox1["min"][0] >= bbox2["max"][0] - eps or bbox2["min"][0] >= bbox1["max"][0] - eps:
        return False
    
    # 检查 y 轴方向
    if bbox1["min"][1] >= bbox2["max"][1] - eps or bbox2["min"][1] >= bbox1["max"][1] - eps:
        return False
    
    return True

def check_mesh_overlap_bvh(obj1, obj2):
    """使用BVH树精确检测两个物体的网格是否重叠"""
    import bmesh
    from mathutils.bvhtree import BVHTree
    
    # 确保物体有网格数据
    if obj1.type != 'MESH' or obj2.type != 'MESH':
        return False
    
    # 创建 bmesh 并应用世界变换
    bm1 = bmesh.new()
    bm1.from_mesh(obj1.data)
    bm1.transform(obj1.matrix_world)
    
    bm2 = bmesh.new()
    bm2.from_mesh(obj2.data)
    bm2.transform(obj2.matrix_world)
    
    try:
        # 创建 BVH 树
        tree1 = BVHTree.FromBMesh(bm1)
        tree2 = BVHTree.FromBMesh(bm2)
        
        # 检测重叠
        overlap = tree1.overlap(tree2)
        
        return len(overlap) > 0
    finally:
        # 清理
        bm1.free()
        bm2.free()

def layout(obj_placement_info_json_path, placeable_area_info_folder, base_fbx_path, fbx_csv_path, output_folder, precomputed_voxel_dir=None, debug=False):
    # if os.path.exists(obj_placement_info_json_path.replace('.json','_s4.json')):
    #     print(f'{obj_placement_info_json_path} 的s4阶段已经完成, 跳过', flush=True)
    #     return
    
    # 设置双重输出
    os.makedirs(output_folder, exist_ok=True)
    scene_name = os.path.splitext(os.path.basename(obj_placement_info_json_path))[0]
    if scene_name.endswith('_placement_info'):
        scene_name = scene_name[:-len('_placement_info')]
    inference_log_path = os.path.join(output_folder, 'inference_log_s4.txt')
    sys.stdout = Logger(inference_log_path)
    
    blender_manager = BlenderManager()
    # 删除所有对象
    blender_manager.clear_scene()
    
    with open(obj_placement_info_json_path, 'r') as f:
        obj_placement_info = json.load(f)
    
    # 资产可摆放区域信息
    asset_placeable_area_json_path_dict = {}
    for file_name in os.listdir(placeable_area_info_folder):
        if file_name.endswith('json'):
            asset_name = file_name.split('.')[0]
            file_abs_path = os.path.join(placeable_area_info_folder, file_name)
            asset_placeable_area_json_path_dict[asset_name]=file_abs_path
    
    # 用于计算缩放 - 读取缩放策略 (Scaling Strategy)
    # 新命名体系：ISOTROPIC, RADIAL, ALIGNED_ANISOTROPIC, SORTED_ANISOTROPIC
    df = pd.read_csv(fbx_csv_path, skiprows=0)
    model_name_en_list = df['name_en'].tolist()
    scaling_strategy_list = df['scaling_strategy'].tolist()
    fbx_scaling_strategy = {
        str(model_name_en): str(scaling_strategy) if scaling_strategy and str(scaling_strategy) != 'nan' else None
        for model_name_en, scaling_strategy in zip(model_name_en_list, scaling_strategy_list)
    }

    # 读取 alignToWallNormal 属性
    align_to_wall_normal_list = df['alignToWallNormal'].tolist() if 'alignToWallNormal' in df.columns else [0] * len(df)
    fbx_align_to_wall_normal = {
        str(model_name_en): int(align_val) if align_val and str(align_val) != 'nan' else 0
        for model_name_en, align_val in zip(model_name_en_list, align_to_wall_normal_list)
    }
    
    # 设置相机
    scene_camera_name = "scene_camera"
    scene_camera = blender_manager.setup_camera(scene_camera_name)
    scene_camera.location = (0, 0, 0)

    resolution_x, resolution_y  = [1024, 1024]
    # resolution_x, resolution_y  = [1440, 1080]
    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y
        
    # 导入地面并获取其变换矩阵
    ground_name = obj_placement_info['reference_obj']
    ground = create_cuboid(ground_name, [10, 10, 0.04])
    
    # 获取地面相对于相机的变换矩阵
    ground_matrix = Matrix(obj_placement_info['obj_info'][ground_name]['pose_matrix_for_blender'])

    # 计算地面变换矩阵的逆矩阵
    ground_matrix_inv = ground_matrix.inverted()

    # 将地面设置为世界坐标系
    ground.matrix_world = Matrix.Identity(4)

    # 导入墙壁并应用变换
    wall_name_list = []
    for name in obj_placement_info['obj_info'].keys():
        if re.match(r'^(wall)_\d+$', name):
            wall_name_list.append(name)
    
    for wall_id in wall_name_list:
        wall = create_cuboid(wall_id, [10, 10, 0.04])  # 使用与地面相同的尺寸，您可以根据需要调整
        wall_matrix = Matrix(obj_placement_info['obj_info'][wall_id]['pose_matrix_for_blender'])
        # 将墙壁的变换矩阵转换为相对于地面的坐标系
        wall.matrix_world = ground_matrix_inv @ wall_matrix
        
    # 设置相机的变换
    camera_matrix = Matrix.Identity(4)  # 相机的初始变换矩阵
    scene_camera.matrix_world = ground_matrix_inv @ camera_matrix

    # 旋转相机180度
    rotation_angle_rad = math.radians(90)
    scene_camera.rotation_euler[0] += rotation_angle_rad

    # 更新场景
    bpy.context.view_layer.update()
    
    # 对每个墙体应用对齐
    wall_objects = [bpy.data.objects[wall_id] for wall_id in wall_name_list]
    align_wall_to_axes(wall_objects, ground)

    for obj_name, obj_info in obj_placement_info['obj_info'].items():
        # 初始化物体和support物体的摆放关系  默认是on  下面会有on和inside的判断
        obj_info['SpatialRel'] = 'on' if obj_info['supported'] else None
        
        if re.match(r'^(wall|floor)_\d+$', obj_name):
            continue
        if 'scene_camera' in obj_name.lower():
            continue
        
        retrieved_asset = obj_info["retrieved_asset"]

        # 注入 alignToWallNormal 属性
        if retrieved_asset in fbx_align_to_wall_normal:
            obj_info['alignToWallNormal'] = fbx_align_to_wall_normal[retrieved_asset]

        fbx_path = os.path.normpath(os.path.join(base_fbx_path, f'{retrieved_asset}.fbx'))
        
        # 导入FBX文件
        obj = blender_manager.import_fbx(fbx_path)
        blender_manager.ensure_object_visible(obj)
        obj.name = obj_name

        # 设置初始位姿
        pose = Matrix(obj_info["pose_matrix_for_blender"])
        obj.matrix_world = ground_matrix_inv @ pose
        # if obj_info.get("againstWall", None):
        #     process_rotation_against_wall(obj_name, obj_info, obj_info["againstWall"])
        # else:
        #     obj_info["againstWall"]=None
        if obj_info.get("againstWall", None) is None:
            obj_info["againstWall"]=None
            obj_info["isAgainstWall"]=False
        
        # 天花板上的物体取消勾选阴影投射
        parent=obj_info.get('supported')
        if re.match(r'^ceiling_\d+', parent):
            obj.visible_shadow = False
            print(f"Disabled shadow casting for ceiling object: {obj_name}")
            
        bpy.context.view_layer.objects.active = obj
        # ⚡ 性能优化：移除循环内的视图更新，改为批量更新
        # bpy.context.view_layer.update()
    
    # ⚡ 性能优化：批量更新场景（替代循环内的多次更新）
    bpy.context.view_layer.update()

    # Apply retrieved textures to Wall, Floor, Ceiling
    print("\nApplying retrieved textures...")
    s3_dir = os.path.dirname(obj_placement_info_json_path)
    result_root = os.path.dirname(s3_dir)
    texture_json_path = os.path.join(result_root, 'S2_3d_retrieval_results', 'texture_retrieval_results.json')
    
    if os.path.exists(texture_json_path):
        print(f"Loading texture retrieval results from {texture_json_path}")
        try:
            with open(texture_json_path, 'r') as f:
                texture_results = json.load(f)
                
            for obj_name, texture_path in texture_results.items():
                obj = bpy.data.objects.get(obj_name)
                if obj:
                    print(f"Applying texture to {obj_name}: {texture_path}")
                    apply_texture_from_path(obj, texture_path)
                else:
                    print(f"Warning: Object {obj_name} not found for texture application.")
        except Exception as e:
             print(f"Error applying textures: {e}")
    else:
        print(f"Texture retrieval results not found at {texture_json_path}")
    
    bpy.context.view_layer.update()

    # 取消所有三级及以上的摆放关系, 先得到所有一级摆放物体list, 再将所有三级及以上的摆放物体向二级合并
    obj_placement_info = simplify_placement(obj_placement_info)
    
    # 创建输出字典
    output_data_s1 = {
        "reference_obj": obj_placement_info['reference_obj'],
        "scene_camrea_name": scene_camera_name,
        "obj_info": obj_placement_info['obj_info']
    }
    output_data_s1['obj_info'][scene_camera_name] = {}
    
    # 遍历场景中的所有网格对象, 更新dict中的位姿信息
    for obj in bpy.data.objects:
        world_matrix = obj.matrix_world
        matrix_list = [list(row) for row in world_matrix]
        output_data_s1["obj_info"][obj.name]['pose_matrix_for_blender'] = matrix_list
            
    # 将字典转换为 JSON 格式并保存
    output_path = os.path.join(output_folder, f'{scene_name}_placement_info_s1.json')
    with open(output_path, 'w') as f:
        json.dump(output_data_s1, f, indent=2)
    
    # 开始渲染
    bpy.context.scene.camera = bpy.data.objects[scene_camera_name]
    output_path = os.path.join(output_folder, f'{scene_name}_render_s1.png')
    blender_manager.render_scene(output_path, resolution_x, resolution_y)
    print(f"S3 render_s1 poses saved to: {output_path}", flush=True)
    
    # s2
    # 处理位姿, 对齐, scale
    tree_sons = {}
    processed_matrix = {}
    obj_list = {}
    output_data_s2 = copy.deepcopy(output_data_s1)

    # 初始化对象列表和尺寸，处理父物体不为墙的那些物体，立正
    for obj_name, obj_info in output_data_s2['obj_info'].items():
        obj = bpy.data.objects[obj_name]
        
        if obj_name == scene_camera_name or obj_info.get("SpatialRel") == "inside":
            continue
        
        obj_list[obj_name] = obj
        processed_matrix[obj_name] = obj.matrix_world
        
        ### 只处理父物体不为墙的那些物体
        parent = obj_info.get('supported')
        if parent and not re.match(r'^wall_\d+', parent) and obj_info.get("SpatialRel") == "on":
            tree_sons.setdefault(parent, []).append(obj_name)
        if not re.match(r'^wall_\d+', obj.name):  # 所有物体都立正
            align_closest_axis_to_world_z(obj)
            
    # 检查tree_sons, 处理循环依赖
    tree_sons, output_data_s2["obj_info"] = process_circular_dependencies(tree_sons, output_data_s2["obj_info"])
    
    # 确保父物体是墙的物体具有 againstWall 属性，以便 process_rotation_against_wall_hierarchical 能处理它们
    for obj_name, obj_info in output_data_s2["obj_info"].items():
        parent = obj_info.get('supported')
        if parent and re.match(r'^wall_\d+', parent):
            if not obj_info.get("againstWall"):
                obj_info["againstWall"] = parent

    # 处理靠墙物体的rotation，要依层级顺序旋转：
    # 将所有一级父物体旋转，然后保留其子物体的相对位姿
    # 然后将所有二级父物体旋转，然后保留其子物体的相对位姿
    # 。。。迭代至没有父物体；然后将剩余没有处理的所有物体都进行旋转
    blender_manager.process_rotation_against_wall_hierarchical(output_data_s2["obj_info"], obj_list, tree_sons)
    bpy.context.view_layer.update()

    # 遍历场景中的所有网格对象, 更新dict中的位姿信息
    for obj in bpy.data.objects:
        if obj.name in output_data_s2["obj_info"]:
            world_matrix = obj.matrix_world
            matrix_list = [list(row) for row in world_matrix]
            output_data_s2["obj_info"][obj.name]['pose_matrix_for_blender'] = matrix_list
    
    # 处理缩放
    for obj_name, obj_info in output_data_s2['obj_info'].items():
        if obj_name == scene_camera_name:
            continue
        
        obj = bpy.data.objects[obj_name]
        retrieved_asset = obj_info["retrieved_asset"]
        
        if retrieved_asset:
            mask_is_truncated = obj_info.get("mask_is_truncated", None)
            retrieved_asset_bbox_size = bpy.data.objects[obj_name].dimensions
            pcd_obb_size = obj_info['pcd_obb_size']
            scaling_strategy = fbx_scaling_strategy[retrieved_asset]

            boxes = obj_info['boxes']
            bbox_size = [max(abs(boxes[2]- boxes[0]), 1), max(abs(boxes[3]- boxes[1]), 1)]
            pose_matrix_list = [list(row) for row in obj.matrix_world]
            scale_factors = estimate_scale_factors_for_object(obj_name, pcd_obb_size, pose_matrix_list, retrieved_asset_bbox_size, bbox_size,
                            scene_camera_name, scaling_strategy, mask_is_truncated)
            obj_info['scale'] = scale_factors
            obj.scale = scale_factors
        
        bpy.context.view_layer.objects.active = obj
        # ⚡ 性能优化：移除循环内的视图更新
        # bpy.context.view_layer.update()
    
    # ⚡ 性能优化：批量更新场景（替代缩放循环内的多次更新）
    bpy.context.view_layer.update()
    
    # 根据scene graph的group关系，让同组的物体的scale保持一致
    groups = defaultdict(list)
    for obj_name, obj_info in output_data_s2['obj_info'].items():
        if 'group' in obj_info:
            groups[obj_info['group']].append((obj_name, obj_info['scale']))
    # 对每个组进行处理
    for group, objects in groups.items():
        if len(objects) <= 1:
            continue  # 跳过只有一个物体的组

        # 收集所有scale
        all_scales = [np.array(obj[1]) for obj in objects]

        # 找出最频繁的scale（与其他scale平均距离最小的scale）
        min_avg_distance = float('inf')
        most_frequent_scale = None

        for scale in all_scales:
            avg_distance = np.mean([np.linalg.norm(scale - other_scale) for other_scale in all_scales])
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                most_frequent_scale = scale

        # 将最频繁的scale应用到组内所有物体
        for obj_name, _ in objects:
            output_data_s2['obj_info'][obj_name]['scale'] = most_frequent_scale.tolist()
            obj = bpy.data.objects[obj_name]
            obj.scale = most_frequent_scale.tolist()
            bpy.context.view_layer.objects.active = obj
            # ⚡ 性能优化：移除循环内的视图更新
            # bpy.context.view_layer.update()
    
    # ⚡ 性能优化：批量更新场景（替代组内循环的多次更新）
    bpy.context.view_layer.update()
        
    # 处理直接面对关系（如果需要）
    blender_manager.process_directly_facing(output_data_s2['obj_info'], fbx_scaling_strategy)
    
    # # 处理wall, 调整位置，以减少与其他物体的重叠, 但是墙的移动可能会很夸张, 后面可能会导致场景与参考图片差异较大
    # for obj_name, obj_info in output_data_s2["obj_info"].items():
    #     if re.match(r"wall_\d+", obj_name):
    #         blender_manager.process_wall(obj_name, obj_list, ground_name)
    # bpy.context.view_layer.update()  
    
    # 遍历场景中的所有网格对象, 更新dict中的位姿信息
    for obj in bpy.data.objects:
        if obj.name in output_data_s2["obj_info"]:
            world_matrix = obj.matrix_world
            matrix_list = [list(row) for row in world_matrix]
            output_data_s2["obj_info"][obj.name]['pose_matrix_for_blender'] = matrix_list
    
    # 只考虑旋转纠正， 不考虑二级物体的靠墙位移纠正
    base_level_pattern = r'^(wall|floor|ceiling|carpet|rug)_\d+'
    for obj_name, obj_info in output_data_s2['obj_info'].items():
        parent = obj_info.get('supported', None)
        againstWall = obj_info.get("againstWall", None)
        if againstWall:
            if parent is None or not re.match(base_level_pattern, parent):
                obj_info["isAgainstWall"] = False
                obj_info["againstWall"] = None
        
    # 处理靠墙物体的translation
    relativePoseManager = RelativePoseManager(obj_list, tree_sons, output_data_s2)
    relativePoseManager.record_all()
    blender_manager.process_translation_against_wall(output_data_s2["obj_info"], obj_list)
    bpy.context.view_layer.update()
    relativePoseManager.restore_all()
    bpy.context.view_layer.update()

    # 遍历场景中的所有网格对象, 更新dict中的位姿信息
    for obj in bpy.data.objects:
        if obj.name in output_data_s2["obj_info"]:
            world_matrix = obj.matrix_world
            matrix_list = [list(row) for row in world_matrix]
            output_data_s2["obj_info"][obj.name]['pose_matrix_for_blender'] = matrix_list
        
    # 处理内部摆放
    items_failed_and_del = []
    for obj_name, obj_info in output_data_s2['obj_info'].items():
        retrieved_asset = obj_info.get('retrieved_asset')
        if retrieved_asset in asset_placeable_area_json_path_dict.keys():
            parent_name = obj_name
            placeable_area_info = json.load(open(asset_placeable_area_json_path_dict[obj_info['retrieved_asset']], 'r'))
            
            subspaces_info = []
            closest_subspace_mapping = {}
            for subspace in placeable_area_info:
                name = subspace['name']
                transform_matrix = subspace['transform_matrix']
                scale_ratio = subspace['scale_ratio']
                create_subspace(name, parent_name, transform_matrix, scale_ratio)
                subspaces_info.append(subspace)
                closest_subspace_mapping[name] = []
            
            sub_objs_info_list = []
            for name, value in output_data_s2['obj_info'].items():
                supported = value.get('supported')
                if supported == parent_name:
                    SpatialRel, closest_subspace_info = get_closest_subspace(name, parent_name, subspaces_info)
                    value['SpatialRel'] = SpatialRel
                    print(f'检测到 {name} 要 {SpatialRel} 于 {parent_name}')
                    if SpatialRel == "inside":
                        sub_objs_info_list.append((name, closest_subspace_info))
                        closest_subspace_mapping[closest_subspace_info['name']].append(name)
            
            # [Fix] 构建完整的 obj_list 以确保 RelativePoseManager 能找到所有物体
            full_obj_list = {}
            for o in bpy.data.objects:
                if o.type == 'MESH':
                    full_obj_list[o.name] = o

            # 使用 RelativePoseManager 来管理子物体跟随
            pose_manager = RelativePoseManager(full_obj_list, tree_sons, output_data_s2)

            for sub_objs_info in sub_objs_info_list:
                name, closest_subspace_info = sub_objs_info
                
                # [Fix] 在移动物体前，记录其子物体的相对位姿
                obj = bpy.data.objects[name]
                current_sons = tree_sons.get(name, [])
                pose_manager.relative_poses = {} # 清空之前的记录
                pose_manager.record_relative_poses(obj, current_sons)

                align_obj_to_closest_subspace(name, closest_subspace_info)
                
                # [Fix] 移动后，恢复子物体的相对位姿 (即带动子物体一起移动)
                pose_manager.restore_relative_poses(obj, current_sons)
                
                # ⚡ 性能优化：移除循环内的视图更新
                # bpy.context.view_layer.update()
            
            # ⚡ 性能优化：批量更新场景（替代对齐循环内的多次更新）
            if sub_objs_info_list:
                bpy.context.view_layer.update()
            
            # 解决碰撞
            for subspace_name, obj_list in closest_subspace_mapping.items():
                if not obj_list: continue
                subspace_obj = bpy.data.objects[subspace_name]
                objects_in_subspace = [bpy.data.objects[name] for name in obj_list]
                items_failed_and_del.extend(resolve_collisions_in_subspace(objects_in_subspace, subspace_obj))
                # ⚡ 性能优化：移除循环内的视图更新
                # bpy.context.view_layer.update()
            
            # ⚡ 性能优化：批量更新场景（替代碰撞循环内的多次更新）
            if closest_subspace_mapping:
                bpy.context.view_layer.update()
                
            # 删除所有子空间对象
            for subspace in subspaces_info:
                bpy.data.objects.remove(bpy.data.objects[subspace['name']], do_unlink=True)
            bpy.context.view_layer.update()

    # 从output_data_s2['obj_info']中删除items_failed_and_del
    for key in items_failed_and_del:
        output_data_s2['obj_info'].pop(key, None)
    print(f'内部摆放阶段删除了{items_failed_and_del}')
 
    obj_list={}
    for obj_name, obj_info in output_data_s2['obj_info'].items():
        obj = bpy.data.objects[obj_name]
        if obj_name == scene_camera_name or obj_info.get("SpatialRel") == "inside":
            continue
        obj_list[obj_name] = obj

    # 处理on的z轴空间关系
    blender_manager.process_z(ground_name, obj_list, tree_sons, 0)
    bpy.context.view_layer.update()
    
    # 更新并保存结果
    for instance_id, obj_info in output_data_s2['obj_info'].items():
        obj = bpy.data.objects.get(instance_id)
        if obj and instance_id != ground_name:
            obj_info["pose_matrix_for_blender"] = [list(row) for row in blender_manager.get_matrix_world(obj)]
            obj_info["bbox"] = [list(point) for point in blender_manager.get_world_bound_box(obj)]
            obj_info["length"] = list(obj.dimensions)

    bpy.context.view_layer.update()

    # 保存结果
    output_path = os.path.join(output_folder, f'{scene_name}_placement_info_s2.json')
    with open(output_path, 'w') as f:
        json.dump(output_data_s2, f, indent=2)

    # 渲染场景
    bpy.context.scene.camera = bpy.data.objects[scene_camera_name]
    output_path = os.path.join(output_folder, f'{scene_name}_render_s2.png')
    blender_manager.render_scene(output_path, resolution_x, resolution_y)
    print(f"S3 render_s2 poses saved to: {output_path}", flush=True)

    # s3
    output_data_s3 = copy.deepcopy(output_data_s2)
    obj_manager = ObjManager(precomputed_voxel_dir=precomputed_voxel_dir)
    obj_manager.obj_info = output_data_s3["obj_info"].copy()
    obj_manager.ground_name = output_data_s3["reference_obj"]
    
    # 加正则表达式模式
    skip_pattern = r'^(wall|floor|ceiling|carpet|rug)_\d+'
    
    # 构建obj_dict和wall_dict
    for instance_id, info in obj_manager.obj_info.items():
        # 跳过墙体、地面等
        if re.match(skip_pattern, instance_id) or instance_id == scene_camera_name:
            print(f"Skipping {instance_id}")
            obj_manager.wall_dict[instance_id] = {"pose_matrix_for_blender": info["pose_matrix_for_blender"]}
            continue
        
        # 跳过内部摆放物体
        if info.get("SpatialRel") == "inside":
            print(f"Skipping inside object: {instance_id}")
            continue
        
        obj = Obj(instance_id, info, base_fbx_path)
        obj_manager.obj_dict[instance_id] = obj
    
    # 体素化
    print("\n初始化体素网格...")
    obj_manager.voxel_manager.initialize_scene_bounds(obj_manager.obj_dict, obj_manager.wall_dict)
    
    # 使用多线程加载体素数据（预计算的体素加载速度快，可以并行）
    print("\n开始体素化（多线程加载）...")
    import time
    start_time = time.time()
    
    def voxelize_single_object(args):
        """单个物体的体素化任务"""
        instance_id, obj, voxel_manager = args
        mesh_path = Path(obj.fbx_path)
        pose = obj.pose_3d
        try:
            voxel_manager.voxelize_object(mesh_path, instance_id, pose, scale=[1.1, 1.1, 1.0])
            return f"  体素化完成: {instance_id}"
        except Exception as e:
            return f"  体素化失败: {instance_id} - {str(e)}"
    
    # 准备任务列表
    tasks = [(instance_id, obj, obj_manager.voxel_manager) for instance_id, obj in obj_manager.obj_dict.items()]
    
    # 使用线程池并行处理（I/O密集型任务）
    max_workers = min(8, len(tasks))  # 最多8个线程
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(voxelize_single_object, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            print(result)
    
    voxel_time = time.time() - start_time
    
    # 打印统计信息
    stats = obj_manager.voxel_manager.voxel_load_stats
    print(f"\n体素化统计:")
    print(f"  预计算加载: {stats['precomputed']} 个")
    print(f"  实时计算: {stats['realtime']} 个")
    print(f"  加载失败: {stats['failed']} 个")
    print(f"  总耗时: {voxel_time:.2f}秒")
    if stats['precomputed'] > 0:
        avg_time = voxel_time / (stats['precomputed'] + stats['realtime'])
        print(f"  平均耗时: {avg_time:.3f}秒/物体")
    
    # 初始化重叠检测
    print("\n初始化重叠检测...")
    obj_manager.init_overlap()
    
    # 计算初始状态
    print("\n初始状态:")
    print(f"  Initial Overlap: {obj_manager.calc_overlap_area(debug_mode=True)}")
    print(f"  Initial Constraints: {obj_manager.calc_constraints()}")
    
    # 运行模拟退火优化
    print("\n开始模拟退火优化...")
    initial_temp = 100.0
    alpha = 0.99
    max_iterations = 5000
    penalty_factor = 1000.0
    
    final_energy = obj_manager.simulated_annealing(initial_temp, alpha, max_iterations, penalty_factor)
    
    # 输出优化结果
    print("\n" + "="*60)
    print("优化完成!")
    print("="*60)
    
    final_position = {}
    for inst_id, obj in obj_manager.obj_dict.items():
        final_position[inst_id] = {
            "x": float(obj.current_pos[0]),
            "y": float(obj.current_pos[1])
        }
        moved_dist = math.sqrt(
            (obj.current_pos[0] - obj.original_pos[0])**2 + 
            (obj.current_pos[1] - obj.original_pos[1])**2
        )
        print(f"{inst_id} 移动距离: {moved_dist:.4f}")
    
    print(f"\nFinal Overlap: {obj_manager.calc_overlap_area(debug_mode=True)}")
    print(f"Final Constraints: {obj_manager.calc_constraints()}")
    print(f"Final Energy: {final_energy}")
    
    # 保存体素可视化图像（仅在debug模式下）
    if debug:
        voxel_output_path = os.path.join(output_folder, f'{scene_name}_final_voxel_visualization.png')
        save_voxel_debug_img_plt(obj_manager, voxel_output_path)
    
    # 更新Blender场景中的物体位置
    print("\n更新Blender场景...")
    for instance_id, info in output_data_s3["obj_info"].items():
        if instance_id == scene_camera_name or instance_id == ground_name:
            continue
        if instance_id in final_position:
            obj = bpy.data.objects.get(instance_id)
            if obj:
                refined_pose = info['pose_matrix_for_blender']
                refined_pose[0][3] = final_position[instance_id]['x']  # 更新X位置
                refined_pose[1][3] = final_position[instance_id]['y']  # 更新Y位置
                
                info['pose_matrix_for_blender'] = refined_pose
                obj.matrix_world = Matrix(refined_pose)
    
    bpy.context.view_layer.update()
    
    # 更新内部摆放物体的位姿
    for instance_id, info in output_data_s3['obj_info'].items():
        if info.get("SpatialRel", None) == "inside":
            parent_name = info['supported']
            
            ori_obj_pose = Matrix(output_data_s2['obj_info'][instance_id]['pose_matrix_for_blender'])
            ori_parent_pose = Matrix(output_data_s2['obj_info'][parent_name]['pose_matrix_for_blender'])
            
            parent = bpy.data.objects.get(parent_name)
            if parent:
                relative_transform = ori_parent_pose.inverted() @ ori_obj_pose
                
                obj = bpy.data.objects.get(instance_id)
                if obj:
                    obj.matrix_world = parent.matrix_world @ relative_transform
    
    bpy.context.view_layer.update()
    
    # 保存最终的位姿信息
    for instance_id, info in output_data_s3['obj_info'].items():
        obj = bpy.data.objects.get(instance_id)
        if obj:
            output_data_s3['obj_info'][instance_id]['pose_matrix_for_blender'] = [
                list(row) for row in blender_manager.get_matrix_world(obj)
            ]

    output_path = os.path.join(output_folder, f'{scene_name}_placement_info_s3.json')
    with open(output_path, 'w') as f:
        json.dump(output_data_s3, f, indent=2)
        
    
    # 开始渲染
    bpy.context.scene.camera = bpy.data.objects[scene_camera_name]
    output_path = os.path.join(output_folder, f'{scene_name}_render_s3.png')
    blender_manager.render_scene(output_path, resolution_x, resolution_y)
    print(f"S3 render_s3 poses saved to: {output_path}", flush=True)
    

    # 使用 run_drop_simulation 进行物理仿真
    print("[PhysicsSimulation] 开始使用 run_drop_simulation 进行物理仿真...")
    
    # 收集所有有directlyFacing关系的物体ID
    directly_facing_objects = set()
    for instance_id, info in output_data_s3['obj_info'].items():
        if info.get("directlyFacing"):
            directly_facing_objects.add(instance_id)
            directly_facing_objects.add(info["directlyFacing"])
    
    # 1. 先找出所有inside_objects（内部摆放物体）
    inside_objects = []
    for instance_id, info in output_data_s3['obj_info'].items():
        if instance_id == scene_camera_name:
            continue
        if info.get("SpatialRel", None) == 'inside':
            inside_objects.append(instance_id)
    
    # 2. 找active物体（需要下落模拟的物体）
    active_objects = []
    for instance_id, info in output_data_s3['obj_info'].items():
        if instance_id == scene_camera_name:
            continue
        if instance_id in inside_objects:
            continue

        # 如果父物体是墙体或天花板，则作为passive
        parent_id = info['supported']
        if parent_id is None:
            continue
        if re.match(r'(wall|ceiling)_\d+', parent_id):
            continue
        if re.match(r'(carpet|rug)_\d+', instance_id):
            continue
        
        # 而对于地面的物体需要进一步判断
        is_first_level = re.match(r'(ground|floor|carpet|rug)_\d+', parent_id)
        
        if is_first_level:
            # 一级物体：如果有agentwall或directlyFacing关系，不是active
            if info.get("againstWall") or instance_id in directly_facing_objects:
                continue
            else:
                # 否则作为active
                obj = bpy.data.objects[instance_id]
                current_matrix = Matrix(info["pose_matrix_for_blender"])
                current_matrix[2][3] += 0.01  # z坐标 +0.01
                obj.matrix_world = current_matrix
                info["pose_matrix_for_blender"] = [list(row) for row in current_matrix]
                active_objects.append(obj)
        else:
            # 非一级物体（二级及以上）也作为active
            obj = bpy.data.objects[instance_id]
            current_matrix = Matrix(info["pose_matrix_for_blender"])
            current_matrix[2][3] += 0.01  # z坐标 +0.01
            obj.matrix_world = current_matrix
            info["pose_matrix_for_blender"] = [list(row) for row in current_matrix]
            active_objects.append(obj)
    
    # 3. 剩下的所有物体作为passive物体（排除inside_objects和carpet/rug）
    active_obj_names = set([obj.name for obj in active_objects])
    passive_objects = []
    for instance_id, info in output_data_s3['obj_info'].items():
        if instance_id == scene_camera_name:
            continue
        if instance_id in inside_objects:
            continue
        # 排除carpet/rug物体
        if re.match(r'(carpet|rug)_\d+', instance_id):
            continue
        if instance_id not in active_obj_names:
            obj = bpy.data.objects[instance_id]
            passive_objects.append(obj)
    
    # 4. 检测active物体与其他物体的干涉，有干涉的转为passive
    print("[PhysicsSimulation] 检测物体干涉...")
    
    # 收集所有非active物体（passive + inside）用于碰撞检测
    other_objects = passive_objects.copy()
    for instance_id in inside_objects:
        obj = bpy.data.objects[instance_id]
        other_objects.append(obj)
    
    # 预计算所有物体的bbox信息
    print(f"[PhysicsSimulation] 预计算bbox信息...")
    active_bboxes = {obj: get_bbox_info(obj) for obj in active_objects}
    other_bboxes = {obj: get_bbox_info(obj) for obj in other_objects}
    
    # 检查每个active物体是否与其他物体（包括其他active物体）有干涉
    active_to_remove = []
    collision_count = 0
    
    for i, active_obj in enumerate(active_objects):
        has_collision = False
        active_bbox = active_bboxes[active_obj]
        
        # 检测对象列表：包括其他active物体（避免重复检测）+ passive/inside物体
        check_objects = []
        
        # 1. 添加其他active物体（只检测索引更大的，避免重复）
        for j in range(i + 1, len(active_objects)):
            check_objects.append(active_objects[j])
        
        # 2. 添加passive和inside物体
        check_objects.extend(other_objects)
        
        # 第一阶段：bbox快速预筛选
        candidates = []
        for other_obj in check_objects:
            # 获取bbox信息
            if other_obj in active_bboxes:
                other_bbox = active_bboxes[other_obj]
            else:
                other_bbox = other_bboxes[other_obj]
            
            if check_bbox_overlap_fast(active_bbox, other_bbox):
                candidates.append(other_obj)
        
        # 第二阶段：使用BVH树精确检测
        if candidates:
            for other_obj in candidates:
                if check_mesh_overlap_bvh(active_obj, other_obj):
                    has_collision = True
                    collision_count += 1
                    print(f"[PhysicsSimulation] 检测到干涉: {active_obj.name} <-> {other_obj.name}")
                    # 如果碰撞的是另一个active物体，也把它标记为需要移除
                    if other_obj in active_objects and other_obj not in active_to_remove:
                        active_to_remove.append(other_obj)
                    break
        
        if has_collision:
            active_to_remove.append(active_obj)
    
    # 将有干涉的物体从active转移到passive
    if active_to_remove:
        print(f"[PhysicsSimulation] 将 {len(active_to_remove)} 个有干涉的物体转为passive（共检测到 {collision_count} 对碰撞）")
        for obj in active_to_remove:
            # 检查对象是否还在active_objects中（避免重复移除）
            if obj in active_objects:
                active_objects.remove(obj)
                passive_objects.append(obj)
                # 恢复原始位置（取消+0.01的抬高）
                current_matrix = obj.matrix_world.copy()
                current_matrix[2][3] -= 0.01
                obj.matrix_world = current_matrix
    else:
        print(f"[PhysicsSimulation] 未检测到干涉")
    
    duration = 0.5  # 默认1秒
    print(f"[PhysicsSimulation] 最终Active物体数: {len(active_objects)}, Passive物体数: {len(passive_objects)}, Inside物体数: {len(inside_objects)}")
    
    # 执行物理模拟（使用默认的world_settings）
    success = run_drop_simulation(
        objects=active_objects,
        colliders=passive_objects,
        duration=duration,
        scene=bpy.context.scene
    )
    
    if success:
        print(f"[PhysicsSimulation] 物理仿真完成！")
    else:
        print("[PhysicsSimulation] 物理仿真失败！")
    
    # 更新内部摆放物体的位姿
    print("[PhysicsSimulation] 更新内部摆放物体的位姿...")
    for instance_id in inside_objects:
        info = output_data_s3['obj_info'][instance_id]
        parent_name = info['supported']
        # 应该使用s2最终的内部摆放的相对pose
        ori_obj_pose = Matrix(output_data_s2['obj_info'][instance_id]['pose_matrix_for_blender'])
        ori_parent_pose = Matrix(output_data_s2['obj_info'][parent_name]['pose_matrix_for_blender'])
        parent = bpy.data.objects[parent_name]
        relative_transform = ori_parent_pose.inverted() @ ori_obj_pose
        obj = bpy.data.objects[instance_id]
        obj.matrix_world = parent.matrix_world @ relative_transform
        bpy.context.view_layer.update()
    print("[PhysicsSimulation] 内部摆放物体位姿更新完成。")

    
    # 设置GPU渲染
    bpy.context.scene.cycles.device = 'GPU'

    output_data_s4 = output_data_s3.copy()
    # 获取场景中的所有物体
    for obj in bpy.context.scene.objects:
        # 只处理刚体物体，可以根据需要修改类型或添加条件
        if obj.type == 'MESH' and obj.rigid_body:
            # 获取物体的世界坐标
            pose = np.array(obj.matrix_world).tolist()
            print(f"Object: {obj.name}\n World Pose: {pose}")
            output_data_s4['obj_info'][obj.name]['pose_matrix_for_blender'] = pose

    # Determine output paths
    save_path = os.path.join(output_folder, f'{scene_name}_placement_info_s4.json')
    output_path = os.path.join(output_folder, f'{scene_name}_render_simu.png')
    
    # 将仿真数据写入 JSON 文件
    with open(save_path, "w") as json_file:
        json.dump(output_data_s4, json_file, indent=2)

    print(f"仿真数据已保存到: {save_path}")
    
    # 开始渲染（使用 Cycles 渲染器以获得高质量结果）
    bpy.context.scene.camera = bpy.data.objects[scene_camera_name]
    blender_manager.render_scene(output_path, resolution_x, resolution_y, samples=256)
    
    print(f"Final poses saved to: {output_path}", flush=True)

    
def main(args):
    # 从配置中读取基础路径
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    params = config["S4_blender_layout_and_corr"]

    placeable_area_info_folder = params['placeable_area_info_folder']
    base_fbx_path = params['base_fbx_path']
    fbx_csv_path = params['fbx_csv_path']
    precomputed_voxel_dir = params.get('precomputed_voxel_dir', None)
    
    obj_placement_info_json_path = args.obj_placement_info_json_path
    output_folder = args.output_folder
    debug = args.debug
    layout(obj_placement_info_json_path, placeable_area_info_folder, base_fbx_path, fbx_csv_path, output_folder, precomputed_voxel_dir, debug)

argv = sys.argv
if "--" not in argv:
    argv = []
else:
   argv = argv[argv.index("--") + 1:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = '3D Layout Processing Script',
        prog = "blender -b -python "+__file__+" --",
        )
    parser.add_argument('--obj_placement_info_json_path', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for S4 results')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to save voxel visualization images')
    
    try:
        args = parser.parse_args(argv)
        main(args)
    except SystemExit as e:
        print(repr(e))

'''
blender --background --python S4_blender_layout_and_corr.py -- --obj_placement_info_json_path "saved_results/demo_result/S3_pose_inference/demo_placement_info.json"
'''