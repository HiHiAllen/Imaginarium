import bpy
import math
import os
import bmesh
import numpy as np
from mathutils import Vector, Matrix
import shutil 
import argparse
import sys
from tqdm import tqdm

def setup_scene(gpu_id=0):
    """设置渲染场景和GPU"""
    # 设置渲染引擎为 Cycles
    bpy.context.scene.render.engine = 'CYCLES'
    # 设置渲染最大采样数为 32
    bpy.context.scene.cycles.samples = 32
    
    bpy.context.scene.cycles.use_persistent_data = True
    # 设置渲染滤波阈值为 0.1
    bpy.context.scene.cycles.denoising_threshold = 0.1
    
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.max_bounces = 4

    # 配置 Cycles 设置
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
    cycles_prefs.compute_device_type = 'CUDA'  # 使用 CUDA
    cycles_prefs.get_devices()  # 刷新设备列表
    
    # 设置场景使用 GPU 计算
    bpy.context.scene.cycles.device = 'GPU'
    
    # 启用所有可用的 GPU 设备
    print(f"可用的计算设备:")
    for device in cycles_prefs.devices:
        print(f"  - {device.name} ({device.type}): {'启用' if device.type == 'CUDA' else '禁用'}")
        if device.type == 'CUDA':
            device.use = True
        else:
            device.use = False
                
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    
    # 删除默认的立方体和光源
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

    # 添加新的相机
    if not bpy.context.scene.camera:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        bpy.context.scene.camera = camera
    
    # 添加环境光
    bpy.ops.object.light_add(type='SUN')
    sun = bpy.context.object
    sun.location = (0, 0, 10)
    sun.data.energy = 0.5

    # 设置世界背景为白色并调整强度
    world = bpy.context.scene.world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (1, 1, 1, 1)  # 颜色
        bg_node.inputs[1].default_value = 0.5  # 强度

def get_camera_positions(nSubDiv):
    """
    构造一个icosphere并进行细分，生成相机位置
    """
    bpy.ops.mesh.primitive_ico_sphere_add(location=(0, 0, 0), enter_editmode=True)
    icos = bpy.context.object
    me = icos.data

    # 切除下半部分
    bm = bmesh.from_edit_mesh(me)
    sel = [v for v in bm.verts if v.co[2] < 0]
    bmesh.ops.delete(bm, geom=sel, context="FACES")
    bmesh.update_edit_mesh(me)

    # 细分并将新顶点移动到球面上
    for i in range(nSubDiv):
        bpy.ops.mesh.subdivide()
        bm = bmesh.from_edit_mesh(me)
        for v in bm.verts:
            l = math.sqrt(v.co[0] ** 2 + v.co[1] ** 2 + v.co[2] ** 2)
            v.co[0] /= l
            v.co[1] /= l
            v.co[2] /= l
        bmesh.update_edit_mesh(me)

    # 切除零高度点
    bm = bmesh.from_edit_mesh(me)
    sel = [v for v in bm.verts if v.co[2] <= 0]
    bmesh.ops.delete(bm, geom=sel, context="FACES")
    bmesh.update_edit_mesh(me)

    # 转换顶点位置为角度
    positions = []
    angles = []
    bm = bmesh.from_edit_mesh(me)
    for v in bm.verts:
        x = v.co[0]
        y = v.co[1]
        z = v.co[2]
        az = math.atan2(x, y)
        el = math.atan2(z, math.sqrt(x**2 + y**2))
        angles.append((el, az))
        positions.append((x, y, z))

    bpy.ops.object.editmode_toggle()
    
    # 删除icosphere
    bpy.ops.object.delete()
    
    # 排序位置
    data = zip(angles, positions)
    positions = sorted(data)
    positions = [y for x, y in positions]
    angles = sorted(angles)
    
    # 坐标系转换
    positions = [(x, z, y) for x, y, z in positions]
    return angles, positions

def get_camera_pose(camera):
    """获取相机的变换矩阵"""
    pose = np.array(camera.matrix_world)
    return pose

def ensure_object_visible(obj):
    """确保物体可见，设置材质"""
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

def setup_camera_and_scene(obj):
    """设置相机和场景参数"""
    camera = bpy.context.scene.camera
    
    # 计算物体的包围盒半径
    bbx_radius = max(obj.dimensions) / 2

    # 设置相机参数
    camera.data.clip_start = bbx_radius * 0.1
    camera.data.clip_end = bbx_radius * 100
    camera.data.lens = 50  # 使用50mm镜头

    # 使相机不可选
    camera.hide_select = True

    # 设置视口着色为材质预览
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

def adjust_camera_to_fit_object(camera, obj):
    """调整相机位置和角度以适配物体"""
    # 计算物体的边界框
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # 计算边界框的中心和大小
    bbox_center = sum(bbox_corners, Vector()) / 8
    bbox_size = max((corner - bbox_center).length for corner in bbox_corners)

    # 设置相机位置
    camera_distance = bbox_size / math.tan(camera.data.angle / 2)
    camera_vector = camera.location - bbox_center
    camera_vector.normalize()
    camera.location = bbox_center + camera_vector * camera_distance * 1.1  # 稍微往后移动一点

    # 让相机看向物体中心
    direction = bbox_center - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

def render_object(obj, output_dir, positions):
    """渲染物体的所有视角"""
    camera = bpy.context.scene.camera
    
    ensure_object_visible(obj)
    setup_camera_and_scene(obj)
    
    poses = []
        
    for i, pos in enumerate(positions):
        # 设置相机位置
        camera.location = Vector(pos)
        
        # 调整相机以适应物体
        adjust_camera_to_fit_object(camera, obj)
        
        # 确保更新场景
        bpy.context.view_layer.update()
        
        pose = get_camera_pose(camera)
        poses.append(pose)
        
        filepath = os.path.join(output_dir, f"{i:06d}.png")
        bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
    
    # 保存相机姿态
    poses_array = np.array(poses)
    np.save(os.path.join(output_dir, "camera_poses.npy"), poses_array)
    
def import_fbx(filepath):
    """导入FBX文件"""
    bpy.ops.import_scene.fbx(filepath=filepath)
    return bpy.context.selected_objects[0]

def clear_scene_wo_camera():
    """清空场景但保留相机"""
    # 取消选择所有对象
    bpy.ops.object.select_all(action='DESELECT')
    
    # 选择所有非相机对象
    for obj in bpy.context.scene.objects:
        if obj.type != 'CAMERA':
            obj.select_set(True)
    
    # 删除选中的对象
    bpy.ops.object.delete()

def move_object_to_origin(obj):
    """将物体移动到原点"""
    # 计算物体的边界框
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # 计算边界框的中心
    bbox_center = sum(bbox_corners, Vector()) / 8
    
    # 计算需要移动的距离（使几何中心与原点重合）
    translation = -bbox_center
    
    # 移动物体
    obj.location += translation
    
    # 更新场景
    bpy.context.view_layer.update()
    
def process_file(fbx_path, base_output_dir, positions):
    """处理单个FBX文件"""
    object_name = os.path.splitext(os.path.basename(fbx_path))[0]
    output_dir = os.path.join(base_output_dir, object_name)
    
    # 检查输出目录是否存在，如果存在，则清空该目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    clear_scene_wo_camera()
    obj = import_fbx(fbx_path)
    move_object_to_origin(obj)
    
    print(f"开始渲染物体: {object_name}")
    render_object(obj, output_dir, positions)
    print(f"完成渲染: {object_name}")

def read_txt_to_list(file_path):
    """从文本文件读取FBX路径列表"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines if line.strip()]
    return lines
        
def main(args):
    """主函数"""
    print(f"\n{'='*60}")
    print(f"Blender渲染脚本")
    print(f"GPU ID: {args.gpu_id}")
    print(f"文件列表: {args.file_path}")
    print(f"输出目录: {args.base_output_dir}")
    print(f"{'='*60}\n")
    
    os.makedirs(args.base_output_dir, exist_ok=True)
    
    # 读取FBX文件列表
    fbx_paths_list = read_txt_to_list(args.file_path)
    
    num_views = 162
    fbx_paths_to_render = []

    # 收集需要渲染的FBX文件
    for path in fbx_paths_list:
        filename = path.split('/')[-1]
        object_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(args.base_output_dir, object_name)
        
        existing_pngs = [file for file in os.listdir(output_dir) if file.endswith('.png')] if os.path.exists(output_dir) else []
        if len(existing_pngs) < num_views:
            fbx_paths_to_render.append(path)
        else:
            print(f"跳过已渲染: {object_name}")
    
    if not fbx_paths_to_render:
        print("没有需要渲染的文件")
        return
    
    print(f"总共需要渲染 {len(fbx_paths_to_render)} 个FBX文件\n")
    
    # 设置场景
    setup_scene(args.gpu_id)
    
    # 生成相机位置 (nSubDiv=1生成162个视角)
    _, positions = get_camera_positions(nSubDiv=1)
    
    print(f"生成了 {len(positions)} 个相机位置\n")
        
    # 渲染每个FBX文件
    for idx, fbx_path in enumerate(fbx_paths_to_render):
        print(f"\n进度: [{idx+1}/{len(fbx_paths_to_render)}]")
        try:
            process_file(fbx_path, args.base_output_dir, positions)
        except Exception as e:
            print(f"渲染失败 {fbx_path}: {e}")
            continue

# 解析命令行参数
argv = sys.argv
if "--" not in argv:
    argv = []
else:
   argv = argv[argv.index("--") + 1:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='3D Layout渲染脚本 (单GPU版本)',
        prog="blender -b -python "+__file__+" --",
    )
    parser.add_argument('--file_path', type=str, required=True, 
                        help='包含FBX路径的文本文件')
    parser.add_argument('--base_output_dir', type=str, required=True, 
                        help='输出目录')
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help='GPU ID (用于日志显示)')
    
    try:
        args = parser.parse_args(argv)
        main(args)
    except SystemExit as e:
        print(repr(e))

