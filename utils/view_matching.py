import numpy as np
from scipy.spatial.transform import Rotation as R

def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def rotation_matrix_from_axis_angle(axis, angle):
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.eye(3)
    return I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

def compute_new_rotation_matrix(pose_matrix_1, pos2):
    '''
    作用：物体视角渲染图的位姿是受其所在位置影响的,这个函数的作用是已知匹配到的视角图的位姿是pose_matrix_1,可以抽取出pos1, R1, 
    现在想在pos2的位置上,使物体的旋转和视角渲染图的成像相近,需要重新修正得到新的R2
    # Example usage
    pose_matrix_1 = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1]
    ])
    pos2 = np.array([-1.9375, 4.7787, -1.3437])
    pose_matrix_2 = compute_new_rotation_matrix(pose_matrix_1, pos2)
    print(pose_matrix_2)
    '''
    # Extract position and rotation matrix from 4x4 transform matrix
    pos1 = pose_matrix_1[:3, 3]
    R1 = pose_matrix_1[:3, :3]
    
    # Normalize view vectors
    view1 = normalize(-pos1)
    view2 = normalize(-pos2)

    # Calculate rotation axis and angle
    axis = np.cross(view1, view2)
    angle = np.arccos(np.clip(np.dot(view1, view2), -1.0, 1.0))

    # Construct rotation matrix
    if np.linalg.norm(axis) != 0:
        axis = normalize(axis)
        R_fix = rotation_matrix_from_axis_angle(axis, angle)
    else:
        R_fix = np.eye(3)  # No rotation needed if view vectors are the same

    # Calculate new rotation matrix
    R2 = R_fix @ R1
    
    # Construct the new 4x4 transform matrix
    pose_matrix_2 = np.eye(4)
    pose_matrix_2[:3, :3] = R2
    pose_matrix_2[:3, 3] = pos2

    return pose_matrix_2

def convert_obb_pose_to_blender_coordinates(obb_pose_matrix):
    '''
    作用是把点云obb的位姿对应到blender中，认为观测方向和图片的相机观测方向一致，在blender里都是看向Y正半轴
    '''
    # Extract location and rotation matrix from the input obb_pose_matrix
    obb_pose_matrix = np.array(obb_pose_matrix)
    location = obb_pose_matrix[:3, 3]
    rotation_matrix = obb_pose_matrix[:3, :3]

    # Convert location
    blender_location = np.array([location[0], location[2], -location[1]])
    
    # Convert rotation matrix and swap Y and Z columns
    blender_rotation_matrix = np.array([
        rotation_matrix[0, :3],
        rotation_matrix[2, :3],
        -rotation_matrix[1, :3]
    ])
    
    # Construct the new 4x4 obb_pose_matrixation matrix
    blender_obb_pose_matrix = np.eye(4)
    blender_obb_pose_matrix[:3, :3] = blender_rotation_matrix
    blender_obb_pose_matrix[:3, 3] = blender_location
    
    return blender_obb_pose_matrix
    

def convert_camera_pose_of_render_view_to_obj_pose_to_blender_coordinates(camera_pose_matrix, place_position=None):
    '''
    对物体视角渲染图的相机位姿进行转换，得到物体的位姿，并使相机处在世界原点，拍摄方向为Y正方向，输出此时物体视角渲染图在blender对应的4*4变换矩阵
    如果place_position不为None，那就用compute_new_rotation_matrix进行修正; place_position是一个list或array [x,y,z]
    注意place_position是blender坐标系中的的, obb的坐标要先做转换
    '''
    pose_matrix = np.array(camera_pose_matrix)

    # 绕 x 轴旋转 90 度，因为相机初始拍摄方向为Z负方向
    rotation_x_90 = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])

    # 计算新的变换矩阵
    new_matrix_world = pose_matrix @ rotation_x_90
    # 假设相机在世界原点, 转化为物体的pose
    obj_pose_matrix = np.linalg.inv(new_matrix_world) 
    
    if place_position is not None:
        obj_pose_matrix = compute_new_rotation_matrix(obj_pose_matrix, place_position)
    
    return obj_pose_matrix

def orthogonalize_rotation_matrix(R):
    """
    使用极分解将矩阵 R 分解为正交矩阵（旋转矩阵）和对称正定矩阵。
    返回正交矩阵部分。
    """
    U, s, Vt = np.linalg.svd(R)
    R_orthogonal = np.dot(U, Vt)
    # 确保行列式为 1
    if np.linalg.det(R_orthogonal) < 0:
        U[:, -1] *= -1
        R_orthogonal = np.dot(U, Vt)
    return R_orthogonal

def rotation_matrix_to_angle_diff(R1, R2):
    # 对旋转矩阵进行正交化
    R1_orthogonal = orthogonalize_rotation_matrix(R1)
    R2_orthogonal = orthogonalize_rotation_matrix(R2)
    
    # 计算相对旋转矩阵
    relative_rotation = np.dot(R2_orthogonal.T, R1_orthogonal)
    
    # 将相对旋转矩阵转换为旋转对象
    rotation = R.from_matrix(relative_rotation)
    
    # 获取旋转向量（轴-角表示）
    rotation_vector = rotation.as_rotvec()
    
    # 旋转向量的范数即为旋转角度（弧度）
    angle = np.linalg.norm(rotation_vector)
    
    # 转换为角度制
    angle_degrees = np.degrees(angle)
    
    return angle, angle_degrees

def ensure_no_reflection(matrix):
    # 计算行列式
    det = np.linalg.det(matrix[:3, :3])
    # 如果行列式为负，说明存在反射，需要调整
    if det < 0:
        # 反转一个轴来消除反射
        matrix[:3, 0] *= -1  # 反转 x 轴，例如
    return matrix

def find_view_best_match_obb(camera_poses_array, obb_pose_matrix):
    '''
    # camera_poses_array为 (162, 4, 4)
    # obb_pose_matrix是点云拟合输出的初始尺寸
    # blender_obb_pose_matrix只是个obb,它的正方向是未知的，它的xyz正负轴对应的八个方向都有可能是真正的方向，我希望对于这八个方向中的每一个，都从162个视角对应的obj_pose_matrix中找到最契合该方向的那个
    # 我希望最后输出的是[(162中的哪个view_id, 将blender_obb_pose_matrix某个方向作为正方向的pose_matrix),...]
    '''
    blender_obb_pose_matrix = convert_obb_pose_to_blender_coordinates(obb_pose_matrix)
    # 定义八个可能的正方向
    # 定义四个可能方向，假设物体都是正立在地面上的
    directions = [
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),  # (x, y, z)
        np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), # (x, -y, z)
        np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),  # (y, x, z)
        np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # (-y, x, z)
    ]
    # 确保每个方向矩阵没有反射
    directions = [ensure_no_reflection(direction) for direction in directions]

    place_position = blender_obb_pose_matrix[:3, 3]
    best_matches = []

    for direction in directions:
        best_view_id = None
        best_obj_pose_matrix = None
        best_score = float('inf')
        transformed_obb = blender_obb_pose_matrix @ direction
                    
        for view_id, camera_pose_matrix in enumerate(camera_poses_array):
            # 转换相机位姿到物体位姿
            obj_pose_matrix = convert_camera_pose_of_render_view_to_obj_pose_to_blender_coordinates(camera_pose_matrix, place_position=place_position)

            # 计算与当前方向的角度差异
            angle, angle_degrees = rotation_matrix_to_angle_diff(transformed_obb[:3, :3], obj_pose_matrix[:3, :3])
            score = angle_degrees
            
            if score < best_score:
                best_score = score
                best_view_id = view_id
                best_obj_pose_matrix = obj_pose_matrix

        best_matches.append((best_view_id, transformed_obb, best_obj_pose_matrix, best_score))

    return best_matches