from PIL import Image, ImageDraw
import os
import math
import io

def resize_images(images, max_width, max_height):
    resized_images = []
    for img in images:
        # 计算保持比例的新尺寸，这里简单地保持原始尺寸
        resized_images.append(img)
    return resized_images

def calculate_grid_size(num_images):
    cols = round(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    return rows, cols

def draw_border(img, border_width, color):
    # 在图片的右侧和底部绘制边框
    draw = ImageDraw.Draw(img)
    width, height = img.size
    # 右侧边框
    draw.line((width - border_width, 0, width - border_width, height), fill=color, width=border_width)
    # 底部边框
    draw.line((0, height - border_width, width, height - border_width), fill=color, width=border_width)
    return img

def resize_with_aspect_ratio(img, target_size):
    """
    等比缩放图像到目标尺寸，保持宽高比，不变形
    target_size: (width, height)
    """
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    # 创建目标尺寸的新图像，背景为白色
    new_img = Image.new('RGB', target_size, 'white')
    # 计算居中位置
    x_offset = (target_size[0] - img.size[0]) // 2
    y_offset = (target_size[1] - img.size[1]) // 2
    new_img.paste(img, (x_offset, y_offset))
    return new_img

def split_comparison_image(comparison_img):
    """
    将 comparison_img 拆分成左右两部分
    返回: (segmentation_img, pose_img)
    """
    width, height = comparison_img.size
    mid_point = width // 2
    # 左侧是分割图
    segmentation_img = comparison_img.crop((0, 0, mid_point, height))
    # 右侧是位姿图
    pose_img = comparison_img.crop((mid_point, 0, width, height))
    return segmentation_img, pose_img

def calculate_optimal_grid_size(num_pairs):
    """
    根据 pair 的数量计算最优的 m 和 n
    m 是偶数，n 是偶数（确保每行有整数个 pair），m 和 n 尽量接近
    num_pairs: pair 的数量
    返回: (m, n) 其中 m 是行数（偶数），n 是列数（偶数）
    """
    num_grids = num_pairs * 2  # 每个 pair 需要两个 grid
    
    # 从 sqrt(num_grids) 开始寻找最优的 m 和 n
    # m 和 n 都必须是偶数
    ideal_size = int(math.sqrt(num_grids))
    
    # 确保起始值是偶数
    if ideal_size % 2 == 0:
        start_size = ideal_size
    else:
        start_size = ideal_size + 1
    
    best_m, best_n = None, None
    min_diff = float('inf')
    
    # 尝试不同的 m 值（必须是偶数）
    for m in range(start_size, num_grids + 1, 2):
        n = math.ceil(num_grids / m)
        # 确保 n 是偶数
        if n % 2 != 0:
            n = n + 1
        # 确保 m * n >= num_grids
        if m * n < num_grids:
            continue
        diff = abs(m - n)
        if diff < min_diff:
            min_diff = diff
            best_m, best_n = m, n
        # 如果已经找到很接近的值，可以提前退出
        if diff <= 2:
            break
    
    # 如果没有找到合适的值，使用默认值
    if best_m is None or best_n is None:
        # 确保 m 和 n 都是偶数
        m = start_size if start_size % 2 == 0 else start_size + 1
        n = math.ceil(num_grids / m)
        if n % 2 != 0:
            n = n + 1
        best_m, best_n = m, n
    
    return best_m, best_n

def stitch_images_grid(directory, save_path, images_list=None, plot_num=12):
    """
    将 comparison_images 重新组织成 m*n 的 grid 布局
    每个 grid 是 128x128，每个物体是相邻的两个 grid（左侧分割图，右侧位姿图）
    """
    if images_list is None:
        # 如果没有提供图像列表,则从目录加载
        image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg')) and 'comparison' in f])
        if not image_files:
            print("No comparison images found in the directory.")
            return
        comparison_images = [Image.open(f) for f in image_files]
    else:
        # 如果提供了图像列表,则直接使用
        if not images_list:
            print("Provided images list is empty.")
            return
        comparison_images = images_list

    if not comparison_images:
        print("No images to stitch.")
        return

    num_pairs = len(comparison_images)
    # 计算最优的 m 和 n
    m, n = calculate_optimal_grid_size(num_pairs)
    print(f"Calculated grid size: {m} rows x {n} cols for {num_pairs} pairs")
    
    grid_size = 128  # 每个 grid 的大小
    border_width = 2  # pair 边框的宽度
    border_color = 'green'
    
    # 创建画布（grid 之间紧密排列，没有间距）
    total_width = n * grid_size
    total_height = m * grid_size
    new_image = Image.new('RGB', (total_width, total_height), 'white')
    
    # 存储每个 pair 的位置信息，用于后续绘制边框
    pair_positions = []
    
    grid_index = 0
    for row in range(m):
        for col in range(n):
            if grid_index >= num_pairs * 2:
                break
            
            # 计算当前 grid 的位置（grid 之间紧密排列）
            x = col * grid_size
            y = row * grid_size
            
            # 计算当前 grid 属于哪个 pair，以及是 pair 中的第几个 grid
            pair_index = grid_index // 2
            grid_in_pair = grid_index % 2  # 0: 左侧（分割图）, 1: 右侧（位姿图）
            
            if pair_index >= num_pairs:
                break
            
            # 获取当前 pair 的 comparison_img
            comparison_img = comparison_images[pair_index]
            
            # 拆分成左右两部分
            segmentation_img, pose_img = split_comparison_image(comparison_img)
            
            if grid_in_pair == 0:
                # 左侧 grid：分割图
                resized_img = resize_with_aspect_ratio(segmentation_img, (grid_size, grid_size))
                # 记录 pair 的起始位置
                pair_positions.append({
                    'pair_index': pair_index,
                    'x': x,
                    'y': y,
                    'width': grid_size * 2,  # 两个 grid 的宽度
                    'height': grid_size
                })
            else:
                # 右侧 grid：位姿图
                resized_img = resize_with_aspect_ratio(pose_img, (grid_size, grid_size))
            
            # 粘贴到画布
            new_image.paste(resized_img, (x, y))
            
            grid_index += 1
    
    # 为每个 pair 绘制绿色边框
    draw = ImageDraw.Draw(new_image)
    for pos_info in pair_positions:
        x = pos_info['x']
        y = pos_info['y']
        width = pos_info['width']
        height = pos_info['height']
        
        # 绘制 pair 的边框（上、下、左、右）
        # 上边框
        draw.line((x, y, x + width, y), fill=border_color, width=border_width)
        # 下边框
        draw.line((x, y + height - border_width, x + width, y + height - border_width), fill=border_color, width=border_width)
        # 左边框
        draw.line((x, y, x, y + height), fill=border_color, width=border_width)
        # 右边框
        draw.line((x + width - border_width, y, x + width - border_width, y + height), fill=border_color, width=border_width)
    
    new_image.save(save_path)
    print(f"Image stitching completed. Grid size: {m}x{n}, {num_pairs} pairs.")
