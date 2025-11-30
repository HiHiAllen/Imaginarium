#!/usr/bin/env python
# coding=utf-8
import os
import torch
import numpy as np
from PIL import Image
import argparse
import lpips
from tqdm import tqdm
import json
from torchvision import transforms
import sys

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置命令行参数
parser = argparse.ArgumentParser(description='Calculate LPIPS similarity between output images and training set')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Directory containing output images organized in subfolders')
parser.add_argument('--train_dir', type=str, required=True,
                    help='Directory containing training images')
parser.add_argument('--results_file', type=str, default='similarity_results.json',
                    help='File to save the similarity results')
args = parser.parse_args()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载LPIPS模型
print("Loading LPIPS model...")
lpips_model = lpips.LPIPS(net='vgg').to(device).eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 预处理图像
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        return tensor
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# 计算LPIPS距离
def calculate_lpips(img1_tensor, img2_tensor):
    if img1_tensor is None or img2_tensor is None:
        return float('inf')
    
    with torch.no_grad():
        distance = lpips_model(img1_tensor, img2_tensor).item()
    return distance

# 获取目录中的所有图像文件
def get_all_images(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_files.append(os.path.join(root, file))
    return image_files

# 预处理并缓存训练集图像
def preprocess_training_images(train_dir):
    print(f"Preprocessing training images from {train_dir}...")
    train_images = get_all_images(train_dir)
    train_features = {}
    
    for img_path in tqdm(train_images):
        tensor = preprocess_image(img_path)
        if tensor is not None:
            train_features[img_path] = tensor
    
    print(f"Processed {len(train_features)} training images")
    return train_features

# 获取输出目录中的所有子文件夹
def get_subfolders(directory):
    return [os.path.join(directory, d) for d in os.listdir(directory) 
            if os.path.isdir(os.path.join(directory, d))]

# 主函数：计算相似度并保存结果
def calculate_similarities(output_dir, train_dir, results_file):
    # 预处理训练集图像
    train_features = preprocess_training_images(train_dir)
    
    # 获取所有子文件夹
    subfolders = get_subfolders(output_dir)
    print(f"Found {len(subfolders)} subfolders in output directory")
    
    # 记录结果
    all_results = {}
    
    # 处理每个子文件夹
    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        print(f"\nProcessing subfolder: {subfolder_name}")
        
        output_images = get_all_images(subfolder)
        subfolder_results = []
        
        for output_path in tqdm(output_images):
            output_tensor = preprocess_image(output_path)
            if output_tensor is None:
                continue
                
            # 找到最相似的训练集图像
            best_distance = float('inf')
            best_match = None
            
            for train_path, train_tensor in train_features.items():
                distance = calculate_lpips(output_tensor, train_tensor)
                if distance < best_distance:
                    best_distance = distance
                    best_match = train_path
            
            # 记录结果
            subfolder_results.append({
                'output_image': output_path,
                'best_match': best_match,
                'lpips_distance': best_distance
            })
        
        # 计算该子文件夹的平均LPIPS距离
        if subfolder_results:
            avg_distance = np.mean([result['lpips_distance'] for result in subfolder_results])
            min_distance = np.min([result['lpips_distance'] for result in subfolder_results])
            
            all_results[subfolder_name] = {
                'avg_lpips_distance': float(avg_distance),
                'min_lpips_distance': float(min_distance),
                'individual_results': subfolder_results
            }
            
            print(f"Subfolder {subfolder_name}: Avg LPIPS = {avg_distance:.4f}, Min LPIPS = {min_distance:.4f}")
    
    # 保存结果
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    calculate_similarities(args.output_dir, args.train_dir, args.results_file)

