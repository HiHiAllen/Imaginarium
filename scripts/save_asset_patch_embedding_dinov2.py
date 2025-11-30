from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import os
import pickle
from tqdm import tqdm
import pandas as pd
import argparse
import sys
from torch.utils.data import Dataset, DataLoader

# --- Model Initialization ---
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)

def crop_render_img(img_path):
    """Crops the transparent background from a rendered image."""
    try:
        image = Image.open(img_path).convert("RGBA")
        bbox = image.getbbox()
        if bbox:
            cropped_image = image.crop(bbox)
        else:
            # If no bounding box, the image might be empty or single-colored.
            cropped_image = image
        # Convert to RGB for the model
        return cropped_image.convert("RGB")
    except FileNotFoundError:
        print(f"Warning: Image file not found at {img_path}. Skipping.")
        return None

class AssetDataset(Dataset):
    def __init__(self, asset_list, assets_render_imgs_folder, view_id_for_embedding, save_folder, processor):
        self.assets_render_imgs_folder = assets_render_imgs_folder
        self.view_id_for_embedding = view_id_for_embedding
        self.save_folder = save_folder
        self.processor = processor
        
        # Filter assets
        self.asset_list = []
        print("Scanning assets for unprocessed items...")
        for asset in asset_list:
            asset_str = str(asset)
            output_path = os.path.join(save_folder, f"{asset_str}.pt")
            if not os.path.exists(output_path):
                # Check if source dir exists
                asset_dir = os.path.join(assets_render_imgs_folder, asset_str)
                if os.path.isdir(asset_dir):
                    self.asset_list.append(asset_str)
        print(f"Found {len(self.asset_list)} unprocessed assets out of {len(asset_list)} total.")

    def __len__(self):
        return len(self.asset_list)

    def __getitem__(self, idx):
        asset = self.asset_list[idx]
        asset_render_imgs_folder = os.path.join(self.assets_render_imgs_folder, asset)
        
        img_paths = [os.path.join(asset_render_imgs_folder, f'{view_id:06}.png') for view_id in self.view_id_for_embedding]
        
        images_pil = [crop_render_img(img_path) for img_path in img_paths]
        # Filter out any images that failed to load
        images_pil = [img for img in images_pil if img is not None]

        if not images_pil:
            return None
        
        # Process images to tensor (on CPU)
        try:
            inputs = self.processor(images=images_pil, return_tensors="pt")
            return inputs['pixel_values'], asset
        except Exception as e:
            print(f"Error processing asset {asset}: {e}")
            return None

def collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    # Since batch_size=1, we just return the first item
    return batch[0]

def save_asset_patch_embedding(asset_list, assets_render_imgs_folder, view_id_for_embedding, save_folder, num_workers=8):
    """
    Extracts and saves patch-level features for each asset's rendered views.
    Saves a single .pt file per asset, containing a tensor of shape 
    (num_views, num_patches, feature_dim).
    """
    
    # Save the view order information, crucial for retrieval mapping.
    # We do this first as it doesn't depend on the processing loop.
    pkl_path = os.path.join(save_folder, "assets_imgs_order.pkl")
    assets_imgs_order = [f'{view_id:06}.png' for view_id in view_id_for_embedding]
    with open(pkl_path, 'wb') as file:
        pickle.dump(assets_imgs_order, file)
    print(f"View order info saved to: {pkl_path}")

    dataset = AssetDataset(asset_list, assets_render_imgs_folder, view_id_for_embedding, save_folder, processor)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_fn,
        prefetch_factor=2
    )

    print(f"Start processing with {num_workers} workers...")
    
    for batch in tqdm(dataloader, desc="Processing assets"):
        if batch is None:
            continue
            
        pixel_values, asset = batch
        output_path = os.path.join(save_folder, f"{asset}.pt")
        
        # Move to GPU
        pixel_values = pixel_values.to(device)
        
        # pixel_values shape: [num_views, 3, height, width]
        # Because we used batch_size=1 and collate_fn unwrapped it
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            # This is the key change: save the full last_hidden_state
            # which contains the features for all patches.
            patch_features = outputs.last_hidden_state
            # Discard the [CLS] token (at index 0) to keep only the 256 patch features.
            patch_features = patch_features[:, 1:, :]
        
        # Save the tensor containing patch features for all views.
        torch.save(patch_features, output_path)

    print(f"\nAsset patch embeddings saved to: {save_folder}")


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="生成 Asset Patch Embeddings 工具")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="asset_data/imaginarium_assets_render_results",
        help="渲染结果所在的输入文件夹路径"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="asset_data/imaginarium_assets_patch_embedding",
        help="Embedding 结果保存路径"
    )
    
    parser.add_argument(
        "--view_ids", 
        type=int, 
        nargs='+',  # '+' 表示接受一个或多个参数并将它们组成列表
        default=[3, 25, 28],
        help="用于生成 Embedding 的视角 ID 列表 (例如: --view_ids 3 25 28)"
    )

    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=8,
        help="用于数据加载的线程数 (默认为 8)"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        sys.exit(1)

    # 获取资源列表 (建议排序以保证顺序一致)
    asset_list = sorted(os.listdir(args.input_dir))
    
    # 创建保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*50)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"选择视角: {args.view_ids}")
    print(f"Workers:  {args.num_workers}")
    print(f"待处理资产数量: {len(asset_list)}")
    print("="*50)

    # 调用处理函数
    save_asset_patch_embedding(
        asset_list, 
        args.input_dir, 
        args.view_ids, 
        args.output_dir,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()

# def main():

#     asset_list = os.listdir("asset_data/imaginarium_assets_render_results")
#     assets_render_result_folder = "asset_data/imaginarium_assets_render_results"
#     # Define the folder to save the new patch-level embeddings
#     save_folder = "asset_data/imaginarium_assets_patch_embedding"
#     os.makedirs(save_folder, exist_ok=True)
    

#     # Define which camera views to use for generating embeddings
#     view_id_for_embedding = [3, 25, 28]   # Front, front-diagonal, back-diagonal
    
#     save_asset_patch_embedding(asset_list, assets_render_result_folder, view_id_for_embedding, save_folder)

# if __name__ == "__main__":
#     main()