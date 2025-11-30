import torch
import os
from tqdm import tqdm
import argparse
import sys
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules._s3_legacy_functions import mp_preprocess_template_rgb_and_mask, Transforms, load_ae_net

class AssetDataset(Dataset):
    def __init__(self, input_folder, transforms):
        self.input_folder = input_folder
        self.transforms = transforms
        
        all_folders = sorted(os.listdir(input_folder))
        self.folder_names = []
        
        # Filter out processed folders
        print("Scanning input folder for unprocessed items...")
        for folder_name in all_folders:
            template_dir = os.path.join(input_folder, folder_name)
            ae_features_path = os.path.join(template_dir, 'template_imgs_ae_features.pt')
            if not os.path.exists(ae_features_path):
                self.folder_names.append(folder_name)
        
        print(f"Found {len(self.folder_names)} unprocessed folders out of {len(all_folders)}.")
        
    def __len__(self):
        return len(self.folder_names)
        
    def __getitem__(self, idx):
        folder_name = self.folder_names[idx]
        template_dir = os.path.join(self.input_folder, folder_name)
        ae_features_path = os.path.join(template_dir, 'template_imgs_ae_features.pt')
        
        try:
            # Use 'cpu' device for data loading in workers
            src_imgs, _ = mp_preprocess_template_rgb_and_mask(
                template_dir, 
                self.transforms, 
                'cpu', 
                return_images=True
            )
            return src_imgs, ae_features_path, folder_name
        except Exception as e:
            print(f"Error processing {folder_name}: {e}")
            return None

def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    
    src_imgs_list = [x[0] for x in batch]
    paths = [x[1] for x in batch]
    names = [x[2] for x in batch]
    
    # Stack images: [B, 162, 3, 224, 224]
    src_imgs = torch.stack(src_imgs_list)
    return src_imgs, paths, names

def process_subfolders_and_save(input_folder, ae_net, num_workers=8):
    if ae_net is None:
        print("ae_net is None, skipping processing.")
        return

    transforms = Transforms()
    device = torch.device('cuda')
    
    # Setup Dataset and DataLoader
    dataset = AssetDataset(input_folder, transforms)
    
    # Use num_workers > 0 to enable multiprocessing
    # batch_size=1 because each item is already a batch of 162 images
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_fn,
        prefetch_factor=2
    )
    
    print(f"Start processing with {len(dataset)} folders...")
    
    for batch in tqdm(dataloader):
        if batch is None:
            continue
            
        src_imgs_batch, paths, names = batch
        
        # Move to GPU
        # src_imgs_batch shape: [B, 162, 3, 224, 224]
        # We process one "asset" (162 images) at a time if B=1
        
        for i, src_imgs in enumerate(src_imgs_batch):
            asset_name = names[i]
            save_path = paths[i]
            
            # src_imgs shape: [162, 3, 224, 224]
            src_imgs = src_imgs.to(device)
            
            try:
                print(f'正在处理: {asset_name}')
                with torch.no_grad():
                    stacked_src_ae_features = ae_net(src_imgs)
                
                torch.save(stacked_src_ae_features, save_path)
                print(f'AE嵌入保存在{save_path}')
            except Exception as e:
                print(f"Error during inference/saving for {asset_name}: {e}")

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="生成 Asset Patch Embeddings 工具")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="渲染结果所在的输入文件夹路径"
    )
    
    parser.add_argument(
        "--ae_net_weights_path", 
        type=str, 
        default="weights/ae_net_pretrained_weights.pth"
    )
    
    parser.add_argument(
        "--ori_dino_weights_path", 
        type=str, 
        default="weights/dinov2_vitl14.pth"
    )
    
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=8,
        help="用于生成 AE 嵌入的线程数"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    ae_net_weights_path = args.ae_net_weights_path
    ori_dino_weights_path = args.ori_dino_weights_path
    input_folder = args.input_dir
    num_workers = args.num_workers
    ae_net = load_ae_net(ae_net_weights_path, ori_dino_weights_path)
    process_subfolders_and_save(input_folder, ae_net, num_workers)


if __name__ == "__main__":
    main()
