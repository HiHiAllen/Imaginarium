import os
import json
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class LocalJsonDataset(Dataset):
    def __init__(self, data_root, resolution=1024):
        self.data_root = data_root
        self.resolution = resolution
        self.data_items = []

        # 扫描目录
        if not os.path.exists(data_root):
            raise ValueError(f"Data root {data_root} does not exist!")

        files = os.listdir(data_root)
        png_files = [f for f in files if f.endswith('.png')]

        print(f"Found {len(png_files)} images in {data_root}")

        for png_file in png_files:
            base_name = os.path.splitext(png_file)[0]
            # 假设 json 文件名规则: bathroom_01.png -> bathroom_01_meta.json
            json_file = f"{base_name}_meta.json"
            json_path = os.path.join(data_root, json_file)
            img_path = os.path.join(data_root, png_file)

            if os.path.exists(json_path):
                self.data_items.append({
                    "img_path": img_path,
                    "json_path": json_path
                })
            else:
                print(f"Warning: Meta json not found for {png_file}, skipping.")

        # Flux 图像预处理: Resize -> CenterCrop -> Normalize to [-1, 1]
        self.transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution), 
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        # 1. 读取图片
        try:
            image = Image.open(item["img_path"]).convert("RGB")
            pixel_values = self.transforms(image)
        except Exception as e:
            print(f"Error loading image {item['img_path']}: {e}")
            # 出错时返回下一个数据，避免训练中断
            return self.__getitem__((idx + 1) % len(self))

        # 2. 读取 JSON 并构建 Prompt
        prompt = ""
        try:
            with open(item["json_path"], 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            scene_name = meta.get("scene_name", "")
            brief = meta.get("caption_en_brief", "")
            
            # 正则提取: "bathroom_03" -> "bathroom"
            # 提取逻辑:以此开头，直到遇到非字母字符(如_或数字)
            match = re.match(r"^([a-zA-Z]+)", scene_name)
            category = match.group(1) if match else "object"
            
            # 组合 prompt: "bathroom [brief description]"
            # prompt = f"{category} {brief}"
            prompt = f"{category}"
            
        except Exception as e:
            print(f"Error parsing json {item['json_path']}: {e}")
            prompt = "a photo of a room" # Fallback

        return {
            "pixel_values": pixel_values,
            "prompt": prompt
        }

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    return {"pixel_values": pixel_values, "prompts": prompts}