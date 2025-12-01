import os
import sys
import json
import re
from tqdm import tqdm
from glob import glob
from PIL import Image

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from utils.llm_api import GPTApi

def is_texture_image(filename):
    """
    Filter to keep only color/diffuse texture images.
    """
    lower_name = filename.lower()
    # Extensions check
    if not lower_name.endswith(('.jpg', '.jpeg', '.png', '.webp')):
        return False
        
    # Exclude non-color maps
    exclude_keywords = ['_nrm', '_disp', '_gloss', '_ao', '_bump', '_rough', '_nor', '_spec', '_metal', '_mask']
    for kw in exclude_keywords:
        if kw in lower_name:
            return False
            
    return True

def main():
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    cfg = Config(config_path)
    gpt_params = {
        'model': cfg.shared.gpt_model,
        'GPT_KEY': cfg.shared.gpt_key,
        'GPT_ENDPOINT': cfg.shared.gpt_endpoint,
        'use_openai_client': cfg.shared.use_openai_client
    }
    
    gpt = GPTApi(**gpt_params)
    
    dataset_root = cfg.shared.get('background_texture_dataset_path', "asset_data/background_texture_dataset")
    output_json = os.path.join(dataset_root, "texture_captions.json")
    
    captions = {}
    if os.path.exists(output_json):
        print(f"Loading existing captions from {output_json}")
        with open(output_json, 'r', encoding='utf-8') as f:
            captions = json.load(f)
            
    # Walk through directories
    categories = ['wall', 'floor', 'ceiling']
    files_to_process = []
    
    for cat in categories:
        cat_dir = os.path.join(dataset_root, cat)
        if not os.path.exists(cat_dir):
            continue
            
        for root, dirs, files in os.walk(cat_dir):
            for file in files:
                if is_texture_image(file):
                    rel_path = os.path.relpath(os.path.join(root, file), dataset_root)
                    if rel_path not in captions:
                        files_to_process.append(rel_path)

    print(f"Found {len(files_to_process)} new textures to caption.")
    
    prompt = (
        "Please generate a English caption for this texture in about 15 words. "
        "Describe its color, material, texture, and style. "
        "Format: 'English description.'"
    )
    
    for rel_path in tqdm(files_to_process, desc="Captioning textures"):
        full_path = os.path.join(dataset_root, rel_path)
        try:
            with Image.open(full_path) as img:
                img_resized = img.resize((512, 512))
                response = gpt.get_response(prompt, image=img_resized)
            if response:
                captions[rel_path] = response
                # Save incrementally
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(captions, f, ensure_ascii=False, indent=2)
            else:
                print(f"Failed to get response for {rel_path}")
        except Exception as e:
            print(f"Error processing {rel_path}: {e}")
            
    print(f"Done. Captions saved to {output_json}")

if __name__ == "__main__":
    main()

