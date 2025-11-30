import os
import torch
import pickle
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import glob

def get_texture_files(root_dir):
    """
    Traverse the directory and find all valid texture files.
    Structure: root_dir/{category}/*
    Returns a dict where key is the category and value is a list of paths to the main (color/diffuse) texture files.
    """
    texture_files = {}
    categories = ['wall', 'floor', 'ceiling']
    
    # Keywords that identify a texture as a "color" or "diffuse" map
    color_keywords = ['_COL_', '_diff_', '_diffuse', '_Albedo', 'COL', 'diff']
    
    for category in categories:
        cat_dir = os.path.join(root_dir, category)
        if not os.path.exists(cat_dir):
            print(f"Warning: Directory {cat_dir} does not exist.")
            continue
            
        files = []
        # Recursive search for all images
        for ext in ['jpg', 'png', 'jpeg', 'exr', 'tif', 'tiff']:
            files.extend(glob.glob(os.path.join(cat_dir, f"*.{ext}")))
            files.extend(glob.glob(os.path.join(cat_dir, "**", f"*.{ext}"), recursive=True))
            
        valid_files = []
        seen_bases = set()
        
        # Sort files to ensure consistent processing
        files = sorted(list(set(files)))
        
        for f in files:
            filename = os.path.basename(f)
            
            # Skip preview/thumbnails
            if 'preview' in filename.lower():
                continue

            # Check if this is a color map
            is_color = False
            for kw in color_keywords:
                # Check for keyword (case-insensitive)
                # We look for _COL_ or _diff_ usually, but sometimes it might be at the end or start?
                # Based on user examples: Tiles15_COL_VAR1_6K.jpg
                if kw.lower() in filename.lower():
                    is_color = True
                    break
            
            if is_color:
                valid_files.append(f)
        
        texture_files[category] = valid_files
        print(f"Found {len(texture_files[category])} color textures for category '{category}'")
        
    return texture_files

def main():
    root_dir = "asset_data/background_texture_dataset"
    output_file = "asset_data/background_texture_dataset/texture_embeddings.pkl"
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load DINOv2 model
    print("Loading DINOv2 model...")
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
    model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)
    model.eval()
    
    texture_files = get_texture_files(root_dir)
    all_embeddings = {}
    
    for category, files in texture_files.items():
        all_embeddings[category] = {}
        print(f"Processing {category}...")
        
        for filepath in tqdm(files):
            try:
                image = Image.open(filepath).convert("RGB")
                
                # Preprocess
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Save Patch features (skip CLS token at index 0)
                    # Shape: [1, 256, 1024] (assuming 224x224 input -> 16x16 patches)
                    patch_features = outputs.last_hidden_state[:, 1:, :].cpu()
                    
                # Store embedding (detached, on CPU)
                filename = os.path.basename(filepath)
                all_embeddings[category][filename] = {
                    'path': filepath,
                    'embedding': patch_features # Tensor [1, 256, 1024]
                }
                
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                
    # Save to file
    print(f"Saving embeddings to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(all_embeddings, f)
    print("Done!")

if __name__ == "__main__":
    main()

