import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import re
import cv2
import matplotlib.pyplot as plt

def draw_mask(mask, draw, random_color=True, color=(30, 144, 255, 153)):
    """
    Draw mask on PIL ImageDraw.
    (From S1_scene_parsing_op.py)
    """
    if random_color:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            153,
        )
    
    nonzero_coords = np.transpose(np.nonzero(mask))
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

def adjust_brightness(color, factor=1.5):
    return tuple(min(int(c * factor), 255) for c in color)

def visualize_detection(image_pil: Image.Image, result: dict, font_path: str = None) -> Image.Image:
    """
    Visualize detection results (Boxes + Masks + Labels).
    (Refactored from S1_scene_parsing_op.py visualize function)
    """
    if isinstance(image_pil, np.ndarray):
        image_pil = Image.fromarray(image_pil)
    
    draw = ImageDraw.Draw(image_pil)
    mask_image = Image.new("RGBA", image_pil.size, color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)
    
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("categorys", [])
    masks = result.get("masks", [])
    
    try:
        font = ImageFont.truetype(font_path, 15) if font_path else ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    cate2color = {}
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if label not in cate2color:
            base = tuple(np.random.randint(0, 255, size=3).tolist())
            cate2color[label] = adjust_brightness(base)
        color = cate2color[label]
        
        # Draw Box
        x0, y0, x1, y1 = map(int, box)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        
        # Draw Text
        text = f"{label} {scores[i]:.2f}" if i < len(scores) else label
        draw.text((x0, y0), text, fill=color, font=font)
        
        # Draw Mask
        if i < len(masks):
            mask_np = np.array(masks[i])
            mask_np = mask_np if mask_np.ndim == 2 else mask_np[:, :, -1]
            draw_mask(mask_np, mask_draw, random_color=False, color=color+(150,))

    image_pil = Image.alpha_composite(image_pil.convert("RGBA"), mask_image).convert("RGB")
    return image_pil

def stitch_images(images_list, grid_size=None):
    """
    Stitch multiple images into a grid.
    (Simplified from S3 logic)
    """
    if not images_list:
        return None
        
    n = len(images_list)
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_size
        
    w, h = images_list[0].size
    grid_img = Image.new('RGB', (cols * w, rows * h))
    
    for i, img in enumerate(images_list):
        r = i // cols
        c = i % cols
        grid_img.paste(img, (c * w, r * h))
        
    return grid_img

