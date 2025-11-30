import os
import json
import pickle
import numpy as np
import cv2
from typing import Any

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_json(data: Any, path: str, indent: int = 2):
    """Save data as JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_json(path: str) -> Any:
    """Load JSON data."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_pickle(data: Any, path: str):
    """Save data as Pickle."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path: str) -> Any:
    """Load Pickle data."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_image(img: np.ndarray, path: str):
    """Save numpy image (BGR)."""
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def save_text(text: str, path: str):
    """Save text string."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

