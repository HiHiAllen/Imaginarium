import base64
from PIL import Image
from io import BytesIO
import os

import numpy as np
from pycocotools import mask as mask_utils
from typing import Dict, List
from typing import Tuple
import concurrent.futures
import tempfile
import numpy as np
from PIL import Image
from easydict import EasyDict

def string2rle(rle_str: str) -> List[int]:
    p = 0
    cnts = []

    while p < len(rle_str) and rle_str[p]:
        x = 0
        k = 0
        more = 1

        while more:
            c = ord(rle_str[p]) - 48
            x |= (c & 0x1f) << 5 * k
            more = c & 0x20
            p += 1
            k += 1

            if not more and (c & 0x10):
                x |= -1 << 5 * k

        if len(cnts) > 2:
            x += cnts[len(cnts) - 2]
        cnts.append(x)
    return cnts


def rle2mask(rle: Dict, size: Tuple[int, int], label=1):
    h, w = size
    img = np.zeros((h, w), dtype=np.uint8)

    ps = 0
    cnts = rle
    for i in range(0, len(cnts) -1, 2):
        ps += cnts[i]

        for j in range(cnts[i + 1]):
            x = (ps + j) % w
            y = (ps + j) // w

            if y < h and x < w:
                img[y, x] = label
            else:
                break

        ps += cnts[i + 1]

    return img

def rle2rgba(mask_obj) -> Image.Image:
    """
    Convert the compressed RLE string of mask object to png image object.

    :param mask_obj: The :class:`Mask <dds_cloudapi_sdk.tasks.ivp.IVPObjectMask>` object detected by this task
    """
    mask_array = mask_utils.decode(mask_obj)

    # convert the array to a 4-channel RGBA image
    mask_alpha = np.where(mask_array == 1, 255, 0).astype(np.uint8)
    mask_rgba = np.stack((255 * np.ones_like(mask_alpha),
                            255 * np.ones_like(mask_alpha),
                            255 * np.ones_like(mask_alpha),
                            mask_alpha),
                            axis=-1)
    image = Image.fromarray(mask_rgba, "RGBA")
    return image


def postprocess(result, task, return_mask):
    """Postprocess the result from the API call

    Args:
        result (TaskResult): Task result with the following keys:
            - objects (List[DetectionObject]): Each DetectionObject has the following keys:
                - bbox (List[float]): Box in xyxy format
                - category (str): Detection category
                - score (float): Detection score
                - mask (DetectionObjectMask): Use mask.counts to parse RLE mask 
        task (DetectionTask): The task object
        return_mask (bool): Whether to return mask

    Returns:
        (Dict): Return dict in format:
            {
                "scores": (List[float]): A list of scores for each object
                "categorys": (List[str]): A list of categorys for each object
                "boxes": (List[List[int]]): A list of boxes for each object
                "masks": (List[PIL.Image]): A list of masks in the format of PIL.Image
            }
    """
    def process_object_with_mask(object):
        box = object.bbox
        score = object.score
        category = object.category
        # import pdb; pdb.set_trace();
        mask = rle2rgba(object.mask)

        # Crop mask with bbox as per user's suggestion
        mask_array = np.array(mask)
        x0, y0, x1, y1 = [int(c) for c in box]
        h, w, _ = mask_array.shape
        
        # Ensure coordinates are within image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)

        # Create a new blank mask and copy the cropped part
        cropped_mask_array = np.zeros_like(mask_array)
        if y1 > y0 and x1 > x0:
            cropped_mask_array[y0:y1, x0:x1] = mask_array[y0:y1, x0:x1]
        
        mask = Image.fromarray(cropped_mask_array, "RGBA")
        return box, score, category, mask
    
    def process_object_without_mask(object):
        box = object.bbox
        score = object.score
        category = object.category
        mask = None
        return box, score, category, mask
    
    boxes, scores, categorys, masks = [], [], [], []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        if return_mask:
            process_object = process_object_with_mask
        else:
            process_object = process_object_without_mask
        futures = [executor.submit(process_object, obj) for obj in result.objects]
        for future in concurrent.futures.as_completed(futures):
            box, score, category, mask = future.result()
            boxes.append(box)
            scores.append(score)
            categorys.append(category)
            if mask is not None:
                masks.append(mask)

    return dict(boxes=boxes, categorys=categorys, scores=scores, masks=masks)


def array_to_base64(image_array):
    # 将numpy数组转换为PIL Image
    if isinstance(image_array, np.ndarray):
        img = Image.fromarray(image_array)
    else:
        img = image_array  # 如果已经是PIL Image
    # 转换为 RGB 或 RGBA（保持 PNG 支持透明度）
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    # 将图像转为二进制数据
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    # 转换为 Base64 并添加前缀
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"

def path_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"

def dino_api(prompts, token):
    import time
    import requests
    
    image_data = prompts['image']
    if isinstance(image_data, str):
        base64_image = path_to_base64(image_data)
    elif isinstance(image_data, np.ndarray):
        base64_image = array_to_base64(image_data)
    else:
        raise ValueError(f"dino_api: image_data type error: {type(image_data)}")

    headers = {
        "Content-Type": "application/json",
        "Token"       : token
    }

    body = {
        "model": "DINO-X-1.0",
        "image": base64_image,
        "prompt": {
            "type":"text",
            "text":prompts['prompt']
        },
        "mask_format": "coco_rle",
        "targets": ["bbox", "mask"],
        "bbox_threshold": 0.25,
        "iou_threshold": 0.8
}
    # 2. 发起算法调用
    resp = requests.post(
        url='https://api.deepdataspace.com/v2/task/dinox/detection', # 这里是举例，具体算法路径查看具体API说明
        json= body,
        headers=headers
    )
    json_resp = resp.json()
    print(json_resp)
    # # 3. 获取 task_uuid
    task_uuid = json_resp["data"]["task_uuid"]

    # # 4. 轮询任务状态
    while True:
        resp = requests.get(f'https://api.deepdataspace.com/v2/task_status/{task_uuid}', headers=headers)
        json_resp = resp.json()
        if json_resp["data"]["status"] not in ["waiting", "running"]:
            break
        time.sleep(1)

    if json_resp["data"]["status"] == "failed":
        raise Exception(json_resp)
    elif json_resp["data"]["status"] == "success":
        obj_resp = EasyDict(json_resp)
        results =  postprocess(obj_resp.data.result,obj_resp,True)
        return results