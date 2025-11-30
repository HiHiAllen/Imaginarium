import base64
import json
import requests
import openai
import time
import io
import numpy as np
import cv2
import multiprocessing
import re
import ast
import threading
import os
from PIL import Image, ImageOps

# 线程锁，用于打印输出，避免混乱
print_lock = threading.Lock()

# 全局变量，用于在工作进程中存储 agent 实例
worker_agent = None

IMAGE_PLACEHOLDER = '<image-placeholder>'

def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with print_lock:
        print(*args, **kwargs, flush=True)

def init_worker(agent_params):
    """初始化工作进程，创建 agent 实例"""
    global worker_agent
    worker_agent = GPTApi(**agent_params)
        
class BaseApi:
    def __init__(self) -> None:
        pass

    @staticmethod
    def encode_image(image_input):
        if isinstance(image_input, str):
            # 读取图像文件并统一转换为JPEG格式，确保与API的image/jpeg声明一致
            img = Image.open(image_input)
            # 如果是RGBA模式，转换为RGB
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            # 转换为JPEG格式的字节流
            with io.BytesIO() as buffer:
                img.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode("utf-8")
        elif isinstance(image_input, np.ndarray):
            _, buffer = cv2.imencode('.jpg', image_input)
            image_bytes = buffer.tobytes()
            return base64.b64encode(image_bytes).decode('utf-8')
        else:
            with io.BytesIO() as buffer:
                if image_input.mode == 'RGBA':
                    image_input = image_input.convert('RGB')
                image_input.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')

    def get_response(self, prompt, image=None, **kwargs):
        raise NotImplementedError


class GPTApi(BaseApi):
    """
    GPT API Wrapper for Imaginarium.
    """
    def __init__(self, model, GPT_KEY, GPT_ENDPOINT, use_openai_client=False) -> None:
        self.GPT_KEY = GPT_KEY
        self.GPT_ENDPOINT = GPT_ENDPOINT
        self.use_openai_client = use_openai_client
        
        if self.use_openai_client:
            if '/v1/chat/completions' in self.GPT_ENDPOINT:
                base_url = self.GPT_ENDPOINT.replace('/v1/chat/completions', '')
            else:
                base_url = self.GPT_ENDPOINT
            
            self.openai_client = openai.OpenAI(
                api_key=self.GPT_KEY,
                base_url=base_url
            )
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.GPT_KEY}",
        }
        self.models = ['gpt-5', 'anthropic/claude-sonnet-4-5-20250929', 'google/gemini-2.5-pro']
        self.current_model = model
        self.system_message = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a visual AI assistant, please follow the user's instructions and answer the questions carefully.",
                },
            ],
        }

    def _get_response_with_openai_client(self, prompt, image=None, history=None, return_history=None, only_return_request=False, **kwargs):
        try:
            messages = []
            if not history:
                messages.append({
                    "role": "system",
                    "content": "You are a visual AI assistant, please follow the user's instructions and answer the questions carefully."
                })
            
            if history:
                messages.extend(history)
            
            user_content = []
            
            if isinstance(image, list):
                if not isinstance(prompt, list):
                    prompts, images = prompt.split(IMAGE_PLACEHOLDER), image
                else:
                    prompts, images = prompt, image
                    
                if images is None:
                    images = [None] * len(prompts)

                if len(prompts) != len(images) + 1:
                    raise ValueError(f"prompts and images must have the same length, {len(prompts)} != {len(images)}")
                
                user_content.append({"type": "text", "text": prompts[0]})
                for prompt_part, image_part in zip(prompts[1:], images):
                    encoded_image = GPTApi.encode_image(image_part)
                    if encoded_image:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        })
                    user_content.append({"type": "text", "text": prompt_part})
            else:
                if image is not None:
                    encoded_image = GPTApi.encode_image(image)
                    if encoded_image:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        })
                user_content.append({"type": "text", "text": prompt})
            
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            if only_return_request:
                return messages
            
            temperature = kwargs.get("temperature", 0)
            max_tokens = kwargs.get("max_tokens", 16384)
            max_retries = kwargs.get("max_retries", 5)
            
            for retry in range(max_retries):
                try:
                    start_time = time.time()
                    response = self.openai_client.chat.completions.create(
                        model=self.current_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    # print(f"GPT Time cost {time.time() - start_time}s.")
                    
                    full_text = response.choices[0].message.content
                    if full_text:
                        if not return_history:
                            return full_text
                        else:
                            cur_answer = {"role": "assistant", "content": [{"type": "text", "text": full_text}]}
                            messages.append(cur_answer)
                            return full_text, messages
                    
                except Exception as e:
                    print(f"Error (Retry {retry+1}): {e}", flush=True)
                    if retry < max_retries - 1:
                        time.sleep(1)
            return None
            
        except Exception as e:
            print(f"Error in OpenAI client method: {e}")
            return None
    
    def get_response(self, prompt, image=None, history=None, return_history=None, only_return_request=False, **kwargs):
        if self.use_openai_client:
            return self._get_response_with_openai_client(prompt, image, history, return_history, only_return_request, **kwargs)
        
        # Fallback to requests if needed (omitted for brevity, assuming OpenAI client is preferred)
        raise NotImplementedError("Only OpenAI Client mode is fully migrated for this refactor.")

def parallel_processing_requests(agent_params,all_image_list, all_prompt_list, return_list, return_json, return_dict, num_processes=8):
    args_list = [(prompt, image_list) for prompt, image_list in zip(all_prompt_list, all_image_list)]
    
    with multiprocessing.Pool(processes=min(len(args_list), num_processes), initializer=init_worker, initargs=(agent_params,)) as pool:
        if return_list:
            all_results = pool.map(process_single_request_and_return_list, args_list)
        elif return_json:
            all_results = pool.map(process_single_request_and_return_json, args_list)
        elif return_dict:
            all_results = pool.map(process_single_request_and_return_dict, args_list)
        else:
            raise  ValueError("return_list or return_json, choose one")
        
    return all_results

def process_single_request_and_return_list(args):
    prompt, image_list = args
    final_res = 'error'
    for _ in range(3):
        res = worker_agent.get_response(prompt, image=image_list)
        final_res = extract_list_with_re(res)
        if final_res!='error': break
    return final_res

def process_single_request_and_return_json(args):
    prompt, image_list = args
    final_res = 'error'
    for _ in range(3):
        res = worker_agent.get_response(prompt, image=image_list)
        final_res = extract_json_with_re(res)
        if final_res!='error': break
    return final_res

def process_single_request_and_return_dict(args):
    prompt, image_list = args
    final_res = 'error'
    for _ in range(3):
        res = worker_agent.get_response(prompt, image=image_list)
        final_res = extract_dict_with_re(res)
        if final_res!='error': break
    return final_res

def extract_list_with_re(output):
    try:
        list_pattern = r'\[.*?\]'
        list_matches = re.findall(list_pattern, output, re.DOTALL)
        if list_matches:
            list_str = list_matches[-1]
            try:
                list_data = ast.literal_eval(list_str)
                if isinstance(list_data, list):
                    return list_data
            except (SyntaxError, ValueError):
                pass
        return []
    except Exception:
        return 'error'
    
def extract_dict_with_re(output):
    try:
        dict_pattern = r'\{.*\}'
        dict_match = re.search(dict_pattern, output, re.DOTALL)
        if dict_match:
            dict_str = dict_match.group(0)
            dict_str = dict_str.replace('None', 'None')
            dict_data = ast.literal_eval(dict_str)
            return dict_data
        else:
            raise ValueError("No dict found")
    except Exception:
        return 'error'
    
def extract_json_with_re(output):
    json_match = re.search(r'\{[\s\S]*\}', output)
    if json_match:
        json_str = json_match.group()
        json_str = re.sub(r'//.*', '', json_str)
        json_str = json_str.replace('False', 'false').replace('True', 'true').replace('None', 'null')
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}
    else:
        return 'error'

