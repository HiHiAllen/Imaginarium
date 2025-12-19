import argparse
import logging
import math
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Diffusers & PEFT
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from peft import LoraConfig, get_peft_model_state_dict
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, T5EncoderModel

from flux_dataset import LocalJsonDataset, collate_fn

# --- DDP Helper ---
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# --- Helpers ---
def encode_prompt(tokenizer, text_encoder, tokenizer_2, text_encoder_2, prompt, device):
    text_inputs = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=False).pooler_output
    
    text_inputs_2 = tokenizer_2(prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    prompt_embeds_2 = text_encoder_2(text_inputs_2.input_ids.to(device))[0]
    
    return prompt_embeds_2, prompt_embeds

def log_validation(transformer, args, rank, device, step, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, scheduler, writer):
    if rank != 0: return
    
    print(f"Running validation... Step {step}")
    unwrapped_transformer = transformer.module if hasattr(transformer, "module") else transformer
    
    # 切换到评估模式
    unwrapped_transformer.eval()
    
    pipeline = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=unwrapped_transformer,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    validation_prompts = [
        "A musician's bedroom with a wooden bed, desk, guitar and bookshelf", 
        "Modern minimalist bedroom",
        "a cozy living room",
        "A minimalist living room with abstract art, white sofas, and a floor lamp, emphasizing simplicity and elegance ",
        "A vibrant florist shop filled with diverse potted plants and a wooden display shelf showcasing vibrant greenery. ",
        "A modern L - shaped kitchen with walnut wood cabinets and white marble countertops , featuring a kitchen island with three wooden bar stools , white microwave , and decorative potted plants. ",
        "An industrial storage space with pallets , barrels ,and various industrial equipment. ",
        "A modern conference room with a large oval table , ergonomic chairs , and wall - mounted display. ",
        "An entertainment room with pool table and arcade machines ",
        "A warm dining room with a chandelier , modern table , and decorative shelving for a cozy dining experience"
    ]
    
    generator = torch.Generator(device=device).manual_seed(args.seed)
    save_dir = os.path.join(args.output_dir, "validation", f"step_{step}")
    os.makedirs(save_dir, exist_ok=True)
    
    images_np = []
    
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        for i, prompt in enumerate(validation_prompts):
            image = pipeline(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=3.5,
                generator=generator,
                height=args.resolution,
                width=args.resolution,
            ).images[0]
            
            image.save(os.path.join(save_dir, f"{i}.png"))
            images_np.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))

    del pipeline
    torch.cuda.empty_cache()
    
    # 恢复训练模式
    unwrapped_transformer.train()

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="flux-ddp-output")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. DDP 初始化
    global_rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    else:
        writer = None

    # 设置随机种子
    torch.manual_seed(args.seed + global_rank)
    np.random.seed(args.seed + global_rank)

    # 2. 加载模型
    dtype = torch.bfloat16
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=dtype).to(device)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=dtype).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder_2 = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
    tokenizer_2 = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    
    transformer = FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=dtype).to(device)

    # 冻结模型
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    transformer.requires_grad_(False)

    # 3. LoRA 设置
    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]
    )
    transformer.add_adapter(lora_config)
    transformer.enable_gradient_checkpointing()
    
    # 确保 LoRA 权重可训练且为 float32
    for p in transformer.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    # 4. Wrap DDP
    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(transformer_ddp.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # 6. 数据集与 DistributedSampler
    dataset = LocalJsonDataset(data_root=args.data_root, resolution=args.resolution)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True, drop_last=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.train_batch_size,
        sampler=sampler, 
        collate_fn=collate_fn, 
        num_workers=4,
        pin_memory=True
    )

    # 7. 训练循环
    global_step = 0
    steps_per_epoch = len(dataloader)
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    epochs = math.ceil(args.max_train_steps / steps_per_epoch)
    
    if global_rank == 0:
        print(f"Start training: {args.max_train_steps} steps, World Size: {world_size}, Epochs: {epochs}")
    
    progress_bar = tqdm(range(args.max_train_steps), disable=(global_rank != 0))
    
    transformer_ddp.train()

    try:
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            
            for step, batch in enumerate(dataloader):
                # 准备输入
                pixel_values = batch["pixel_values"].to(device, dtype=dtype)
                prompts = batch["prompts"]

                # 编码
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        tokenizer, text_encoder, tokenizer_2, text_encoder_2, prompts, device
                    )

                # 噪声 & 时间
                bsz = latents.shape[0]
                noise = torch.randn_like(latents)
                timesteps = torch.rand((bsz,), device=device)
                shift = 5
                timesteps = sd3_time_shift(shift,timesteps)
                
                # Flow Matching 前向过程
                timesteps_bc = timesteps.view(-1, 1, 1, 1)
                noisy_input = (1.0 - timesteps_bc) * latents + timesteps_bc * noise

                # Pack latent
                packed_input = FluxPipeline._pack_latents(
                    noisy_input, bsz, noisy_input.shape[1], noisy_input.shape[2], noisy_input.shape[3]
                )
                packed_ids = FluxPipeline._prepare_latent_image_ids(
                    bsz, noisy_input.shape[2] // 2, noisy_input.shape[3] // 2, device, dtype
                )
                txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)

                # 前向传播
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = transformer_ddp(
                        hidden_states=packed_input,
                        timestep=timesteps,
                        guidance=torch.tensor([1.0], device=device, dtype=dtype),
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=txt_ids,
                        img_ids=packed_ids,
                        return_dict=False,
                    )[0]
                    
                    target = noise - latents
                    packed_target = FluxPipeline._pack_latents(
                        target, bsz, target.shape[1], target.shape[2], target.shape[3]
                    )
                    
                    loss = F.mse_loss(pred.float(), packed_target.float())

                # 反向传播
                loss.backward()

                # 梯度裁剪 & 优化
                grad_norm = torch.nn.utils.clip_grad_norm_(transformer_ddp.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                progress_bar.update(1)

                # 记录日志
                dist.all_reduce(loss,op=dist.ReduceOp.AVG)
                if global_rank == 0:
                    current_loss = loss.item()
                    writer.add_scalar("train/loss", current_loss, global_step)
                    writer.add_scalar("train/grad_norm", grad_norm, global_step)
                    progress_bar.set_postfix(loss=current_loss, grad_norm=grad_norm)
                    
                    # 验证
                    if global_step % args.validation_steps == 0:
                        log_validation(transformer_ddp, args, global_rank, device, global_step,
                                       vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, noise_scheduler, writer)

                    # 保存
                    if global_step % args.save_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        unwrapped_model = transformer_ddp.module
                        lora_state_dict = get_peft_model_state_dict(unwrapped_model)
                        torch.save(lora_state_dict, os.path.join(save_path, "lora_weights.pt"))
                        print(f"Saved to {save_path}")

                if global_step >= args.max_train_steps:
                    break
                    
    except Exception as e:
        print(f"Rank {global_rank} encountered error: {e}")
        raise
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    main()