import argparse
import os
import re
import torch
import torch.distributed as dist
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, T5EncoderModel

from peft import LoraConfig
from peft.utils import set_peft_model_state_dict


# ---------------- DDP helpers ----------------
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_rank0():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def shard_list(items, rank, world_size):
    # rank 0 gets 0,8,16... ; rank1 gets 1,9,17...
    return [(i, items[i]) for i in range(rank, len(items), world_size)]


# ---------------- naming helpers ----------------
def prompt_to_prefix(prompt: str, max_words: int = 6, max_len: int = 60) -> str:
    """
    Turn prompt into a safe filename-ish prefix:
    - take first max_words words
    - lower, keep alnum/_/-
    """
    words = prompt.strip().split()
    head = " ".join(words[:max_words]).lower()
    head = re.sub(r"[^a-z0-9\s_-]+", "", head)  # drop punctuation
    head = re.sub(r"\s+", "_", head).strip("_")
    if not head:
        head = "prompt"
    return head[:max_len]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/share/project/huangxu/models/flux1")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="flux-ddp-infer-output")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--base_seed", type=int, default=42)

    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=3.5)

    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # roll config
    parser.add_argument("--rolls", type=int, default=4)
    parser.add_argument("--seed_stride", type=int, default=1000)  # seed = base + idx*stride + k

    # naming
    parser.add_argument("--prefix_words", type=int, default=6)
    parser.add_argument("--prefix_use_index", action="store_true",
                        help="prepend pXX_ to avoid collisions (recommended).")

    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    if is_rank0():
        os.makedirs(args.output_dir, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    # ---------------- load base model parts ----------------
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=dtype
    ).to(device)

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device)
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )

    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=dtype
    ).to(device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    transformer.requires_grad_(False)

    # ---------------- attach LoRA + load weights ----------------
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    transformer.add_adapter(lora_config)

    lora_state = torch.load(args.lora_path, map_location="cpu")
    set_peft_model_state_dict(transformer, lora_state, adapter_name="default")

    transformer.eval()

    pipe = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=transformer,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # ---------------- prompts (20) ----------------
    prompts = [
        "A cozy living room with a sectional sofa, warm floor lamp, and a large window with afternoon light",
        "Modern minimalist living room with abstract art, white sofa, and a single floor lamp",
    ]

    # ---------------- distribute prompts across ranks ----------------
    my_work = shard_list(prompts, rank, world_size)

    # per-rank output folder (avoids write collisions)
    rank_dir = os.path.join(args.output_dir, f"rank_{rank:02d}")
    os.makedirs(rank_dir, exist_ok=True)

    if is_rank0():
        print(f"DDP inference | world_size={world_size} | prompts={len(prompts)} | rolls={args.rolls}")
        print(f"LoRA: {args.lora_path} | r={args.lora_rank} alpha={args.lora_alpha}")
        print(f"Seed rule: seed = base_seed + prompt_idx*{args.seed_stride} + roll_k")
        print(f"Output: {args.output_dir}")

    iterator = my_work
    if is_rank0():
        iterator = tqdm(my_work, desc="Prompts on rank0", total=len(my_work))

    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        for idx, prompt in iterator:
            prefix = prompt_to_prefix(prompt, max_words=args.prefix_words)
            if args.prefix_use_index:
                prefix = f"p{idx:02d}_{prefix}"

            for k in range(args.rolls):
                seed = args.base_seed + idx * args.seed_stride + k
                gen = torch.Generator(device=device).manual_seed(seed)

                image = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=gen,
                    height=args.resolution,
                    width=args.resolution,
                ).images[0]

                # naming: prefix_0/1/2/3.png
                out_path = os.path.join(rank_dir, f"{prefix}_{k}.png")
                image.save(out_path)

    if dist.is_initialized():
        dist.barrier()

    if is_rank0():
        print("Done.")

    cleanup_ddp()


if __name__ == "__main__":
    main()
