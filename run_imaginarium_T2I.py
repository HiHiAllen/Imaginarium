#!/usr/bin/env python
# coding=utf-8
import os
import argparse
import torch
import datetime
from pathlib import Path
from diffusers import FluxPipeline, FluxTransformer2DModel
from peft import LoraConfig, set_peft_model_state_dict

# Set HF mirror for better connectivity in some regions
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def parse_args():
    parser = argparse.ArgumentParser(description="Imaginarium Text-to-Image Generation (Stage 1)")
    
    # Required arguments matching the README example
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for image generation",
    )
    
    # Optional arguments
    parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_images",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Path to the base model or model identifier",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="weights/imaginarium_finetuned_flux.pth",
        help="Path to the LoRA weights file",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Image resolution (height and width)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation",
    )
    
    return parser.parse_args()

def setup_pipeline(args):
    """
    Setup the Flux pipeline with LoRA weights.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use float16 for CUDA, float32 for CPU
    weight_dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading transformer from {args.base_model}...")
    try:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.base_model,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        print("Please ensure you have access to the model and a valid HF token if required.")
        raise

    # Configure LoRA (using configuration from training/inference scripts)
    # Rank and Alpha are set to 16 as per provided flux_train_eval scripts
    transformer_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_bias=False,
    )
    transformer.add_adapter(transformer_lora_config)
    
    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora_path}...")
    if os.path.exists(args.lora_path):
        try:
            lora_state_dict = torch.load(args.lora_path, map_location="cpu")
            incompatible_keys = set_peft_model_state_dict(
                transformer, lora_state_dict, adapter_name="default"
            )
            if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
                print(f"Warning: Incompatible keys in LoRA weights: {incompatible_keys}")
        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
    else:
        print(f"Warning: LoRA weights file not found at {args.lora_path}. Using base model without LoRA.")

    # Create the full pipeline
    print("Creating Flux pipeline...")
    pipeline = FluxPipeline.from_pretrained(
        args.base_model,
        transformer=transformer,
        torch_dtype=weight_dtype,
    )
    
    pipeline = pipeline.to(device)
    return pipeline, weight_dtype

def main():
    args = parse_args()
    
    # Initialize pipeline
    pipeline, weight_dtype = setup_pipeline(args)
    
    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Generating {args.num} images for prompt: '{args.prompt}'")
    
    for i in range(args.num):
        # Handle seeding
        current_seed = None
        generator = None
        if args.seed is not None:
            current_seed = args.seed + i
            generator = torch.Generator(device=pipeline.device).manual_seed(current_seed)
        
        print(f"  Generating image {i+1}/{args.num}...")
        
        with torch.autocast(pipeline.device.type, dtype=weight_dtype):
            image = pipeline(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                max_sequence_length=512,
                height=args.resolution,
                width=args.resolution,
            ).images[0]
            
        # Save image
        filename = f"img_{timestamp}_{i+1}.png"
        save_path = output_dir / filename
        image.save(save_path)
        print(f"  Saved to: {save_path}")

    print("Generation complete.")

if __name__ == "__main__":
    main()

