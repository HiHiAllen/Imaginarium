#!/bin/bash

# --- 配置区域 ---
# 可见显卡，例如使用前两张卡

# nohup bash run.sh > run_flux.log 2>&1 &
# export WANDB_DISABLED=true
# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_MODE=online
# export MASTER_ADDR="localhost"
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA="mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107"
export NCCL_IB_GID_INDEX=7

export CUDA_VISIBLE_DEVICES=0,1,2,3
unset PET_NNODES
unset PET_NODE_RANK
unset PET_MASTER_ADDR

# Huggingface 模型 ID
MODEL_NAME="Your FLUX DIR"
# 你的数据集文件夹路径 (包含 png 和 json)
DATA_DIR="Your DATA DIR"
# 输出路径
OUTPUT_DIR="./flux_full_v2_rank32"

# 训练参数
RESOLUTION=1024
TRAIN_BATCH_SIZE=1  # Flux 显存占用大，通常设为 1
GRAD_ACCUM=1        
LR=5e-5
MAX_STEPS=2000
VALIDATION_STEPS=200 # 每多少步生成一次预览图
RANK=32              # LoRA Rank

# 启动训练
# --mixed_precision=bf16 是必须的，Flux 在 fp16 下可能会溢出
torchrun --nproc_per_node=4 --master_port=29502 \
    train.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --data_root=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --resolution=$RESOLUTION \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --learning_rate=$LR \
    --max_train_steps=$MAX_STEPS \
    --validation_steps=$VALIDATION_STEPS \
    --save_steps=500 \
    --rank=$RANK \
    # --gradient_accumulation_steps=$GRAD_ACCUM \