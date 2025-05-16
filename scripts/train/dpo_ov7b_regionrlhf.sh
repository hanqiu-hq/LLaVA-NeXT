#!/bin/bash

export OMP_NUM_THREADS=8
#export NCCL_IB_DISABLE=0
#export NCCL_IB_GID_INDEX=3
# export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
#export NCCL_SOCKET_IFNAME=eth0
#export NCCL_DEBUG=INFO

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

# DPO Stage
PROMPT_VERSION="qwen_1_5"
SFT_MODEL="lmms-lab/llava-onevision-qwen2-7b-ov"
EPOCH=1
beta=0.1

NUM_GPUS=$1
DATA_PATH=$2
OUTPUT_DIR=$3
LEARNING_RATE=$4
# default batch size 2
ACC_BATCH_SIZE=$(echo "scale=0; 2 / $NUM_GPUS" | bc)
if [ "$ACC_BATCH_SIZE" -lt 1 ]; then
  ACC_BATCH_SIZE=1
fi


MASTER_PORT=$((RANDOM % 101 + 20001))

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    llava/train/train_dpo.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --deepspeed scripts/zero2.json \
    --model_name_or_path=${SFT_MODEL} \
    --dpo_alpha=1.0 \
    --beta=${beta} \
    --gamma=0 \
    --version $PROMPT_VERSION \
    --data_path=$DATA_PATH \
    --image_folder "all_data" \
    --unfreeze_mm_vision_tower True \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name "llava_ov_region_rlhf" \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $ACC_BATCH_SIZE \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 1 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --image_max_pixels 1690000 \
    --image_min_pixels 147456 \
    --shuffle_data True \
    ${@:5}
