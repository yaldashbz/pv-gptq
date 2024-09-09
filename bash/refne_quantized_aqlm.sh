#!/bin/bash
#SBATCH --job-name=mainx2mmlu
#SBATCH --output=../sbatch-logs/mainx2mmlu.out
#SBATCH --error=../sbatch-logs/mainx2mmlu.err

#number of CPUs to be used
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

#Define the number of hours the job should run.
#SBATCH --time=36:00:00

#Define the amount of system RAM used by your job in GigaBytes
#SBATCH --mem=1000G
#SBATCH --no-requeue

#Define the "gpu" partition for GPU-accelerated jobs
#SBATCH --partition=gpu100
#Define the number of GPUs used by your job
#SBATCH --gres=gpu:8


# Llama-3.1-8B
export NAME='Meta-Llama-3.1-8B-Instruct-gptq4-128-True-seed1_auxmmlu_gptqv2'
# export NAME='Meta-Llama-3.1-8B-Instruct-gptq4-128-True-seed1_mse_static'
export SEQLEN=8192
export SAVE_NAME=$NAME\_1e-4_mmlu_$SEQLEN
export MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
export QUANTIZED_MODEL_PATH=/nfs/scistore19/alistgrp/yshabanz/saved/$NAME # path to the model created by initial calibration
export TOKENIZED_DATASET_PATH=/nfs/scistore19/alistgrp/yshabanz/data/mmluaux_tokenized_llama3.1-8b-instruct_$SEQLEN  # yet again, red pajama adviced
export CACHE_DIR=/nfs/scistore19/alistgrp/yshabanz/cache_dir
export SNAPSHOT_PATH=/nfs/scistore19/alistgrp/yshabanz/saved/$SAVE_NAME
export NUM_GPUS=8

export WANDB_PROJECT=pv-gptq
export WANDB_NAME=$SAVE_NAME
export HUGGINGFACE_TOKEN='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'
export HF_TOKEN=$HUGGINGFACE_TOKEN


torchrun --nproc-per-node=$NUM_GPUS finetune_fsdp.py \
    --base_model $MODEL_PATH --quantized_model $QUANTIZED_MODEL_PATH \
    --model_seqlen=$SEQLEN --block_type LlamaDecoderLayer --limit_parallel_inits 4 \
    --load_dtype bfloat16 --amp_dtype bfloat16 --code_dtype int16 \
    --straight_through_buffer_dtype float32 \
    --dataset_name=$TOKENIZED_DATASET_PATH --split none --seed 1337 \
    --preprocessing_chunk_length 100000 --cache_dir=$CACHE_DIR --trust_remote_code \
    --update_codes --update_codebooks_and_scales --update_non_quantized_parameters \
    --lamb --debias --lr 1e-4 --adam_beta1 0.9 --adam_beta2 0.95 \
    --discrete_lr 0 --code_lr 3e-3 --code_beta1 0.0 --code_beta2 0.95 --beam_size 1 --delta_decay 0 \
    --max_code_change_per_step 5e-5 --code_trust_ratio 1e-2 --code_selection_temperature 0 \
    --batch_size=128 --microbatch_size=1 --max_epochs 10 --gradient_checkpointing \
    --print_every_steps=1 --verbose_optimizer  --eval_every_steps=10 --keep_best_model --wandb \
    --save $SNAPSHOT_PATH --save_every_steps 100