# This script quantizes the model and then tests its performance in terms of perplexity using WikiText2, and C4 datasets. 
# For this, we adapt the original code from AQLM and modify it to better support VQ.

export MODEL_PATH=meta-llama/Llama-2-7b-hf  # path or huggingface id of the base model
# export MODEL_PATH=facebook/opt-125m
export DATASET_PATH=wikitext2
export MODEL_SEQLEN=2048    # model-specific maximal sequence length, 4096 for llama2, 8192 for mistral
export NBITS_PER_CODEBOOK=16
export GROUP_SIZE=16
export HUGGINGFACE_TOKEN='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'
# this corresponds to having a single 16-bit codebook for 16-dimensional vectors

export BLOCKWISE_FINETUNE_EPOCHS=25
# set to 0 to disable blockwise finetuning during calibration

export CUDA_VISIBLE_DEVICES=0   # or e.g. 0,1,2,3
export SAVE_PATH=./
export WANDB_PROJECT=pv-tuning
export WANDB_NAME=initial_llama27bhf_wikitext2_gptq

python main.py \
    $MODEL_PATH \
    $DATASET_PATH \
    --nsamples=2048 \
    --val_size=256 \
    --model_seqlen=$MODEL_SEQLEN \
    --num_codebooks=1 \
    --nbits_per_codebook=$NBITS_PER_CODEBOOK \
    --out_group_size=1 \
    --in_group_size=$GROUP_SIZE \
    --beam_size=1 \
    --relative_mse_tolerance=0.01 \
    --max_epochs=100 \
    --finetune_lr=1e-4 \
    --finetune_adam_beta1=0.90 \
    --finetune_adam_beta2=0.999 \
    --finetune_keep_best \
    --finetune_batch_size=64 \
    --local_batch_size=4 \
    --finetune_max_epochs=$BLOCKWISE_FINETUNE_EPOCHS \
    --finetune_early_stop=3 \
    --offload_activations \
    --save $SAVE_PATH \
    --wandb --resume \
    --trust_remote_code
