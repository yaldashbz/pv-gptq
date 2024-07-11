export MODEL_PATH=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T  # path or huggingface id of the base model
export QUANTIZED_MODEL_PATH=../PV-GPTQ/Tiny-llama-4bit-v2 # path to the model created by initial calibration
export TOKENIZED_DATASET_PATH=../PV-GPTQ/pajama_tokenized_tinyllama-v2  # yet again, red pajama adviced
export CACHE_DIR=../PV-GPTQ/cache_dir
export SNAPSHOT_PATH=../PV-GPTQ/pv_model-v2
export SEQLEN=2048
export NUM_GPUS=4

export WANDB_PROJECT=pv-gptq
export WANDB_NAME=pv-gptq_tinyllama_pajama-v2-loadcheck
export HUGGINGFACE_TOKEN='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'


torchrun --nproc-per-node=$NUM_GPUS finetune_fsdp.py \
    --base_model $MODEL_PATH --quantized_model $QUANTIZED_MODEL_PATH \
    --model_seqlen=$SEQLEN --block_type LlamaDecoderLayer --limit_parallel_inits 4 \
    --load_dtype bfloat16 --amp_dtype bfloat16 --code_dtype int16 \
    --straight_through_buffer_dtype float32 \
    --dataset_name=$TOKENIZED_DATASET_PATH --split none --seed 1337 \
    --preprocessing_chunk_length 100000 --cache_dir=$CACHE_DIR --trust_remote_code \
    --update_codes --update_codebooks_and_scales --update_non_quantized_parameters \
    --lamb --debias --lr 3e-4 --adam_beta1 0.9 --adam_beta2 0.95 \
    --code_lr 3e-3 --code_beta1 0.0 --code_beta2 0.95 --beam_size 1 --delta_decay 0 \
    --max_code_change_per_step 1e-2 --code_trust_ratio 1e-2 --code_selection_temperature 0 \
    --batch_size=256 --microbatch_size=8 --max_epochs 10 --gradient_checkpointing \
    --print_every_steps=1 --verbose_optimizer  --eval_every_steps=10 --keep_best_model --wandb \
    --save $SNAPSHOT_PATH --save_every_steps 100