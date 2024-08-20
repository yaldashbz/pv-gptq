TARGET_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct  # used for tokenization
SEQLEN=8192
# TARGET_MODEL=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T  # used for tokenization
# SEQLEN=2048
DATASET=togethercomputer/RedPajama-Data-1T-Sample
OUTPUT_PATH=/nfs/scistore19/alistgrp/yshabanz/data/pajama_tokenized_llama3.1-8b-instruct_$SEQLEN
CACHE_DIR=/nfs/scistore19/alistgrp/yshabanz/cache_dir

export HUGGINGFACE_TOKEN='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'

# --dtype -> --load_dtype

CUDA_VISIBLE_DEVICES=1 HF_HOME='../../hf' OMP_NUM_THREADS=16 torchrun --master-port 3456 --nproc-per-node=1 finetune_fsdp.py \
    --dataset_config_name=$SUBSET --base_model $TARGET_MODEL --quantized_model ./doesnt_matter \
    --load_dtype bfloat16 --block_type LlamaDecoderLayer --dataset_name=$DATASET --split train \
    --cache_dir=$CACHE_DIR --trust_remote_code --model_seqlen=$SEQLEN --preprocessing_num_workers=64 \
    --save_dataset_and_exit $OUTPUT_PATH \
    # --preprocessing_chunk_length 100000

# tar -cvf tokenized_data_tinyllama.tar $OUTPUT_PATH   # optionally pack for distribution