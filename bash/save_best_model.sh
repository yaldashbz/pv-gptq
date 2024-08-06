# Llama-3-8B
export NAME=Meta-Llama-3-8B-Instruct-gptq4-128-True-seed0_cosmopedia_static
export MODEL_PATH=/nfs/scistore19/alistgrp/huggingface/hub/Meta-Llama-3-8B-Instruct  # path or huggingface id of the base model
export QUANTIZED_MODEL_PATH=/nfs/scistore19/alistgrp/amoeini/saved/$NAME # path to the model created by initial calibration
export SNAPSHOT_PATH=/nfs/scistore19/alistgrp/yshabanz/saved/pv_model_llama_cosmopedia_static
export CONVERTED_CHECKPOINT_PATH=/nfs/scistore19/alistgrp/yshabanz/saved/$NAME\_p-step

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TinyLlama
# export MODEL_PATH=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T  # path or huggingface id of the base model
# export QUANTIZED_MODEL_PATH=/nfs/scistore19/alistgrp/yshabanz/codes/Tiny-llama-4bit-v2 # path to the model created by initial calibration
# export TOKENIZED_DATASET_PATH=/nfs/scistore19/alistgrp/yshabanz/codes/pajama_tokenized_tinyllama-v2  # yet again, red pajama adviced
# export CACHE_DIR=/nfs/scistore19/alistgrp/yshabanz/cache_dir
# export SNAPSHOT_PATH=/nfs/scistore19/alistgrp/yshabanz/saved/pv_model_tinyllama_test
# export SEQLEN=2048
# export NUM_GPUS=4


python convert_fsdp_model_format.py\
    --base_model ./doesnt_matter \
    --pv_fsdp_dir $SNAPSHOT_PATH \
    --quantized_model $QUANTIZED_MODEL_PATH \
    --save $CONVERTED_CHECKPOINT_PATH
