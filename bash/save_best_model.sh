# Llama-3-8B
export NAME='Meta-Llama-3.1-8B-Instruct-gptq4-128-True-seed1_mse_static'
export SAVED_NAME=$NAME\_nf4_test_8192
export MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct  # path or huggingface id of the base model
export QUANTIZED_MODEL_PATH=/nfs/scistore19/alistgrp/amoeini/saved/$NAME # path to the model created by initial calibration
export SNAPSHOT_PATH=/nfs/scistore19/alistgrp/yshabanz/saved/$SAVED_NAME
export CONVERTED_CHECKPOINT_PATH=/nfs/scistore19/alistgrp/yshabanz/saved/$SAVED_NAME\_converted


python convert_fsdp_model_format.py\
    --base_model ./doesnt_matter \
    --pv_fsdp_dir $SNAPSHOT_PATH \
    --quantized_model $QUANTIZED_MODEL_PATH \
    --save $CONVERTED_CHECKPOINT_PATH
