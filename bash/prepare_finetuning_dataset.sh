TARGET_MODEL=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T  # used for tokenization
# TARGET_MODEL=facebook/opt-125m
SEQLEN=2048
DATASET=togethercomputer/RedPajama-Data-1T-Sample
# DATASET_CONFIG=wikitext-2-raw-v1
# DATASET=wikitext
OUTPUT_PATH=../PV-GPTQ/pajama_tokenized_tinyllama-v2

export HUGGINGFACE_TOKEN='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'

# --dtype -> --load_dtype

CUDA_VISIBLE_DEVICES=0 HF_HOME=../PV-GPTQ/hf OMP_NUM_THREADS=16 torchrun --master-port 3456 --nproc-per-node=1 finetune_fsdp.py --dataset_config_name=$DATASET_CONFIG --base_model $TARGET_MODEL --quantized_model ./doesnt_matter --load_dtype bfloat16 --block_type LlamaDecoderLayer --dataset_name=$DATASET --split train --cache_dir=./cache_dir --trust_remote_code --model_seqlen=$SEQLEN --preprocessing_num_workers=64 --preprocessing_chunk_length 100000 --save_dataset_and_exit $OUTPUT_PATH

tar -cvf tokenized_data_qwen2.tar $OUTPUT_PATH   # optionally pack for distribution