import time
from argparse import Namespace

import torch
from accelerate.utils import get_max_memory
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

from src.datautils import get_loaders, evaluate_perplexity

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False


@torch.no_grad()
def get_quantized_model(args: Namespace):
    quantization_config = BaseQuantizeConfig(
        bits=args.bits, group_size=args.group_size, true_sequential=args.true_sequential
    )
    quantized_model = AutoGPTQForCausalLM.from_pretrained(
        args.model_name_or_path, quantize_config=quantization_config, max_memory=get_max_memory()
    )
    # Prepare train data
    train_data = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.model_seqlen or quantized_model.config.max_position_embeddings,
        eval_mode=False,
        model_path=args.model_name_or_path,
        use_fast_tokenizer=args.use_fast_tokenizer,
    )
    train_examples = [{"input_ids": example, "attention_mask": torch.ones_like(example)} for example in train_data]

    print("Starting GPTQ Quantization ...")
    start_tick = time.perf_counter()
    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    quantized_model.quantize(train_examples)
    end_tick = time.perf_counter()
    if args.wandb:
        wandb.log({"max_cuda_mem_quantize": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
    print(f"Quantize: {torch.cuda.max_memory_allocated()=:,}")
    print(f"Quantization time: {end_tick - start_tick:.1f}")
    return quantized_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--new_eval",
        action="store_true",
        help="if this is set, evaluate on new (and slightly more realistic!) val dataset versions",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="Number of calibration data samples.If None take all calibration data.",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=None,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    # Quantization parameters
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="each codebook will contain 2 ** nbits_per_codebook vectors",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Quantization group size",
    )
    parser.add_argument(
        "--true_sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--no_quant",
        action="store_true",
        help="Load model without quantization.",
    )
    # Eval arguments
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["wikitext2", "c4"],
        help="Datasets to run evaluation on",
    )
    # Save arguments
    parser.add_argument("--save", type=str, default=None, help="Path to save quantized model.")
    # Misc arguments
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "eager", "flash_attention_2", "sdpa"],
        help="Attention implementation.",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer (some models have only fast tokenizer).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )

    torch.set_num_threads(min(16, torch.get_num_threads()))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    args = parser.parse_args()

    if args.wandb:
        wandb.init(config=args)

    # Get original or quantized model
    if args.no_quant:
        print("\n============ Loading model... ============")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            torch_dtype=args.dtype,
            low_cpu_mem_usage=True,
            attn_implementation=args.attn_implementation,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        print("\n============ Quantizing model... ============")
        model = get_quantized_model(args)
    # Turn off cache_usage
    model.config.use_cache = False
    print("\n============ Evaluating perplexity... ============")
    torch.cuda.reset_peak_memory_stats()
    eval_seqlen = args.model_seqlen or model.config.max_position_embeddings
    for dataset_name in args.eval_datasets:
        test_loaders = get_loaders(
            dataset_name,
            seed=args.seed,
            model_path=args.model_name_or_path,
            seqlen=eval_seqlen,
            eval_mode=True,
            use_fast_tokenizer=args.use_fast_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
        ppl = evaluate_perplexity(model, test_loaders, seqlen=eval_seqlen, device=device)
        print(f"Perplexity on {dataset_name} {ppl:.3f}")
        if args.wandb:
            wandb.log({dataset_name: ppl})
    # Save model after evaluation
    if args.save:
        model.save_quantized(args.save)
        # Save tokenizer as well
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=args.use_fast_tokenizer, trust_remote_code=args.trust_remote_code
        )
        tokenizer.save_pretrained(args.save)
