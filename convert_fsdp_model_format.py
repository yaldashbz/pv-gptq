import argparse
import os
from pathlib import Path

import torch
from auto_gptq import AutoGPTQForCausalLM


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="path or name of the teacher model",
    )
    parser.add_argument(
        "--quantized_model",
        type=str,
        required=True,
        help="path to quantized model",
    )
    parser.add_argument(
        "--load_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--code_dtype",
        type=str,
        default=None,
        help="if specified, cast quantized layers' codes to this dtype; default = keep loaded dtype",
    )
    parser.add_argument(
        "--pv_fsdp_dir",
        type=str,
        default=None,
        help="path to quantized model state dict saved by the old FSDP finetuning code",
    )
    parser.add_argument("--save", type=str, required=True, help="Save the converted quantized model here")

    args = parser.parse_args()
    device = torch.device("cuda:0")

    quantized_model = AutoGPTQForCausalLM.from_quantized(args.quantized_model, device=device)
    weights_dir = Path(os.path.join(args.pv_fsdp_dir, "best_model")).expanduser()
    attributes = ["scales"]

    for weight_file in weights_dir.glob("*.pth"):
        state_dict = torch.load(weight_file)
        if "non_quantized_state_dict" not in weight_file.stem:
            for attr in attributes:
                key_name = weight_file.stem.replace(".weight", f".{attr}")
                if key_name in quantized_model.state_dict():
                    quantized_model.state_dict()[key_name].copy_(getattr(state_dict, attr))
                else:
                    print(f"Key {key_name} not found in the model's state_dict")
        else:
            # Non quantized - for bias
            for name, tensor in state_dict.items():
                if name in quantized_model.state_dict():
                    quantized_model.state_dict()[name].copy_(tensor)
                else:
                    print(f"Key {name} not found in the model's state_dict")

    # Save the updated model's state_dict
    quantized_model.save_quantized(args.save)


if __name__ == "__main__":
    main()
