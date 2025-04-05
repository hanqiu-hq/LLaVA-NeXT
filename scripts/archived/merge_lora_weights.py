import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):
    if "onevision" in args.model_base or "ov" in args.model_base:
        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = 2
        overwrite_config["mm_spatial_pool_mode"] = "bilinear"

        # Model
        llava_model_args = {
            "multimodal": True,
            "overwrite_config": overwrite_config,
            # "attn_implementation": best_fit_attn_implementation,
        }
    else:
        llava_model_args = {}

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map="cpu", **llava_model_args)
    model._hf_peft_config_loaded = False
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)
