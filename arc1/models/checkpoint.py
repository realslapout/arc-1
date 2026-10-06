"""ARC-1 checkpoint loading.

A checkpoint folder holds:
    arc1_config.json   architecture, sequence format and calibration temperatures
    model.safetensors  weights (backbone + decision heads, LoRA already merged)
    tokenizer/         tokenizer files
    encoder/           backbone config, so the model is rebuilt without downloading the base model
"""
import json
import os


def load_tokenizer(path_or_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path_or_id)


def load_checkpoint(ckpt_dir: str, device: str = "cuda"):
    """(model, tokenizer, cfg) for an ARC-1 checkpoint folder."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel
    from arc1.models.branched import BranchedDecisionModel
    cfg = json.load(open(os.path.join(ckpt_dir, "arc1_config.json")))
    if cfg.get("arch") != "branched":
        raise ValueError("this release only loads ARC-1 'branched' checkpoints, got arch=%r" % cfg.get("arch"))
    bb = AutoModel.from_config(AutoConfig.from_pretrained(os.path.join(ckpt_dir, "encoder")), attn_implementation="sdpa")
    model = BranchedDecisionModel(bb, set_layers=cfg.get("set_layers", 1))
    model.load_state_dict(load_file(os.path.join(ckpt_dir, "model.safetensors")), strict=True)
    tok = load_tokenizer(os.path.join(ckpt_dir, "tokenizer"))
    return model.to(device).eval(), tok, cfg
