"""
Adapted from https://github.com/mit-han-lab/llm-awq/blob/main/awq/entry.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, modeling_utils
import sys
sys.path.append("../BitDistiller-Fork")
from quantization.quantizer import real_quantize_model_weight
from inference.modules.fused_mlp import make_fused_mlp

import accelerate
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)

import os

q_config = {
    "zero_point": True,  # by default True
    "q_group_size": 128,  # whether to use group quantization
}

def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module

def simple_dispatch_model(model, device_map):
    from accelerate.hooks import add_hook_to_module, AlignDevicesHook

    if "" in device_map:
        d = device_map[""]
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model

    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {
        "cpu",
        "disk",
    }:
        main_device = "cpu"
    else:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    cpu_offload_group = [(n, d) for n, d in device_map.items() if d == "cpu"]
    prev_hook = None
    for idx, (n, d) in enumerate(cpu_offload_group):
        m = get_module_by_name_suffix(model, n)
        _, prev_hook = accelerate.cpu_offload_with_hook(
            m, execution_device=main_device, prev_module_hook=prev_hook
        )
    # set first cpu offload module's prev_module_hook to the last cpu offload module's hook
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(
            model, cpu_offload_group[0][0]
        )._hf_hook.prev_module_hook = prev_hook

    for n, d in device_map.items():
        m = get_module_by_name_suffix(model, n)
        if d != "cpu":
            d = torch.device(d)
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=True)
            add_hook_to_module(m, hook)
    accelerate.utils.modeling.retie_parameters(model, tied_params)
    model.hf_device_map = device_map

    return model

def get_int2_model(model_path, w_bit, load_quant):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code = True)
    config.use_cache = False
    print("Loading pre-computed quantized weights...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config, torch_dtype=torch.float16, trust_remote_code=True
        )
    real_quantize_model_weight(
        model, w_bit=w_bit, q_config=q_config, init_only=True
    )

    model.tie_weights()

    # Infer device map
    kwargs =  {}
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=[
            "OPTDecoderLayer",
            "LlamaDecoderLayer",
            "BloomBlock",
            "MPTBlock",
            "DecoderLayer",
        ],
        **kwargs,
    )
    # Load checkpoint in the model
    load_checkpoint_in_model(
        model,
        checkpoint=load_quant,
        device_map=device_map,
        offload_state_dict=True,
    )
    # Dispatch model
    model = simple_dispatch_model(model, device_map=device_map)
    model = make_fused_mlp(model)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    return model, tokenizer