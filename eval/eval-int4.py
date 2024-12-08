from executorch.extension.pybindings.portable_lib import _load_for_executorch
from executorch.extension.llm.custom_ops.sdpa_with_kv_cache import custom_ops_lib  # noqa
from executorch.kernels import quantized

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import tqdm

from constants import DEVICE, OUTPUT_OFFSET, SAFE_STR, UNSAFE_STR, DBG, MAX_EVAL_ITERATIONS

from preprocessor import get_preprocessed_dummy_prompts_and_labels, get_preprocessed_toxic_chat_data

UNSAFE_TOKEN_ID = 39257
SAFE_TOKEN_ID = 19193
END_TOKEN_ID = 128009

def load_model_and_tokenizer(model_path, original_model):
    model = _load_for_executorch(model_path) # model_file is the path to ET model
    tokenizer = AutoTokenizer.from_pretrained(original_model)

    return model, tokenizer

def get_prompts_and_labels(tokenizer, use_dummy_data):
    prompts = None
    labels = None
    if use_dummy_data:
        prompts, labels = get_preprocessed_dummy_prompts_and_labels(tokenizer, device=None)
    else:
        prompts, labels = get_preprocessed_toxic_chat_data(tokenizer, device=None)

    return prompts, labels

#Taken from https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/1B/ET_INSTRUCTIONS.md 
def generate(model, token_ids):
    pos = 0  # position in KV cache
    out_tokens = []  # contains the output tokens
    input = (
        token_ids,
        torch.tensor([pos], dtype=torch.long),
    )
    logits = model.forward(input)
    out_token = logits[0].argmax(-1).item()  # first_token
    out_tokens.append(out_token)

    max_seq_len = 100
    new_pos = len(token_ids) # New position in the KV cache (After the first n tokens)

    # Loop until the stop token is reached or the maximum sequence length is exceeded

    while out_token != END_TOKEN_ID and len(out_tokens) < max_seq_len:
        new_input = (
            torch.tensor([[out_token]], dtype=torch.long),
            torch.tensor([new_pos], dtype=torch.long),
        )

        new_logits = model.forward(new_input)
        out_token = new_logits[0].argmax(-1).item()
        out_tokens.append(out_token)
        new_pos += 1
    print(out_tokens)
    return out_tokens

def eval_and_bench_model(model, tokenizer, prompts, labels):
    total_time_s = 0.0
    iters = 0
    correct = 0
    tokens = 0

    total_time_s = 0.0
    iters = 0
    correct = 0
    tokens = 0

    for i in tqdm.tqdm(range(10), "warmup"):
        input_ids = prompts[0]
        output = generate(model, input_ids)

    for i in tqdm.tqdm(range(len(prompts)), "eval"):
        input_ids = prompts[i]

        start = time.perf_counter()
        output = generate(model, input_ids)
        end = time.perf_counter()

        total_time_s += end - start

        tokens += len(output)

        if output[0] == SAFE_TOKEN_ID:
            if DBG: print("SAFE")
            if labels[i] == 0:
                correct += 1
        elif output[0] == UNSAFE_TOKEN_ID:
            if DBG: print("UNSAFE")
            if labels[i] == 1:
                correct += 1

        iters += 1
        if DBG:            
            print(labels[i])
            if iters == MAX_EVAL_ITERATIONS:
                break

    print("Avg Time Per Token (s):", total_time_s / tokens)
    print("Binary Accuracy:", correct / iters)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="model evaluation parser")
    parser.add_argument("--model_path", help="model path")
    parser.add_argument("--dummy", help="use dummy dataset", action="store_true")
    parser.add_argument("--original_model", type=str, default=None, help="original model path (for the tokenizer)")

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.original_model)
    print(model)

    prompts, labels = get_prompts_and_labels(tokenizer, args.dummy)

    eval_and_bench_model(model, tokenizer, prompts, labels)
