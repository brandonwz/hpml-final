#WIP -- decoding step adapted from https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/1B/ET_INSTRUCTIONS.md 

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

def load_model_and_tokenizer(model_path, original_model):
    model = _load_for_executorch(model_path) # model_file is the path to ET model
    tokenizer = AutoTokenizer.from_pretrained(original_model)

    return model, tokenizer

def get_prompts_and_labels(tokenizer, use_dummy_data):
    prompts = None
    labels = None
    if use_dummy_data:
        prompts, labels = get_preprocessed_dummy_prompts_and_labels(tokenizer)
    else:
        prompts, labels = get_preprocessed_toxic_chat_data(tokenizer)

    return prompts, labels

def eval_and_bench_model(model, tokenizer, prompts, labels):
    #model.eval()
    total_time_s = 0.0
    iters = 0
    correct = 0
    tokens = 0

    total_time_s = 0.0
    iters = 0
    correct = 0
    tokens = 0

    prompt = "you are very cool"
    wrapped_prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                    ]
            }
        ]
    token_ids = tokenizer.apply_chat_template(wrapped_prompt, return_tensors="pt")
    pos = 0  # position in KV cache
    out_tokens = []  # contains the output tokens
    input = (
    token_ids,
    torch.tensor([pos], dtype=torch.long),
    )
    print("hello")
    logits = model.forward(input)
    out_token = logits[0].argmax(-1).item()  # first_token
    print(out_token)  # 19193 is safe, 39257 is unsafe
    out_tokens.append(out_token)

    # for i in tqdm.tqdm(range(100), "warmup"):
    #     input_ids = prompts[0]
    #     config = {'input_ids':input_ids, 'max_new_tokens':100, 'pad_token_id':0, 'use_cache' : True}
    #     if lookup:
    #         config = {'input_ids':input_ids, 'max_new_tokens':100, 'pad_token_id':0, 'use_cache' : True, 'do_sample':True, 'temperature':0.9, 'prompt_lookup_num_tokens':3}
    #     output = model.generate(**config)

    # for i in tqdm.tqdm(range(len(prompts)), "eval"):
    #     input_ids = prompts[i]
    #     config = {'input_ids':input_ids, 'max_new_tokens':100, 'pad_token_id':0, 'use_cache' : True}
    #     if lookup:
    #         config = {'input_ids':input_ids, 'max_new_tokens':100, 'pad_token_id':0, 'use_cache' : True, 'do_sample':True, 'temperature':0.9, 'prompt_lookup_num_tokens':3}
    #     if DEVICE == "cuda":
    #         torch.cuda.synchronize()
    #     start = time.perf_counter()
    #     output = model.generate(**config)

    #     if DEVICE == "cuda":
    #         torch.cuda.synchronize()
    #     end = time.perf_counter()

    #     total_time_s += end - start

    #     prompt_len = input_ids.shape[-1]
    #     output_decoded = tokenizer.decode(output[0][prompt_len:])
    #     tokens += len(output_decoded)
    #     output_trimmed = output_decoded[OUTPUT_OFFSET:]

    #     if output_trimmed[0:len(SAFE_STR)] == SAFE_STR:
    #         if DBG: print("SAFE")
    #         if labels[i] == 0:
    #             correct += 1
    #     elif output_trimmed[0:len(UNSAFE_STR)] == UNSAFE_STR:
    #         if DBG: print("UNSAFE")
    #         if labels[i] == 1:
    #             correct += 1

    #     iters += 1
    #     if DBG:            
    #         print(labels[i])
    #         if iters == MAX_EVAL_ITERATIONS:
    #             break

    # print("Avg Time Per Token (s):", total_time_s / tokens)
    # print("Binary Accuracy:", correct / iters)

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
