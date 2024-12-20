"""
Main evaluation engine for the project. Contains benchmarking code for quantized and unquantized models
for the Llama Guard 3 1B and Llama 3.2 Instruct 1B variants, across ToxicChat and IFEval respectively.
"""

import argparse
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import tqdm

from constants import DEVICE, OUTPUT_OFFSET, INSTRUCT_OFFSET, SAFE_STR, UNSAFE_STR, DBG, MAX_EVAL_ITERATIONS, EOT_ID, PROFILE_EVAL_LEN
from generate_configs import get_guard3_configs, get_guard3_lookup_configs, get_instruct_configs, get_instruct_lookup_configs, get_int2_instruct_configs, get_int2_instruct_lookup_configs

from preprocessor import get_preprocessed_dummy_prompts_and_labels, get_preprocessed_toxic_chat_data, get_preprocessed_ifeval_data
from test_quant import get_int2_model

"""
Helper function to load the appropriate model and its tokenizer
"""
def load_model_and_tokenizer(model_type, path, w_bit, load_quant, original_model):
    model = None
    tokenizer = None
    if model_type == "1B-BF16":
        if w_bit != None or load_quant != None or original_model != None:
            print("Warning: either w_bit, load_quant, or original_model was configured for an unsupported model type. Ignoring those settings.")
        model = AutoModelForCausalLM.from_pretrained(
                    path,
                    load_in_4bit=False,
                    load_in_8bit=False,
                    torch_dtype=torch.bfloat16,
                    device_map=DEVICE
                )
        tokenizer = AutoTokenizer.from_pretrained(path)
    elif model_type == "1B-INT2":
        model, tokenizer = get_int2_model(path, w_bit, load_quant)
    else:
        raise Exception("Model type unsupported, exiting.")

    return model, tokenizer

"""
Helper function to get prompts and labels in preparation to pass into model.generate
"""
def get_prompts_and_labels(model_type, tokenizer, original_tokenizer, dataset, instruct):
    prompts = None
    labels = None
    if model_type == "1B-BF16":
        if dataset == 'dummy':
            prompts, labels = get_preprocessed_dummy_prompts_and_labels(tokenizer)
        elif dataset == 'ifeval':
            prompts, labels = get_preprocessed_ifeval_data(tokenizer, instruct)
        else:
            prompts, labels = get_preprocessed_toxic_chat_data(tokenizer)
    else:
        # We use the original model tokenizer to get the prompts in text form with the chat template and then encode
        # them with the tokenizer from the quantized model
        if dataset == 'dummy':
            prompts, labels = get_preprocessed_dummy_prompts_and_labels(original_tokenizer, tokenize=False)
        elif dataset == 'ifeval':
            prompts, labels = get_preprocessed_ifeval_data(original_tokenizer, instruct, tokenize=False)
        else:
            prompts, labels = get_preprocessed_toxic_chat_data(original_tokenizer, tokenize=False)

        for i in range(len(prompts)):
            prompts[i] = tokenizer.encode(prompts[i], return_tensors="pt").to(DEVICE)

    return prompts, labels

"""
Gets inference times and Llama Guard 3 1B binary accuracy scores. Intended for use with
the ToxicChat and dummy datasets.
"""
def eval_and_bench_model(model, tokenizer, prompts, labels, lookup, profile=False):
    model.eval()
    total_time_s = 0.0
    iters = 0
    correct = 0
    tokens = 0

    # Warmup phase, INT2 models use triton autotune so let it run for a bit before benchmarking
    for i in tqdm.tqdm(range(100), "warmup"):
        input_ids = prompts[0]
        config = get_guard3_configs(input_ids)
        if lookup:
            config = get_guard3_lookup_configs(input_ids)
        output = model.generate(**config)

    eval_len = len(prompts)

    # The profiler can run out of memory if left for too long, so we cap it to a specific number of iterations if enabled
    if profile:
        eval_len = min(eval_len, PROFILE_EVAL_LEN)
    for i in tqdm.tqdm(range(eval_len), "eval"):
        input_ids = prompts[i]
        config = get_guard3_configs(input_ids)
        if lookup:
            config = get_guard3_lookup_configs(input_ids)

        # Synchronize cuda kernels before starting and ending the counter for accurate measurements
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        output = model.generate(**config)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        total_time_s += end - start

        prompt_len = input_ids.shape[-1]
        output_decoded = tokenizer.decode(output[0][prompt_len:])
        tokens += len(output[0][prompt_len:])
        output_trimmed = output_decoded[OUTPUT_OFFSET:]

        if output_trimmed[0:len(SAFE_STR)] == SAFE_STR:
            if DBG: print("SAFE")
            if labels[i] == 0:
                correct += 1
        elif output_trimmed[0:len(UNSAFE_STR)] == UNSAFE_STR:
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

"""
Gets the inference speed and jsonl file for Llama 3.2 Instruct 1B, intended for use with IFEval benchmarking code.
Not intended for use with ToxicChat/dummy datasets. The jsonl file is saved to the filepath provided in response_path.
"""
def generate_response(model, tokenizer, prompts, original_prompts, lookup, response_path, instruct, model_type):
    total_time_s = 0.0
    tokens = 0

    final_prompts = []
    final_decoded = []

    # Warmup phase, INT2 models use triton autotune so let it run for a bit before benchmarking
    for i in tqdm.tqdm(range(100), "warmup"):
        input_ids = prompts[0]
        config = get_instruct_configs(input_ids)
        if lookup:
            config = get_instruct_lookup_configs(input_ids)
        output = model.generate(**config)

    for i in tqdm.tqdm(range(len(prompts)), "eval"):
        input_ids = prompts[i]
        config = get_int2_instruct_configs(input_ids)
        if lookup:
            config = get_int2_instruct_lookup_configs(input_ids)
        
        # The repetition penalty, sampling, and top_k parameters were for experimentation with the INT2 quantized versions, so don't use those for the unquantized version
        if model_type == "1B-BF16":
            config = get_instruct_configs(input_ids)
            if lookup:
                config = get_instruct_lookup_configs(input_ids)

        # Synchronize cuda kernels before starting and ending the counter for accurate measurements
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
    
        output = model.generate(**config)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        total_time_s += end - start

        # Extract the output and decode it from id to text
        prompt_len = input_ids.shape[-1]
        tokens += len(output[0][prompt_len:])

        # Format properly for IFEval benchmarking
        eot_id = EOT_ID

        if instruct:
            output_start_idx = prompt_len + INSTRUCT_OFFSET
        else:
            output_start_idx = prompt_len
        output_decoded = tokenizer.decode(output[0][output_start_idx:])
        if output_decoded[len(output_decoded)-len(eot_id):len(output_decoded)] == eot_id:
            output_decoded = output_decoded[:len(output_decoded) - len(eot_id)]

        final_prompts.append(original_prompts[i])
        final_decoded.append(output_decoded)

    # Write it into a jsonl at the end in the correct format for IFEval. The IFEval score is not generated
    # here, the jsonl is given back to the user and then passed into the IFEval evaluation framework. However,
    # we still report inference speed scores.
    with open(response_path, 'a') as the_file:
        for i in range(len(final_prompts)):
            data = {"prompt": final_prompts[i], "response": final_decoded[i]}
            the_file.write(f"{json.dumps(data)}\n")

    print("Avg Time Per Token (s):", total_time_s / tokens)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="model evaluation parser")
    parser.add_argument("--model", help="which model to use. options: 1B-BF16, 1B-INT2")
    parser.add_argument("--path", help="model path")
    parser.add_argument("--dataset", help="dataset options: dummy|toxic|ifeval")

    # Only relevant if --model is 1B-INT2
    parser.add_argument("--w_bit", type=int, default=None)
    parser.add_argument("--lookup", action="store_true")
    parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
    parser.add_argument("--original_model", type=str, default=None, help="original model path (for the tokenizer)")

    # Used for Instruct 3.2 1B to generate IFEval jsonl for evaluation
    parser.add_argument("--response", help="response save location")
    parser.add_argument("--instruct", help="to indicate if the model is an instruct or not", action="store_true")

    parser.add_argument("--profile", help="whether to enable pytorch profiling", action="store_true")

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.path, args.w_bit, args.load_quant, args.original_model)
    model = model.to(DEVICE)
    print(model)

    original_tokenizer = None
    # For the quantized model, we want to use the original model's tokenizer for the chat template. But the actual
    # tokenization will be done using the quantized model's tokenizer.
    if args.original_model:
        original_tokenizer = AutoTokenizer.from_pretrained(args.original_model)
    prompts, labels = get_prompts_and_labels(args.model, tokenizer, original_tokenizer, args.dataset, args.instruct)

    if (DEVICE == "cuda" or DEVICE == "cpu") and args.profile:
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        ) as p:
            with torch.profiler.record_function("model_inference"):
                if not args.response:
                    eval_and_bench_model(model, tokenizer, prompts, labels, args.lookup, profile=True)
                else:
                    print("Profiler not supported for this eval type.")
                    exit(1)

        print(p.key_averages().table(sort_by=DEVICE + "_time_total"))

    else:
        if not args.response:
            eval_and_bench_model(model, tokenizer, prompts, labels, args.lookup)
        else:
            print(f"generating and saving response at location: {args.response}...")
            generate_response(model, tokenizer, prompts, labels, args.lookup, args.response, args.instruct,
                                args.model)


