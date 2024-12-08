import argparse
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import tqdm

from constants import DEVICE, OUTPUT_OFFSET, SAFE_STR, UNSAFE_STR, DBG, MAX_EVAL_ITERATIONS

from preprocessor import get_preprocessed_dummy_prompts_and_labels, get_preprocessed_toxic_chat_data, get_preprocessed_mutox_data, get_preprocessed_ifeval_data
from test_quant import get_int2_model

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

def get_prompts_and_labels(model_type, tokenizer, original_tokenizer, dataset):
    prompts = None
    labels = None
    if model_type == "1B-BF16":
        if dataset == 'dummy':
            prompts, labels = get_preprocessed_dummy_prompts_and_labels(tokenizer)
        elif dataset == 'mutox':
            prompts, labels = get_preprocessed_mutox_data(tokenizer)
        elif dataset == 'ifeval':
            prompts = get_preprocessed_ifeval_data(tokenizer)
        else:
            prompts, labels = get_preprocessed_toxic_chat_data(tokenizer)
    else:
        if dataset == 'dummy':
            prompts, labels = get_preprocessed_dummy_prompts_and_labels(original_tokenizer, tokenize=False)
        elif dataset == 'mutox':
            prompts, labels = get_preprocessed_mutox_data(original_tokenizer, tokenize=False)
        elif dataset == 'ifeval':
            prompts = get_preprocessed_ifeval_data(original_tokenizer, tokenize=False)
        else:
            prompts, labels = get_preprocessed_toxic_chat_data(original_tokenizer, tokenize=False)

        # We use the original llama guard 3 1b tokenizer to get the prompts in text form and then encode
        # them with the tokenizer from the quantized model
        for i in range(len(prompts)):
            prompts[i] = tokenizer.encode(prompts[i], return_tensors="pt").to(DEVICE)

    return prompts, labels

def eval_and_bench_model(model, tokenizer, prompts, labels, lookup):
    model.eval()
    total_time_s = 0.0
    iters = 0
    correct = 0
    tokens = 0

    for i in tqdm.tqdm(range(100), "warmup"):
        input_ids = prompts[0]
        config = {'input_ids':input_ids, 'max_new_tokens':100, 'pad_token_id':0, 'use_cache' : True}
        if lookup:
            config = {'input_ids':input_ids, 'max_new_tokens':100, 'pad_token_id':0, 'use_cache' : True, 'do_sample':True, 'temperature':0.9, 'prompt_lookup_num_tokens':3}
        output = model.generate(**config)

    for i in tqdm.tqdm(range(len(prompts)), "eval"):
        input_ids = prompts[i]
        config = {'input_ids':input_ids, 'max_new_tokens':100, 'pad_token_id':0, 'use_cache' : True}
        if lookup:
            config = {'input_ids':input_ids, 'max_new_tokens':100, 'pad_token_id':0, 'use_cache' : True, 'do_sample':True, 'temperature':0.9, 'prompt_lookup_num_tokens':3}
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
        tokens += len(output_decoded)
        output_trimmed = output_decoded[OUTPUT_OFFSET:]
        #print(f"output_trimmed : {output_trimmed}")
        #print(f"labels[i]: {labels[i]}")

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

def generate_response(model, tokenizer, prompts, lookup, response_path):
    total_time_s = 0.0
    tokens = 0

    for i in tqdm.tqdm(range(100), "warmup"):
        input_ids = prompts[0]
        config = {'input_ids':input_ids, 'max_new_tokens': 100, 'pad_token_id':0, 'use_cache' : True}
        if lookup:
            config = {'input_ids':input_ids, 'max_new_tokens': 100, 'pad_token_id':0, 'use_cache' : True, 'do_sample':True, 'temperature':0.9, 'prompt_lookup_num_tokens':3}
        output = model.generate(**config)

    for i in tqdm.tqdm(range(len(prompts)), "eval"):
        input_ids = prompts[i]
        config = {'input_ids':input_ids, 'max_new_tokens': 100, 'pad_token_id':0, 'use_cache' : True}
        if lookup:
            config = {'input_ids':input_ids, 'max_new_tokens': 100, 'pad_token_id':0, 'use_cache' : True, 'do_sample':True, 'temperature':0.9, 'prompt_lookup_num_tokens':3}
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = model.generate(**config)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        total_time_s += end - start

        prompt_len = input_ids.shape[-1]

        input_decoded = tokenizer.decode(output[0][1:prompt_len])
        #print("input_decoded:" + input_decoded)

        output_decoded = tokenizer.decode(output[0][prompt_len:])
        tokens += len(output_decoded)
        #print("output_decoded:" + output_decoded)

        with open(response_path, 'a') as the_file:
            data = {"prompt": input_decoded, "response": output_decoded}
            the_file.write(f"{json.dumps(data)}\n")

    print("Avg Time Per Token (s):", total_time_s / tokens)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="model evaluation parser")
    parser.add_argument("--model", help="which model to use. options: 1B-BF16, 1B-INT2")
    parser.add_argument("--path", help="model path")
    parser.add_argument("--dataset", help="dataset options: dummy|toxic|mutox|ifeval")
    # Only relevant if --model is 1B-INT2
    parser.add_argument("--w_bit", type=int, default=None)
    parser.add_argument("--lookup", action="store_true")
    parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
    parser.add_argument("--original_model", type=str, default=None, help="original model path (for the tokenizer)")
    parser.add_argument("--response", help="response save location")

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.path, args.w_bit, args.load_quant, args.original_model)
    model = model.to(DEVICE)
    print(model)

    original_tokenizer = None
    if args.original_model:
        original_tokenizer = AutoTokenizer.from_pretrained(args.original_model)
    prompts, labels = get_prompts_and_labels(args.model, tokenizer, original_tokenizer, args.dataset)

    if not args.response:
        eval_and_bench_model(model, tokenizer, prompts, labels, args.lookup)
    else:
        print(f"generating and saving response at location: {args.response}...")
        generate_response(model, tokenizer, prompts, args.lookup, args.response)

