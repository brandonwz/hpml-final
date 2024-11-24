import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

from constants import DEVICE, OUTPUT_OFFSET, SAFE_STR, UNSAFE_STR, DBG, MAX_EVAL_ITERATIONS

from preprocessor import get_preprocessed_dummy_prompts_and_labels, get_preprocessed_toxic_chat_data
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

def get_prompts_and_labels(model_type, tokenizer, original_tokenizer, use_dummy_data):
    prompts = None
    labels = None
    if model_type == "1B-BF16":
        print("hi")
        if use_dummy_data:
            prompts, labels = get_preprocessed_dummy_prompts_and_labels(tokenizer)
        else:
            prompts, labels = get_preprocessed_toxic_chat_data(tokenizer)
    else:
        if use_dummy_data:
            prompts, labels = get_preprocessed_dummy_prompts_and_labels(original_tokenizer, tokenize=False)
        else:
            prompts, labels = get_preprocessed_toxic_chat_data(original_tokenizer, tokenize=False)

        # We use the original llama guard 3 1b tokenizer to get the prompts in text form and then encode
        # them with the tokenizer from the quantized model
        for i in range(len(prompts)):
            prompts[i] = tokenizer.encode(prompts[i], return_tensors="pt").to(DEVICE)
    return prompts, labels

# TODO: add some sort of evaluation metric,
# e.g. binary accuracy or f1
def eval_and_bench_model(model, tokenizer, prompts, labels):
    model.eval()
    total_time_s = 0.0
    iters = 0
    correct = 0
    for i in range(len(prompts)):
        input_ids = prompts[i]
        start = time.perf_counter()
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        end = time.perf_counter()

        total_time_s += end - start

        prompt_len = input_ids.shape[-1]
        output_decoded = tokenizer.decode(output[0][prompt_len:])
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

    print("Avg Time Per Sample (s):", total_time_s / len(prompts))
    print("Binary Accuracy:", correct / iters)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="model evaluation parser")
    parser.add_argument("--model", help="which model to use. options: 1B-BF16, 1B-INT2")
    parser.add_argument("--path", help="model path")
    parser.add_argument("--dummy", help="use dummy dataset", action="store_true")
    # Only relevant if --model is 1B-INT2
    parser.add_argument("--w_bit", type=int, default=None)
    parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
    parser.add_argument("--original_model", type=str, default=None, help="original model path (for the tokenizer)")

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.path, args.w_bit, args.load_quant, args.original_model)
    model = model.to(DEVICE)
    print(model)

    original_tokenizer = None
    if args.original_model:
        original_tokenizer = AutoTokenizer.from_pretrained(args.original_model)
    prompts, labels = get_prompts_and_labels(args.model, tokenizer, original_tokenizer, args.dummy)

    eval_and_bench_model(model, tokenizer, prompts, labels)