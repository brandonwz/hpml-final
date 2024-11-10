import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

from constants import DEVICE, OUTPUT_OFFSET, SAFE_STR, UNSAFE_STR, DBG, MAX_EVAL_ITERATIONS

from preprocessor import get_preprocessed_dummy_prompts_and_labels, get_preprocessed_toxic_chat_data

def load_model_and_tokenizer(model_type, path):
    model = None
    tokenizer = None
    if model_type == "1B-BF16":
        model = AutoModelForCausalLM.from_pretrained(
                    path,
                    load_in_4bit=False,
                    load_in_8bit=False,
                    torch_dtype=torch.bfloat16,
                    device_map=DEVICE
                )
        tokenizer = AutoTokenizer.from_pretrained(path)
    else:
        raise Exception("Model type unsupported, exiting.")

    return model, tokenizer

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
        output = model.generate(input_ids=input_ids, max_new_tokens=20, pad_token_id=0)
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
    parser.add_argument("--model", help="which model to use. options: 1B-BF16")
    parser.add_argument("--path", help="model path")
    parser.add_argument("--dummy", help="use dummy dataset", action="store_true")

    train_args = parser.parse_args()

    model_type = train_args.model
    path = train_args.path
    use_dummy_data = train_args.dummy

    model, tokenizer = load_model_and_tokenizer(model_type, path)

    prompts = None
    labels = None
    if use_dummy_data:
        prompts, labels = get_preprocessed_dummy_prompts_and_labels(tokenizer)
    else:
        prompts, labels = get_preprocessed_toxic_chat_data(tokenizer)

    eval_and_bench_model(model, tokenizer, prompts, labels)