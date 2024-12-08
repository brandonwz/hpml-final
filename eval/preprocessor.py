from constants import DEVICE, CATEGORIES
import torch
import pandas as pd

#
# Returns a dummy dataset for local testing and debugging
#
def get_dummy_prompts():
    X = ["Hi! I like your shirt.",
        "High-performance machine learning is my favorite class",
        "That was pretty boring"]
    Y = [0,
        0,
        0]
    
    return X, Y

def get_toxic_chat_data():
    test_csv = pd.read_csv("./data/toxic-chat/test.csv")
    inputs = test_csv["user_input"]
    outputs = test_csv["toxicity"]

    return inputs.tolist(), outputs.tolist()

def get_mutox_data():
    test_csv = pd.read_csv("./data/mutox/mutox.tsv", sep='\t')
    test_csv_eng = test_csv[test_csv.lang == 'eng']  # only include english
    inputs = test_csv_eng["audio_file_transcript"]
    outputs = test_csv_eng["label"]

    return inputs.tolist(), outputs.tolist()

def get_ifeval_data():
    df = pd.read_json("./../instruction_following_eval/data/input_data.jsonl", lines=True)
    inputs = df["prompt"]

    return inputs.tolist()

def preprocess_prompts(tokenizer, prompts, tokenize=True):
    preprocessed_prompts = []
    for prompt in prompts:
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
        preprocessed_prompt = None
        if tokenize:
            preprocessed_prompt = tokenizer.apply_chat_template(
                                            wrapped_prompt,
                                            return_tensors="pt",
                                            categories = CATEGORIES
                                        ).to(DEVICE)
        else:
            preprocessed_prompt = tokenizer.apply_chat_template(
                                            wrapped_prompt,
                                            tokenize = False,
                                            categories = CATEGORIES
                                        )
        preprocessed_prompts.append(preprocessed_prompt)
    return preprocessed_prompts

def get_preprocessed_toxic_chat_data(tokenizer, tokenize=True):
    prompts, labels = get_toxic_chat_data()
    preprocessed_prompts = preprocess_prompts(tokenizer, prompts, tokenize)
    return preprocessed_prompts, labels

def get_preprocessed_mutox_data(tokenizer, tokenize=True):
    prompts, labels = get_mutox_data()
    preprocessed_prompts = preprocess_prompts(tokenizer, prompts, tokenize)
    return preprocessed_prompts, labels

def get_preprocessed_ifeval_data(tokenizer, tokenize=True):
    prompts = get_ifeval_data()
    for i in range(len(prompts)):
        prompts[i] = tokenizer.encode(prompts[i], return_tensors="pt").to(DEVICE)
    return prompts

def get_preprocessed_dummy_prompts_and_labels(tokenizer, tokenize=True):
    prompts, labels = get_dummy_prompts()
    preprocessed_prompts = preprocess_prompts(tokenizer, prompts, tokenize)
    return preprocessed_prompts, labels
