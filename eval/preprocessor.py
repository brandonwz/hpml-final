from constants import DEVICE, CATEGORIES
import pandas as pd

"""
Returns a dummy dataset for local testing and debugging
"""
def get_dummy_prompts():
    X = ["Hi! I like your shirt.",
        "High-performance machine learning is my favorite class",
        "That was pretty boring"]
    Y = [0,
        0,
        0]
    
    return X, Y

"""
Gets toxic chat data and reformats them as a python list
"""
def get_toxic_chat_data():
    test_csv = pd.read_csv("./data/toxic-chat/test.csv")
    inputs = test_csv["user_input"]
    outputs = test_csv["toxicity"]

    return inputs.tolist(), outputs.tolist()

"""
Gets ifeval dataset and reformats the prompts as a list
"""
def get_ifeval_data():
    df = pd.read_json("./../instruction_following_eval/data/input_data.jsonl", lines=True)
    inputs = df["prompt"]

    return inputs.tolist()

"""
Preprocessor for Llama Guard 3 1B data
"""
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


"""
Preprocessor for Llama Guard 3.2 1B Instruct data
"""
def get_preprocessed_ifeval_data(tokenizer, instruct, tokenize=True):
    prompts = get_ifeval_data()

    preprocessed_prompts = []

    for i in range(len(prompts)):
        if not instruct:
            if tokenize:
                preprocessed_prompts.append(tokenizer.encode(prompts[i], return_tensors="pt").to(DEVICE))
            else:
                # If there's no need to apply a template or anything specific to the original tokenizer, don't need to do anything here
                preprocessed_prompts.append(prompts[i])
        else:
            wrapped_prompt = [
                {
                    "role": "user",
                    "content": prompts[i]
                }
            ]
            if tokenize:
                preprocessed_prompt = tokenizer.apply_chat_template(
                                                wrapped_prompt,
                                                return_tensors="pt"
                                            ).to(DEVICE)
            else:
                preprocessed_prompt = tokenizer.apply_chat_template(
                                                wrapped_prompt,
                                                tokenize = False
                                            )
            preprocessed_prompts.append(preprocessed_prompt)
    return preprocessed_prompts, prompts

def get_preprocessed_dummy_prompts_and_labels(tokenizer, tokenize=True):
    prompts, labels = get_dummy_prompts()
    preprocessed_prompts = preprocess_prompts(tokenizer, prompts, tokenize)
    return preprocessed_prompts, labels
