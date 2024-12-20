"""
Helper file that defines all the configs passed into generate
for each model type to help modularize the code
"""

def get_guard3_configs(input_ids):
    return {
                'input_ids':input_ids,
                'max_new_tokens':100,
                'pad_token_id':0,
                'use_cache' : True
    }

def get_guard3_lookup_configs(input_ids):
    return {
                'input_ids':input_ids,
                'max_new_tokens':100,
                'pad_token_id':0,
                'use_cache' : True,
                'do_sample':True,
                'temperature':0.9,
                'prompt_lookup_num_tokens':3
    }

def get_instruct_configs(input_ids):
    return {
                'input_ids':input_ids,
                'max_new_tokens': 100,
                'pad_token_id':0,
                'use_cache' : True
    }

def get_instruct_lookup_configs(input_ids):
    return {
                'input_ids':input_ids,
                'max_new_tokens': 100,
                'pad_token_id':0,
                'use_cache' : True,
                'do_sample':True,
                'temperature':0.9,
                'prompt_lookup_num_tokens':3
    }

def get_int2_instruct_configs(input_ids):
    return {
                'input_ids':input_ids,
                'max_new_tokens': 100,
                'pad_token_id':0,
                'use_cache' : True,
                'repetition_penalty':1.2,
                'do_sample':True,
                'temperature':0.9,
                'top_k':40
    }

def get_int2_instruct_lookup_configs(input_ids):
    return {
                'input_ids':input_ids,
                'max_new_tokens': 100,
                'pad_token_id':0,
                'use_cache' : True,
                'do_sample':True,
                'temperature':0.9,
                'prompt_lookup_num_tokens':3,
                'repetition_penalty':1.2,
                'top_k':40
    }