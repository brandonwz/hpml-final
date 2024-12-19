# hpml-final


## Prerequisites

### Packages
1) All the requirements in requirements.txt in BitDistiller-Fork
2) `pip install wandb` to run the wandb upload script

### Models

1) Llama Guard 3 1B: `$ huggingface-cli download meta-llama/Llama-Guard-3-1B  --local-dir Llama-Guard-3-1B`
2) Llama Instruct 3.2 1B: `$ huggingface-cli download meta-llama/Llama-3.2-1B-Instruct  --local-dir Llama-Guard-3-1B`

## Training
For training, we used LambdaLabs to provision H100 80GB GPUs.

### Setup on LambdaLabs GPU:
0) python -m venv venv; . venv/bin/activate <p>
1) unzip hpml-final.zip <p>
2) cd hpml-final/BitDistiller-Fork/train <p>
3) pip install -r ../requirement.txt <p>
4) pip install -U "huggingface_hub[cli]" <p>
5) huggingface-cli login <p>
6) huggingface-cli download meta-llama/Llama-Guard-3-1B  --local-dir /home/ubuntu/Llama-Guard-3-1B <p>
7) export CLIP_PATH=../quantization/clip_cache/hf-llama3-1b/int2-g128.pt <p>
8) export MODEL_PATH=/home/ubuntu/Llama-Guard-3-1B <p>
9) sh train.sh ../data/generation/datasets/llama-guard-3-1b/toxicchat_T0.7_N1024_S42_3000.json save log 2 <p>

### Scp and Ssh with LambaLabs:
0) select "Lauch instance", select the instance type, and create a new file system <p>
1) download the ssh key to your local when you create the instance <p>
2) scp -i <path to pem>/my-ssh-key.pem <path to py package>/hpml-final.zip ubuntu@<gpu ip>:/home/ubuntu <p>
3) ssh -i <path to pem>/my-ssh-key.pem ubuntu@<gpu ip> <p>

### To Overcome the CUDA capability issue:
```
NVIDIA H100 80GB HBM3 with CUDA capability sm_90 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86.
```
Upgrade the torch version as:
```
pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/test/cu121
```

## Evaluation

### Monitoring GPU during Evaluation
`watch -n 1 nvidia-smi`

### Evaluating Llama Guard 3 1B on ToxicChat
`$ cd eval` <p>
Unquantized model: `$ python eval.py --model 1B-BF16 --path Llama-Guard-3-1B --dataset toxic`<p>
Quantized model: ```$ python eval.py --model 1B-INT2 --path <PATH_TO_UNQUANTIZED_MODEL_POST_QAT> --w_bit 2 --load_quant <PATH_TO_QUANTIZED_MODEL_CHECKPOINT> --original_model <PATH_TO_ORIGINAL_UNQUANTIZED_MODEL> --dataset toxic``` <p>
Quantized model with Prompt Lookup Decoding: ```$ python eval.py --model 1B-INT2 --path <PATH_TO_UNQUANTIZED_MODEL_POST_QAT> --w_bit 2 --load_quant <PATH_TO_QUANTIZED_MODEL_CHECKPOINT> --original_model <PATH_TO_ORIGINAL_UNQUANTIZED_MODEL> --dataset toxic --lookup``` <p>

### Evaluating Llama Instruct 3.2 1B on IFEval
`$ cd eval` <p>
Unquantized model: `$ python eval.py --model 1B-BF16 --path Llama-3.2-1B --dataset ifeval --response result/Llama-3.2-1B_ifeval/response.jsonl` <p>
Quantized model: `$ python eval.py --model 1B-INT2 --path <PATH_TO_UNQUANTIZED_MODEL_POST_QAT> --w_bit 2 --load_quant <PATH_TO_QUANTIZED_MODEL_CHECKPOINT>   --original_model <PATH_TO_ORIGINAL_UNQUANTIZED_MODEL> --dataset ifeval --response result/Llama-3.2-1B_ifeval/response.jsonl` <p>
Quantized model with Prompt Lookup Decoding: `$ python eval.py --model 1B-INT2 --path <PATH_TO_UNQUANTIZED_MODEL_POST_QAT> --w_bit 2 --load_quant <PATH_TO_QUANTIZED_MODEL_CHECKPOINT>   --original_model <PATH_TO_ORIGINAL_UNQUANTIZED_MODEL> --dataset ifeval --response result/Llama-3.2-1B_ifeval/response.jsonl --lookup` <p>
`$ cd ..` <p>
`$ python3 -m instruction_following_eval.evaluation_main --input_data=./instruction_following_eval/data/input_data.jsonl --input_response_data=./eval/result/Llama-3.2-1B_ifeval/response.jsonl --output_dir=./eval/result/Llama-3.2-1B_ifeval/`

## Results

Output result is located at and as follows:
```
Avg Time Per Token (s): 0.03453764048901507
Binary Accuracy: 0.8373008066102695
```
## Interpretation of Results

## Sources



### py packaging:
1) zip -r hpml-final.zip hpml-final -x hpml-final/instruction_following_eval/venv/\* -x hpml-final/eval/__pycache__/\* -x hpml-final/eval/Llama-3.2-1B-Instruct/\* -x hpml-final/eval/Llama-3.2-1B/\* -x hpml-final/eval/Llama-Guard-3-1B/\* -x hpml-final/.idea/\* -x hpml-final/.git/\* -x hpml-final/result/\*





### The output result by using Lambda Labs GPU gpu_1x_h100_sxm5
```
result/output_llama-guard-3-1b
```

### Training with clip hf-llama3-2-1b and model Llama-3.2-1B
0) huggingface-cli download meta-llama/Llama-3.2-1B --local-dir /home/ubuntu/Llama-3.2-1B <p>
1) export CLIP_PATH=/home/ubuntu/hpml-final/BitDistiller-Fork/quantization/clip_cache/hf-llama3-2-1b/llama-3-2-int2-g128.pt <p>
2) export MODEL_PATH=/home/ubuntu/Llama-3.2-1B <p>
3) sh train.sh ../data/generation/datasets/hf-llama-3-2-1b/wikitext_T0.7_N1024_S42_3000.json save log 2 <p>

### The output result for clip hf-llama3-2-1b and model Llama-3-2-1B by using Lambda Labs GPU gpu_1x_h100_pcie
```
result/output_llama-3-2-1b
```

### Evaluating Quantized Model Example

```
CUDA_VISIBLE_DEVICES=0 python eval.py --model 1B-INT2 --path <PATH_TO_UNQUANTIZED_MODEL_POST_QAT> --w_bit 2 --load_quant <PATH_TO_QUANTIZED_MODEL_CHECKPOINT> --original_model <PATH_TO_ORIGINAL_UNQUANTIZED_MODEL> --dummy --lookup
```

### eval ifeval data with Llama-3.2-1B-Instruct model and mps:
0) $ cd eval <p>
1) $ huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir Llama-3.2-1B-Instruct <p>
2) $ python eval.py --model 1B-BF16 --path Llama-3.2-1B-Instruct --dataset ifeval --response result/Llama-3.2-1B-Instruct_ifeval/response.jsonl --instruct <p>
3) $ cd .. <p>
4) $ python3 -m instruction_following_eval.evaluation_main --input_data=./instruction_following_eval/data/input_data.jsonl --input_response_data=./eval/result/Llama-3.2-1B-Instruct_ifeval/response.jsonl --output_dir=./eval/result/Llama-3.2-1B-Instruct_ifeval/

Output result is located at and as follows:
```
eval/result/Llama-3.2-1B-Instruct_ifeval

Avg Time Per Token (s): 0.012948842507713415
Accuracy scores of keywords: 0.5214723926380368 
```

### eval ifeval data with Llama-3.2-1B model and mps:
0) $ cd eval <p>
1) $ huggingface-cli download meta-llama/Llama-3.2-1B --local-dir Llama-3.2-1B <p>
2) $ python eval.py --model 1B-BF16 --path Llama-3.2-1B --dataset ifeval --response result/Llama-3.2-1B_ifeval/response.jsonl <p>
3) $ cd .. <p>
4) $ python3 -m instruction_following_eval.evaluation_main --input_data=./instruction_following_eval/data/input_data.jsonl --input_response_data=./eval/result/Llama-3.2-1B_ifeval/response.jsonl --output_dir=./eval/result/Llama-3.2-1B_ifeval/

Output result is located at and as follows:
```
eval/result/Llama-3.2-1B_ifeval

Avg Time Per Token (s): 0.011015437406368827
Accuracy scores of keywords: 0.4539877300613497
```


