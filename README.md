# hpml-final

### eval Llama-Guard-3-1B model with mps:
$ cd eval
$ huggingface-cli download meta-llama/Llama-Guard-3-1B  --local-dir Llama-Guard-3-1B
$ python eval.py --model 1B-BF16 --path Llama-Guard-3-1B
Output result is as follows:
```
Avg Time Per Sample (s): 0.5747905081089929
Binary Accuracy: 0.8341530592169978
```

### monitor gpu:
nvidia-smi

### py packaging:
1) zip -r hpml-final.zip  hpml-final -x hpml-final/BitDistiller-Fork/venv/\* -x hpml-final/eval/__pycache__/\* -x hpml-final/eval/Llama-Guard-3-1B/\* -x hpml-final/.idea/\* -x hpml-final/.git/\*

### scp and ssh with lambda labs:
0) select "Lauch instance", select the instance type, and create a new file system <p>
1) download the ssh key to your local when you create the instance <p>
2) scp -i <path to pem>/my-ssh-key.pem <path to py package>/hpml-final.zip ubuntu@<gpu ip>:/home/ubuntu <p>
3) ssh -i <path to pem>/my-ssh-key.pem ubuntu@<gpu ip> <p>

### setup on lambda labs gpu:
0) python -m venv venv; . venv/bin/activate <p>
1) unzip hpml-final.zip <p>
2) cd hpml-final/BitDistiller-Fork/train <p>
3) pip install -r ../requirement.txt <p>
4) pip install -U "huggingface_hub[cli]" <p>
5) huggingface-cli login <p>
6) huggingface-cli download meta-llama/Llama-Guard-3-1B  --local-dir /home/ubuntu/Llama-Guard-3-1B <p>
7) sh train.sh ../data/generation/datasets/llama-guard-3-1b/toxicchat_T0.7_N1024_S42_3000.json save log 2 <p>

### To overcome the CUDA capability issue:
```
NVIDIA H100 80GB HBM3 with CUDA capability sm_90 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86.
```
Upgrade the torch version as:
```
pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/test/cu121
```

### The output result by using Lambda Labs GPU gpu_1x_h100_sxm5
```
result/output_llama-guard-3-1b
```

### Training with clip hf-llama3-2-1b

Modify train.sh with
```
--clip /home/ubuntu/hpml-final/BitDistiller-Fork/quantization/clip_cache/hf-llama3-2-1b/int2-g128.pt
```

With the following 
```
sh train.sh ../data/generation/datasets/hf-llama-3-2-1b/wikitext_T0.7_N1024_S42_3000.json save log 2
```

### The output result for clip hf-llama3-2-1b by using Lambda Labs GPU gpu_1x_h100_pcie
```
result/output_hf-llama-3-2-1b
```

### Evaluating Quantized Model Example

```
CUDA_VISIBLE_DEVICES=0 python eval.py --model 1B-INT2 --path <PATH_TO_UNQUANTIZED_MODEL_POST_QAT> --w_bit 2 --load_quant <PATH_TO_QUANTIZED_MODEL_CHECKPOINT> --original_model <PATH_TO_ORIGINAL_UNQUANTIZED_MODEL> --dummy --lookup
```
