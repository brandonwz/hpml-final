# hpml-final

$ cd eval
$ huggingface-cli download meta-llama/Llama-Guard-3-1B  --local-dir Llama-Guard-3-1B
$ python eval.py --model 1B-BF16 --path Llama-Guard-3-1B

Avg Time Per Sample (s): 0.5747905081089929
Binary Accuracy: 0.8341530592169978

### monitor gpu:
nvidia-smi

### py packaging:
1)zip -r hpml-final.zip  hpml-final -x hpml-final/BitDistiller-Fork/venv/\* -x hpml-final/eval/__pycache__/\* -x hpml-final/eval/Llama-Guard-3-1B/\* -x hpml-final/.idea/\* -x hpml-final/.git/\*

### scp and ssh with lambda labs:
0)select "Lauch instance", select the instance type, and create a new file system
1)download the ssh key to your local when you create the instance
2)scp -i <path to pem>/my-ssh-key.pem <path to py package>/hpml-final.zip ubuntu@<gpu ip>:/home/ubuntu
3)ssh -i <path to pem>/my-ssh-key.pem ubuntu@<gpu ip>

### setup on lambda labs gpu:
1)pip install -U "huggingface_hub[cli]"
2)huggingface-cli login
3)huggingface-cli download meta-llama/Llama-Guard-3-1B  --local-dir Llama-Guard-3-1B
4)unzip hpml-final.zip
5)cd hpml-final/BitDistiller-Fork/train
6)pip install -r ../requirement.txt
7)sh train.sh ../data/generation/datasets/llama-guard-3-1b/toxicchat_T0.7_N1024_S42_3000.json save log 1

