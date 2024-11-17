# hpml-final

$ cd eval
$ huggingface-cli download meta-llama/Llama-Guard-3-1B  --local-dir Llama-Guard-3-1B
$ python eval.py --model 1B-BF16 --path Llama-Guard-3-1B

Avg Time Per Sample (s): 0.5747905081089929
Binary Accuracy: 0.8341530592169978
