export MODEL_PATH='/home/brandon/hpml/hpmlfinal/Llama-Guard-3-1B/' #CHANGE TO YOUR PATH
export SAVE_PATH=$2
export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true  

deepspeed --num_gpus=1 train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $1 \
    --model_max_length 131072 \
    --output_dir $SAVE_PATH \
    --logging_dir $3 \
    --num_train_epochs $4 \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 4 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 4 \
    --save_total_limit 15 \
    --learning_rate 8e-6 \
    --lr_scheduler_type "constant" \
    --weight_decay 0. \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --deepspeed config/zero.json \
    --bits 2 \
    --quant_type int2-asym \
    --q_group_size 128 \
    --train_kd True \
    --kd_loss_type "cakld" \
    --max_train_samples 999999 \
    --clip /home/brandon/hpml/hpmlfinal/BitDistiller/quantization/clip_cache/hf-llama3-1b/int2-g128.pt #CHANGE TO YOUR PATH WITH THE CLIPPINGS, INCLUDED IN quantization/clip_cache/hf-llama3-1b


# data_path: the path to the generated toxic chat teacher dataset in data/generation/datasets/llama-guard-3-1b/toxicchat_<suffix>
# SAVE_PATH: whatever you want
# logging_dir: whatever you want
# epochs: either 2 or 4, maybe 2 for now