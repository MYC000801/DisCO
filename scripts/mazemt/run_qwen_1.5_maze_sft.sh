#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/multiturn/train.parquet \
    data.val_files=./data/multiturn/test.parquet \
    data.max_length=6000 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.enable_thinking_key=false \
    data.micro_batch_size=2 \
    model.partial_pretrain=Qwen/Qwen2.5-1.5B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn-sft-qwen-2.5-1.5b-maze \
    trainer.logger=['console'] \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true