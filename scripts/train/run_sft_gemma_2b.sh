# Tested with 2 & 4 GPUs

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_gemma_2b.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/projectnb/rlhf/mingyuc/TinyZero/TinyZero_old/data/gsm8k/train.parquet \
    data.val_files=/projectnb/rlhf/mingyuc/TinyZero/TinyZero_old/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=8 \
    model.partial_pretrain=Qwen/Qwen2.5-1.5B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-gemma-2b-it \
    trainer.total_epochs=2 \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null $@