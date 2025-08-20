# SFT Dataset Generation

python scripts/dataset/generate_maze_sft.py \
    --save_path /projectnb/rlhf/mingyuc/verl_github/verl/data/maze_mt/train10000.parquet \
    --num_samples 10000 \
    --maze_size 9 \
    --step_per_traj 20 \
    --traj_per_chat 5

python scripts/dataset/split_maze_sft.py --local_dir ./data/multiturn \
    --file_path /projectnb/rlhf/mingyuc/verl_github/verl/data/maze_mt/train10000.parquet

# SFT Training
bash scripts/mazemt/run_qwen_7_maze_sft.sh 4 ./sft


# RL Dateset Generation
python scripts/dataset/generate_maze_rl.py --local_path ./data/maze_test \
    --num_samples 200000 \
    --maze_size 9 \
    --step_per_traj 20 \
    --traj_per_chat 5


# RL Training
bash scripts/mazemt/run_qwen_7_maze_rl.sh 