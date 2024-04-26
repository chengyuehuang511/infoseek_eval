#!/bin/bash
#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie

export PYTHONUNBUFFERED=TRUE
cd /nethome/chuang475/flash/projects/infoseek_eval

# Define the parameters
name="blip2_t5"  # blip2_t5 | blip2_opt
model_type="pretrain_flant5xxl"  # pretrain_flant5xxl | pretrain_opt2.7b
batch_size=8
accumulation_steps=2
target_modules="v q qkv"
use_lora=0
ratio="10%"
split="val_seen"
val_print_freq=100

# Create the output directory name
output_dir="experiments_${split}_${ratio}/${name}_${model_type}_bs${batch_size}_as${accumulation_steps}_lora${use_lora}_target${target_modules}_$(date +%Y%m%d_%H%M%S)"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Run the training command
srun -u /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python -u run_training_lavis.py \
                                                        --name "$name" \
                                                        --model_type "$model_type" \
                                                        --batch_size $batch_size \
                                                        --accumulation_steps $accumulation_steps \
                                                        --target_modules $target_modules \
                                                        --output_dir "$output_dir" \
                                                        --use_lora $use_lora \