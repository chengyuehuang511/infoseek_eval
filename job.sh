#!/bin/bash

#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie

export PYTHONUNBUFFERED=TRUE
cd /nethome/chuang475/flash/projects/infoseek_eval

# Run the training command
srun -u /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python -u run_training_lavis.py \
                                                        --name "$name" \
                                                        --model_type "$model_type" \
                                                        --batch_size $batch_size \
                                                        --accumulation_steps $accumulation_steps \
                                                        --target_modules $target_modules \
                                                        --output_dir "$output_dir" \
                                                        --val_print_freq $val_print_freq \
                                                        --epoch $epoch \
                                                        --opt "$opt" \
                                                        --use_lora $use_lora \