#!/bin/bash

#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax
#SBATCH --mem-per-gpu=45G

export PYTHONUNBUFFERED=TRUE
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
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
                                                        --lora_alpha $lora_alpha \
                                                        --lora_rank $lora_rank \
                                                        # --best_model_task "$best_model_task" \