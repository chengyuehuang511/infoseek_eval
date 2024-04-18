cd /nethome/chuang475/flash/projects/infoseek_eval
# /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python run_evaluation.py
# /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python run_blip2_infoseek.py --split val --output_dir predictions_try --batch_size 32
# /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python run_training_lavis.py --name instruct --batch_size 8 --accumulation_steps 2
# /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python run_training_lavis.py --name pretrain --batch_size 1 --accumulation_steps 2
# /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python blip2_arch.py --name pretrain --batch_size 8 --accumulation_steps 2
# /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python try_lavis.py


# Define the parameters
name="blip2_opt"
model_type="pretrain_opt2.7b"
batch_size=8
accumulation_steps=2
target_modules="v q qkv"
use_lora=True

# Create the output directory name
output_dir="experiments/${name}_${model_type}_bs${batch_size}_as${accumulation_steps}_lora${use_lora}_target${target_modules}_$(date +%Y%m%d_%H%M%S)"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Run the training command
/nethome/chuang475/flash/miniconda3/envs/lavis/bin/python -m run_training_lavis \
                                                          --name "$name" \
                                                          --model_type "$model_type" \
                                                          --batch_size $batch_size \
                                                          --accumulation_steps $accumulation_steps \
                                                          --target_modules $target_modules \
                                                          --output_dir "$output_dir" \
                                                          --use_lora

                                                          