#!/bin/bash
cd /nethome/chuang475/flash/projects/infoseek_eval

# Define the parameters
name="blip2_t5"  # blip2_t5 | blip2_opt
model_type="pretrain_flant5xxl"  # pretrain_flant5xxl | pretrain_opt2.7b
batch_size=8
accumulation_steps=2
target_modules=""
use_lora=0
ratio="10%"
split="val_seen"
val_print_freq=1000000
lora_rank=16
lora_alpha=32

epoch=0
opt="adamp"

# Create the output directory name
# job_name="${name}_${model_type}_epoch${epoch}_bs${batch_size}_as${accumulation_steps}_lora${use_lora}_target${target_modules}_$(date +%Y%m%d_%H%M%S)"
# output_dir="experiments/wd_0.3_new_20/experiments_${opt}_${split}_${ratio}_slurm/${job_name}"

# # Ensure the output directory exists
# mkdir -p "$output_dir"

# # Submit the job
# sbatch --export "ALL,name=${name},model_type=${model_type},batch_size=${batch_size},accumulation_steps=${accumulation_steps},target_modules=${target_modules},use_lora=${use_lora},ratio=${ratio},split=${split},val_print_freq=${val_print_freq},output_dir=${output_dir},epoch=${epoch},opt=${opt},lora_alpha=${lora_alpha},lora_rank=${lora_rank}" --job-name="${opt}_${split}_${ratio}_${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" job.sh


# use_lora=1, target_modules="v q qkv"
# use_lora=1, target_modules="v q"
# use_lora=1, target_modules="qkv"
# if epoch>0, then for the above 3 cases, create a loop to run sbatch
# if [ $epoch -eq 0 ]; then
#     exit
# fi

use_lora=1
for target_modules in "qkv"  # "qkv" "v q qkv" "v q"
do
    job_name="${name}_${model_type}_epoch${epoch}_bs${batch_size}_as${accumulation_steps}_lora${use_lora}_target${target_modules}_$(date +%Y%m%d_%H%M%S)"
    output_dir="experiments/experiments_${opt}_${split}_${ratio}_slurm/${job_name}"
    mkdir -p "$output_dir"
    sbatch --export "ALL,name=${name},model_type=${model_type},batch_size=${batch_size},accumulation_steps=${accumulation_steps},target_modules=${target_modules},use_lora=${use_lora},ratio=${ratio},split=${split},val_print_freq=${val_print_freq},output_dir=${output_dir},epoch=${epoch},opt=${opt},lora_alpha=${lora_alpha},lora_rank=${lora_rank}" --job-name="${opt}_${split}_${ratio}_${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" job.sh
done

# for best_model_task in "Q" "VQ" "QL" "VQL" "Q_ftp" "VQ_ftp" "QL_ftp" "VQL_ftp"
# do
#     job_name="${name}_${model_type}_epoch${epoch}_bs${batch_size}_as${accumulation_steps}_lora${use_lora}_best_model_task${best_model_task}_target${target_modules}_$(date +%Y%m%d_%H%M%S)"
#     output_dir="experiments_${opt}_${split}_${ratio}_slurm/${job_name}"
#     mkdir -p "$output_dir"
#     sbatch --export "ALL,name=${name},model_type=${model_type},batch_size=${batch_size},accumulation_steps=${accumulation_steps},best_model_task=${best_model_task},target_modules=${target_modules},use_lora=${use_lora},ratio=${ratio},split=${split},val_print_freq=${val_print_freq},output_dir=${output_dir},epoch=${epoch},opt=${opt}" --job-name="${opt}_${split}_${ratio}_${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" job.sh
# done