cd /nethome/chuang475/flash/projects/infoseek_eval
# /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python run_evaluation.py
# /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python run_blip2_infoseek.py --split val --output_dir predictions_try --batch_size 32
# /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python run_training_lavis.py --name instruct --batch_size 8 --accumulation_steps 2
/nethome/chuang475/flash/miniconda3/envs/lavis/bin/python run_training_lavis.py --name pretrain --batch_size 1 --accumulation_steps 2
# /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python blip2_arch.py --name pretrain --batch_size 8 --accumulation_steps 2
# /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python try_lavis.py