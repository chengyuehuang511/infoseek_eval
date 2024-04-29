""" Infoseek Validation Set Evaluation script."""
from infoseek_eval import evaluate, evaluate_seen
from utils import *

if __name__ == "__main__":
    # for split in ["val"]:
    #     for model in ["instruct", "pretrain"]:
    #         print(f"===BLIP2 {model} Flan T5 XXL===")
    #         # pred_path = f"predictions/zeroshot_blip2_t5_{model}_flant5xxl_{split}.jsonl"
    #         pred_path = "development/blip2_t5_pretrain_flant5xxl_val_399.jsonl"
    #         reference_path = f"infoseek_data/infoseek_{split}.jsonl"
    #         reference_qtype_path = f"infoseek_data/infoseek_{split}_qtype.jsonl"

    #         result = evaluate(pred_path, reference_path, reference_qtype_path)
    #         final_score = result["final_score"]
    #         unseen_question_score = result["unseen_question_score"]["score"]
    #         unseen_entity_score = result["unseen_entity_score"]["score"]
    #         print(f"{split} final score: {final_score}")
    #         print(f"{split} unseen question score: {unseen_question_score}")
    #         print(f"{split} unseen entity score: {unseen_entity_score}")
    
    set_logger('/nethome/chuang475/flash/projects/infoseek_eval/logfile.log')

    split = "val"

    best_models = {
        "zeroshot": None,
        "Q": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora0_targetv q qkv_20240426_013427/blip2_t5_pretrain_flant5xxl_9999_val=66.89_epoch=2.pt",
        "VQ": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetqkv_20240426_013428/blip2_t5_pretrain_flant5xxl_19999_val=72.45_epoch=4.pt",
        "QL": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetv q_20240426_013428/blip2_t5_pretrain_flant5xxl_24055_val=76.8_epoch=4.pt",
        "VQL": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetv q qkv_20240426_013427/blip2_t5_pretrain_flant5xxl_24055_val=79.28_epoch=4.pt",
        
        "Q_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora0_targetv q qkv_20240426_044130/blip2_t5_pretrain_flant5xxl_24055_val=72.49_epoch=4.pt",
        "VQ_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetqkv_20240426_044134/blip2_t5_pretrain_flant5xxl_24055_val=72.49_epoch=4.pt",
        "QL_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetv q_20240426_044133/blip2_t5_pretrain_flant5xxl_24055_val=75.13_epoch=4.pt",
        "VQL_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetv q qkv_20240426_044132/blip2_t5_pretrain_flant5xxl_24055_val=75.24_epoch=4.pt",
    }

    dict_seen = {
        "zeroshot": "experiments_adam_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch0_bs8_as2_lora0_targetv q qkv_20240426_034317/blip2_t5_pretrain_flant5xxl_val_seen_0_epoch=0.jsonl",
        "Q": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora0_targetv q qkv_20240426_013427/blip2_t5_pretrain_flant5xxl_val_seen_9999_epoch=2.jsonl",
        "VQ": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetqkv_20240426_013428/blip2_t5_pretrain_flant5xxl_val_seen_19999_epoch=4.jsonl",
        "QL": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetv q_20240426_013428/blip2_t5_pretrain_flant5xxl_val_seen_24055_epoch=4.jsonl",
        "VQL": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetv q qkv_20240426_013427/blip2_t5_pretrain_flant5xxl_val_seen_24055_epoch=4.jsonl",
        
        "Q_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora0_targetv q qkv_20240426_044130/blip2_t5_pretrain_flant5xxl_val_seen_24055_epoch=4.jsonl",
        "VQ_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetqkv_20240426_044134/blip2_t5_pretrain_flant5xxl_val_seen_24055_epoch=4.jsonl",
        "QL_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetv q_20240426_044133/blip2_t5_pretrain_flant5xxl_val_seen_24055_epoch=4.jsonl",
        "VQL_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetv q qkv_20240426_044132/blip2_t5_pretrain_flant5xxl_val_seen_24055_epoch=4.jsonl",
    }

    dict_unseen = {
        "zeroshot": "experiments_adam_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch0_bs8_as2_lora0_targetv q qkv_20240426_034317/blip2_t5_pretrain_flant5xxl_val_unseen_0_epoch=0.jsonl",
        "Q": "",
        "VQ": "",
        "QL": "",
        "VQL": "",
        
        "Q_ftp": "",
        "VQ_ftp": "",
        "QL_ftp": "",
        "VQL_ftp": "",
    }

    reference_path_dict = {
        "seen": "infoseek/infoseek_val_seen_10%.jsonl",
        "unseen": "infoseek/infoseek_val_unseen_10%.jsonl",
    }
    
    # task_list = ["zeroshot", "Q", "VQ", "QL", "VQL", "Q_ftp", "VQ_ftp", "QL_ftp", "VQL_ftp"]
    task_list = ["zeroshot"]
    for task in task_list:
        logging.info(f"===Task: {task}===")
        
        logging.info(f"===Validation unseen===")
        reference_path = reference_path_dict["unseen"]
        pred_path = dict_unseen[task]
        result = evaluate(pred_path, reference_path, reference_path)
        logging.info(f"Validation unseen result: {result}")
        final_score = result["final_score"]
        unseen_question_score = result["unseen_question_score"]["score"]
        unseen_entity_score = result["unseen_entity_score"]["score"]
        logging.info(f"{split} final score: {final_score}")
        logging.info(f"{split} unseen question score: {unseen_question_score}")
        logging.info(f"{split} unseen entity score: {unseen_entity_score}")
    
        logging.info(f"===Validation seen===")
        reference_path = reference_path_dict["seen"]
        pred_path = dict_seen[task]
        result = evaluate_seen(pred_path, reference_path)
        logging.info(f"Validation seen result: {result}")
        seen_score = result["seen_score"]["score"]
        logging.info(f"{split} seen score: {seen_score}")
