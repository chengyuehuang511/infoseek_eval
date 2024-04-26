""" Infoseek Validation Set Evaluation script."""
from infoseek_eval import evaluate, evaluate_seen
from utils import *

if __name__ == "__main__":
    for split in ["val"]:
        for model in ["instruct", "pretrain"]:
            print(f"===BLIP2 {model} Flan T5 XXL===")
            # pred_path = f"predictions/zeroshot_blip2_t5_{model}_flant5xxl_{split}.jsonl"
            pred_path = "development/blip2_t5_pretrain_flant5xxl_val_399.jsonl"
            reference_path = f"infoseek_data/infoseek_{split}.jsonl"
            reference_qtype_path = f"infoseek_data/infoseek_{split}_qtype.jsonl"

            result = evaluate(pred_path, reference_path, reference_qtype_path)
            final_score = result["final_score"]
            unseen_question_score = result["unseen_question_score"]["score"]
            unseen_entity_score = result["unseen_entity_score"]["score"]
            print(f"{split} final score: {final_score}")
            print(f"{split} unseen question score: {unseen_question_score}")
            print(f"{split} unseen entity score: {unseen_entity_score}")
    
    set_logger('/nethome/chuang475/flash/projects/infoseek_eval/logfile.log')
    
    for pred_path, reference_path in zip(
        [
            "experiments_adam_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch0_bs8_as2_lora0_targetv q qkv_20240426_034317/blip2_t5_pretrain_flant5xxl_val_seen_0_epoch=0.jsonl",
            "experiments_adam_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch0_bs8_as2_lora0_targetv q qkv_20240426_034317/blip2_t5_pretrain_flant5xxl_val_unseen_0_epoch=0.jsonl",
        ],
        [
            "infoseek/infoseek_val_seen_10%.jsonl",
            "infoseek/infoseek_val_unseen_10%.jsonl",
        ],
    ):
        if "unseen" in reference_path:
            result = evaluate(pred_path, reference_path, reference_path)
            logging.info(f"Validation seen result: {result}")
            final_score = result["final_score"]
            unseen_question_score = result["unseen_question_score"]["score"]
            unseen_entity_score = result["unseen_entity_score"]["score"]
            print(f"{split} final score: {final_score}")
            print(f"{split} unseen question score: {unseen_question_score}")
            print(f"{split} unseen entity score: {unseen_entity_score}")
        else:
            result = evaluate_seen(pred_path, reference_path)
            logging.info(f"Validation unseen result: {result}")
