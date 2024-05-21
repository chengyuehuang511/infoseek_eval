import os
import time
import pandas as pd
import itertools
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import copy
from lavis.models import load_model_and_preprocess
import torch
from PIL import Image
import json
from infoseek_eval import evaluate as evaluate_infoseek
from infoseek_eval import evaluate_seen
import argparse
from infoseek_data.data_path import INFOSEEK_SPLIT2DATA, ID2IMAGE, IMAGES, OVEN_SPLIT2DATA
from peft import LoraConfig, get_peft_model
from utils import set_logger, AverageMeter
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
from FTP import SGDP, AdamP
from adamh import AdamH
import warnings
warnings.filterwarnings("ignore")

def create_eval_data(split):
    # Read the input JSONL file
    with open(split2data[split], 'r') as f:
        batch_data = [json.loads(line) for line in f]

    clean_batch_data = []
    not_exit = []
    for idx, item in enumerate(batch_data):
        if idx % 10000 == 0:
            logging.info(f"Processing {idx}/{len(batch_data)}")
        path = id2path[item["image_id"]]
        # check path exists
        if not os.path.exists(path):
            not_exit.append(item["image_id"])
        else:
            clean_batch_data.append(item)
    return clean_batch_data

def load_and_process_image(item):
    # Load and preprocess the image
    raw_image = Image.open(id2path[item["image_id"]]).convert("RGB").resize((224, 224))    
    processed_image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    return processed_image, item["question"], item["data_id"]

def process_images_in_batches(model, batch_data, batch_size, prompt):
    # Create a pool of workers
    # Monitor the progress of the pool
    
    output = []
    logging.info("Generate predictions...")
    # Process images in batches
    for idx, i in enumerate(range(0, len(batch_data), batch_size)):
        if (idx + 1) % 100 == 0:
            logging.info(f"Processing batch {idx}/{len(batch_data)/batch_size}")
        # Subset results for the current batch
        batch_subset = batch_data[i:i+batch_size]

        # Separate the images, questions, and ids
        batch_images, batch_questions, batch_ids = [], [], []

        # Load and preprocess the images
        for item in batch_subset:
            tmp_img, tmp_q, tmp_id = load_and_process_image(item)
            batch_images.append(tmp_img)
            batch_questions.append(tmp_q)
            batch_ids.append(tmp_id)

        # Concatenate the batch images
        image_batch = torch.cat(batch_images, dim=0)
        
        # add prompt to questions
        batch_questions = [prompt.format(q) for q in batch_questions]
        # Generate predictions for the batch
        
        answers = model.generate({"image": image_batch, "prompt": batch_questions},
                                 length_penalty=-1)  # default: num_beams=5
        # print(batch_questions)
        # print(answers)
        
        for idx, ans in zip(batch_ids, answers):
            output.append({"data_id": idx, "prediction": ans})
    return output

def evaluate_model(split, model, batch_size, step, prompt, args, epoch):
    # Create evaluate data
    batch_data = create_eval_data(split)
    # Process the data in batches
    output = process_images_in_batches(model, batch_data, batch_size, prompt)

    # Save the predictions
    # development_{args.batch_size}_all_lora
    pred_path = f"{args.output_dir}/{args.name}_{args.model_type}_{split}_{step}_epoch={epoch}.jsonl"
    
    # ref_path = f"infoseek_data/infoseek_{split}.jsonl"
    ref_path = split2data[split]
    # ref_qtype_path = f"infoseek_data/infoseek_{split}_qtype.jsonl"
    
    with open(pred_path, "w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")

    if split == "val_seen" or split == "test_seen":
        result = evaluate_seen(pred_path, ref_path)
    else:
        result = evaluate_infoseek(pred_path, ref_path, ref_path)
    return result


class BLIP2Dataset(torch.utils.data.Dataset):
    def __init__(self, split, processor, PROMPT="Question: {} Short answer:"):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        self.image_path = []
        self.question = []
        self.answer = []
        with open(split2data[split], "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                image_id = line["image_id"]
                path = id2path[image_id]
                self.image_path.append(path)
                self.question.append(line["question"])
                self.answer.append(line["answer"][0])

        self.vis_processor = processor
        self.prompt = PROMPT
 
    def __getitem__(self, idx):
        raw_image = Image.open(self.image_path[idx]).convert("RGB").resize((224, 224))
        question = self.prompt.format(self.question[idx])
        answer = self.answer[idx]
        processed_image = self.vis_processor["train"](raw_image).unsqueeze(0)
        inputs = {"image": processed_image, "text_input": question, "text_output": answer}
        return inputs
 
    def __len__(self):
        return len(self.question)


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val_seen", help="val, test, or human")
    parser.add_argument("--name", type=str, default="blip2_t5", help="blip2_t5 | blip2_t5_instruct | blip2_opt | blip2_vicuna_instruct")
    parser.add_argument("--model_type", type=str, default="pretrain_flant5xxl", help="pretrain_flant5xxl ｜ flant5xxl ｜ pretrain_opt2.7b")
    parser.add_argument("--output_dir", type=str, default="predictions", help="output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="accumulation size")
    parser.add_argument("--use_lora", type=int, help="use lora")
    parser.add_argument("--target_modules", type=str, default=None, nargs='*', help="target modules")
    parser.add_argument("--ratio", type=str, default="10%", help="ratio")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--early_stop", type=int, default=20, help="early stop")
    parser.add_argument("--val_print_freq", type=int, default=1000, help="val print freq")
    parser.add_argument("--epoch", type=int, default=10, help="epoch")
    parser.add_argument("--opt", type=str, default="adam", help="optimizer")
    parser.add_argument("--wd", type=float, default=0.01, help="weight decay")
    parser.add_argument("--best_model_task", type=str, default=None, help="best model task")
    parser.add_argument("--lora_alpha", type=int, default=32, help="lora alpha")
    parser.add_argument("--lora_rank", type=int, default=16, help="lora rank")

    args = parser.parse_args()

    set_seed(args.seed)
    set_logger(args.output_dir + "/train.log")
    logging.info("Initialize Processor...")
    
    if args.ratio == "100%":
        split2data = {
            "val_seen": "infoseek/infoseek_val_seen.jsonl",
            "val_unseen": "infoseek/infoseek_val_unseen.jsonl",
            "test_seen": "infoseek/infoseek_test_seen.jsonl",
            "test_unseen": "infoseek/infoseek_test_unseen.jsonl",
            "train": "infoseek/infoseek_train.jsonl"
        }
    else:
        split2data = {
            "val_seen": f"infoseek/infoseek_val_seen_{args.ratio}.jsonl",
            "val_unseen": f"infoseek/infoseek_val_unseen_{args.ratio}.jsonl",
            "test_seen": "infoseek/infoseek_test_seen.jsonl",
            "test_unseen": "infoseek/infoseek_test_unseen.jsonl",
            "train": f"infoseek/infoseek_train_{args.ratio}.jsonl"
        }

    id2path = dict()

    # load image paths
    with open(ID2IMAGE, "r") as f:
        for line in f:
            line = json.loads(line)
            image_id = line["image_id"]
            path = line["image_path"]
            id2path[image_id] = path

    model, vis_processors, _ = load_model_and_preprocess(name=args.name,
                                                         model_type=args.model_type, 
                                                         is_eval=False, 
                                                         device="cuda")
    
    logging.info("target modules: {}".format(args.target_modules))
    logging.info(f"if use lora: {args.use_lora}")  
    logging.info(f"lora alpha: {args.lora_alpha}")
    logging.info(f"lora rank: {args.lora_rank}")
    logging.info(f"optimizer: {args.opt}")
    
    if args.use_lora == 1:
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=args.target_modules,  # ['v', 'q', 'qkv'],  # qformer, qkv
        )
        
        logging.info(config)
        model = get_peft_model(model, config)

    blip_dataset = BLIP2Dataset(
        split="train",
        processor=vis_processors,
        PROMPT="Question: {} Short answer:"
    )
    logging.info("Initialize Dataloader...")
    # Padding dataloader
    train_dataloader = DataLoader(
        blip_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6
    )

    # # freeze everything except qformer
    logging.info("Freeze Model...")
    for name, param in model.named_parameters():
        if "Qformer" in name:
            param.requires_grad = True
        else:
            if args.use_lora == 0:
                if_freeze = True
                for target_module in args.target_modules:
                    if f".{target_module}." in name:
                        param.requires_grad = True
                        if_freeze = False
                        logging.info(name)
                        break
                if if_freeze:
                    param.requires_grad = False
    
    # use lora to train the visual and text encoder
    if args.use_lora == 1:
        logging.info(model.print_trainable_parameters())

    # optmizer adamw for all parameters require grad
    # optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # load best model according to best model name
    # best_models = {
    #     "zeroshot": None,
    #     "Q": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora0_targetv q qkv_20240426_013427/blip2_t5_pretrain_flant5xxl_9999_val=66.89_epoch=2.pt",
    #     "VQ": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetqkv_20240426_013428/blip2_t5_pretrain_flant5xxl_19999_val=72.45_epoch=4.pt",
    #     "QL": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetv q_20240426_013428/blip2_t5_pretrain_flant5xxl_24055_val=76.8_epoch=4.pt",
    #     "VQL": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetv q qkv_20240426_013427/blip2_t5_pretrain_flant5xxl_24055_val=79.28_epoch=4.pt",
        
    #     "Q_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora0_targetv q qkv_20240426_044130/blip2_t5_pretrain_flant5xxl_24055_val=72.49_epoch=4.pt",
    #     "VQ_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetqkv_20240426_044134/blip2_t5_pretrain_flant5xxl_24055_val=72.49_epoch=4.pt",
    #     "QL_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetv q_20240426_044133/blip2_t5_pretrain_flant5xxl_24055_val=75.13_epoch=4.pt",
    #     "VQL_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetv q qkv_20240426_044132/blip2_t5_pretrain_flant5xxl_24055_val=75.24_epoch=4.pt",

    #     "Q_h": "experiments_adamh_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora0_targetv q qkv_20240426_234538/blip2_t5_pretrain_flant5xxl_24055_val=67.25_epoch=4.pt",
    #     "VQ_h": "experiments_adamh_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetqkv_20240426_234539/blip2_t5_pretrain_flant5xxl_24055_val=66.53_epoch=4.pt",
    #     "QL_h": "experiments_adamh_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetv q_20240426_234539/blip2_t5_pretrain_flant5xxl_24055_val=74.98_epoch=4.pt",
    #     "VQL_h": "experiments_adamh_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetv q qkv_20240426_234538/blip2_t5_pretrain_flant5xxl_24055_val=75.66_epoch=4.pt",
    
    # }

    best_models = {
        "zeroshot": None,
        "Q": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora0_targetv q qkv_20240426_013427/blip2_t5_pretrain_flant5xxl_9999_val=66.89_epoch=2.pt", 
        "VQ": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetqkv_20240426_013428/blip2_t5_pretrain_flant5xxl_19999_val=72.45_epoch=4.pt",
        "QL": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetv q_20240426_013428/blip2_t5_pretrain_flant5xxl_38488_val=79.46_epoch=7.pt",
        "VQL": "experiments_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_bs8_as2_lora1_targetv q qkv_20240426_013427/blip2_t5_pretrain_flant5xxl_24055_val=79.28_epoch=4.pt",

        "Q_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora0_targetv q qkv_20240426_044130/blip2_t5_pretrain_flant5xxl_48110_val=77.79_epoch=9.pt",
        "VQ_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetqkv_20240426_044134/blip2_t5_pretrain_flant5xxl_48110_val=77.82_epoch=9.pt",
        "QL_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetv q_20240426_044133/blip2_t5_pretrain_flant5xxl_48110_val=79.42_epoch=9.pt",
        "VQL_ftp": "experiments_adamp_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch10_bs8_as2_lora1_targetv q qkv_20240426_044132/blip2_t5_pretrain_flant5xxl_48110_val=78.7_epoch=9.pt",

        "Q_h": "experiments/wd_0.3_new/experiments_adamh_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch20_bs8_as2_lora0_target_20240515_052826/blip2_t5_pretrain_flant5xxl_48110_val=77.94_epoch=9.pt",
        "VQ_h": "experiments/wd_0.3_new/experiments_adamh_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch20_bs8_as2_lora1_targetqkv_20240515_052827/blip2_t5_pretrain_flant5xxl_48110_val=76.88_epoch=9.pt",
        "QL_h": "experiments/wd_0.3_new/experiments_adamh_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch20_bs8_as2_lora1_targetv q_20240515_052826/blip2_t5_pretrain_flant5xxl_38488_val=81.07_epoch=7.pt",
        "VQL_h": "experiments/wd_0.3_new/experiments_adamh_val_seen_10%_slurm/blip2_t5_pretrain_flant5xxl_epoch20_bs8_as2_lora1_targetv q qkv_20240515_052827/blip2_t5_pretrain_flant5xxl_43299_val=80.2_epoch=8.pt",
    }
    
    if args.epoch == 0:  # use best model
        if args.opt == "adam":
            tmp = ""
        elif args.opt == "adamp":
            tmp = "_ftp"
        elif args.opt == "adamh":
            tmp = "_h"
        
        if args.use_lora == 1:
            if args.target_modules == ["v", "q", "qkv"]:
                args.best_model_task = "VQL" + tmp
            elif args.target_modules == ["v", "q"]:
                args.best_model_task = "QL" + tmp
            elif args.target_modules == ["qkv"]:
                args.best_model_task = "VQ" + tmp
        else:
            args.best_model_task = "Q" + tmp

    w0 = {key: value.to('cpu') for key, value in model.state_dict().items()}
    state_dict = torch.load(best_models[args.best_model_task], map_location='cpu')
    state_dict = {key: value.to('cpu') for key, value in state_dict.items()}
    lora = {key: (value - w0[key]) * 16 / 32 for key, value in state_dict.items()}

    new_w = {key: value + lora[key] * args.lora_alpha / args.lora_rank for key, value in w0.items()}
    
    if args.best_model_task is not None:
        logging.info("best model task: {}".format(args.best_model_task))
        model.load_state_dict(new_w)
    model.eval()

    if args.epoch == 0 and args.best_model_task is None:
        logging.info("Zero-shot evaluation ...")
    
    logging.info("Validation seen ...")
    val_seen_result = evaluate_model(split="val_seen", model=model, batch_size=args.batch_size, step=0, prompt="Question: {} Short answer:",
                                args=args, epoch=0)
    logging.info(f"Validation seen result: {val_seen_result}")

    logging.info("Validation unseen ...")
    val_unseen_result = evaluate_model(split="val_unseen", model=model, batch_size=args.batch_size, step=0, prompt="Question: {} Short answer:",
                                args=args, epoch=0)
    logging.info(f"Validation unseen result: {val_unseen_result}")