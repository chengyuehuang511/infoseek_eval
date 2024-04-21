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
from lavis.models import load_model_and_preprocess
import torch
from PIL import Image
import json
from infoseek_eval import evaluate as evaluate_infoseek
import argparse
from infoseek_data.data_path import INFOSEEK_SPLIT2DATA, ID2IMAGE, IMAGES, OVEN_SPLIT2DATA
from peft import LoraConfig, get_peft_model
from utils import set_logger, AverageMeter
from torch.utils.tensorboard import SummaryWriter
import logging

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
    ref_path = f"infoseek_data/infoseek_{split}.jsonl"
    ref_qtype_path = f"infoseek_data/infoseek_{split}_qtype.jsonl"
    with open(pred_path, "w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")

    result = evaluate_infoseek(pred_path, ref_path, ref_qtype_path)
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
    

if __name__ == "__main__":
    logging.info("Initialize Processor...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", help="val, test, or human")
    parser.add_argument("--name", type=str, default="blip2_t5", help="blip2_t5 | blip2_t5_instruct | blip2_opt | blip2_vicuna_instruct")
    parser.add_argument("--model_type", type=str, default="pretrain_flant5xxl", help="pretrain_flant5xxl ｜ flant5xxl ｜ pretrain_opt2.7b")
    parser.add_argument("--output_dir", type=str, default="predictions", help="output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="accumulation size")
    parser.add_argument("--use_lora", type=int, help="use lora")
    parser.add_argument("--target_modules", type=str, default=["v", "q", "qkv"], nargs='*', help="target modules")
    parser.add_argument("--ratio", type=str, default="10%", help="ratio")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--early_stop", type=int, default=20, help="early stop")


    args = parser.parse_args()

    set_logger(args.output_dir + "/train.log")
    
    if args.ratio == "100%":
        split2data = {
            "val": "infoseek_data/infoseek_val.jsonl",
            "test": "infoseek_data/infoseek_test.jsonl",
            "human": "infoseek_data/infoseek_human.jsonl",
            "train": "infoseek_data/infoseek_train.jsonl"
        }
    else:
        split2data = {
                "val": f"infoseek_data/infoseek_val_{args.ratio}.jsonl",
                "test": f"infoseek_data/infoseek_test_{args.ratio}.jsonl",
                "human": f"infoseek_data/infoseek_human_{args.ratio}.jsonl",
                "train": f"infoseek_data/infoseek_train_{args.ratio}.jsonl"
            }
        # /coc/pskynet6/ychen3411/multimodal/infoseek/infoseek_qtype

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
    logging.info(f"if use lora: {args.use_lora}")  
    if args.use_lora == 1:
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=args.target_modules,  # ['v', 'q', 'qkv'],  # qformer, qkv
        )
        
        logging.info(config)
        model = get_peft_model(model, config)
        

    # raw_image = Image.open("aircraft.png").convert("RGB")
    # image = vis_processors["eval"](raw_image).unsqueeze(0).to("cuda")
    # output = model.generate({"image": image, "prompt": "Question: what is the date this aircraft took the first flight? Answer:"})
    # print(output)

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
                param.requires_grad = False
    
    # use lora to train the visual and text encoder
    if args.use_lora == 1:
        logging.info(model.print_trainable_parameters())

    # optmizer adamw for all parameters require grad
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    accum_iter = args.accumulation_steps

    writer = SummaryWriter(args.output_dir)
    optimization_step = 0
    best_val_score = 0
    early_stop = args.early_stop
    
    for epoch in range(20):
        start_time = time.time()
        train_loss = AverageMeter("train_loss", ":.4e")

        logging.info(f"=============== Epoch: {epoch} ===============")
        for idx, batch in enumerate(tqdm(train_dataloader)):
            batch["image"] = batch["image"].squeeze(1).to(device)
            output = model(samples=batch)
            loss = output["loss"]
            train_loss.update(loss.item(), batch["image"].size(0))
            # Gradient accumulation
            loss = loss / accum_iter
            loss.backward()
            # print(loss.item())
            if (idx + 1) % accum_iter == 0 or idx == len(train_dataloader) - 1:
                optimization_step += 1
                optimizer.step()
                optimizer.zero_grad()

                if (optimization_step + 1) % 1000 == 0 or idx == len(train_dataloader) - 1:
                    writer.add_scalar("loss/train_loss", train_loss.avg, optimization_step)
                    
                    logging.info(f"Step: {optimization_step} | Train Loss: {train_loss.avg}")
                    
                    logging.info("Evaluation...")
                    model.eval()
                    val_result = evaluate_model(split="val", model=model, batch_size=args.batch_size, step=optimization_step, prompt="Question: {} Short answer:",
                                                args=args, epoch=epoch)      
                    # logging.info("Step:", idx)
                    logging.info("Validation result:", val_result)
                    cur_val_score = val_result["final_score"]
                    
                    writer.add_scalar("score/val_score", cur_val_score, optimization_step)
                    writer.add_scalar("score/val_unseen_question_score", val_result["unseen_question_score"]["score"], optimization_step)
                    writer.add_scalar("score/val_unseen_entity_score", val_result["unseen_entity_score"]["score"], optimization_step)
                    
                    if cur_val_score > best_val_score:
                        best_val_score = cur_val_score
                        early_stop = args.early_stop
                        torch.save(model.state_dict(), f"{args.output_dir}/{args.name}_{args.model_type}_{optimization_step}_val={cur_val_score}_epoch={epoch}.pt")
                        logging.info("-------- Save Best Model! --------")
                    else:
                        early_stop -= 1
                        logging.info("Early Stop Left: {}".format(early_stop))
                    if early_stop == 0:
                        logging.info("-------- Early Stop! --------")
                        break
                    model.train()

"""
v q qkv
trainable params: 127,526,400 || all params: 12,251,930,496 || trainable%: 1.0408678048054119
0: 1: 
{'final_score': 16.09, 'unseen_question_score': {'score': 18.42, 'score_time': 5.19, 'score_num': 9.75, 'score_string': 21.95}, 'unseen
_entity_score': {'score': 14.29, 'score_time': 1.13, 'score_num': 10.81, 'score_string': 16.04}} 
0: 15.73 1: 15.738 2: 15.762

v q
0: 15.122 1: 15.347

qkv
0: 14.884 1: 14.68

q
0: 14.622 1: 14.385 2: 14.407

"""