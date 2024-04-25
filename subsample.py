import os
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
import random

split2data = {
        "val": "infoseek_data/infoseek_val.jsonl",
        "test": "infoseek_data/infoseek_test.jsonl",
        "human": "infoseek_data/infoseek_human.jsonl",
        "train": "infoseek_data/infoseek_train.jsonl"
    }

subset_output_path = {
    "val": "infoseek_data/infoseek_val_10%.jsonl",
    "test": "infoseek_data/infoseek_test_10%.jsonl",
    "human": "infoseek_data/infoseek_human_10%.jsonl",
    "train": "infoseek_data/infoseek_train_10%.jsonl"
}
# /coc/pskynet6/ychen3411/multimodal/infoseek/infoseek_qtype

train_image_id = []
val_image_id = []
val_data_split = []

# with open(split2data["train"], "r", encoding="utf-8") as f:    
#     for line in f:
#         line = json.loads(line)
#         train_image_id.append(line["image_id"])

# with open(split2data["val"], 'r') as f:
#     for line in f:
#         line = json.loads(line)
#         val_image_id.append(line["image_id"])
#         val_data_split.append(line["data_split"])

# train_image_id_set = set(train_image_id)
# found = False  # Flag to track if any match is found

# val_image_id_new = []
# val_image_id_ = []
# for idx, split in enumerate(val_data_split):
#     if split == "val_unseen_question":
#         val_image_id_new.append(val_image_id[idx])
#     else:
#         val_image_id_.append(val_image_id[idx])

# train_image_id_set = set(train_image_id)
# val_image_id_set = set(val_image_id_new)
# val_image_id_set_ = set(val_image_id_)
# print("Number of train_image_id:", len(train_image_id_set))
# print("Number of val_image_id:", len(val_image_id_set))
# print("Number of val_image_id_:", len(val_image_id_set_))

# # 计算差集，找出仅在 val_image_id_set 中的元素
# unseen_in_train = val_image_id_set.difference(train_image_id_set)

# # 打印差集结果
# # print("Elements in val_image_id not in train_image_id:", unseen_in_train)

# # 判断是否所有val_image_id都在train_image_id中
# if not unseen_in_train:
#     print("All val_image_id elements are present in train_image_id.")
# else:
#     print(f"Not all val_image_id elements are present in train_image_id.  Not in # {len(unseen_in_train)}")

def select_random_subset(input_file, output_file, percentage=0.1):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Calculate the number of lines to sample
    sample_size = int(len(lines) * percentage)

    # Randomly sample lines
    sampled_lines = random.sample(lines, sample_size)

    # Write the selected lines to a new jsonl file
    with open(output_file, 'w') as outfile:
        for line in sampled_lines:
            outfile.write(line)

# Usage
# for split in ["train", "val", "test", "human"]:
#     select_random_subset(split2data[split], subset_output_path[split], 0.1)
