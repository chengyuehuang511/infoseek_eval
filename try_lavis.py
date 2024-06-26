from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
from lavis.models import load_model_and_preprocess
import logging
from utils import set_logger

if_lora = True
if_qformer = True
# model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=False, device="cuda")
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=False, device="cuda")

# print("Freeze Model...")
# for name, param in model.named_parameters():
#     param.requires_grad = True
    # if param.requires_grad == True:
    #     print(name)
#     if "Qformer" in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# Let's define the LoraConfig
if if_lora:
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=['v', 'q', 'qkv'],  # qformer, qkv, language_model
    )
    model = get_peft_model(model, config)

"""
config_qkv = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=['qkv'],  # qformer, qkv, language_model
)

config_vq = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=['v', 'q'],  # qformer, qkv, language_model
)

# Get our peft model and print the number of trainable parameters
model = get_peft_model(model, config_qkv)
model = get_peft_model(model, config_vq)
"""

set_logger(f"arch_lavis_blip2_t5_lora_{if_lora}_qformer_{if_qformer}.log")
for name, param in model.named_parameters():
    if if_qformer:
        if "Qformer" in name:
            param.requires_grad = True
    if if_lora == False:
        if (".qkv." in name) or (".v." in name) or (".q." in name):
            param.requires_grad = True
        else:
            param.requires_grad = False
    if param.requires_grad == True:
        logging.info(name) 

model.print_trainable_parameters()
# print(model)
# with open('arch_lavis_blip2_t5.txt', 'w') as file:
#     print(model, file=file)


##### "v_proj", "q_proj" #####
# trainable params: 5,242,880 || all params: 3,749,867,904 || trainable%: 0.1398150584026546

# trainable params: 5,242,880 || all params: 3,749,922,816 || trainable%: 0.13981301102065136

##### "v_proj", "q_proj", "qkv" #####
# trainable params: 8,757,248 || all params: 3,753,382,272 || trainable%: 0.23331617632790908
# V+Q+L
# trainable params: 113,894,912 || all params: 3,753,382,272 || trainable%: 3.0344607542282334
# (t5) trainable params: 127,526,400 || all params: 12,251,930,496 || trainable%: 1.0408678048054119

# trainable params: 8,757,248 || all params: 3,753,437,184 || trainable%: 0.2333127629611078

# The results are the same no matter if I set all params requires_grad to True or Not before using lora.