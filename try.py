from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
from lavis.models import load_model_and_preprocess

#pretrain_model_path="/workspace/model/blip2-opt-2.7b"
pretrain_model_path = "Salesforce/blip2-opt-2.7b"
#train_dataset_path = "/workspace/data/pytorch_data/multimodal/blip2/ybelkada___football-dataset/default-80f5618dafa96df9/0.0.0/0111277fb19b16f696664cde7f0cb90f833dec72db2cc73cfdf87e697f78fe02"


# We load our model and processor using `transformers`
model = AutoModelForVision2Seq.from_pretrained(pretrain_model_path)
# processor = AutoProcessor.from_pretrained(pretrain_model_path)

# model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device="cuda")

# print("Freeze Model...")
for name, param in model.named_parameters():
    # param.requires_grad = True
    # if "Qformer" in name:
    #     param.requires_grad = True
    # else:
    #     param.requires_grad = False
    if param.requires_grad == True:
        print(name)

# target_modules = [None, ["v_proj", "q_proj"], ["v_proj", "q_proj", "qkv"]]
target_modules = [None]
# target_modules = [["v_proj", "q_proj"]]
# target_modules = [["v_proj", "q_proj", "qkv"]]
for t in target_modules:
    print(t)
    # Let's define the LoraConfig
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        # target_modules=["v_proj", "q_proj", "qkv"],  # qformer, qkv, language_model
        # target_modules=t,
    )

    # Get our peft model and print the number of trainable parameters
    model_lora = get_peft_model(model, config)
    model_lora.print_trainable_parameters()

    for name, param in model_lora.named_parameters():
        if param.requires_grad == True:
            print(name)

    del model_lora
    del config


# print(model)
# with open('arch_wo8bit.txt', 'w') as file:
#     print(model, file=file)

# "v_proj", "q_proj"
# trainable params: 5,242,880 || all params: 3,749,922,816 || trainable%: 0.13981301102065136

# "v_proj", "q_proj", "qkv"
# trainable params: 8,757,248 || all params: 3,753,437,184 || trainable%: 0.2333127629611078

# "language_model"
# not supported

"""
None
trainable params: 5,242,880 || all params: 3,749,922,816 || trainable%: 0.13981301102065136
['v_proj', 'q_proj']
trainable params: 5,242,880 || all params: 3,749,922,816 || trainable%: 0.13981301102065136
['v_proj', 'q_proj', 'qkv']
trainable params: 8,757,248 || all params: 3,753,437,184 || trainable%: 0.2333127629611078
"""