# A PEFT LoRA adapter for the tiny qwen2 model, with transformers' logits of the adapted model.
#   pip install torch transformers peft
#   python make_lora.py <this folder>
import json, os, sys, torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
out = sys.argv[1]
torch.manual_seed(1)
model = AutoModelForCausalLM.from_pretrained(f"{out}/qwen2", dtype=torch.float32).eval()
peft = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"],
                                        lora_dropout=0.0, task_type="CAUSAL_LM"))
with torch.no_grad():
    for n, p in peft.named_parameters():
        if "lora_" in n:
            p.normal_(0, 0.3)
peft.save_pretrained(f"{out}/qwen2-lora")
for f in os.listdir(f"{out}/qwen2-lora"):
    if f not in ("adapter_config.json", "adapter_model.safetensors"):
        os.remove(f"{out}/qwen2-lora/{f}")
ids = [[1, 5, 9, 3, 7, 2, 11, 4]]
with torch.no_grad():
    logits = peft(torch.tensor(ids)).logits[0]
json.dump({"ids": ids[0], "logits": logits.numpy().round(6).tolist()}, open(f"{out}/qwen2-lora/expected.json", "w"))
