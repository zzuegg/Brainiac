# Writes <model>.expected.json next to each Hugging Face model directory in a folder: the
# token ids of the test prompt, transformers' logits for the next token, and its greedy
# answer. PretrainedModelsTest (-PtestModels=<folder>) compares Brainiac with them.
#   pip install torch transformers
#   python reference_models.py <folder>
import json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
folder = sys.argv[1]
question = "What is the capital of France? Answer in one word."
for name in sorted(os.listdir(folder)):
    d = os.path.join(folder, name)
    if not os.path.isfile(os.path.join(d, "config.json")):
        continue
    tok = AutoTokenizer.from_pretrained(d)
    if tok.chat_template:
        prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                         add_generation_prompt=True)
    else:
        prompt = "Q: " + question + "\nA:"
    ids = tok(prompt, add_special_tokens=True)["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        x = torch.tensor([ids])
        logits = model(x).logits[0, -1]
        out = model.generate(x, max_new_tokens=24, do_sample=False)[0, len(ids):]
    json.dump({"ids": ids, "logits": logits.numpy().round(5).tolist(), "greedy": out.tolist()},
              open(d + ".expected.json", "w"))
    print(name, repr(tok.decode(out)))
