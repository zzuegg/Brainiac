# A Qwen2 tokenizer saved the slow way (vocab.json + merges.txt, no tokenizer.json), from the
# byte-level fixture's vocabulary, with ids from transformers' Qwen2TokenizerFast (which
# converts it as Qwen2Converter does).
# Run: python make_qwen2_slow.py <this folder>
import json, os, sys
from transformers import AutoTokenizer

here = sys.argv[1]
src = json.load(open(f"{here}/bytelevel-bpe.json"))
out = f"{here}/qwen2-slow"
os.makedirs(out, exist_ok=True)
json.dump(src["model"]["vocab"], open(f"{out}/vocab.json", "w"), ensure_ascii=False)
with open(f"{out}/merges.txt", "w") as f:
    f.write("#version: 0.2\n")
    for m in src["model"]["merges"]:
        f.write((m if isinstance(m, str) else " ".join(m)) + "\n")
specials = {str(a["id"]): {"content": a["content"], "lstrip": False, "normalized": False, "rstrip": False,
                           "single_word": False, "special": True} for a in src["added_tokens"]}
json.dump({"tokenizer_class": "Qwen2Tokenizer", "added_tokens_decoder": specials, "eos_token": "<|im_end|>",
           "bos_token": None}, open(f"{out}/tokenizer_config.json", "w"), indent=1)
tok = AutoTokenizer.from_pretrained(out)
assert type(tok).__name__ == "Qwen2TokenizerFast", type(tok)
texts = ["Hello world", "<|im_start|>assistant\nHi there, 12345 times!<|im_end|>\n<|im_start|>assistant\n",
         "  leading spaces and\ttabs\n\nnew lines", "Ünïcödé café 東京 ÀB", "it's they're I'll we've",
         "numbers 2026 and 3.14159", "é composed?"]
json.dump([{"text": t, "ids": tok(t)["input_ids"]} for t in texts], open(f"{out}/expected.json", "w"),
          ensure_ascii=False, indent=0)
print(open(f"{out}/expected.json").read()[:600])
