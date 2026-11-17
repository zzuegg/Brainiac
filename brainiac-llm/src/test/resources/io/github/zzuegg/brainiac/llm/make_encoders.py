"""Tiny BERT and XLM-R encoders with transformers' outputs, for EncoderTest.

Run with transformers 5: python make_encoders.py
"""
import json, os, torch
from transformers import (BertConfig, BertModel, BertForSequenceClassification, BertTokenizerFast,
                          XLMRobertaConfig, XLMRobertaForSequenceClassification)

here = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)
words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "cat", "sat", "on", "mat", "a", "dog", "ran",
         "##s", "fast", "is", "where", "?", ".", "kitten", "##ing", "play", "stock", "market"]


def dump(name, model, tok=None, extra=None):
    d = os.path.join(here, name)
    model.save_pretrained(d, safe_serialization=True)
    if tok is not None:
        tok.save_pretrained(d)
    with open(os.path.join(d, "expected.json"), "w") as f:
        json.dump(extra, f)


def run(model, ids, mask, types=None):
    with torch.no_grad():
        return model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask),
                     token_type_ids=None if types is None else torch.tensor(types))


ids = [[2, 5, 6, 7, 8, 5, 9, 3], [2, 11, 12, 14, 3, 0, 0, 0]]
mask = [[1] * 8, [1, 1, 1, 1, 1, 0, 0, 0]]
types = [[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0, 0, 0]]

tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(words)}, do_lower_case=True)

cfg = BertConfig(vocab_size=len(words), hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                 intermediate_size=48, max_position_embeddings=16, type_vocab_size=2)
bert = BertModel(cfg).eval()
out = run(bert, ids, mask, types)
enc = tok("the cat sat", "on the mat")
dump("bert", bert, tok, {"ids": ids, "mask": mask, "types": types,
                         "hidden": out.last_hidden_state.flatten().tolist(),
                         "pooled": out.pooler_output.flatten().tolist(),
                         "pair_ids": enc["input_ids"], "pair_types": enc["token_type_ids"]})

rerank = BertForSequenceClassification(BertConfig(**{**cfg.to_dict(), "num_labels": 1})).eval()
out = run(rerank, ids, mask, types)
dump("bert-rerank", rerank, tok, {"ids": ids, "mask": mask, "types": types,
                                  "logits": out.logits.flatten().tolist()})

xcfg = XLMRobertaConfig(vocab_size=len(words), hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                        intermediate_size=48, max_position_embeddings=20, type_vocab_size=1, pad_token_id=0,
                        num_labels=1)
x = XLMRobertaForSequenceClassification(xcfg).eval()
out = run(x, ids, mask)
dump("xlmr-rerank", x, None, {"ids": ids, "mask": mask, "logits": out.logits.flatten().tolist()})
print("ok")
