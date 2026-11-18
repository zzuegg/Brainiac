"""Writes tiny random T5 folders (t5-relu: original T5 with tied embeddings; t5-gated: Flan-T5
style gated-GELU, untied) with encoder states, teacher-forced logits and greedy generations
from transformers (expected.safetensors); run from this folder."""
import json, torch
from safetensors.torch import save_file
from transformers import T5Config, T5ForConditionalGeneration

torch.manual_seed(0)
ids = torch.tensor([[5, 17, 42, 9, 1, 0], [7, 3, 1, 0, 0, 0]])
mask = (ids != 0).long()
mask[0, 4] = 1
dec = torch.tensor([[0, 11, 23, 4], [0, 8, 8, 30]])
base = dict(vocab_size=64, d_model=32, d_kv=8, d_ff=48, num_layers=2, num_decoder_layers=3, num_heads=4,
            relative_attention_num_buckets=8, relative_attention_max_distance=16, dropout_rate=0.0,
            decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
cases = {
    "t5-relu": T5Config(**base, feed_forward_proj="relu", tie_word_embeddings=True),
    "t5-gated": T5Config(**base, feed_forward_proj="gated-gelu", tie_word_embeddings=False),
}
for name, config in cases.items():
    model = T5ForConditionalGeneration(config)
    with torch.no_grad():
        # random weights large enough that every layer matters (the embedding alone would
        # make a tied model repeat its input)
        for n, p in model.named_parameters():
            if "layer_norm" in n:
                p.copy_(1 + torch.randn_like(p) * 0.1)
            else:
                p.copy_(torch.randn_like(p) * (1.0 if n == "shared.weight" else 0.3))
        if not config.scale_decoder_outputs:
            # transformers 5 always ties T5's head; Flan-T5 checkpoints have their own,
            # unscaled, so give this one its own head as those do
            model.lm_head = torch.nn.Linear(32, 64, bias=False)
            model.lm_head.weight.copy_(torch.randn(64, 32) * 0.3)
    model.eval()
    model.save_pretrained(name)
    if not config.scale_decoder_outputs:
        weights = {k: v.clone().contiguous() for k, v in model.state_dict().items()
                   if not k.endswith("embed_tokens.weight")}
        save_file(weights, name + "/model.safetensors", metadata={"format": "pt"})
        c = json.load(open(name + "/config.json"))
        c.pop("scale_decoder_outputs")
        c["tie_word_embeddings"] = False
        json.dump(c, open(name + "/config.json", "w"), indent=2)
    with torch.no_grad():
        enc = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        logits = model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec).logits
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6, do_sample=False, num_beams=1,
                             min_new_tokens=0, use_cache=False)
    out = {"encoded": enc, "logits": logits, "generated": gen.float()}
    save_file({k: v.contiguous() for k, v in out.items()}, name + "/expected.safetensors")
    print(name, gen.tolist())
