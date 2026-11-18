"""Writes tiny random BART, mBART and Marian folders (bart, mbart, marian) with encoder states,
teacher-forced logits and greedy generations from transformers (expected.safetensors); run
from this folder."""
import torch
from safetensors.torch import save_file
from transformers import (BartConfig, BartForConditionalGeneration, MBartConfig, MBartForConditionalGeneration,
                          MarianConfig, MarianMTModel)

torch.manual_seed(0)
ids = torch.tensor([[5, 17, 42, 9, 2, 1], [7, 3, 2, 1, 1, 1]])
mask = (ids != 1).long()
dec = torch.tensor([[2, 11, 23, 4], [2, 8, 8, 30]])
base = dict(vocab_size=64, d_model=32, encoder_layers=2, decoder_layers=3, encoder_attention_heads=4,
            decoder_attention_heads=2, encoder_ffn_dim=48, decoder_ffn_dim=40, max_position_embeddings=32,
            dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, pad_token_id=1, bos_token_id=0,
            eos_token_id=2, decoder_start_token_id=2, forced_bos_token_id=None, forced_eos_token_id=None)
cases = {
    "bart": BartForConditionalGeneration(BartConfig(**base)),
    "mbart": MBartForConditionalGeneration(MBartConfig(**base, scale_embedding=True, activation_function="relu")),
    "marian": MarianMTModel(MarianConfig(**{**base, "decoder_start_token_id": 1}, activation_function="swish",
                                         scale_embedding=True)),
}
for name, model in cases.items():
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n and n.endswith("weight"):
                p.copy_(1 + torch.randn_like(p) * 0.1)
            elif "embed_positions" in n and name == "marian":
                pass
            else:
                p.copy_(torch.randn_like(p) * (1.0 if "embed" in n or "shared" in n else 0.3))
        model.final_logits_bias.copy_(torch.randn_like(model.final_logits_bias))
    model.eval()
    model.save_pretrained(name)
    if name == "bart":
        # as facebook/bart-large-cnn: the tied table only under the decoder's name
        from safetensors.torch import load_file
        w = load_file(name + "/model.safetensors")
        w["model.decoder.embed_tokens.weight"] = w.pop("model.shared.weight")
        save_file(w, name + "/model.safetensors", metadata={"format": "pt"})
    with torch.no_grad():
        enc = model.get_encoder()(input_ids=ids, attention_mask=mask).last_hidden_state
        logits = model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec).logits
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6, do_sample=False, num_beams=1,
                             min_length=0, no_repeat_ngram_size=0, forced_bos_token_id=None,
                             forced_eos_token_id=None, bad_words_ids=None)
    out = {"encoded": enc, "logits": logits, "generated": gen.float()}
    save_file({k: v.contiguous() for k, v in out.items()}, name + "/expected.safetensors")
    print(name, gen.tolist())
