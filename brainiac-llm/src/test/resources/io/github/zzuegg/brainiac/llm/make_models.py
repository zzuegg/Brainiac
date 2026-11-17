# Tiny random Llama-family models saved as Hugging Face would, with reference logits from
# transformers, and a GGUF copy written the way llama.cpp's converter writes it.
import json, os, sys, torch, numpy as np, gguf
import gguf.quants as q
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
out = sys.argv[1]
torch.manual_seed(0)
ids = [[1, 5, 9, 3, 7, 2, 11, 4]]
def save(name, model):
    d = f"{out}/{name}"
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    for f in os.listdir(d):
        if f not in ("config.json", "model.safetensors"): os.remove(f"{d}/{f}")
    with torch.no_grad():
        logits = model(torch.tensor(ids)).logits[0]
        gen = model.generate(torch.tensor(ids), max_new_tokens=6, do_sample=False, pad_token_id=0)[0, len(ids[0]):]
        beam = model.generate(torch.tensor(ids), max_new_tokens=6, do_sample=False, num_beams=3, pad_token_id=0,
                              eos_token_id=None, length_penalty=1.0)[0, len(ids[0]):]
    json.dump({"ids": ids[0], "logits": logits.float().numpy().round(6).tolist(), "greedy": gen.tolist(), "beam3": beam.tolist()},
              open(f"{d}/expected.json", "w"))
    return model
def init(model):
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.2)
        for n, p in model.named_parameters():
            if "norm" in n: p.uniform_(0.5, 1.5)
    return model.eval()
base = dict(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=128, rms_norm_eps=1e-5, rope_theta=10000.0)
llama = save("llama", init(LlamaForCausalLM(LlamaConfig(**base, tie_word_embeddings=False))))
save("llama3-rope", init(LlamaForCausalLM(LlamaConfig(**base, tie_word_embeddings=True, rope_scaling={
    "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
    "original_max_position_embeddings": 16}))))
save("qwen2", init(Qwen2ForCausalLM(Qwen2Config(**base, tie_word_embeddings=True))))

# GGUF of the llama model: Q8_0 matrices, q/k permuted for ggml's rope, as convert_hf_to_gguf does
def permute(w, n_head):
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)
sd = {k: v.float().numpy() for k, v in llama.state_dict().items()}
w = gguf.GGUFWriter(f"{out}/llama.gguf", arch="llama")
w.add_context_length(128); w.add_embedding_length(32); w.add_block_count(2); w.add_feed_forward_length(64)
w.add_head_count(4); w.add_head_count_kv(2); w.add_rope_freq_base(10000.0); w.add_layer_norm_rms_eps(1e-5)
w.add_rope_dimension_count(8); w.add_vocab_size(48)
names = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
         "lm_head.weight": "output.weight"}
for i in range(2):
    for a, b in (("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
                 ("self_attn.o_proj", "attn_output"), ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"),
                 ("mlp.down_proj", "ffn_down"), ("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm")):
        names[f"model.layers.{i}.{a}.weight"] = f"blk.{i}.{b}.weight"
def unpermute(w, n_head):
    return w.reshape(n_head, w.shape[0] // n_head // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)
deq = {}
for k, v in sd.items():
    heads = 4 if "q_proj" in k else 2 if "k_proj" in k else 0
    if heads: v = permute(v, heads)
    if v.ndim == 2 and "embed" not in k:
        qv = q.quantize(v, gguf.GGMLQuantizationType.Q8_0)
        w.add_tensor(names[k], qv, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        v = q.dequantize(qv, gguf.GGMLQuantizationType.Q8_0)
    else:
        w.add_tensor(names[k], v.astype(np.float32))
    deq[k] = unpermute(v, heads) if heads else v
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
ref = LlamaForCausalLM(llama.config).eval()
ref.load_state_dict({k: torch.tensor(v) for k, v in deq.items()})
with torch.no_grad():
    logits = ref(torch.tensor(ids)).logits[0]
json.dump({"ids": ids[0], "logits": logits.numpy().round(6).tolist()}, open(f"{out}/llama.gguf.expected.json", "w"))
print("max diff gguf vs f32:", (logits - llama(torch.tensor(ids)).logits[0]).abs().max().item())
