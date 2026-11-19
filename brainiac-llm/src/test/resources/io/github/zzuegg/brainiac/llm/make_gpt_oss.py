# A tiny random gpt-oss written as llama.cpp's converter writes it (gpt-oss.gguf): experts in
# MXFP4 stacks with their biases, router bias, attention sinks, q/k/v/o biases, alternating
# sliding-window layers; and transformers' logits of the same (dequantized) weights.
#   pip install torch transformers gguf
#   python make_gpt_oss.py <this folder>
import json, sys, torch, numpy as np, gguf
import gguf.quants as q
import transformers as tf
out = sys.argv[1]
torch.manual_seed(0)
ids = [[1, 5, 9, 3, 7, 2, 11, 4, 6, 8]]

c = tf.GptOssConfig(vocab_size=48, hidden_size=64, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=16, num_local_experts=4, num_experts_per_tok=2,
                    sliding_window=4, max_position_embeddings=128, rms_norm_eps=1e-5, pad_token_id=0,
                    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
                    layer_types=["sliding_attention", "full_attention"], tie_word_embeddings=False)
c._attn_implementation = "eager"
model = tf.GptOssForCausalLM(c).eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        p.normal_(0, 0.2) if "norm" not in n else p.uniform_(0.5, 1.5)
sd = {k: v.float().numpy().copy() for k, v in model.state_dict().items()}

MX = gguf.GGMLQuantizationType.MXFP4
w = gguf.GGUFWriter(f"{out}/gpt-oss.gguf", arch="gpt-oss")
w.add_context_length(128); w.add_embedding_length(64); w.add_block_count(2); w.add_feed_forward_length(32)
w.add_head_count(4); w.add_head_count_kv(2); w.add_layer_norm_rms_eps(1e-5); w.add_vocab_size(48)
w.add_key_length(16); w.add_value_length(16)
w.add_uint32("gpt-oss.expert_count", 4); w.add_uint32("gpt-oss.expert_used_count", 2)
w.add_uint32("gpt-oss.expert_feed_forward_length", 32); w.add_uint32("gpt-oss.attention.sliding_window", 4)
w.add_float32("gpt-oss.rope.freq_base", 10000.0)


def put(name, arr, mxfp4=False):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if mxfp4:
        qv = q.quantize(arr, MX)
        w.add_tensor(name, qv, raw_dtype=MX)
        return q.dequantize(qv, MX).reshape(arr.shape)
    w.add_tensor(name, arr)
    return arr


put("token_embd.weight", sd["model.embed_tokens.weight"])
put("output_norm.weight", sd["model.norm.weight"])
put("output.weight", sd["lm_head.weight"])
f = 32
for i in range(2):
    p, g = f"model.layers.{i}.", f"blk.{i}."
    put(g + "attn_norm.weight", sd[p + "input_layernorm.weight"])
    put(g + "post_attention_norm.weight", sd[p + "post_attention_layernorm.weight"])
    for hf, gn in [("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")]:
        put(g + gn + ".weight", sd[p + "self_attn." + hf + ".weight"])
        put(g + gn + ".bias", sd[p + "self_attn." + hf + ".bias"])
    put(g + "attn_sinks.weight", sd[p + "self_attn.sinks"])
    put(g + "ffn_gate_inp.weight", sd[p + "mlp.router.weight"])
    put(g + "ffn_gate_inp.bias", sd[p + "mlp.router.bias"])
    # transformers keeps gate and up interleaved, [experts, in, 2 * ffn]; llama.cpp splits them
    # and stores each expert as [out, in]
    gu, gub = sd[p + "mlp.experts.gate_up_proj"], sd[p + "mlp.experts.gate_up_proj_bias"]
    if gu.shape[1] != 64:
        gu = gu.transpose(0, 2, 1)   # a layout of [experts, 2 * ffn, in]
    gate = put(g + "ffn_gate_exps.weight", gu[:, :, ::2].transpose(0, 2, 1), mxfp4=True)
    up = put(g + "ffn_up_exps.weight", gu[:, :, 1::2].transpose(0, 2, 1), mxfp4=True)
    put(g + "ffn_gate_exps.bias", gub[:, ::2]); put(g + "ffn_up_exps.bias", gub[:, 1::2])
    down = sd[p + "mlp.experts.down_proj"]
    flip = down.shape[1] == f   # [experts, ffn, hidden]: in, out
    d = put(g + "ffn_down_exps.weight", down.transpose(0, 2, 1) if flip else down, mxfp4=True)
    put(g + "ffn_down_exps.bias", sd[p + "mlp.experts.down_proj_bias"])
    # the reference multiplies the values the file holds
    both = np.empty_like(gu)
    both[:, :, ::2] = gate.transpose(0, 2, 1)
    both[:, :, 1::2] = up.transpose(0, 2, 1)
    sd[p + "mlp.experts.gate_up_proj"] = both if sd[p + "mlp.experts.gate_up_proj"].shape == both.shape \
        else both.transpose(0, 2, 1)
    sd[p + "mlp.experts.down_proj"] = d.transpose(0, 2, 1) if flip else d
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

model.load_state_dict({k: torch.tensor(v) for k, v in sd.items()})
with torch.no_grad():
    logits = model(torch.tensor(ids)).logits[0]
json.dump({"ids": ids[0], "logits": logits.float().numpy().round(6).tolist()},
          open(f"{out}/gpt-oss.gguf.expected.json", "w"))
print("wrote gpt-oss.gguf", logits.shape)
