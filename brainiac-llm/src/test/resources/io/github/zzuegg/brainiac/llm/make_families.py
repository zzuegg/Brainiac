# Tiny random models of the other Llama-like families (Mistral, Qwen 3, Phi-3, Gemma 1, 2 and 3)
# saved as Hugging Face would, with transformers' logits, greedy and beam answers, and GGUF
# copies written the way llama.cpp's converter writes them (Q8_0 matrices).
#   pip install torch transformers gguf
#   python make_families.py <this folder>
import copy, json, os, sys, torch, numpy as np, gguf
import gguf.quants as q
import transformers as tf
out = sys.argv[1]
torch.manual_seed(0)
ids = [[1, 5, 9, 3, 7, 2, 11, 4]]


def expected(model, path, beams=True):
    with torch.no_grad():
        logits = model(torch.tensor(ids)).logits[0]
        e = {"ids": ids[0], "logits": logits.float().numpy().round(6).tolist()}
        if beams:
            e["greedy"] = model.generate(torch.tensor(ids), max_new_tokens=6, do_sample=False,
                                         pad_token_id=0)[0, len(ids[0]):].tolist()
            e["beam3"] = model.generate(torch.tensor(ids), max_new_tokens=6, do_sample=False, num_beams=3,
                                        pad_token_id=0, eos_token_id=None,
                                        length_penalty=1.0)[0, len(ids[0]):].tolist()
    json.dump(e, open(path, "w"))
    return logits


def save(name, model):
    d = f"{out}/{name}"
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    for f in os.listdir(d):
        if f not in ("config.json", "model.safetensors"):
            os.remove(f"{d}/{f}")
    expected(model, f"{d}/expected.json")
    return model


def init(model):
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.2)
        for n, p in model.named_parameters():
            if "norm" in n:
                # Gemma's norms scale by 1 + w
                p.uniform_(-0.5, 0.5) if "gemma" in type(model).__name__.lower() else p.uniform_(0.5, 1.5)
    return model.eval()


base = dict(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=128, rms_norm_eps=1e-5, pad_token_id=0)
mistral = save("mistral", init(tf.MistralForCausalLM(tf.MistralConfig(**base, sliding_window=4,
                                                                       tie_word_embeddings=False))))
qwen3 = save("qwen3", init(tf.Qwen3ForCausalLM(tf.Qwen3Config(**base, head_dim=8, tie_word_embeddings=True))))
phi3 = save("phi3", init(tf.Phi3ForCausalLM(tf.Phi3Config(**base, sliding_window=5, tie_word_embeddings=False,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5}))))
gemma = save("gemma", init(tf.GemmaForCausalLM(tf.GemmaConfig(**base, head_dim=8))))
gemma2 = save("gemma2", init(tf.Gemma2ForCausalLM(tf.Gemma2Config(**base, head_dim=8, sliding_window=4,
    query_pre_attn_scalar=16, attn_logit_softcapping=2.0, final_logit_softcapping=3.0))))
g3 = dict(base, num_hidden_layers=6)
gemma3 = save("gemma3", init(tf.Gemma3ForCausalLM(tf.Gemma3TextConfig(**g3, head_dim=8, sliding_window=4,
    query_pre_attn_scalar=8, rope_parameters={
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "linear", "rope_theta": 1000.0, "factor": 2.0}}))))

moe = dict(base, num_hidden_layers=2)
save("mixtral", init(tf.MixtralForCausalLM(tf.MixtralConfig(**moe, num_local_experts=4, num_experts_per_tok=2,
                                                            tie_word_embeddings=False))))
save("qwen3moe", init(tf.Qwen3MoeForCausalLM(tf.Qwen3MoeConfig(**moe, head_dim=8, num_experts=4, num_experts_per_tok=2,
                                                               moe_intermediate_size=16, norm_topk_prob=True,
                                                               tie_word_embeddings=False))))
save("qwen2moe", init(tf.Qwen2MoeForCausalLM(tf.Qwen2MoeConfig(**moe, num_experts=4, num_experts_per_tok=2,
                                                               moe_intermediate_size=16, shared_expert_intermediate_size=24,
                                                               norm_topk_prob=False, tie_word_embeddings=False))))

# ---------------------------------------------------------------- GGUF copies

LAYER = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
         "self_attn.o_proj": "attn_output", "self_attn.qkv_proj": "attn_qkv", "self_attn.q_norm": "attn_q_norm",
         "self_attn.k_norm": "attn_k_norm", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
         "mlp.gate_up_proj": "ffn_up", "mlp.down_proj": "ffn_down", "input_layernorm": "attn_norm"}


def gguf_name(arch, k):
    top = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
           "lm_head.weight": "output.weight"}
    if k in top:
        return top[k]
    parts = k.split(".")
    i, rest = parts[2], ".".join(parts[3:-1])
    gemma = arch in ("gemma2", "gemma3")
    names = dict(LAYER)
    names["post_attention_layernorm"] = "post_attention_norm" if gemma else "ffn_norm"
    names["pre_feedforward_layernorm"] = "ffn_norm"
    names["post_feedforward_layernorm"] = "post_ffw_norm"
    return f"blk.{i}.{names[rest]}.{parts[-1]}"


def write_gguf(name, arch, model, kv, ref_config=None):
    """The model as llama.cpp writes it, and the logits of its dequantized weights."""
    sd = {k: v.float().numpy() for k, v in model.state_dict().items()}
    w = gguf.GGUFWriter(f"{out}/{name}.gguf", arch=arch)
    c = model.config
    w.add_context_length(128); w.add_embedding_length(32); w.add_block_count(c.num_hidden_layers)
    w.add_feed_forward_length(64); w.add_head_count(4); w.add_head_count_kv(2); w.add_layer_norm_rms_eps(1e-5)
    w.add_vocab_size(48); hd = getattr(c, "head_dim", None) or 8; w.add_key_length(hd); w.add_value_length(hd)
    for k, v in kv.items():
        if isinstance(v, float):
            w.add_float32(k, v)
        elif isinstance(v, str):
            w.add_string(k, v)
        else:
            w.add_uint32(k, v)
    deq = {}
    for k, v in sd.items():
        if k == "lm_head.weight" and c.tie_word_embeddings:
            continue
        shift = 1 if arch.startswith("gemma") and k.endswith("norm.weight") else 0
        if v.ndim == 2 and "embed" not in k:
            qv = q.quantize(v, gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor(gguf_name(arch, k), qv, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            deq[k] = q.dequantize(qv, gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(gguf_name(arch, k), (v + shift).astype(np.float32))
            deq[k] = v
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    ref = type(model)(ref_config or c).eval()
    ref.load_state_dict({k: torch.tensor(v) for k, v in deq.items()}, strict=False)
    if c.tie_word_embeddings:
        ref.tie_weights()
    expected(ref, f"{out}/{name}.gguf.expected.json", beams=False)


write_gguf("qwen3", "qwen3", qwen3, {"qwen3.rope.freq_base": 10000.0})
write_gguf("phi3", "phi3", phi3, {"phi3.rope.freq_base": 10000.0, "phi3.rope.dimension_count": 4,
                                  "phi3.attention.sliding_window": 5})
# llama.cpp scales Gemma 2 and 3 queries by 1 / sqrt(head_dim) (but the 27B models)
g2 = copy.deepcopy(gemma2.config); g2.query_pre_attn_scalar = 8
write_gguf("gemma2", "gemma2", gemma2, {"gemma2.rope.freq_base": 10000.0, "gemma2.attention.sliding_window": 4,
                                        "gemma2.attn_logit_softcapping": 2.0,
                                        "gemma2.final_logit_softcapping": 3.0}, g2)
write_gguf("gemma3", "gemma3", gemma3, {"gemma3.rope.freq_base": 1000.0, "gemma3.attention.sliding_window": 4,
                                        "gemma3.rope.scaling.type": "linear", "gemma3.rope.scaling.factor": 2.0})

# ---------------------------------------------------------------- mixtures of experts as GGUF


def write_moe_gguf(name, arch, path, kv):
    """Qwen 2 and 3 MoE as llama.cpp writes them: each projection's experts in one 3-d stack."""
    model = getattr(tf, {"qwen2moe": "Qwen2MoeForCausalLM", "qwen3moe": "Qwen3MoeForCausalLM"}[arch]) \
        .from_pretrained(path).eval()
    sd = {k: v.float().numpy() for k, v in model.state_dict().items()}
    c = model.config
    w = gguf.GGUFWriter(f"{out}/{name}.gguf", arch=arch)
    w.add_context_length(128); w.add_embedding_length(32); w.add_block_count(c.num_hidden_layers)
    w.add_feed_forward_length(64); w.add_head_count(4); w.add_head_count_kv(2); w.add_layer_norm_rms_eps(1e-5)
    w.add_vocab_size(48); hd = getattr(c, "head_dim", None) or 8; w.add_key_length(hd); w.add_value_length(hd)
    w.add_uint32(f"{arch}.expert_count", c.num_experts); w.add_uint32(f"{arch}.expert_used_count", c.num_experts_per_tok)
    w.add_uint32(f"{arch}.expert_feed_forward_length", c.moe_intermediate_size)
    for k, v in kv.items():
        w.add_float32(k, v) if isinstance(v, float) else w.add_uint32(k, v)
    deq = {}

    def put(gname, hf_keys, q8=True):
        arr = np.stack([sd[k] for k in hf_keys]) if len(hf_keys) > 1 else sd[hf_keys[0]]
        if q8 and arr.ndim >= 2 and arr.shape[-1] % 32 == 0:
            qv = q.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor(gname, qv, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            d = q.dequantize(qv, gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(gname, arr.astype(np.float32))
            d = arr
        for i, k in enumerate(hf_keys):
            deq[k] = d[i] if len(hf_keys) > 1 else d

    put("token_embd.weight", ["model.embed_tokens.weight"], q8=False)
    put("output_norm.weight", ["model.norm.weight"], q8=False)
    if not c.tie_word_embeddings:
        put("output.weight", ["lm_head.weight"])
    for i in range(c.num_hidden_layers):
        p, g = f"model.layers.{i}.", f"blk.{i}."
        put(g + "attn_norm.weight", [p + "input_layernorm.weight"], q8=False)
        put(g + "ffn_norm.weight", [p + "post_attention_layernorm.weight"], q8=False)
        for hf, gn in [("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")]:
            put(g + gn + ".weight", [p + "self_attn." + hf + ".weight"])
            if p + "self_attn." + hf + ".bias" in sd:
                put(g + gn + ".bias", [p + "self_attn." + hf + ".bias"], q8=False)
        if arch == "qwen3moe":
            put(g + "attn_q_norm.weight", [p + "self_attn.q_norm.weight"], q8=False)
            put(g + "attn_k_norm.weight", [p + "self_attn.k_norm.weight"], q8=False)
        put(g + "ffn_gate_inp.weight", [p + "mlp.gate.weight"], q8=False)
        # transformers 5 keeps the experts stacked too: gate and up in one, gate first
        gu, f = sd[p + "mlp.experts.gate_up_proj"], c.moe_intermediate_size
        sd[p + "#gate"], sd[p + "#up"] = gu[:, :f], gu[:, f:]
        put(g + "ffn_gate_exps.weight", [p + "#gate"])
        put(g + "ffn_up_exps.weight", [p + "#up"])
        put(g + "ffn_down_exps.weight", [p + "mlp.experts.down_proj"])
        deq[p + "mlp.experts.gate_up_proj"] = np.concatenate([deq.pop(p + "#gate"), deq.pop(p + "#up")], axis=1)
        if arch == "qwen2moe":
            for hf, gn in [("gate_proj", "ffn_gate_shexp"), ("up_proj", "ffn_up_shexp"), ("down_proj", "ffn_down_shexp")]:
                put(g + gn + ".weight", [f"{p}mlp.shared_expert.{hf}.weight"])
            w.add_tensor(g + "ffn_gate_inp_shexp.weight", sd[p + "mlp.shared_expert_gate.weight"][0].astype(np.float32))
            deq[p + "mlp.shared_expert_gate.weight"] = sd[p + "mlp.shared_expert_gate.weight"]
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    model.load_state_dict({k: torch.tensor(np.asarray(v)) for k, v in deq.items()}, strict=False)
    expected(model, f"{out}/{name}.gguf.expected.json", beams=False)


write_moe_gguf("qwen3moe", "qwen3moe", f"{out}/qwen3moe", {"qwen3moe.rope.freq_base": 10000.0})
write_moe_gguf("qwen2moe", "qwen2moe", f"{out}/qwen2moe", {"qwen2moe.rope.freq_base": 10000.0,
                                                          "qwen2moe.expert_shared_feed_forward_length": 24})
