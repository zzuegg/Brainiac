# Tiny random DeepSeek-V2 and DeepSeek-V3 models (multi-head latent attention, fine-grained
# experts with shared experts, group-limited routing, YaRN), saved as Hugging Face would, with
# transformers' logits and greedy and beam answers. Kimi K2 is DeepSeek-V3 under another name.
#
# The GGUF copies are written by llama.cpp's own converter (convert_hf_to_gguf.py, which splits
# kv_b_proj into k_b and v_b for the absorbed attention), and their expected logits come from
# llama.cpp itself through llama-cpp-python, so they check the GGUF reading against an
# independent implementation:
#   pip install torch transformers gguf llama-cpp-python
#   python make_deepseek.py <this folder> <llama.cpp checkout>
import json, os, sys, torch
import transformers as tf

out = sys.argv[1]
llama_cpp_dir = sys.argv[2] if len(sys.argv) > 2 else None
torch.manual_seed(0)
ids = [[1, 5, 9, 3, 7, 2, 11, 4]]


def expected(model, path):
    with torch.no_grad():
        logits = model(torch.tensor(ids)).logits[0]
        e = {"ids": ids[0], "logits": logits.float().numpy().round(6).tolist()}
        e["greedy"] = model.generate(torch.tensor(ids), max_new_tokens=6, do_sample=False,
                                     pad_token_id=0)[0, len(ids[0]):].tolist()
        e["beam3"] = model.generate(torch.tensor(ids), max_new_tokens=6, do_sample=False, num_beams=3,
                                    pad_token_id=0, eos_token_id=None,
                                    length_penalty=1.0)[0, len(ids[0]):].tolist()
    json.dump(e, open(path, "w"))


def init(model):
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.2)
        for n, p in model.named_parameters():
            if "norm" in n:
                p.uniform_(0.5, 1.5)
        for n, b in model.named_buffers():
            if "e_score_correction_bias" in n:
                b.normal_(0, 0.1)
    return model.eval()


def save(name, model):
    d = f"{out}/{name}"
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    for f in os.listdir(d):
        if f not in ("config.json", "model.safetensors"):
            os.remove(f"{d}/{f}")
    expected(model, f"{d}/expected.json")
    return d


base = dict(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128, rms_norm_eps=1e-5, pad_token_id=0, bos_token_id=1, eos_token_id=2,
            kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=6, n_routed_experts=8,
            n_shared_experts=2, num_experts_per_tok=3, moe_intermediate_size=12, first_k_dense_replace=1,
            n_group=4, topk_group=2, tie_word_embeddings=False)

# V2: queries without a low-rank step, softmax routing limited to the groups with the largest
# expert, YaRN with mscale = mscale_all_dim (cos and sin unscaled, the scores scaled by mscale^2)
v2 = save("deepseek-v2", init(tf.DeepseekV2ForCausalLM(tf.DeepseekV2Config(
    **base, q_lora_rank=None, topk_method="group_limited_greedy", routed_scaling_factor=1.5, norm_topk_prob=False,
    rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
                     "original_max_position_embeddings": 32, "beta_fast": 32.0, "beta_slow": 1.0,
                     "mscale": 0.707, "mscale_all_dim": 0.707}))))

# V3 (and Kimi K2): low-rank queries, sigmoid routing with a selection bias, groups ranked by
# their two best experts, the chosen weights normalized then scaled
v3 = save("deepseek-v3", init(tf.DeepseekV3ForCausalLM(tf.DeepseekV3Config(
    # scoring_func and topk_method as DeepSeek's own config.json has them: transformers routes V3
    # by sigmoid either way, but llama.cpp's converter writes the gating function only from them
    **base, q_lora_rank=12, routed_scaling_factor=2.5, norm_topk_prob=True, num_mtp_layers=0,
    scoring_func="sigmoid", topk_method="noaux_tc",
    rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
                     "original_max_position_embeddings": 32, "beta_fast": 32.0, "beta_slow": 1.0,
                     "mscale": 1.0, "mscale_all_dim": 1.0}))))

# V3 as DeepSeek releases it: projections in 8-bit floats with a scale per 128 x 128 block
# (weight_scale_inv), the router, norms, embeddings and output layer as they were. The expected
# logits are transformers' from the weights the 8-bit floats stand for.
def fp8(name, src):
    from safetensors.torch import load_file, save_file
    sd = load_file(f"{src}/model.safetensors")
    out_sd, deq = {}, {}
    for k, v in sd.items():
        if k.endswith(".weight") and v.ndim == 2 and "embed" not in k and "lm_head" not in k and ".gate." not in k \
                and not k.endswith("mlp.gate.weight"):
            rows, cols = v.shape
            br, bc = (rows + 127) // 128, (cols + 127) // 128
            scale = torch.zeros(br, bc)
            q = torch.zeros_like(v, dtype=torch.float8_e4m3fn)
            d = torch.zeros_like(v)
            for i in range(br):
                for j in range(bc):
                    blk = v[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]
                    s_ = blk.abs().max().clamp(min=1e-12) / 448.0
                    qb = (blk / s_).to(torch.float8_e4m3fn)
                    q[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = qb
                    d[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = qb.float() * s_
                    scale[i, j] = s_
            out_sd[k] = q
            out_sd[k[:-len("weight")] + "weight_scale_inv"] = scale
            deq[k] = d
        else:
            out_sd[k] = v
            deq[k] = v
    d = f"{out}/{name}"
    os.makedirs(d, exist_ok=True)
    save_file(out_sd, f"{d}/model.safetensors", metadata={"format": "pt"})
    cfg = json.load(open(f"{src}/config.json"))
    cfg["quantization_config"] = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                  "weight_block_size": [128, 128]}
    json.dump(cfg, open(f"{d}/config.json", "w"), indent=2)
    # through a folder: transformers 5 stacks the experts when it loads, so their per-expert names
    # would not load into a model directly
    ref = f"{out}/{name}-dequantized"
    os.makedirs(ref, exist_ok=True)
    save_file(deq, f"{ref}/model.safetensors", metadata={"format": "pt"})
    json.dump(json.load(open(f"{src}/config.json")), open(f"{ref}/config.json", "w"))
    model = tf.DeepseekV3ForCausalLM.from_pretrained(ref).eval()
    import shutil
    shutil.rmtree(ref)
    with torch.no_grad():
        logits = model(torch.tensor(ids)).logits[0]
    json.dump({"ids": ids[0], "logits": logits.float().numpy().round(6).tolist()}, open(f"{d}/expected.json", "w"))


fp8("deepseek-v3-fp8", v3)

# ---------------------------------------------------------------- GGUF through llama.cpp

if llama_cpp_dir:
    sys.path.insert(0, llama_cpp_dir)
    sys.path.insert(0, os.path.join(llama_cpp_dir, "gguf-py"))
    import gguf
    import convert_hf_to_gguf as conv
    from conversion.deepseek import DeepseekV2Model

    def tiny_vocab(self):
        # the tiny models have no tokenizer: 48 made-up tokens and a merge llama.cpp insists on
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre("deepseek-v3")
        self.gguf_writer.add_token_list([f"<t{i}>" for i in range(self.hparams["vocab_size"])])
        self.gguf_writer.add_token_types([gguf.TokenType.NORMAL] * self.hparams["vocab_size"])
        self.gguf_writer.add_token_merges(["<t3> <t4>"])
        self.gguf_writer.add_bos_token_id(1)
        self.gguf_writer.add_eos_token_id(2)

    DeepseekV2Model.set_vocab = tiny_vocab
    from llama_cpp import Llama

    for name, d in [("deepseek-v2", v2), ("deepseek-v3", v3)]:
        gguf_path = f"{out}/{name}.gguf"
        sys.argv = ["convert_hf_to_gguf.py", d, "--outtype", "f32", "--outfile", gguf_path]
        conv.main()
        if name == "deepseek-v2":
            # llama.cpp tells V2-Lite's direct query projection by its 27 (or 26) blocks; with 3 it
            # looks for q_lora_rank, so this file is checked against transformers' logits instead
            e = json.load(open(f"{d}/expected.json"))
            json.dump({"ids": e["ids"], "logits": e["logits"]}, open(f"{gguf_path}.expected.json", "w"))
            continue
        llm = Llama(model_path=gguf_path, logits_all=True, n_ctx=64, n_batch=64, verbose=False, seed=0)
        llm.eval(ids[0])
        logits = [list(map(float, llm.scores[i])) for i in range(len(ids[0]))]
        json.dump({"ids": ids[0], "logits": [[round(x, 6) for x in row] for row in logits]},
                  open(f"{gguf_path}.expected.json", "w"))
