"""GPTQ and AWQ checkpoints of the tiny qwen2 model, packed as AutoGPTQ (v1, with act-order
g_idx) and AutoAWQ (GEMM) pack them, with transformers' logits of the dequantized model.

Run with transformers 5: python make_quantized.py
"""
import json, os, shutil, torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

here = os.path.dirname(os.path.abspath(__file__))
GROUP = 16
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]
ids = [1, 5, 9, 3, 7, 2, 11, 4]


def quantize(w, groups):
    """Asymmetric 4-bit round-to-nearest per (group, output): w [out, in], groups [in] -> q, scale, zero."""
    out, n = w.shape
    g = int(groups.max()) + 1
    scale = torch.zeros(g, out)
    zero = torch.zeros(g, out, dtype=torch.int64)
    for k in range(g):
        cols = w[:, groups == k]
        lo = torch.clamp(cols.min(1).values, max=0)
        hi = torch.clamp(cols.max(1).values, min=0)
        s = torch.clamp((hi - lo) / 15, min=1e-8)
        scale[k] = s
        # at least 1: GPTQ v1 stores zero - 1 in four bits
        zero[k] = torch.clamp(torch.round(-lo / s), 1, 15).long()
    q = torch.clamp(torch.round(w / scale[groups].t()) + zero[groups].t(), 0, 15).long()   # [out, in]
    return q, scale.half(), zero


def to_int32(x):
    return torch.from_numpy(x.numpy().astype("uint32").view("int32"))


def pack_rows(q, per):   # q [rows, cols] -> [rows / per, cols], value j of a group at bits 4j
    out = torch.zeros(q.shape[0] // per, q.shape[1], dtype=torch.int64)
    for j in range(per):
        out |= (q[j::per] & 0xF) << (4 * j)
    return to_int32(out)


def pack_cols(q, order=None):   # q [rows, cols] -> [rows, cols / 8]
    out = torch.zeros(q.shape[0], q.shape[1] // 8, dtype=torch.int64)
    for i in range(8):
        out |= (q[:, (order[i] if order else i)::8] & 0xF) << (4 * i)
    return to_int32(out)


def build(name, awq):
    model = AutoModelForCausalLM.from_pretrained(os.path.join(here, "qwen2"), dtype=torch.float32).eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    tensors = {}
    gen = torch.Generator().manual_seed(1)
    for key, w in state.items():
        if not any(p in key for p in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                                      "gate_proj.weight", "up_proj.weight", "down_proj.weight"]):
            continue
        prefix = key[:-len("weight")]
        out, n = w.shape
        if awq:
            groups = torch.arange(n) // GROUP
        else:   # act-order: groups follow a permutation of the inputs
            perm = torch.randperm(n, generator=gen)
            groups = torch.empty(n, dtype=torch.int64)
            groups[perm] = torch.arange(n) // GROUP
        q, scale, zero = quantize(w, groups)
        deq = (q - zero[groups].t()).float() * scale.float()[groups].t()
        state[key] = deq   # the model transformers runs for the expected logits
        if awq:
            tensors[prefix + "qweight"] = pack_cols(q.t(), AWQ_ORDER)          # [in, out / 8]
            tensors[prefix + "qzeros"] = pack_cols(zero, AWQ_ORDER)            # [groups, out / 8]
        else:
            tensors[prefix + "qweight"] = pack_rows(q.t(), 8)                  # [in / 8, out]
            tensors[prefix + "qzeros"] = pack_cols(zero - 1)                   # v1 stores zero - 1
            tensors[prefix + "g_idx"] = groups.int()
        tensors[prefix + "scales"] = scale
    for key, v in state.items():
        if not any(key.startswith(p[:-len("qweight")]) and key.endswith("weight")
                   for p in tensors if p.endswith("qweight")):
            if key not in [p[:-len("qweight")] + "weight" for p in tensors if p.endswith("qweight")]:
                tensors[key] = v.contiguous()
    model.load_state_dict(state)
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0]
    d = os.path.join(here, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    config = json.load(open(os.path.join(here, "qwen2", "config.json")))
    config["quantization_config"] = ({"quant_method": "awq", "bits": 4, "group_size": GROUP, "zero_point": True,
                                      "version": "gemm"} if awq else
                                     {"quant_method": "gptq", "bits": 4, "group_size": GROUP, "desc_act": True,
                                      "sym": False, "checkpoint_format": "gptq"})
    json.dump(config, open(os.path.join(d, "config.json"), "w"), indent=2)
    save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(d, "model.safetensors"))
    json.dump({"ids": ids, "logits": logits.tolist()}, open(os.path.join(d, "expected.json"), "w"))


build("qwen2-gptq", False)
build("qwen2-awq", True)
print("ok")
