"""Writes llama.cpp-style projector files (mmproj.gguf) for the vlm-llava and vlm-gemma3
fixtures from their safetensors, the way llama.cpp's converters lay them out:

- vlm-llava: the older LLaVA layout (convert_image_encoder_to_gguf.py): projector "mlp",
  the last vision layer and the final norm cut off (LLaVA reads the second-to-last layer),
  fc1 stored as ffn_down and fc2 as ffn_up, the projector as mm.0 and mm.2.
- vlm-gemma3: the convert_hf_to_gguf.py layout: projector "gemma3", fc1 as ffn_up, the soft
  embedding norm with its 1 added, the input projection stored [text, vision].

Run from this folder after make_vlm.py.
"""
import json
import gguf
import numpy as np
from safetensors.numpy import load_file


def write(folder, arch_projector, tensors, vision, mean, std, extra=None):
    w = gguf.GGUFWriter(f"{folder}/mmproj.gguf", "clip")
    w.add_string("clip.projector_type", arch_projector)
    w.add_string("clip.vision.projector_type", arch_projector)
    w.add_bool("clip.has_vision_encoder", True)
    w.add_uint32("clip.vision.image_size", vision["image_size"])
    w.add_uint32("clip.vision.patch_size", vision["patch_size"])
    w.add_uint32("clip.vision.embedding_length", vision["hidden_size"])
    w.add_uint32("clip.vision.feed_forward_length", vision["intermediate_size"])
    w.add_uint32("clip.vision.block_count", vision["blocks"])
    w.add_uint32("clip.vision.attention.head_count", vision["num_attention_heads"])
    w.add_float32("clip.vision.attention.layer_norm_epsilon", vision["eps"])
    w.add_array("clip.vision.image_mean", mean)
    w.add_array("clip.vision.image_std", std)
    for k, v in (extra or {}).items():
        if isinstance(v, bool):
            w.add_bool(k, v)
        else:
            w.add_uint32(k, v)
    for name, t in tensors.items():
        w.add_tensor(name, np.ascontiguousarray(t.astype(np.float32)))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def tower(st, prefix, blocks, ffn_up_is_fc1):
    out = {}
    e = prefix + "embeddings."
    out["v.patch_embd.weight"] = st[e + "patch_embedding.weight"]
    if e + "patch_embedding.bias" in st:
        out["v.patch_embd.bias"] = st[e + "patch_embedding.bias"]
    if e + "class_embedding" in st:
        out["v.class_embd"] = st[e + "class_embedding"]
    out["v.position_embd.weight"] = st[e + "position_embedding.weight"]
    for i in range(blocks):
        p = f"{prefix}encoder.layers.{i}."
        b = f"v.blk.{i}."
        for hf, g in [("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
                      ("self_attn.out_proj", "attn_out"), ("layer_norm1", "ln1"), ("layer_norm2", "ln2"),
                      ("mlp.fc1", "ffn_up" if ffn_up_is_fc1 else "ffn_down"),
                      ("mlp.fc2", "ffn_down" if ffn_up_is_fc1 else "ffn_up")]:
            for s in ("weight", "bias"):
                out[b + g + "." + s] = st[p + hf + "." + s]
    return out


def config(folder):
    with open(f"{folder}/config.json") as f:
        return json.load(f)


# LLaVA: three layers in the model, the file keeps two
st = load_file("vlm-llava/model.safetensors")
vc = config("vlm-llava")["vision_config"]
pre = next(p for p in ("model.vision_tower.vision_model.", "vision_tower.vision_model.", "vision_tower.")
           if any(k.startswith(p + "embeddings.") for k in st))
proj = "model.multi_modal_projector." if any(k.startswith("model.multi_modal_projector.") for k in st) \
    else "multi_modal_projector."
t = tower(st, pre, vc["num_hidden_layers"] - 1, False)
t["v.pre_ln.weight"] = st[pre + "pre_layrnorm.weight"]
t["v.pre_ln.bias"] = st[pre + "pre_layrnorm.bias"]
t["mm.0.weight"] = st[proj + "linear_1.weight"]
t["mm.0.bias"] = st[proj + "linear_1.bias"]
t["mm.2.weight"] = st[proj + "linear_2.weight"]
t["mm.2.bias"] = st[proj + "linear_2.bias"]
with open("vlm-llava/preprocessor_config.json") as f:
    pp = json.load(f)
write("vlm-llava", "mlp", t, dict(vc, blocks=vc["num_hidden_layers"] - 1, eps=1e-5), pp["image_mean"],
      pp["image_std"], {"clip.has_llava_projector": True, "clip.use_gelu": False})

# Gemma 3
st = load_file("vlm-gemma3/model.safetensors")
vc = config("vlm-gemma3")["vision_config"]
pre = next(p for p in ("model.vision_tower.vision_model.", "vision_tower.vision_model.", "vision_tower.")
           if any(k.startswith(p + "embeddings.") for k in st))
proj = "model.multi_modal_projector." if any(k.startswith("model.multi_modal_projector.") for k in st) \
    else "multi_modal_projector."
t = tower(st, pre, vc["num_hidden_layers"], True)
t["v.post_ln.weight"] = st[pre + "post_layernorm.weight"]
t["v.post_ln.bias"] = st[pre + "post_layernorm.bias"]
t["mm.input_projection.weight"] = st[proj + "mm_input_projection_weight"].T
t["mm.soft_emb_norm.weight"] = st[proj + "mm_soft_emb_norm.weight"] + 1
with open("vlm-gemma3/preprocessor_config.json") as f:
    pp = json.load(f)
# llama.cpp pools Gemma 3 images to 256 tokens; this tiny tower makes 4, which the file says
write("vlm-gemma3", "gemma3", t, dict(vc, blocks=vc["num_hidden_layers"], eps=1e-6), pp["image_mean"],
      pp["image_std"], {"clip.vision.mm_tokens_per_image": config("vlm-gemma3")["mm_tokens_per_image"]})
print("ok")
