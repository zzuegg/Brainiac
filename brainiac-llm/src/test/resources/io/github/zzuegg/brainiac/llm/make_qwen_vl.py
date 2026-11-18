"""Writes a tiny random Qwen2.5-VL folder (vlm-qwen25: a windowed vision transformer with 2D
rotary positions, a patch merger and a Qwen2 decoder with 3D rotary positions) and the
features, logits and greedy generation transformers computes for vision-image.png
(expected.safetensors); run from this folder after make_vision.py."""
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLImageProcessor

torch.manual_seed(0)
START, PAD, END = 60, 61, 62
vision = dict(depth=3, hidden_size=32, intermediate_size=40, num_heads=4, patch_size=4, spatial_merge_size=2,
              temporal_patch_size=2, window_size=16, out_hidden_size=48, fullatt_block_indexes=[1], hidden_act="silu")
text = dict(vocab_size=64, hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=256, rms_norm_eps=1e-6,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 2, 2]},
            bos_token_id=1, eos_token_id=2, tie_word_embeddings=False)
config = Qwen2_5_VLConfig(vision_config=vision, text_config=text, image_token_id=PAD, video_token_id=63,
                          vision_start_token_id=START, vision_end_token_id=END, tie_word_embeddings=False)
model = Qwen2_5_VLForConditionalGeneration(config)
with torch.no_grad():
    for n, p in model.named_parameters():
        if ("norm" in n or "ln_q" in n) and n.endswith("weight"):
            p.copy_(1 + torch.randn_like(p) * 0.1)
        else:
            p.copy_(torch.randn_like(p) * (1.0 if "embed_tokens" in n else 0.2))
model.eval()
model.save_pretrained("vlm-qwen25")
# 45 x 61 pixels: the budget makes it 24 x 40, so a 6 x 10 patch grid, 3 x 5 tokens and
# windows of 2 x 2 tokens that do not divide it
proc = Qwen2VLImageProcessor(patch_size=4, merge_size=2, temporal_patch_size=2, min_pixels=16 * 16,
                             max_pixels=1000)
proc.save_pretrained("vlm-qwen25")
out = proc(images=Image.open("vision-image.png"), return_tensors="pt")
pixels, grid = out["pixel_values"], out["image_grid_thw"]
tokens = int(grid.prod()) // 4
ids = torch.tensor([[1, 5, 9, START] + [PAD] * tokens + [END, 17, 23, 4]])
types = (ids == PAD).int()   # as the processor marks image tokens, for the 3D positions
with torch.no_grad():
    feats = model.model.get_image_features(pixel_values=pixels, image_grid_thw=grid)
    feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
    if isinstance(feats, (list, tuple)):
        feats = torch.cat(list(feats), 0)
    logits = model(input_ids=ids, pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=types).logits
    gen = model.generate(input_ids=ids, pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=types,
                         max_new_tokens=8, do_sample=False)
save_file({"pixel_values": pixels, "grid": grid.float(), "features": feats.contiguous(), "logits": logits,
           "ids": ids.float(), "generated": gen[:, ids.shape[1]:].float()}, "vlm-qwen25/expected.safetensors")
print(grid.tolist(), pixels.shape, feats.shape, gen[:, ids.shape[1]:].tolist())

# Qwen2-VL: layer norms, a quick-GELU MLP, attention over the whole image in every block
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

torch.manual_seed(1)
vision = dict(depth=2, embed_dim=32, hidden_size=48, mlp_ratio=2, num_heads=4, patch_size=4, spatial_merge_size=2,
              temporal_patch_size=2, hidden_act="quick_gelu")
config = Qwen2VLConfig(vision_config=vision, text_config=text, image_token_id=PAD, video_token_id=63,
                       vision_start_token_id=START, vision_end_token_id=END, tie_word_embeddings=False)
model = Qwen2VLForConditionalGeneration(config)
with torch.no_grad():
    for n, p in model.named_parameters():
        if ("norm" in n or "ln_q" in n) and n.endswith("weight"):
            p.copy_(1 + torch.randn_like(p) * 0.1)
        else:
            p.copy_(torch.randn_like(p) * (1.0 if "embed_tokens" in n else 0.2))
model.eval()
model.save_pretrained("vlm-qwen2")
proc.save_pretrained("vlm-qwen2")
with torch.no_grad():
    feats = model.model.get_image_features(pixel_values=pixels, image_grid_thw=grid)
    feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
    if isinstance(feats, (list, tuple)):
        feats = torch.cat(list(feats), 0)
    logits = model(input_ids=ids, pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=types).logits
save_file({"pixel_values": pixels, "features": feats.contiguous(), "logits": logits, "ids": ids.float()},
          "vlm-qwen2/expected.safetensors")
print(feats.shape)
