"""Writes a tiny random LLaVA folder (vlm-llava: a CLIP image tower, a two-layer projector and
a Llama) with the image features, logits and greedy generation transformers computes for
vision-image.png (expected.safetensors); run from this folder after make_vision.py."""
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import CLIPImageProcessor, LlavaConfig, LlavaForConditionalGeneration

torch.manual_seed(0)
IMAGE = 60
vision = dict(model_type="clip_vision_model", hidden_size=32, intermediate_size=64, num_hidden_layers=3,
              num_attention_heads=4, image_size=32, patch_size=8)
text = dict(model_type="llama", vocab_size=64, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, rms_norm_eps=1e-5,
            tie_word_embeddings=False, bos_token_id=1, eos_token_id=2, pad_token_id=0)
config = LlavaConfig(vision_config=vision, text_config=text, image_token_index=IMAGE, vision_feature_layer=-2,
                     vision_feature_select_strategy="default", projector_hidden_act="gelu")
model = LlavaForConditionalGeneration(config)
with torch.no_grad():
    for n, p in model.named_parameters():
        if "norm" in n and n.endswith("weight"):
            p.copy_(1 + torch.randn_like(p) * 0.1)
        else:
            p.copy_(torch.randn_like(p) * (1.0 if "embed" in n else 0.2))
model.eval()
model.save_pretrained("vlm-llava")
proc = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
proc.save_pretrained("vlm-llava")
pixels = proc(images=Image.open("vision-image.png"), return_tensors="pt")["pixel_values"]
ids = torch.tensor([[1, 5, 9] + [IMAGE] * 16 + [17, 23, 4]])
with torch.no_grad():
    feats = model.get_image_features(pixel_values=pixels)
    feats = feats.pooler_output if hasattr(feats, "pooler_output") and feats.pooler_output is not None else feats
    if isinstance(feats, (list, tuple)):
        feats = torch.cat(list(feats), 0)
    logits = model(input_ids=ids, pixel_values=pixels).logits
    gen = model.generate(input_ids=ids, pixel_values=pixels, max_new_tokens=8, do_sample=False)
save_file({"pixel_values": pixels, "features": feats.reshape(-1, 48).contiguous(), "logits": logits,
           "ids": ids.float(), "generated": gen[:, ids.shape[1]:].float()}, "vlm-llava/expected.safetensors")
print(gen[:, ids.shape[1]:].tolist(), feats.shape)

# Gemma 3: a SigLIP tower whose 4 x 4 patches pool to 2 x 2 tokens, and a Gemma 3 decoder
from transformers import Gemma3Config, Gemma3ForConditionalGeneration, SiglipImageProcessor

torch.manual_seed(1)
SOFT = 62
vision = dict(model_type="siglip_vision_model", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
              num_attention_heads=4, image_size=32, patch_size=8, vision_use_head=False)
text = dict(model_type="gemma3_text", vocab_size=64, hidden_size=48, intermediate_size=96, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2, head_dim=12, max_position_embeddings=128,
            sliding_window=4, layer_types=["sliding_attention", "sliding_attention", "full_attention"],
            query_pre_attn_scalar=12, rms_norm_eps=1e-6, bos_token_id=2, eos_token_id=1, pad_token_id=0)
config = Gemma3Config(vision_config=vision, text_config=text, mm_tokens_per_image=4, boi_token_index=60,
                      eoi_token_index=61, image_token_index=SOFT)
model = Gemma3ForConditionalGeneration(config)
with torch.no_grad():
    for n, p in model.named_parameters():
        if "norm" in n and n.endswith("weight"):
            p.copy_(torch.randn_like(p) * 0.1)
        else:
            p.copy_(torch.randn_like(p) * (1.0 if "embed" in n else 0.2))
model.eval()
model.save_pretrained("vlm-gemma3")
proc = SiglipImageProcessor(size={"height": 32, "width": 32})
proc.save_pretrained("vlm-gemma3")
pixels = proc(images=Image.open("vision-image.png"), return_tensors="pt")["pixel_values"]
# BOS, text, boi, 4 soft tokens, eoi, text: the soft tokens see each other both ways, and
# the window of 4 cuts the image off from later text in the sliding layers
ids = torch.tensor([[2, 5, 9, 60] + [SOFT] * 4 + [61, 17, 23, 4, 8, 30]])
types = (ids == SOFT).long()
with torch.no_grad():
    feats = model.get_image_features(pixel_values=pixels)
    feats = feats.pooler_output if hasattr(feats, "pooler_output") and feats.pooler_output is not None else feats
    logits = model(input_ids=ids, pixel_values=pixels, token_type_ids=types).logits
    gen = model.generate(input_ids=ids, pixel_values=pixels, token_type_ids=types, max_new_tokens=8,
                         do_sample=False)
save_file({"pixel_values": pixels, "features": feats.reshape(-1, 48).contiguous(), "logits": logits,
           "ids": ids.float(), "generated": gen[:, ids.shape[1]:].float()}, "vlm-gemma3/expected.safetensors")
print(gen[:, ids.shape[1]:].tolist(), feats.shape)
