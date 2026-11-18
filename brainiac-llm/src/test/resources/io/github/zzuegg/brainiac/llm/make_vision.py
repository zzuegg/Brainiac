"""Writes tiny random CLIP, SigLIP, ViT and DINOv2 folders (vision-*) with an image and the
outputs transformers computes for it (expected.safetensors); run from this folder."""
import numpy as np, torch
from PIL import Image
from safetensors.torch import save_file
from transformers import (CLIPConfig, CLIPModel, CLIPImageProcessor, SiglipConfig, SiglipModel, SiglipImageProcessor,
                          ViTConfig, ViTForImageClassification, ViTImageProcessor, Dinov2Config, Dinov2Model,
                          BitImageProcessor)

torch.manual_seed(0)
rng = np.random.default_rng(0)
# a smooth test image, 45 x 61, so resizing and cropping matter
y, x = np.mgrid[0:45, 0:61]
img = np.stack([(x * 4) % 256, (y * 5 + x) % 256, ((x - y) * 3) % 256], -1).astype(np.uint8)
img = (img * 0.7 + rng.integers(0, 70, img.shape)).astype(np.uint8)
Image.fromarray(img).save("vision-image.png")
image = Image.open("vision-image.png")

v = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, image_size=32, patch_size=8)
t = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, vocab_size=100,
         max_position_embeddings=16, eos_token_id=99, bos_token_id=98, pad_token_id=0)
ids = torch.tensor([[98, 5, 17, 42, 99, 0, 0, 0], [98, 7, 3, 99, 0, 0, 0, 0]])
mask = (torch.arange(8)[None] < torch.tensor([[5], [4]])).long()
siglip_ids = torch.tensor([[5, 17, 42, 1, 0, 0, 0, 0], [7, 3, 1, 0, 0, 0, 0, 0]])

def init(m):
    # random weights everywhere, including norms and the logit scale
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m.eval()

cases = {
    "vision-clip": (init(CLIPModel(CLIPConfig(vision_config=v, text_config=t, projection_dim=24))),
                    CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})),
    "vision-siglip": (init(SiglipModel(SiglipConfig(vision_config=v, text_config={**t, "eos_token_id": 1}))),
                      SiglipImageProcessor(size={"height": 32, "width": 32})),
    "vision-vit": (init(ViTForImageClassification(ViTConfig(**v, num_labels=3, id2label={0: "cat", 1: "dog", 2: "fish"},
                                                            label2id={"cat": 0, "dog": 1, "fish": 2}))),
                   ViTImageProcessor(size={"height": 32, "width": 32})),
    "vision-dinov2": (init(Dinov2Model(Dinov2Config(**{**v, "image_size": 40}))),
                      BitImageProcessor(size={"shortest_edge": 36}, crop_size={"height": 32, "width": 32},
                                        image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])),
}
for name, (model, proc) in cases.items():
    model.save_pretrained(name)
    proc.save_pretrained(name)
    pixels = proc(images=image, return_tensors="pt")["pixel_values"]
    out = {"pixel_values": pixels}
    with torch.no_grad():
        if name == "vision-clip":
            r = model(input_ids=ids, attention_mask=mask, pixel_values=pixels)
            out.update(image_embeds=r.image_embeds, text_embeds=r.text_embeds, logits=r.logits_per_image,
                       hidden=r.vision_model_output.last_hidden_state)
        elif name == "vision-siglip":
            r = model(input_ids=siglip_ids, pixel_values=pixels)
            out.update(image_embeds=r.image_embeds, text_embeds=r.text_embeds, logits=r.logits_per_image,
                       hidden=r.vision_model_output.last_hidden_state)
        elif name == "vision-vit":
            r = model(pixel_values=pixels, output_hidden_states=True)
            out.update(logits=r.logits, hidden=model.vit(pixels).last_hidden_state)
        else:
            r = model(pixel_values=pixels)
            out.update(hidden=r.last_hidden_state, image_embeds=r.pooler_output)
    save_file({k: v.float().clone().contiguous() for k, v in out.items()}, name + "/expected.safetensors")
