# Writes tiny random Wan image-to-video pipelines and what diffusers' WanImageToVideoPipeline
# makes with them, for brainiac-diffusion's tests:
#   ti2v   Wan 2.2 TI2V-5B style: the image's latents replace the first latent frame, whose
#          patches get timestep 0 (reuses the tiny wan22 pipeline)
#   i2v22  Wan 2.2 I2V-A14B style: a mask and the image's latents ride along as extra input
#          channels, two experts split at a boundary, no image encoder
#   i2v21  Wan 2.1 I2V style: the same channels plus CLIP vision features of the image
#   python wan_i2v_fixtures.py <brainiac-diffusion/src/test/resources/tiny>
# Run wan_fixtures.py and wan22_fixtures.py first.
import sys, numpy as np, torch
from PIL import Image
from safetensors.torch import save_file
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanTransformer3DModel
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, T5TokenizerFast, UMT5EncoderModel

out = sys.argv[1]
torch.manual_seed(2)
rng = np.random.default_rng(2)


def randomize(m, scale=0.15):
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * scale)
    return m.eval()


def picture(w, h):
    # smooth enough that resizing is not the whole story
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([128 + 100 * np.sin(x / 5.0), 128 + 100 * np.cos(y / 7.0), 128 + 60 * np.sin((x + y) / 9.0)], -1)
    return np.clip(base + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8)


def transformer(channels, image_dim=None):
    return randomize(WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
                                           in_channels=channels, out_channels=4, text_dim=32, freq_dim=16, ffn_dim=40,
                                           num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                                           eps=1e-6, image_dim=image_dim,
                                           added_kv_proj_dim=None if image_dim is None else 24))


tokenizer = T5TokenizerFast.from_pretrained(out + "/wan/tokenizer")
umt5 = UMT5EncoderModel.from_pretrained(out + "/wan/text_encoder").eval()
ref = {}

# ---- Wan 2.2 TI2V-5B: the tiny wan22 pipeline, run from an image
vae22 = AutoencoderKLWan.from_pretrained(out + "/wan22/vae").eval()
t22 = WanTransformer3DModel.from_pretrained(out + "/wan22/transformer").eval()
s22 = UniPCMultistepScheduler.from_pretrained(out + "/wan22/scheduler")
ti2v = WanImageToVideoPipeline(tokenizer=tokenizer, text_encoder=umt5, vae=vae22, scheduler=s22, transformer=t22,
                               expand_timesteps=True)
img = picture(64, 64)
ref["ti2v.image"] = torch.tensor(img).float()
with torch.no_grad():
    noise = torch.randn(1, 4, 3, 4, 4)
    ref["ti2v.noise"] = noise
    ref["ti2v.video"] = ti2v(image=Image.fromarray(img), prompt="a red fox runs in snow", negative_prompt="blurry",
                             height=64, width=64, num_frames=9, num_inference_steps=3, guidance_scale=5.0,
                             latents=noise.clone(), max_sequence_length=16, output_type="pt").frames

# ---- Wan 2.2 I2V-A14B: mask and image latents as channels, two experts
vae = AutoencoderKLWan.from_pretrained(out + "/wan/vae").eval()
scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000,
                                    flow_shift=5.0)
i2v22 = WanImageToVideoPipeline(tokenizer=tokenizer, text_encoder=umt5, vae=vae, scheduler=scheduler,
                                transformer=transformer(12), transformer_2=transformer(12), boundary_ratio=0.9)
i2v22.save_pretrained(out + "/wan_i2v22", safe_serialization=True)
img = picture(48, 32)
ref["i2v22.image"] = torch.tensor(img).float()
with torch.no_grad():
    noise = torch.randn(1, 4, 3, 4, 6)
    ref["i2v22.noise"] = noise
    ref["i2v22.video"] = i2v22(image=Image.fromarray(img), prompt="a lighthouse in a storm", negative_prompt="blurry",
                               height=32, width=48, num_frames=9, num_inference_steps=4, guidance_scale=4.0,
                               guidance_scale_2=3.0, latents=noise.clone(), max_sequence_length=16,
                               output_type="pt").frames

# ---- Wan 2.1 I2V: plus CLIP vision features (the penultimate layer's states); diffusers splits
# image from text states assuming 512 text tokens, so this one pads the prompt to 512
clip = randomize(CLIPVisionModel(CLIPVisionConfig(hidden_size=20, intermediate_size=40, num_hidden_layers=3,
                                                  num_attention_heads=2, image_size=32, patch_size=8,
                                                  hidden_act="gelu", layer_norm_eps=1e-5)), 0.2)
processor = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
i2v21 = WanImageToVideoPipeline(tokenizer=tokenizer, text_encoder=umt5, vae=vae, scheduler=scheduler,
                                image_processor=processor, image_encoder=clip, transformer=transformer(12, 20))
i2v21.save_pretrained(out + "/wan_i2v21", safe_serialization=True)
img = picture(48, 32)
ref["i2v21.image"] = torch.tensor(img).float()
with torch.no_grad():
    pixels = processor(images=Image.fromarray(img), return_tensors="pt").pixel_values
    ref["i2v21.pixels"] = pixels
    ref["i2v21.features"] = clip(pixel_values=pixels, output_hidden_states=True).hidden_states[-2]
    noise = torch.randn(1, 4, 3, 4, 6)
    ref["i2v21.noise"] = noise
    ref["i2v21.video"] = i2v21(image=Image.fromarray(img), prompt="a lighthouse in a storm", negative_prompt="blurry",
                               height=32, width=48, num_frames=9, num_inference_steps=3, guidance_scale=5.0,
                               latents=noise.clone(), max_sequence_length=512, output_type="pt").frames

# ---- Pillow's resampling, which both pipelines resize their pictures with
src = Image.fromarray(picture(64, 48))
ref["pil.source"] = torch.tensor(np.asarray(src)).float()
ref["pil.bicubic"] = torch.tensor(np.asarray(src.resize((37, 23), Image.BICUBIC))).float()
ref["pil.lanczos"] = torch.tensor(np.asarray(src.resize((90, 50), Image.LANCZOS))).float()
ref["pil.mixed"] = torch.tensor(np.asarray(src.resize((100, 30), Image.LANCZOS))).float()

save_file({k: v.contiguous().float().clone() for k, v in ref.items()}, out + "/wan_i2v_reference.safetensors")
print({k: tuple(v.shape) for k, v in ref.items()})
