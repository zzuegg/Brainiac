# Writes tiny random Stable Diffusion 1.x and XL pipelines in diffusers layout, and what
# transformers and diffusers compute with them, for brainiac-diffusion's tests.
#   python fixtures.py <brainiac-diffusion/src/test/resources/tiny>
# Needs torch, diffusers, transformers and safetensors.
import json, os, sys, torch
from safetensors.torch import save_file
from diffusers import (UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler, StableDiffusionPipeline,
                       StableDiffusionXLPipeline)
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)


def randomize(m):
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
    return m.eval()


# ---- a small CLIP vocabulary (GPT-2 byte characters, their end-of-word forms, a few merges)
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


chars = list(bytes_to_unicode().values())
merges = ["h e", "l l", "he ll", "hell o</w>", "a t</w>", "c at</w>", "m at</w>", "s at</w>", "t h",
          "th e</w>", "a n", "o n</w>", "p h", "o t", "ph ot", "phot o", "' s</w>", "c a", "ca f"]
vocab = chars + [c + "</w>" for c in chars] + [m.replace(" ", "") for m in merges] + ["<|startoftext|>", "<|endoftext|>"]
os.makedirs(out + "/vocab", exist_ok=True)
with open(out + "/vocab/vocab.json", "w") as f:
    json.dump({t: i for i, t in enumerate(vocab)}, f, ensure_ascii=False)
with open(out + "/vocab/merges.txt", "w") as f:
    f.write("#version: 0.2\n" + "\n".join(merges) + "\n")
tok1 = CLIPTokenizer(out + "/vocab/vocab.json", out + "/vocab/merges.txt", model_max_length=12)
tok2 = CLIPTokenizer(out + "/vocab/vocab.json", out + "/vocab/merges.txt", pad_token="!", model_max_length=12)

cases = []
for p in ["Hello", "the cat sat on the mat", "  Photography's   1st  hello!!", "a café, 12 cats & <|endoftext|> dogs",
          "hello " * 20]:
    cases.append({"prompt": p, "ids": tok1(p, padding="max_length", max_length=16, truncation=True).input_ids,
                  "ids_pad_bang": tok2(p, padding="max_length", max_length=16, truncation=True).input_ids})
with open(out + "/tokenizer_cases.json", "w") as f:
    json.dump(cases, f, ensure_ascii=False, indent=1)

# ---- the schedules diffusers' Euler scheduler makes
sched = []
for spacing in ["leading", "trailing", "linspace"]:
    for karras in [False, True]:
        for steps in [1, 7, 30]:
            s = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                       steps_offset=1, timestep_spacing=spacing, use_karras_sigmas=karras)
            s.set_timesteps(steps)
            sched.append({"spacing": spacing, "karras": karras, "steps": steps, "timesteps": s.timesteps.tolist(),
                          "sigmas": s.sigmas.tolist(), "init": float(s.init_noise_sigma)})
with open(out + "/schedules.json", "w") as f:
    json.dump(sched, f, indent=0)

# ---- models
clip_kw = dict(vocab_size=600, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
               max_position_embeddings=12, projection_dim=24, eos_token_id=2, bos_token_id=1, pad_token_id=1)
enc_l = randomize(CLIPTextModel(CLIPTextConfig(hidden_act="quick_gelu", **clip_kw)))
enc_g = randomize(CLIPTextModelWithProjection(CLIPTextConfig(hidden_act="gelu", **clip_kw)))


def unet(**kw):
    return randomize(UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, block_out_channels=(16, 32),
                                          layers_per_block=1, norm_num_groups=4, **kw))


unet_sd = unet(down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
               cross_attention_dim=32, attention_head_dim=2)
unet_xl = unet(down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
               cross_attention_dim=64, attention_head_dim=(2, 4), transformer_layers_per_block=(1, 2),
               use_linear_projection=True, addition_embed_type="text_time", addition_time_embed_dim=8,
               projection_class_embeddings_input_dim=24 + 6 * 8)
vae = randomize(AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2,
                              up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(16, 32), layers_per_block=1,
                              latent_channels=4, norm_num_groups=8, sample_size=16))
scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", steps_offset=1,
                                   timestep_spacing="leading")

sd = StableDiffusionPipeline(vae=vae, text_encoder=enc_l, tokenizer=tok1, unet=unet_sd, scheduler=scheduler,
                             safety_checker=None, feature_extractor=None, requires_safety_checker=False)
sd.save_pretrained(out + "/sd", safe_serialization=True)
xl = StableDiffusionXLPipeline(vae=vae, text_encoder=enc_l, text_encoder_2=enc_g, tokenizer=tok1, tokenizer_2=tok2,
                               unet=unet_xl, scheduler=scheduler)
xl.save_pretrained(out + "/sdxl", safe_serialization=True)

ref = {}
ids = torch.tensor([[530, 5, 17, 300, 531, 531, 531, 531, 531, 531, 531, 531],
                    [530, 42, 531, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
with torch.no_grad():
    for name, m in [("text_encoder", enc_l), ("text_encoder_2", enc_g)]:
        o = m(input_ids=ids, output_hidden_states=True)
        ref[name + ".ids"] = ids.float()
        ref[name + ".last"] = o.last_hidden_state
        ref[name + ".penultimate"] = o.hidden_states[-2]
        ref[name + ".pooled"] = o.text_embeds if name.endswith("2") else o.pooler_output
    torch.manual_seed(1)
    x = torch.randn(2, 4, 8, 16)
    ref["unet.x"], ref["unet.ctx"], ref["unet.t"] = x, torch.randn(2, 5, 32), torch.tensor([981.0, 981.0])
    ref["unet.out"] = unet_sd(x, ref["unet.t"], encoder_hidden_states=ref["unet.ctx"]).sample
    ref["unet_xl.x"], ref["unet_xl.ctx"], ref["unet_xl.t"] = x.clone(), torch.randn(2, 5, 64), torch.tensor([500.0, 37.0])
    ref["unet_xl.pooled"] = torch.randn(2, 24)
    ref["unet_xl.time_ids"] = torch.tensor([[64., 128., 0., 0., 64., 128.]] * 2)
    ref["unet_xl.out"] = unet_xl(x, ref["unet_xl.t"], encoder_hidden_states=ref["unet_xl.ctx"], added_cond_kwargs={
        "text_embeds": ref["unet_xl.pooled"], "time_ids": ref["unet_xl.time_ids"]}).sample
    ref["vae.z"] = torch.randn(1, 4, 6, 10)
    ref["vae.decoded"] = vae.decode(ref["vae.z"]).sample
    ref["vae.image"] = torch.rand(1, 3, 12, 20) * 2 - 1
    ref["vae.mean"] = vae.encode(ref["vae.image"]).latent_dist.mean

    # whole pipelines from given noise, deterministic Euler sampling
    noise = torch.randn(1, 4, 8, 16)
    ref["pipe.noise"] = noise
    for name, pipe, neg in [("sd", sd, "a cat"), ("sdxl", xl, None), ("sdxl_neg", xl, "the mat")]:
        img = pipe(prompt="hello the cat", negative_prompt=neg, height=64, width=128, num_inference_steps=4,
                   guidance_scale=3.0, latents=noise.clone(), output_type="pt").images
        ref["pipe." + name] = img
    ref["pipe.sd_unguided"] = sd(prompt="hello the cat", height=64, width=128, num_inference_steps=3,
                                 guidance_scale=1.0, latents=noise.clone(), output_type="pt").images

save_file({k: v.contiguous().float().clone() for k, v in ref.items()}, out + "/reference.safetensors")
print({k: tuple(v.shape) for k, v in ref.items()})
