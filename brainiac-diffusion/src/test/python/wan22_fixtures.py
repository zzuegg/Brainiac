# Writes a tiny random Wan 2.2 TI2V-style pipeline (the 16x autoencoder with residual levels and
# 2x2 patches) in diffusers layout, the autoencoder also in Wan's own layout, and what diffusers
# computes with them, for brainiac-diffusion's tests.
#   python wan22_fixtures.py <brainiac-diffusion/src/test/resources/tiny>
# Run wan_fixtures.py first: this reuses its tokenizer and umT5.
import os, sys, torch
from safetensors.torch import save_file
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
from transformers import T5TokenizerFast, UMT5EncoderModel

out = sys.argv[1]
torch.manual_seed(22)


def randomize(m, scale=0.2):
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * scale)
    return m.eval()


tokenizer = T5TokenizerFast.from_pretrained(out + "/wan/tokenizer")
umt5 = UMT5EncoderModel.from_pretrained(out + "/wan/text_encoder").eval()

mean = [0.1, -0.2, 0.3, -0.05]
std = [1.5, 0.8, 2.0, 1.1]
vae = randomize(AutoencoderKLWan(base_dim=8, decoder_base_dim=12, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                                 attn_scales=[], temperal_downsample=[False, True, True], latents_mean=mean,
                                 latents_std=std, is_residual=True, in_channels=12, out_channels=12, patch_size=2,
                                 scale_factor_temporal=4, scale_factor_spatial=16), 0.15)
transformer = randomize(WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
                                              in_channels=4, out_channels=4, text_dim=32, freq_dim=16, ffn_dim=40,
                                              num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                                              eps=1e-6), 0.15)
scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000,
                                    flow_shift=5.0)
pipe = WanPipeline(tokenizer=tokenizer, text_encoder=umt5, transformer=transformer, vae=vae, scheduler=scheduler,
                   expand_timesteps=True)
pipe.save_pretrained(out + "/wan22", safe_serialization=True)

ref = {}
with torch.no_grad():
    # the autoencoder: 9 frames of 64x96 <-> 3 latent frames of 4x6
    video = torch.rand(1, 3, 9, 64, 96) * 2 - 1
    ref["vae.video"] = video
    mu = vae.encode(video).latent_dist.mode()
    m = torch.tensor(mean).view(1, 4, 1, 1, 1)
    s = torch.tensor(std).view(1, 4, 1, 1, 1)
    ref["vae.latents"] = (mu - m) / s
    z = torch.randn(1, 4, 3, 4, 6)
    ref["vae.z"] = z
    ref["vae.decoded"] = vae.decode(z * s + m).sample

    noise = torch.randn(1, 4, 3, 4, 4)
    ref["pipe.noise"] = noise
    frames = pipe(prompt="a red fox runs in snow", negative_prompt="blurry", height=64, width=64, num_frames=9,
                  num_inference_steps=3, guidance_scale=5.0, latents=noise.clone(), max_sequence_length=16,
                  output_type="pt").frames
    ref["pipe.video"] = frames   # [1, frames, 3, h, w] in [0, 1]

save_file({k: v.contiguous().float().clone() for k, v in ref.items()}, out + "/wan22_reference.safetensors")
print({k: tuple(v.shape) for k, v in ref.items()})


# ---- the autoencoder in Wan 2.2's own names (Wan2.2_VAE.pth, ComfyUI's wan2.2_vae.safetensors)
def original_vae(sd, res=2):
    o = {}
    for k, v in sd.items():
        n = k
        n = n.replace("quant_conv", "conv1") if n.startswith("quant_conv") else n
        n = n.replace("post_conv1", "conv2").replace("post_quant_conv", "conv2")
        n = n.replace("encoder.conv_in", "encoder.conv1").replace("decoder.conv_in", "decoder.conv1")
        n = n.replace("norm_out.gamma", "head.0.gamma").replace("conv_out", "head.2")
        n = n.replace("mid_block.resnets.0.", "middle.0.").replace("mid_block.attentions.0.", "middle.1.")
        n = n.replace("mid_block.resnets.1.", "middle.2.")
        for d, kind, last in (("encoder.down_blocks.", "downsamples", res), ("decoder.up_blocks.", "upsamples", res + 1)):
            if n.startswith(d):
                rest = n[len(d):].split(".")
                level = rest[0]
                if rest[1] == "resnets":
                    tail = ".".join(rest[3:])
                    n = f"{d.split('.')[0]}.{kind}.{level}.{kind}.{rest[2]}.{tail}"
                else:   # downsampler / upsampler
                    tail = ".".join(rest[2:])
                    n = f"{d.split('.')[0]}.{kind}.{level}.{kind}.{last}.{tail}"
        for a, b in ((".norm1.gamma", ".residual.0.gamma"), (".conv1.", ".residual.2."), (".norm2.gamma", ".residual.3.gamma"),
                     (".conv2.", ".residual.6."), (".conv_shortcut.", ".shortcut.")):
            if ("samples." in n or "middle." in n) and a in n:
                n = n.replace(a, b)
        o[n] = v
    return o


ov = original_vae(vae.state_dict())
# check against diffusers' own conversion (scripts/convert_wan_to_diffusers.py, convert_vae_22), restated
import importlib.util
back = {}
for key, value in ov.items():
    nk = key
    if key.startswith("encoder.middle.") or key.startswith("decoder.middle."):
        idx = key.split(".")[2]
        nk = key.replace(f"middle.{idx}.", {"0": "mid_block.resnets.0.", "1": "mid_block.attentions.0.",
                                             "2": "mid_block.resnets.1."}[idx])
        nk = (nk.replace(".residual.0.gamma", ".norm1.gamma").replace(".residual.2.", ".conv1.")
              .replace(".residual.3.gamma", ".norm2.gamma").replace(".residual.6.", ".conv2."))
    elif key.startswith("encoder.head.") or key.startswith("decoder.head."):
        nk = key.replace("head.0.gamma", "norm_out.gamma").replace("head.2", "conv_out")
    elif key in ("conv1.weight", "conv1.bias"):
        nk = key.replace("conv1", "quant_conv")
    elif key in ("conv2.weight", "conv2.bias"):
        nk = key.replace("conv2", "post_quant_conv")
    elif key.startswith("encoder.conv1") or key.startswith("decoder.conv1"):
        nk = key.replace("conv1", "conv_in")
    elif key.startswith("encoder.downsamples.") or key.startswith("decoder.upsamples."):
        enc = key.startswith("encoder.")
        nk = key.replace("encoder.downsamples.", "encoder.down_blocks.").replace("decoder.upsamples.", "decoder.up_blocks.")
        if "residual" in nk or "shortcut" in nk:
            nk = nk.replace(".downsamples.", ".resnets.").replace(".upsamples.", ".resnets.")
            nk = (nk.replace(".residual.0.gamma", ".norm1.gamma").replace(".residual.2.", ".conv1.")
                  .replace(".residual.3.gamma", ".norm2.gamma").replace(".residual.6.", ".conv2.")
                  .replace(".shortcut.", ".conv_shortcut."))
        else:
            parts = nk.split(".")
            parts = parts[:3] + ["downsampler" if enc else "upsampler"] + parts[5:]
            nk = ".".join(parts)
    back[nk] = value
assert set(back) == set(vae.state_dict()), (set(back) ^ set(vae.state_dict()))
single = out + "/wan22_single"
os.makedirs(single, exist_ok=True)
save_file({k: v.contiguous() for k, v in ov.items()}, single + "/wan22_vae_tiny.safetensors")
print(len(ov), "original VAE weights")
