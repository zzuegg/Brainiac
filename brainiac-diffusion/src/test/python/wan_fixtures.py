# Writes a tiny random Wan 2.1 text-to-video pipeline in diffusers layout, a tiny image-to-video
# transformer, and what transformers and diffusers compute with them, for brainiac-diffusion's tests.
#   python wan_fixtures.py <brainiac-diffusion/src/test/resources/tiny>
# Needs torch, diffusers, transformers, tokenizers and safetensors.
import json, sys, torch
from safetensors.torch import save_file
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
from tokenizers import Tokenizer, Regex, models, normalizers, pre_tokenizers, processors, decoders
from transformers import T5TokenizerFast, UMT5Config, UMT5EncoderModel

out = sys.argv[1]
torch.manual_seed(0)


def randomize(m, scale=0.2):
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * scale)
    return m.eval()


# ---- a small umT5-style Unigram vocabulary
pieces = [("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁", -2.0)]
pieces += [(c, -5.0) for c in "abcdefghijklmnopqrstuvwxyz,.!'"]
pieces += [("▁" + w, -3.0 - 0.1 * i) for i, w in enumerate(
    ["a", "the", "cat", "red", "fox", "in", "snow", "video", "of", "runs", "mat", "hello", "sat", "blurry"])]
pieces += [(w, -4.0) for w in ["at", "he", "ll", "ow", "ing", "er"]]
tok = Tokenizer(models.Unigram(pieces, unk_id=2, byte_fallback=False))
tok.normalizer = normalizers.Sequence([normalizers.Replace(Regex(" {2,}"), " ")])
tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)
tok.post_processor = processors.TemplateProcessing(single="$A </s>", pair="$A </s> $B </s>",
                                                   special_tokens=[("</s>", 1)])
tokenizer = T5TokenizerFast(tokenizer_object=tok, eos_token="</s>", pad_token="<pad>", unk_token="<unk>",
                            extra_ids=0, model_max_length=16)

umt5 = randomize(UMT5EncoderModel(UMT5Config(vocab_size=len(pieces), d_model=32, d_kv=8, d_ff=48, num_layers=2,
                                             num_heads=4, relative_attention_num_buckets=32,
                                             relative_attention_max_distance=128, feed_forward_proj="gated-gelu",
                                             dropout_rate=0.0, layer_norm_epsilon=1e-6)), 0.3)

mean = [0.1, -0.2, 0.3, -0.05]
std = [1.5, 0.8, 2.0, 1.1]
vae = randomize(AutoencoderKLWan(base_dim=8, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                                 temperal_downsample=[False, True, True], latents_mean=mean, latents_std=std), 0.15)

# head size 12: rotary axes 4 (frames), 4 (rows), 4 (columns)
transformer = randomize(WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
                                              in_channels=4, out_channels=4, text_dim=32, freq_dim=16, ffn_dim=40,
                                              num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
                                              eps=1e-6), 0.15)
i2v = randomize(WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
                                      in_channels=9, out_channels=4, text_dim=32, freq_dim=16, ffn_dim=40,
                                      num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6,
                                      image_dim=20, added_kv_proj_dim=24), 0.15)
i2v.save_pretrained(out + "/wan_i2v_transformer", safe_serialization=True)

scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000,
                                    flow_shift=3.0)
pipe = WanPipeline(tokenizer=tokenizer, text_encoder=umt5, transformer=transformer, vae=vae, scheduler=scheduler)
pipe.save_pretrained(out + "/wan", safe_serialization=True)

ref = {}
with torch.no_grad():
    # umT5: masked padding, as Wan runs it
    ids = tokenizer(["hello the cat", "a video of a red fox in snow"], padding="max_length", max_length=12,
                    return_tensors="pt")
    ref["t5.ids"] = ids.input_ids.float()
    ref["t5.mask"] = ids.attention_mask.float()
    ref["t5.last"] = umt5(ids.input_ids, ids.attention_mask).last_hidden_state

    torch.manual_seed(1)
    x = torch.randn(2, 4, 3, 6, 8)
    ctx = torch.randn(2, 7, 32)
    t = torch.tensor([999.0, 371.0])
    ref["dit.x"], ref["dit.ctx"], ref["dit.t"] = x, ctx, t
    ref["dit.out"] = transformer(hidden_states=x, timestep=t, encoder_hidden_states=ctx).sample
    # a timestep per patch (Wan 2.2 keeps the first frame clean this way)
    tp = torch.full((2, 3 * 3 * 4), 500.0)
    tp[:, :12] = 0
    ref["dit.tp"] = tp
    ref["dit.out_tp"] = transformer(hidden_states=x, timestep=tp, encoder_hidden_states=ctx).sample
    xi = torch.randn(1, 9, 2, 4, 4)
    img = torch.randn(1, 5, 20)
    ctx_i = torch.randn(1, 512, 32)
    ref["i2v.x"], ref["i2v.img"], ref["i2v.ctx"] = xi, img, ctx_i
    ref["i2v.out"] = i2v(hidden_states=xi, timestep=torch.tensor([640.0]), encoder_hidden_states=ctx_i,
                         encoder_hidden_states_image=img).sample

    # the autoencoder: 9 frames <-> 3 latent frames
    video = torch.rand(1, 3, 9, 32, 48) * 2 - 1
    ref["vae.video"] = video
    mu = vae.encode(video).latent_dist.mode()
    m = torch.tensor(mean).view(1, 4, 1, 1, 1)
    s = torch.tensor(std).view(1, 4, 1, 1, 1)
    ref["vae.latents"] = (mu - m) / s
    z = torch.randn(1, 4, 3, 4, 6)
    ref["vae.z"] = z
    ref["vae.decoded"] = vae.decode(z * s + m).sample

    # UniPC on a made-up velocity model: v = x * (0.5 * sigma + 0.2) + 0.1
    sch = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000,
                                  flow_shift=3.0)
    sch.set_timesteps(6)
    ref["unipc.sigmas"] = sch.sigmas.clone()
    ref["unipc.timesteps"] = sch.timesteps.float()
    xs = torch.randn(1, 2, 3, 4)
    ref["unipc.noise"] = xs
    for i, tt in enumerate(sch.timesteps):
        sg = sch.sigmas[i]
        v = xs * (0.5 * sg + 0.2) + 0.1
        xs = sch.step(v, tt, xs).prev_sample
    ref["unipc.out"] = xs

    # the whole pipeline
    noise = torch.randn(1, 4, 3, 4, 4)
    ref["pipe.noise"] = noise
    video = pipe(prompt="a red fox runs in snow", negative_prompt="blurry", height=32, width=32, num_frames=9,
                 num_inference_steps=3, guidance_scale=5.0, latents=noise.clone(), max_sequence_length=16,
                 output_type="pt").frames
    ref["pipe.video"] = video   # [1, frames, 3, h, w] in [0, 1]

save_file({k: v.contiguous().float().clone() for k, v in ref.items()}, out + "/wan_reference.safetensors")
print({k: tuple(v.shape) for k, v in ref.items()})

# ---- the same networks in the original layout (Wan's repositories, ComfyUI's single files), checked
# by converting them back with diffusers' own converters
import os, re, shutil
from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers, convert_wan_vae_to_diffusers

TRANSFORMER_NAMES = [
    ("condition_embedder.time_embedder.linear_1", "time_embedding.0"),
    ("condition_embedder.time_embedder.linear_2", "time_embedding.2"),
    ("condition_embedder.text_embedder.linear_1", "text_embedding.0"),
    ("condition_embedder.text_embedder.linear_2", "text_embedding.2"),
    ("condition_embedder.time_proj", "time_projection.1"),
    (".attn1.", ".self_attn."), (".attn2.", ".cross_attn."), (".to_out.0.", ".o."), (".to_q.", ".q."),
    (".to_k.", ".k."), (".to_v.", ".v."), ("ffn.net.0.proj", "ffn.0"), ("ffn.net.2", "ffn.2"),
    ("proj_out", "head.head")]


def original_transformer(sd):
    out = {}
    for k, v in sd.items():
        if k == "scale_shift_table":
            n = "head.modulation"
        elif re.match(r"blocks\.\d+\.scale_shift_table$", k):
            n = k.replace("scale_shift_table", "modulation")
        elif re.match(r"blocks\.\d+\.norm2\.", k):
            n = k.replace(".norm2.", ".norm3.")
        else:
            n = k
            for a, b in TRANSFORMER_NAMES:
                n = n.replace(a, b)
        out[n] = v
    return out


def original_vae(sd, levels=4, res=2):
    out = {}
    for k, v in sd.items():
        n = k
        n = n.replace("quant_conv", "conv1") if n.startswith("quant_conv") else n
        n = n.replace("post_quant_conv", "conv2") if n.startswith("post_quant_conv") else n
        for c in ["encoder", "decoder"]:
            n = n.replace(c + ".conv_in", c + ".conv1").replace(c + ".norm_out.gamma", c + ".head.0.gamma")
            n = n.replace(c + ".conv_out", c + ".head.2")
            n = n.replace(c + ".mid_block.resnets.0.", c + ".middle.0.")
            n = n.replace(c + ".mid_block.attentions.0.", c + ".middle.1.")
            n = n.replace(c + ".mid_block.resnets.1.", c + ".middle.2.")
        n = n.replace("encoder.down_blocks.", "encoder.downsamples.")
        m = re.match(r"decoder\.up_blocks\.(\d+)\.(resnets\.(\d+)|upsamplers\.0)\.(.*)", n)
        if m:
            level = int(m.group(1))
            j = int(m.group(3)) if m.group(3) is not None else res + 1
            n = "decoder.upsamples.%d.%s" % (level * (res + 2) + j, m.group(4))
        for a, b in [(".norm1.gamma", ".residual.0.gamma"), (".conv1.", ".residual.2."),
                     (".norm2.gamma", ".residual.3.gamma"), (".conv2.", ".residual.6."),
                     (".conv_shortcut.", ".shortcut.")]:
            if ("downsamples." in n or "upsamples." in n or "middle.0." in n or "middle.2." in n):
                n = n.replace(a, b)
        out[n] = v
    return out


t_sd = {k: v.clone() for k, v in transformer.state_dict().items()}
o_sd = original_transformer(t_sd)
back = convert_wan_transformer_to_diffusers({k: v.clone() for k, v in o_sd.items()})
assert set(back) == set(t_sd), (set(back) ^ set(t_sd))
v_sd = {k: v.clone() for k, v in vae.state_dict().items()}
ov_sd = original_vae(v_sd)
back = convert_wan_vae_to_diffusers({k: v.clone() for k, v in ov_sd.items()})
assert set(back) == set(v_sd), (sorted(set(back) - set(v_sd)), sorted(set(v_sd) - set(back)))

single = out + "/wan_single"
os.makedirs(single, exist_ok=True)
save_file({"model.diffusion_model." + k: v.contiguous() for k, v in o_sd.items()}, single + "/wan_t2v_tiny.safetensors")
save_file({k: v.contiguous() for k, v in ov_sd.items()}, single + "/wan_vae_tiny.safetensors")
save_file({k: v.contiguous() for k, v in umt5.state_dict().items() if k != "encoder.embed_tokens.weight"},
          single + "/umt5_tiny.safetensors")
shutil.copy(out + "/wan/tokenizer/tokenizer.json", single + "/tokenizer.json")
print("original layouts check out")
