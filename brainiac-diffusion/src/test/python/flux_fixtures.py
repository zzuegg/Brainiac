# Writes tiny random Flux pipelines (dev-like with guidance, schnell-like without) in diffusers
# layout, and what transformers and diffusers compute with them, for brainiac-diffusion's tests.
#   python flux_fixtures.py <brainiac-diffusion/src/test/resources/tiny>
# Run fixtures.py first: this reuses its CLIP vocabulary. Needs torch, diffusers, transformers,
# tokenizers and safetensors.
import json, os, sys, torch
from safetensors.torch import save_file
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from tokenizers import Tokenizer, Regex, models, normalizers, pre_tokenizers, processors, decoders
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5Config, T5EncoderModel, T5TokenizerFast

out = sys.argv[1]
torch.manual_seed(0)


def randomize(m, scale=0.2):
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * scale)
    return m.eval()


# ---- a small T5-style Unigram vocabulary
pieces = [("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁", -2.0)]
pieces += [(c, -5.0) for c in "abcdefghijklmnopqrstuvwxyz,.!'"]
pieces += [("▁" + w, -3.0 - 0.1 * i) for i, w in enumerate(
    ["a", "the", "cat", "red", "fox", "in", "snow", "photo", "of", "on", "mat", "hello", "sat"])]
pieces += [(w, -4.0) for w in ["at", "he", "ll", "ow", "ing", "er"]]
t5tok = Tokenizer(models.Unigram(pieces, unk_id=2, byte_fallback=False))
t5tok.normalizer = normalizers.Sequence([normalizers.Replace(Regex(" {2,}"), " ")])
t5tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
t5tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)
t5tok.post_processor = processors.TemplateProcessing(single="$A </s>", pair="$A </s> $B </s>",
                                                     special_tokens=[("</s>", 1)])
tok2 = T5TokenizerFast(tokenizer_object=t5tok, eos_token="</s>", pad_token="<pad>", unk_token="<unk>", extra_ids=0,
                       model_max_length=16)
tok1 = CLIPTokenizer(out + "/vocab/vocab.json", out + "/vocab/merges.txt", model_max_length=12)

t5_cases = []
for p in ["hello the cat", "a photo of a red fox in snow", "  the   cat sat on the mat!", "xyz", "fox " * 20]:
    t5_cases.append({"prompt": p, "ids": tok2(p, padding="max_length", max_length=16, truncation=True).input_ids})

clip = randomize(CLIPTextModel(CLIPTextConfig(
    vocab_size=600, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
    max_position_embeddings=12, eos_token_id=2, bos_token_id=1, pad_token_id=1, hidden_act="quick_gelu")))
t5 = randomize(T5EncoderModel(T5Config(vocab_size=len(pieces), d_model=32, d_kv=8, d_ff=48, num_layers=2,
                                       num_heads=4, relative_attention_num_buckets=32,
                                       relative_attention_max_distance=128, feed_forward_proj="gated-gelu",
                                       dropout_rate=0.0, layer_norm_epsilon=1e-6)), 0.3)
vae = randomize(AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2,
                              up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(16, 32), layers_per_block=1,
                              latent_channels=4, norm_num_groups=8, sample_size=16, use_quant_conv=False,
                              use_post_quant_conv=False, scaling_factor=0.3611, shift_factor=0.1159))


def transformer(guidance):
    t = FluxTransformer2DModel(patch_size=1, in_channels=16, num_layers=2, num_single_layers=2, attention_head_dim=16,
                               num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=32,
                               guidance_embeds=guidance, axes_dims_rope=(4, 6, 6))
    return randomize(t, 0.15)


dev_t = transformer(True)
schnell_t = transformer(False)
dev_s = FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
                                        base_image_seq_len=256, max_image_seq_len=4096)
schnell_s = FlowMatchEulerDiscreteScheduler(shift=1.0, use_dynamic_shifting=False)
dev = FluxPipeline(scheduler=dev_s, vae=vae, text_encoder=clip, tokenizer=tok1, text_encoder_2=t5, tokenizer_2=tok2,
                   transformer=dev_t)
schnell = FluxPipeline(scheduler=schnell_s, vae=vae, text_encoder=clip, tokenizer=tok1, text_encoder_2=t5,
                       tokenizer_2=tok2, transformer=schnell_t)
dev.save_pretrained(out + "/flux_dev", safe_serialization=True)
schnell.save_pretrained(out + "/flux_schnell", safe_serialization=True)
with open(out + "/t5_tokenizer_cases.json", "w") as f:
    json.dump(t5_cases, f, ensure_ascii=False, indent=1)


def pack(x):
    b, c, h, w = x.shape
    return x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def unpack(x, c, h, w):
    b = x.shape[0]
    return x.view(b, h // 2, w // 2, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


ref = {}
with torch.no_grad():
    ids = torch.tensor([[24, 5, 9, 1, 0, 0, 0, 0, 0, 0], [40, 41, 42, 43, 44, 45, 46, 47, 48, 1]])
    ref["t5.ids"] = ids.float()
    ref["t5.last"] = t5(input_ids=ids).last_hidden_state
    torch.manual_seed(1)
    x = torch.randn(2, 4, 6, 8)
    ctx = torch.randn(2, 5, 32)
    pooled = torch.randn(2, 32)
    sig = torch.tensor([1.0, 0.37])
    txt_ids = torch.zeros(5, 3)
    img_ids = FluxPipeline._prepare_latent_image_ids(2, 3, 4, "cpu", torch.float32)
    ref["dit.x"], ref["dit.ctx"], ref["dit.pooled"], ref["dit.sigma"] = x, ctx, pooled, sig
    ref["dit.guidance"] = torch.tensor([3.5, 1.0])
    for name, m, g in [("dev", dev_t, ref["dit.guidance"]), ("schnell", schnell_t, None)]:
        o = m(hidden_states=pack(x), encoder_hidden_states=ctx, pooled_projections=pooled, timestep=sig,
              img_ids=img_ids, txt_ids=txt_ids, guidance=g).sample
        ref["dit." + name] = unpack(o, 4, 6, 8)

    noise = torch.randn(1, 4, 16, 32)
    ref["pipe.noise"] = noise
    for name, pipe, steps, g in [("dev", dev, 4, 3.5), ("schnell", schnell, 3, 0.0)]:
        img = pipe(prompt="hello the cat", height=32, width=64, num_inference_steps=steps, guidance_scale=g,
                   latents=pack(noise.clone()), max_sequence_length=16, output_type="pt").images
        ref["pipe." + name] = img

save_file({k: v.contiguous().float().clone() for k, v in ref.items()}, out + "/flux_reference.safetensors")
print({k: tuple(v.shape) for k, v in ref.items()})
