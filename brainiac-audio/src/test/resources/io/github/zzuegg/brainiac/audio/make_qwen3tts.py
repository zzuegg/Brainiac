# Tiny random Qwen3-TTS models saved as Qwen/Qwen3-TTS-12Hz-* folders are, with reference
# outputs, for Qwen3TtsTest:
#   qwen3tts/speech_tokenizer  the 12 Hz codec: a Mimi encoder (transformers' MimiModel) and the
#                              decoder (split RVQ, causal transformer, ConvNeXt upsampling,
#                              SnakeBeta decoder), decoding in chunks of 300 frames
#   qwen3tts                   the talker (a Qwen3 transformer over codec plus text embeddings),
#                              the code predictor (a small Qwen3 transformer writing codebooks
#                              1..15 of each frame) and the ECAPA-TDNN speaker encoder
# The decoder, speaker encoder and generation loop are written here from the qwen-tts package's
# sources (qwen_tts 0.1.1, modeling_qwen3_tts.py and modeling_qwen3_tts_tokenizer_v2.py);
# transformers' Qwen3Model and MimiModel are used as they are. Generation is greedy, with the
# talker's repetition penalty, suppressed control tokens and two-token minimum.
# Run: python make_qwen3tts.py <this folder>   (torch, transformers 4.57, librosa, safetensors)
import json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from librosa.filters import mel as librosa_mel
from safetensors.torch import save_file
from transformers import MimiConfig, MimiModel, Qwen3Config, Qwen3Model

out = sys.argv[1]
torch.manual_seed(7)
np.random.seed(7)


def rnd(*shape, s=0.3):
    return torch.randn(*shape) * s


def save(module_dicts, path):
    tensors = {}
    for prefix, sd in module_dicts:
        for k, v in sd.items():
            tensors[prefix + k] = v.detach().contiguous().clone()
    save_file(tensors, path)


# ------------------------------------------------------------------ codec decoder

class CausalConv(nn.Module):
    def __init__(self, i, o, k, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(i, o, k, stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.k = (k - 1) * dilation + 1
        self.pad = self.k - stride

    def forward(self, x):
        n = (x.shape[-1] - self.k + self.pad) / self.stride + 1
        ideal = (math.ceil(n) - 1) * self.stride + (self.k - self.pad)
        return self.conv(F.pad(x, (self.pad, ideal - x.shape[-1])))


class CausalTrans(nn.Module):
    def __init__(self, i, o, k, stride):
        super().__init__()
        self.conv = nn.ConvTranspose1d(i, o, k, stride=stride)
        self.right = k - stride

    def forward(self, x):
        y = self.conv(x)
        return y[..., : y.shape[-1] - self.right] if self.right > 0 else y


class ConvNeXt(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dwconv = CausalConv(d, d, 7, groups=d)
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.pwconv1 = nn.Linear(d, 4 * d)
        self.pwconv2 = nn.Linear(4 * d, d)
        self.gamma = nn.Parameter(rnd(d))

    def forward(self, x):
        h = self.dwconv(x).permute(0, 2, 1)
        h = self.pwconv2(F.gelu(self.pwconv1(self.norm(h))))
        return x + (self.gamma * h).permute(0, 2, 1)


class SnakeBeta(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.alpha = nn.Parameter(rnd(c, s=0.3))
        self.beta = nn.Parameter(rnd(c, s=0.3))

    def forward(self, x):
        a = torch.exp(self.alpha)[None, :, None]
        b = torch.exp(self.beta)[None, :, None]
        return x + (1.0 / (b + 1e-9)) * torch.sin(x * a) ** 2


class ResUnit(nn.Module):
    def __init__(self, d, dilation):
        super().__init__()
        self.act1 = SnakeBeta(d)
        self.conv1 = CausalConv(d, d, 7, dilation=dilation)
        self.act2 = SnakeBeta(d)
        self.conv2 = CausalConv(d, d, 1)

    def forward(self, x):
        return x + self.conv2(self.act2(self.conv1(self.act1(x))))


class DecBlock(nn.Module):
    def __init__(self, c, i):
        super().__init__()
        din, dout, r = c["decoder_dim"] // 2 ** i, c["decoder_dim"] // 2 ** (i + 1), c["upsample_rates"][i]
        self.block = nn.ModuleList([SnakeBeta(din), CausalTrans(din, dout, 2 * r, r)]
                                   + [ResUnit(dout, d) for d in (1, 3, 9)])

    def forward(self, x):
        for b in self.block:
            x = b(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(1 + rnd(d, s=0.1))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def rope(x, theta):
    d = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, d, 2).float() / d)
    pos = torch.arange(x.shape[-2]).float()
    f = torch.outer(pos, inv)
    emb = torch.cat([f, f], -1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * emb.cos() + torch.cat([-x2, x1], -1) * emb.sin()


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        h, n, kv = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
        self.n, self.kv, self.hd, self.window, self.theta = n, kv, h // n, c["sliding_window"], c["rope_theta"]
        self.q_proj = nn.Linear(h, n * self.hd, bias=False)
        self.k_proj = nn.Linear(h, kv * self.hd, bias=False)
        self.v_proj = nn.Linear(h, kv * self.hd, bias=False)
        self.o_proj = nn.Linear(n * self.hd, h, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        q = rope(self.q_proj(x).view(b, t, self.n, self.hd).transpose(1, 2), self.theta)
        k = rope(self.k_proj(x).view(b, t, self.kv, self.hd).transpose(1, 2), self.theta)
        v = self.v_proj(x).view(b, t, self.kv, self.hd).transpose(1, 2)
        k = k.repeat_interleave(self.n // self.kv, 1)
        v = v.repeat_interleave(self.n // self.kv, 1)
        i = torch.arange(t)[:, None]
        j = torch.arange(t)[None, :]
        allowed = (i - j >= 0) & (i - j < self.window)
        s = q @ k.transpose(2, 3) / math.sqrt(self.hd)
        s = s.masked_fill(~allowed, float("-inf"))
        o = (s.softmax(-1) @ v).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(o)


class Scale(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scale = nn.Parameter(rnd(d, s=0.5))


class Mlp(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c["hidden_size"]
        self.self_attn = Attn(c)
        self.mlp = Mlp(h, c["intermediate_size"])
        self.input_layernorm = RMSNorm(h, c["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(h, c["rms_norm_eps"])
        self.self_attn_layer_scale = Scale(h)
        self.mlp_layer_scale = Scale(h)

    def forward(self, x):
        x = x + self.self_attn_layer_scale.scale * self.self_attn(self.input_layernorm(x))
        return x + self.mlp_layer_scale.scale * self.mlp(self.post_attention_layernorm(x))


class PreTransformer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layers = nn.ModuleList([Layer(c) for _ in range(c["num_hidden_layers"])])
        self.norm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.input_proj = nn.Linear(c["latent_dim"], c["hidden_size"])
        self.output_proj = nn.Linear(c["hidden_size"], c["latent_dim"])

    def forward(self, x):
        x = self.input_proj(x)
        for l in self.layers:
            x = l(x)
        return self.output_proj(self.norm(x))


class Codebook(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.cluster_usage = nn.Parameter(torch.rand(n) * 2 + 0.5)
        self.embedding_sum = nn.Parameter(rnd(n, d, s=1.0))

    def decode(self, codes):
        return F.embedding(codes, self.embedding_sum / self.cluster_usage.clamp(min=1e-5)[:, None])


class VQ(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self._codebook = Codebook(n, d)


class RVQLayers(nn.Module):
    def __init__(self, q, n, d):
        super().__init__()
        self.layers = nn.ModuleList([VQ(n, d) for _ in range(q)])


class RVQ(nn.Module):
    def __init__(self, q, n, d, io):
        super().__init__()
        self.input_proj = nn.Conv1d(io, d, 1, bias=False)
        self.output_proj = nn.Conv1d(d, io, 1, bias=False)
        self.vq = RVQLayers(q, n, d)

    def decode(self, codes):  # [B, K, T]
        z = 0
        for i in range(codes.shape[1]):
            z = z + self.vq.layers[i]._codebook.decode(codes[:, i]).transpose(1, 2)
        return self.output_proj(z)


class SplitRVQ(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["codebook_dim"] // 2
        self.rvq_first = RVQ(1, c["codebook_size"], d, c["codebook_dim"])
        self.rvq_rest = RVQ(c["num_quantizers"] - 1, c["codebook_size"], d, c["codebook_dim"])

    def decode(self, codes):
        return self.rvq_first.decode(codes[:, :1]) + self.rvq_rest.decode(codes[:, 1:])


class Decoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.total = int(np.prod(c["upsample_rates"] + c["upsampling_ratios"]))
        self.pre_transformer = PreTransformer(c)
        self.quantizer = SplitRVQ(c)
        self.pre_conv = CausalConv(c["codebook_dim"], c["latent_dim"], 3)
        self.upsample = nn.ModuleList([nn.ModuleList([CausalTrans(c["latent_dim"], c["latent_dim"], f, f),
                                                      ConvNeXt(c["latent_dim"])]) for f in c["upsampling_ratios"]])
        out_dim = c["decoder_dim"] // 2 ** len(c["upsample_rates"])
        self.decoder = nn.ModuleList([CausalConv(c["latent_dim"], c["decoder_dim"], 7)]
                                     + [DecBlock(c, i) for i in range(len(c["upsample_rates"]))]
                                     + [SnakeBeta(out_dim), CausalConv(out_dim, 1, 7)])

    def forward(self, codes):
        h = self.pre_conv(self.quantizer.decode(codes)).transpose(1, 2)
        h = self.pre_transformer(h).permute(0, 2, 1)
        for blocks in self.upsample:
            for b in blocks:
                h = b(h)
        for b in self.decoder:
            h = b(h)
        return h.clamp(-1, 1)

    def chunked(self, codes, chunk=300, left=25):
        wavs, start = [], 0
        while start < codes.shape[-1]:
            end = min(start + chunk, codes.shape[-1])
            ctx = left if start - left > 0 else start
            w = self(codes[..., start - ctx: end])
            wavs.append(w[..., ctx * self.total:])
            start = end
        return torch.cat(wavs, -1)


dec_cfg = dict(codebook_size=32, codebook_dim=16, hidden_size=32, latent_dim=24, rope_theta=10000,
               num_attention_heads=4, num_key_value_heads=2, sliding_window=5, intermediate_size=48,
               hidden_act="silu", rms_norm_eps=1e-5, num_hidden_layers=2, num_quantizers=4,
               upsample_rates=[2, 3], upsampling_ratios=[2], decoder_dim=32, layer_scale_initial_scale=0.01,
               max_position_embeddings=8000, attention_bias=False)
decoder = Decoder(dec_cfg).eval()
for n, p in decoder.named_parameters():   # weights large enough that every part matters
    if n.endswith("weight") and p.dim() > 1:
        with torch.no_grad():
            p.copy_(torch.randn_like(p) / math.sqrt(p[0].numel()) * 1.2)
with torch.no_grad():
    decoder.decoder[-1].conv.weight.mul_(0.003)     # keep the waveform inside the final clamp

# ------------------------------------------------------------------ Mimi encoder

mimi_cfg = MimiConfig(sampling_rate=24000, frame_rate=2000, audio_channels=1, hidden_size=16, num_filters=4,
                      num_residual_layers=1, upsampling_ratios=[3, 2], kernel_size=7, last_kernel_size=3,
                      residual_kernel_size=3, dilation_growth_rate=2, use_causal_conv=True, pad_mode="constant",
                      compress=2, trim_right_ratio=1.0, codebook_size=32, codebook_dim=8,
                      num_quantizers=8, use_conv_shortcut=False, vector_quantization_hidden_dimension=8,
                      num_semantic_quantizers=1, upsample_groups=16, num_hidden_layers=2, intermediate_size=32,
                      num_attention_heads=2, num_key_value_heads=2, head_dim=8, hidden_act="gelu",
                      max_position_embeddings=1000, initializer_range=0.02, norm_eps=1e-5, use_cache=False,
                      rope_theta=10000.0, sliding_window=7, attention_dropout=0.0, layer_scale_initial_scale=0.2)
mimi = MimiModel(mimi_cfg).eval()
with torch.no_grad():
    for n, p in mimi.named_parameters():
        if "codebook" in n:
            continue
        if p.dim() > 1:
            p.copy_(torch.randn_like(p) / math.sqrt(p[0].numel()))
        elif n.endswith("scale"):
            p.copy_(rnd(*p.shape, s=0.5))
        else:
            p.copy_(rnd(*p.shape, s=0.1) + (1 if "norm.weight" in n or "layernorm.weight" in n else 0))
    for n, b in mimi.named_buffers():
        if n.endswith("embed_sum"):
            b.copy_(rnd(*b.shape, s=1.0))
        elif n.endswith("cluster_usage"):
            b.copy_(torch.rand(b.shape) + 0.5)
hop = 12
assert int(np.prod(mimi_cfg.upsampling_ratios)) * 2 == hop

enc_state = {k: v for k, v in mimi.state_dict().items()
             if not k.startswith(("decoder", "decoder_transformer", "upsample"))
             and not k.endswith(("initialized",))}

codec_dir = f"{out}/qwen3tts/speech_tokenizer"
os.makedirs(codec_dir, exist_ok=True)
save([("encoder.", enc_state), ("decoder.", decoder.state_dict())], f"{codec_dir}/model.safetensors")
mimi_json = mimi_cfg.to_dict()
json.dump({"model_type": "qwen3_tts_tokenizer_12hz", "encoder_config": mimi_json, "decoder_config": dec_cfg,
           "encoder_valid_num_quantizers": 4, "input_sample_rate": 24000, "output_sample_rate": 24000,
           "decode_upsample_rate": hop, "encode_downsample_rate": hop}, open(f"{codec_dir}/config.json", "w"),
          indent=1)

# decode: 330 frames, so two chunks of 300 (the second with 25 frames of context)
frames = 330
codes = torch.tensor([[(7 * t + 13 * q + (t * t) % 5) % 32 for q in range(4)] for t in range(frames)])
with torch.no_grad():
    audio = decoder.chunked(codes.T[None])[0, 0][: frames * hop]
# encode: 1.1 s of a chirp with a little noise
n = 13213
tt = np.arange(n) / 24000
wave = (0.5 * np.sin(2 * np.pi * (180 + 300 * tt) * tt) + 0.05 * np.random.randn(n)).astype(np.float32)
with torch.no_grad():
    enc_codes = mimi.encode(torch.from_numpy(wave)[None, None]).audio_codes[0, :4]
    enc_codes = enc_codes[:, : -(-n // hop)]
json.dump({"codes": codes.tolist(), "audio": audio.numpy().round(6).tolist(),
           "wave": wave.round(6).tolist(), "encoded": enc_codes.T.tolist()},
          open(f"{codec_dir}/expected.json", "w"))
print("codec", audio.shape, float(audio.abs().max()), enc_codes.shape)

# ------------------------------------------------------------------ speaker encoder

class TDNN(nn.Module):
    def __init__(self, i, o, k, d):
        super().__init__()
        self.conv = nn.Conv1d(i, o, k, dilation=d, padding="same", padding_mode="reflect")

    def forward(self, x):
        return F.relu(self.conv(x))


class Res2Net(nn.Module):
    def __init__(self, i, o, scale, k, d):
        super().__init__()
        self.blocks = nn.ModuleList([TDNN(i // scale, o // scale, k, d) for _ in range(scale - 1)])
        self.scale = scale

    def forward(self, x):
        outs = []
        for i, part in enumerate(torch.chunk(x, self.scale, 1)):
            if i == 0:
                y = part
            elif i == 1:
                y = self.blocks[0](part)
            else:
                y = self.blocks[i - 1](part + y)
            outs.append(y)
        return torch.cat(outs, 1)


class SE(nn.Module):
    def __init__(self, i, s, o):
        super().__init__()
        self.conv1 = nn.Conv1d(i, s, 1, padding="same", padding_mode="reflect")
        self.conv2 = nn.Conv1d(s, o, 1, padding="same", padding_mode="reflect")

    def forward(self, x):
        m = x.mean(2, keepdim=True)
        return x * torch.sigmoid(self.conv2(F.relu(self.conv1(m))))


class SERes2Net(nn.Module):
    def __init__(self, i, o, scale, se, k, d):
        super().__init__()
        self.tdnn1 = TDNN(i, o, 1, 1)
        self.res2net_block = Res2Net(o, o, scale, k, d)
        self.tdnn2 = TDNN(o, o, 1, 1)
        self.se_block = SE(o, se, o)

    def forward(self, x):
        return x + self.se_block(self.tdnn2(self.res2net_block(self.tdnn1(x))))


class ASP(nn.Module):
    def __init__(self, c, a):
        super().__init__()
        self.tdnn = TDNN(c * 3, a, 1, 1)
        self.conv = nn.Conv1d(a, c, 1, padding="same", padding_mode="reflect")

    @staticmethod
    def stats(x, m):
        mean = (m * x).sum(2)
        std = torch.sqrt((m * (x - mean.unsqueeze(2)).pow(2)).sum(2).clamp(1e-12))
        return mean, std

    def forward(self, x):
        t = x.shape[-1]
        mean, std = self.stats(x, torch.ones(1, 1, t) / t)
        att = torch.cat([x, mean.unsqueeze(2).repeat(1, 1, t), std.unsqueeze(2).repeat(1, 1, t)], 1)
        att = F.softmax(self.conv(torch.tanh(self.tdnn(att))), 2)
        mean, std = self.stats(x, att)
        return torch.cat([mean, std], 1).unsqueeze(2)


class Ecapa(nn.Module):
    def __init__(self, c):
        super().__init__()
        ch, ks, ds = c["enc_channels"], c["enc_kernel_sizes"], c["enc_dilations"]
        self.blocks = nn.ModuleList([TDNN(c["mel_dim"], ch[0], ks[0], ds[0])]
                                    + [SERes2Net(ch[i - 1], ch[i], c["enc_res2net_scale"], c["enc_se_channels"],
                                                 ks[i], ds[i]) for i in range(1, len(ch) - 1)])
        self.mfa = TDNN(ch[-1], ch[-1], ks[-1], ds[-1])
        self.asp = ASP(ch[-1], c["enc_attention_channels"])
        self.fc = nn.Conv1d(ch[-1] * 2, c["enc_dim"], 1, padding="same", padding_mode="reflect")

    def forward(self, mel):   # [B, T, mels]
        x = mel.transpose(1, 2)
        outs = []
        for b in self.blocks:
            x = b(x)
            outs.append(x)
        x = self.mfa(torch.cat(outs[1:], 1))
        return self.fc(self.asp(x)).squeeze(-1)


def speaker_mel(y):
    basis = torch.from_numpy(librosa_mel(sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000)).float()
    y = F.pad(y[None, None], (384, 384), mode="reflect")[0]
    spec = torch.stft(y, 1024, hop_length=256, win_length=1024, window=torch.hann_window(1024), center=False,
                      normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(basis @ spec, min=1e-5)).transpose(1, 2)


# ------------------------------------------------------------------ talker and code predictor

H, G, CB = 32, 4, 32              # talker width, code groups, codebook size
VOCAB = 1024 + CB                 # the talker's codec vocabulary: codes, then control tokens
TEXT_VOCAB, TEXT_H = 40, 24
spk_cfg = dict(mel_dim=128, enc_dim=H, enc_channels=[16, 16, 16, 16, 48], enc_kernel_sizes=[5, 3, 3, 3, 1],
               enc_dilations=[1, 2, 3, 4, 1], enc_attention_channels=8, enc_res2net_scale=4, enc_se_channels=8,
               sample_rate=24000)
cp_cfg = dict(vocab_size=CB, hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
              num_key_value_heads=1, head_dim=8, hidden_act="silu", max_position_embeddings=64, rms_norm_eps=1e-6,
              rope_theta=10000, num_code_groups=G, tie_word_embeddings=False, attention_bias=False)
talker_cfg = dict(code_predictor_config=cp_cfg, vocab_size=VOCAB, hidden_size=H, intermediate_size=48,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8, hidden_act="silu",
                  max_position_embeddings=256, rms_norm_eps=1e-6, rope_theta=1000000.0,
                  rope_scaling={"interleaved": True, "mrope_section": [1, 1, 2], "rope_type": "default", "type": "default"},
                  num_code_groups=G, text_hidden_size=TEXT_H, text_vocab_size=TEXT_VOCAB, codec_eos_token_id=1036,
                  codec_think_id=1040, codec_nothink_id=1041, codec_think_bos_id=1042, codec_think_eos_id=1043,
                  codec_pad_id=1034, codec_bos_id=1035, tie_word_embeddings=False, attention_bias=False,
                  spk_id={"alice": 1050, "bob": 1051}, spk_is_dialect={"alice": False, "bob": "sichuan_dialect"},
                  codec_language_id={"english": 1045, "chinese": 1046, "sichuan_dialect": 1047})
IM_START, IM_END, TTS_PAD, TTS_BOS, TTS_EOS = 30, 31, 37, 38, 39
model_cfg = dict(architectures=["Qwen3TTSForConditionalGeneration"], model_type="qwen3_tts",
                 talker_config=talker_cfg, speaker_encoder_config=spk_cfg, tokenizer_type="qwen3_tts_tokenizer_12hz",
                 tts_model_size="0b6", tts_model_type="base", im_start_token_id=IM_START, im_end_token_id=IM_END,
                 tts_pad_token_id=TTS_PAD, tts_bos_token_id=TTS_BOS, tts_eos_token_id=TTS_EOS)


def qwen3(c, vocab):
    q = Qwen3Model(Qwen3Config(vocab_size=vocab, hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
                               num_hidden_layers=c["num_hidden_layers"], num_attention_heads=c["num_attention_heads"],
                               num_key_value_heads=c["num_key_value_heads"], head_dim=c["head_dim"],
                               rms_norm_eps=c["rms_norm_eps"], rope_theta=c["rope_theta"],
                               max_position_embeddings=c["max_position_embeddings"], attn_implementation="eager"))
    with torch.no_grad():
        for n, p in q.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn_like(p) / math.sqrt(p.shape[1]) * 1.3)
            else:
                p.copy_(1 + rnd(*p.shape, s=0.2))
    return q.eval()


talker = qwen3(talker_cfg, VOCAB)
cp = qwen3(cp_cfg, CB)
codec_emb = nn.Embedding(VOCAB, H)
text_emb = nn.Embedding(TEXT_VOCAB, TEXT_H)
fc1, fc2 = nn.Linear(TEXT_H, TEXT_H), nn.Linear(TEXT_H, H)
codec_head = nn.Linear(H, VOCAB, bias=False)
cp_emb = nn.ModuleList([nn.Embedding(CB, H) for _ in range(G - 1)])
cp_heads = nn.ModuleList([nn.Linear(16, CB, bias=False) for _ in range(G - 1)])
mtp = nn.Linear(H, 16)
with torch.no_grad():
    for m in [codec_emb, text_emb, *cp_emb]:
        m.weight.copy_(torch.randn_like(m.weight) * 0.6)
    for m in [codec_head, *cp_heads]:
        m.weight.copy_(torch.randn_like(m.weight) / math.sqrt(m.weight.shape[1]) * 3)
speaker = Ecapa(spk_cfg).eval()


def proj(ids):
    return fc2(F.silu(fc1(text_emb(torch.tensor(ids)))))


def emb(ids):
    return codec_emb(torch.tensor(ids))


def generate(input_id, instruct=None, speaker_name=None, language="auto", spk_embed=None, ref_id=None,
             ref_code=None, non_streaming=False, max_new=24, penalty=1.05):
    """Greedy Qwen3TTSForConditionalGeneration.generate for one sequence: the frames' codes."""
    parts = []
    if instruct is not None:
        parts.append(proj(instruct))
    pad_e, bos_e, eos_e = proj([TTS_PAD]), proj([TTS_BOS]), proj([TTS_EOS])
    tc = talker_cfg
    if spk_embed is None and speaker_name:
        spk = emb([tc["spk_id"][speaker_name]])
    else:
        spk = None if spk_embed is None else spk_embed.view(1, -1)
    lang = None if language == "auto" else tc["codec_language_id"][language]
    if language in ("chinese", "auto") and speaker_name and tc["spk_is_dialect"][speaker_name] is not False:
        lang = tc["codec_language_id"][tc["spk_is_dialect"][speaker_name]]
    prefill = ([tc["codec_nothink_id"], tc["codec_think_bos_id"], tc["codec_think_eos_id"]] if lang is None
               else [tc["codec_think_id"], tc["codec_think_bos_id"], lang, tc["codec_think_eos_id"]])
    codec = torch.cat([emb(prefill)] + ([spk] if spk is not None else []) + [emb([tc["codec_pad_id"], tc["codec_bos_id"]])])
    role = proj(input_id[:3])
    body = torch.cat([pad_e.expand(codec.shape[0] - 2, -1), bos_e]) + codec[:-1]
    seq = torch.cat([role, body])
    text = input_id[3:-5]
    if ref_code is not None:
        t_e = torch.cat([proj(ref_id[3:-2] + text), eos_e])
        c_e = torch.stack([emb([ref_code[f][0]])[0] + sum(cp_emb[g - 1](torch.tensor(ref_code[f][g]))
                                                          for g in range(1, G)) for f in range(len(ref_code))])
        c_e = torch.cat([emb([tc["codec_bos_id"]]), c_e])
        if non_streaming:
            icl = torch.cat([t_e + emb([tc["codec_pad_id"]] * len(t_e)), c_e + pad_e])
            trailing = pad_e
        elif len(t_e) > len(c_e):
            icl, trailing = t_e[: len(c_e)] + c_e, t_e[len(c_e):]
        else:
            icl = torch.cat([t_e, pad_e.expand(len(c_e) - len(t_e), -1)]) + c_e
            trailing = pad_e
        seq = torch.cat([seq, icl])
    else:
        seq = torch.cat([seq, proj(text[:1]) + codec[-1:]])
        if non_streaming:
            seq = seq[:-1]
            seq = torch.cat([seq, torch.cat([proj(text), eos_e]) + emb([tc["codec_pad_id"]] * (len(text) + 1)),
                             pad_e + emb([tc["codec_bos_id"]])])
            trailing = pad_e
        else:
            trailing = torch.cat([proj(text[1:]), eos_e])
    parts.append(seq)
    embeds = torch.cat(parts)
    prefill_len = len(embeds)

    frames, firsts = [], []
    hidden = talker(inputs_embeds=embeds[None]).last_hidden_state[0]
    logits = codec_head(hidden[-1])
    step = 0
    while True:
        l = logits.clone()
        for t in set(firsts):
            l[t] = l[t] * penalty if l[t] < 0 else l[t] / penalty
        l[VOCAB - 1024:] = torch.where(torch.arange(VOCAB - 1024, VOCAB) == tc["codec_eos_token_id"],
                                       l[VOCAB - 1024:], torch.tensor(float("-inf")))
        if len(firsts) < 2:
            l[tc["codec_eos_token_id"]] = float("-inf")
        c0 = int(l.argmax())
        firsts.append(c0)
        if c0 == tc["codec_eos_token_id"] or len(firsts) >= max_new:
            break
        # the code predictor writes codebooks 1.. of the frame
        sub, inputs = [], [hidden[-1], codec_emb.weight[c0]]
        for j in range(G - 1):
            h = cp(inputs_embeds=mtp(torch.stack(inputs))[None]).last_hidden_state[0, -1]
            c = int(cp_heads[j](h).argmax())
            sub.append(c)
            inputs.append(cp_emb[j].weight[c])
        frames.append([c0] + sub)
        nxt = codec_emb.weight[c0] + sum(cp_emb[j].weight[sub[j]] for j in range(G - 1))
        nxt = nxt + (trailing[step] if step < len(trailing) else pad_e[0])
        step += 1
        embeds = torch.cat([embeds, nxt[None]])
        hidden = talker(inputs_embeds=embeds[None]).last_hidden_state[0]
        logits = codec_head(hidden[-1])
    return frames, prefill_len


model_dir = f"{out}/qwen3tts"
state = {}
state.update({"talker.model." + k: v for k, v in talker.state_dict().items() if not k.startswith("embed_tokens")})
state["talker.model.codec_embedding.weight"] = codec_emb.weight
state["talker.model.text_embedding.weight"] = text_emb.weight
state.update({"talker.text_projection.linear_fc1." + k: v for k, v in fc1.state_dict().items()})
state.update({"talker.text_projection.linear_fc2." + k: v for k, v in fc2.state_dict().items()})
state["talker.codec_head.weight"] = codec_head.weight
state.update({"talker.code_predictor.model." + k: v for k, v in cp.state_dict().items() if not k.startswith("embed_tokens")})
for j in range(G - 1):
    state[f"talker.code_predictor.model.codec_embedding.{j}.weight"] = cp_emb[j].weight
    state[f"talker.code_predictor.lm_head.{j}.weight"] = cp_heads[j].weight
state.update({"talker.code_predictor.small_to_mtp_projection." + k: v for k, v in mtp.state_dict().items()})
with torch.no_grad():
    for n, p in speaker.named_parameters():
        p.copy_(torch.randn_like(p) / math.sqrt(max(1, p[0].numel())) if p.dim() > 1 else rnd(*p.shape, s=0.2))
state.update({"speaker_encoder." + k: v for k, v in speaker.state_dict().items()})
save([("", state)], f"{model_dir}/model.safetensors")
json.dump(model_cfg, open(f"{model_dir}/config.json", "w"), indent=1)
json.dump({"do_sample": True, "top_k": 50, "top_p": 1.0, "temperature": 0.9, "repetition_penalty": 1.05,
           "subtalker_dosample": True, "subtalker_top_k": 50, "subtalker_top_p": 1.0, "subtalker_temperature": 0.9,
           "max_new_tokens": 8192}, open(f"{model_dir}/generation_config.json", "w"), indent=1)

# prompts: "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n" as token ids
text_ids = [IM_START, 5, 6] + [11, 12, 13, 14, 15, 16, 17] + [IM_END, 6, IM_START, 5, 6]
ref_ids = [IM_START, 5, 6] + [20, 21, 22] + [IM_END, 6]
instruct_ids = [IM_START, 7, 6, 25, 26, 27, IM_END, 6]
ref_code = [[(3 * f + g) % CB for g in range(G)] for f in range(6)]
voice_wave = (0.4 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000)
              + 0.2 * np.sin(2 * np.pi * 523 * np.arange(24000) / 24000)
              + 0.03 * np.random.randn(24000)).astype(np.float32)
with torch.no_grad():
    mel = speaker_mel(torch.from_numpy(voice_wave))
    xvec = speaker(mel)[0]
    cases = {}
    cases["custom"], _ = generate(text_ids, instruct=instruct_ids, speaker_name="alice", language="english",
                                  non_streaming=True)
    cases["dialect"], _ = generate(text_ids, speaker_name="bob", language="auto", non_streaming=True)
    cases["design"], _ = generate(text_ids, instruct=instruct_ids, language="auto", non_streaming=True)
    cases["streaming"], _ = generate(text_ids, speaker_name="alice", language="chinese", non_streaming=False)
    cases["xvector"], _ = generate(text_ids, spk_embed=xvec, language="english")
    cases["icl"], _ = generate(text_ids, spk_embed=xvec, ref_id=ref_ids, ref_code=ref_code, language="english")
    cases["icl_long_text"], _ = generate(text_ids, spk_embed=xvec, ref_id=ref_ids, ref_code=ref_code[:2],
                                         language="english")
    cases["icl_nonstreaming"], _ = generate(text_ids, spk_embed=xvec, ref_id=ref_ids, ref_code=ref_code,
                                            language="english", non_streaming=True)
json.dump({"text_ids": text_ids, "ref_ids": ref_ids, "instruct_ids": instruct_ids, "ref_code": ref_code,
           "voice": voice_wave.round(6).tolist(), "mel": mel[0].numpy().round(5).tolist(),
           "xvector": xvec.numpy().round(6).tolist(), "cases": cases}, open(f"{model_dir}/expected.json", "w"))
for k, v in cases.items():
    print(k, len(v), v[:3])

# The tests feed token ids directly; qwen3tts/ also holds the small Qwen2 tokenizer of
# brainiac-io's tokenizers/qwen2-slow (vocab.json, merges.txt, tokenizer_config.json) so it loads.
