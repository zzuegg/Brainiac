# Tiny random Fish Audio S2-Pro models (fishaudio/s2-pro layout) and the reference outputs the
# Java port is checked against. The modules are transcribed from fish-speech (llama.py's
# DualARTransformer, inference.py's prompt and sampling, modded_dac.py and rvq.py's codec, and
# descript-audio-codec's layers) with the same parameter names, so the tensors are named as in
# the real model-*.safetensors and codec.pth; only the sizes are small.
#
#   python make_fish.py <resources dir>      (torch, safetensors, tokenizers)
import json
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.nn.utils.parametrizations import weight_norm as p_weight_norm
from torch.nn.utils import weight_norm as old_weight_norm

out = sys.argv[1] if len(sys.argv) > 1 else "."
torch.manual_seed(0)
np.random.seed(0)
torch.set_grad_enabled(False)


def tolist(t, digits=6):
    return np.round(t.detach().float().numpy(), digits).tolist()


# ------------------------------------------------------------------ shared transformer pieces

def precompute_freqs_cis(seq_len, n_elem, base=10000):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x, freqs_cis):
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack([
        xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
        xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], -1)
    return x_out2.flatten(3).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim, inter):
        super().__init__()
        self.w1 = nn.Linear(dim, inter, bias=False)
        self.w3 = nn.Linear(dim, inter, bias=False)
        self.w2 = nn.Linear(inter, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def sdpa(q, k, v, mask):
    # q, k, v [b, h, t, d]; mask bool [t_q, t_k] (True = attend)
    s = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


# ------------------------------------------------------------------ the DualAR text-to-semantic model

D, N_HEAD, N_LOCAL, HEAD_DIM, INTER, N_LAYER = 64, 4, 2, 16, 96, 2
FAST_LAYERS = 2
NUM_CODEBOOKS = 4               # 1 semantic + 3 residual
CODEBOOK = 64                   # the audio decoder's vocabulary (the semantic codebook size)
RESIDUAL = 32                   # residual codebooks of the codec
ROPE = 1000000.0
EPS = 1e-6


class Attention(nn.Module):
    def __init__(self, qk_norm):
        super().__init__()
        self.wqkv = nn.Linear(D, (N_HEAD + 2 * N_LOCAL) * HEAD_DIM, bias=False)
        self.wo = nn.Linear(N_HEAD * HEAD_DIM, D, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(HEAD_DIM, EPS)
            self.k_norm = nn.RMSNorm(HEAD_DIM, EPS)

    def forward(self, x, freqs, mask):
        b, t, _ = x.shape
        q, k, v = self.wqkv(x).split([N_HEAD * HEAD_DIM, N_LOCAL * HEAD_DIM, N_LOCAL * HEAD_DIM], dim=-1)
        q = q.view(b, t, N_HEAD, HEAD_DIM)
        k = k.view(b, t, N_LOCAL, HEAD_DIM)
        v = v.view(b, t, N_LOCAL, HEAD_DIM)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)
        q, k, v = map(lambda z: z.transpose(1, 2), (q, k, v))
        k = k.repeat_interleave(N_HEAD // N_LOCAL, dim=1)
        v = v.repeat_interleave(N_HEAD // N_LOCAL, dim=1)
        y = sdpa(q, k, v, mask)
        return self.wo(y.transpose(1, 2).contiguous().view(b, t, N_HEAD * HEAD_DIM))


class Block(nn.Module):
    def __init__(self, qk_norm):
        super().__init__()
        self.attention = Attention(qk_norm)
        self.feed_forward = FeedForward(D, INTER)
        self.ffn_norm = RMSNorm(D, EPS)
        self.attention_norm = RMSNorm(D, EPS)

    def forward(self, x, freqs, mask):
        h = x + self.attention(self.attention_norm(x), freqs, mask)
        return h + self.feed_forward(self.ffn_norm(h))


class DualAR(nn.Module):
    def __init__(self, vocab, sem_begin, sem_end):
        super().__init__()
        self.vocab = vocab
        self.sem_begin, self.sem_end = sem_begin, sem_end
        self.embeddings = nn.Embedding(vocab, D)
        self.codebook_embeddings = nn.Embedding(CODEBOOK * NUM_CODEBOOKS, D)
        self.layers = nn.ModuleList(Block(True) for _ in range(N_LAYER))
        self.norm = RMSNorm(D, EPS)
        self.freqs_cis = precompute_freqs_cis(512, HEAD_DIM, ROPE)
        # the audio decoder (fast transformer); fast_dim == dim, so no projection
        self.fast_embeddings = nn.Embedding(CODEBOOK, D)
        self.fast_layers = nn.ModuleList(Block(False) for _ in range(FAST_LAYERS))
        self.fast_norm = RMSNorm(D, EPS)
        self.fast_output = nn.Linear(D, CODEBOOK, bias=False)
        self.fast_freqs_cis = precompute_freqs_cis(NUM_CODEBOOKS, HEAD_DIM, ROPE)

    def embed(self, inp):
        # inp [1 + num_codebooks, t]
        embeds = [self.codebook_embeddings(inp[i + 1] + i * CODEBOOK) for i in range(NUM_CODEBOOKS)]
        vq_sum = torch.stack(embeds, dim=1).sum(dim=1)
        vq_mask = (inp[0] >= self.sem_begin) & (inp[0] <= self.sem_end)
        vq_sum[~vq_mask] = 0
        x = self.embeddings(inp[0]) + vq_sum
        # scale_codebook_embeddings (fish_qwen3_omni)
        return torch.where(vq_mask[:, None], x / math.sqrt(NUM_CODEBOOKS + 1), x)

    def slow(self, inp):
        """Logits and the normalized hidden state of the last step (norm_fastlayer_input)."""
        x = self.embed(inp)[None]
        t = x.shape[1]
        mask = torch.tril(torch.ones(t, t, dtype=torch.bool))
        for layer in self.layers:
            x = layer(x, self.freqs_cis[:t], mask)
        h = self.norm(x)[:, -1:]
        return F.linear(h, self.embeddings.weight)[0, 0], h      # tied embeddings

    def fast(self, seq):
        """Codebook logits after a fast sequence [1, n, D]."""
        n = seq.shape[1]
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
        x = seq
        for layer in self.fast_layers:
            x = layer(x, self.fast_freqs_cis[:n], mask)
        return self.fast_output(self.fast_norm(x))[0, -1]


def logits_to_probs(logits, temperature, top_p, top_k):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    indices = torch.arange(sorted_logits.shape[-1])
    remove = (cum_probs > top_p) | (indices >= top_k)
    remove[..., 0] = False
    remove = remove.scatter(dim=-1, index=sorted_indices, src=remove)
    logits = torch.where(remove, float("-inf"), logits)
    logits = logits / max(temperature, 1e-5)
    return torch.softmax(logits, dim=-1)


def generate_greedy(model, prompt, max_new, im_end):
    """decode_one_token_ar with top_k = 1: the arg max of the constrained logits, then the fast
    transformer's arg max per codebook (repetition-aware sampling never changes a top-1 draw)."""
    bias = torch.full((model.vocab,), float("-inf"))
    bias[model.sem_begin:model.sem_end + 1] = 0
    bias[im_end] = 0
    seq = prompt.clone()
    frames = []
    for _ in range(max_new):
        logits, h = model.slow(seq)
        main = int((logits + bias).argmax())
        a = min(max(main - model.sem_begin, 0), CODEBOOK - 1)
        codes = [a]
        fast_seq = torch.cat([h, model.fast_embeddings(torch.tensor([[a]]))], dim=1)
        for _ in range(1, NUM_CODEBOOKS):
            c = int(model.fast(fast_seq).argmax())
            codes.append(c)
            fast_seq = torch.cat([fast_seq, model.fast_embeddings(torch.tensor([[c]]))], dim=1)
        col = torch.tensor([main] + codes)[:, None]
        seq = torch.cat([seq, col], dim=1)
        if main == im_end:
            break
        frames.append(codes)
    return frames


# ------------------------------------------------------------------ prompts (conversation.py, content_sequence.py)

def encode_parts(tok, parts, sem_begin):
    """parts: ("text", str) or ("vq", codes [num_codebooks, t]); each text part is tokenized alone."""
    rows = []
    for kind, v in parts:
        if kind == "text":
            ids = tok.encode(v, add_special_tokens=False).ids
            col = torch.zeros(NUM_CODEBOOKS + 1, len(ids), dtype=torch.long)
            col[0] = torch.tensor(ids, dtype=torch.long)
        else:
            col = torch.zeros(NUM_CODEBOOKS + 1, v.shape[1], dtype=torch.long)
            col[0] = v[0] + sem_begin
            col[1:] = v
        rows.append(col)
    return torch.cat(rows, dim=1)


def conversation(ref_text, ref_codes, turns):
    """generate_long's conversation: the system message (with the reference), then for each
    (text, codes) turn a user message and, with codes, the assistant's answer; the last turn
    has no codes and opens the assistant's voice message."""
    parts = [("text", "<|im_start|>system\n")]
    if ref_codes is not None:
        parts.append(("text", "convert the provided text to speech reference to the following:\n\nText:\n"))
        parts.append(("text", "<|speaker:0|>" + ref_text))
        parts.append(("text", "\n\nSpeech:\n"))
        parts.append(("vq", ref_codes))
    else:
        parts.append(("text", "convert the provided text to speech"))
    parts.append(("text", "<|im_end|>\n"))
    for text, codes in turns:
        parts += [("text", "<|im_start|>user\n"), ("text", text), ("text", "<|im_end|>\n")]
        if codes is not None:
            parts += [("text", "<|im_start|>assistant\n<|voice|>"), ("vq", codes), ("text", "<|im_end|>\n")]
    parts.append(("text", "<|im_start|>assistant\n<|voice|>"))
    return parts


# ------------------------------------------------------------------ the codec (modded_dac.py, rvq.py, dac.nn)

def snake(x, alpha):
    return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def extra_padding(length, kernel, stride, padding_total):
    n_frames = (length - kernel + padding_total) / stride + 1
    ideal = (math.ceil(n_frames) - 1) * stride + (kernel - padding_total)
    return ideal - length


class CausalConvNet(nn.Module):
    def __init__(self, cin, cout, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel_size, stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.kernel_size - self.stride

    def forward(self, x):
        extra = extra_padding(x.shape[-1], self.kernel_size, self.stride, self.padding)
        return self.conv(F.pad(x, (self.padding, extra)))

    def wn(self):
        self.conv = p_weight_norm(self.conv)
        return self


class CausalTransConvNet(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(cin, cout, kernel_size, stride=stride)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        return x[..., :x.shape[-1] - pad]

    def wn(self):
        self.conv = p_weight_norm(self.conv)
        return self


class ResidualUnit(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.block = nn.Sequential(Snake1d(dim), CausalConvNet(dim, dim, 7, dilation=dilation).wn(),
                                   Snake1d(dim), CausalConvNet(dim, dim, 1).wn())

    def forward(self, x):
        y = self.block(x)
        pad = x.shape[-1] - y.shape[-1]
        if pad > 0:
            x = x[..., :-pad]
        return x + y


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(1e-2 * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class CAttention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.n_head = n_head

    def forward(self, x, freqs, mask):
        b, t, d = x.shape
        q, k, v = self.wqkv(x).split([d, d, d], dim=-1)
        q, k, v = (z.view(b, t, self.n_head, 64) for z in (q, k, v))
        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)
        q, k, v = map(lambda z: z.transpose(1, 2), (q, k, v))
        y = sdpa(q, k, v, mask)
        return self.wo(y.transpose(1, 2).contiguous().view(b, t, d))


class CBlock(nn.Module):
    def __init__(self, dim, inter):
        super().__init__()
        self.attention = CAttention(dim, dim // 64)
        self.feed_forward = FeedForward(dim, inter)
        self.ffn_norm = RMSNorm(dim, 1e-5)
        self.attention_norm = RMSNorm(dim, 1e-5)
        self.attention_layer_scale = LayerScale(dim)
        self.ffn_layer_scale = LayerScale(dim)

    def forward(self, x, freqs, mask):
        h = x + self.attention_layer_scale(self.attention(self.attention_norm(x), freqs, mask))
        return h + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(h)))


class WindowLimitedTransformer(nn.Module):
    def __init__(self, dim, inter, layers, window):
        super().__init__()
        self.layers = nn.ModuleList(CBlock(dim, inter) for _ in range(layers))
        self.norm = RMSNorm(dim, 1e-5)
        self.window = window
        # the real checkpoint carries these buffers; Brainiac ignores them
        self.register_buffer("freqs_cis", precompute_freqs_cis(64, 64, 10000))
        self.register_buffer("causal_mask", torch.tril(torch.ones(8, 8, dtype=torch.bool)))

    def forward(self, x):
        x = x.transpose(1, 2)
        t = x.shape[1]
        rows = torch.arange(t).view(-1, 1)
        mask = torch.tril(torch.ones(t, t)).bool() & (torch.arange(t) >= (rows - self.window + 1).clamp(min=0))
        freqs = precompute_freqs_cis(t, 64, 10000)
        for layer in self.layers:
            x = layer(x, freqs, mask)
        return self.norm(x).transpose(1, 2)


class EncoderBlock(nn.Module):
    def __init__(self, dim, stride, n_t_layer):
        super().__init__()
        mods = [ResidualUnit(dim // 2, 1), ResidualUnit(dim // 2, 3), ResidualUnit(dim // 2, 9), Snake1d(dim // 2),
                CausalConvNet(dim // 2, dim, 2 * stride, stride=stride).wn()]
        if n_t_layer:
            # the encoder's transformer: window_size falls back to 512 (getattr on a partial)
            mods.append(WindowLimitedTransformer(dim, dim * 3, n_t_layer, 512))
        self.block = nn.Sequential(*mods)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model, strides, d_latent, t_layers):
        super().__init__()
        block = [CausalConvNet(1, d_model, 7).wn()]
        for stride, n in zip(strides, t_layers):
            d_model *= 2
            block.append(EncoderBlock(d_model, stride, n))
        block += [Snake1d(d_model), CausalConvNet(d_model, d_latent, 3).wn()]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        # the reference builds a transformer here for decoder_transformer_layers but never uses it
        self.block = nn.Sequential(Snake1d(cin), CausalTransConvNet(cin, cout, 2 * stride, stride=stride).wn(),
                                   ResidualUnit(cout, 1), ResidualUnit(cout, 3), ResidualUnit(cout, 9))

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, cin, channels, rates):
        super().__init__()
        layers = [CausalConvNet(cin, channels, 7).wn()]
        for i, stride in enumerate(rates):
            layers.append(DecoderBlock(channels // 2 ** i, channels // 2 ** (i + 1), stride))
        out = channels // 2 ** len(rates)
        layers += [Snake1d(out), CausalConvNet(out, 1, 7).wn(), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VectorQuantize(nn.Module):
    def __init__(self, dim, size, cdim):
        super().__init__()
        self.in_proj = old_weight_norm(nn.Conv1d(dim, cdim, 1))
        self.out_proj = old_weight_norm(nn.Conv1d(cdim, dim, 1))
        self.codebook = nn.Embedding(size, cdim)

    def encode(self, z):
        z_e = self.in_proj(z)
        enc = F.normalize(z_e[0].T)
        book = F.normalize(self.codebook.weight)
        dist = enc.pow(2).sum(1, keepdim=True) - 2 * enc @ book.t() + book.pow(2).sum(1, keepdim=True).t()
        idx = (-dist).max(1)[1]
        z_q = self.out_proj(self.codebook(idx).T[None])
        return z_q, idx

    def decode(self, idx):
        return self.out_proj(self.codebook(idx).T[None])


class RVQ(nn.Module):
    def __init__(self, dim, n, size, cdim):
        super().__init__()
        self.quantizers = nn.ModuleList(VectorQuantize(dim, size, cdim) for _ in range(n))

    def encode(self, z):
        residual, codes, z_q = z, [], 0
        for q in self.quantizers:
            zq_i, idx = q.encode(residual)
            z_q = z_q + zq_i
            residual = residual - zq_i
            codes.append(idx)
        return z_q, codes

    def decode(self, codes):
        return sum(q.decode(c) for q, c in zip(self.quantizers, codes))


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = CausalConvNet(dim, dim, 7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        y = self.dwconv(x).permute(0, 2, 1)
        y = self.pwconv2(F.gelu(self.pwconv1(self.norm(y))))
        return x + (self.gamma * y).permute(0, 2, 1)


class Quantizer(nn.Module):
    def __init__(self, dim, n, size, sem_size, cdim, t_inter, t_layers):
        super().__init__()
        self.semantic_quantizer = RVQ(dim, 1, sem_size, cdim)
        self.quantizer = RVQ(dim, n, size, cdim)
        self.downsample = nn.Sequential(*[nn.Sequential(CausalConvNet(dim, dim, 2, stride=2), ConvNeXtBlock(dim))
                                          for _ in range(2)])
        self.upsample = nn.Sequential(*[nn.Sequential(CausalTransConvNet(dim, dim, 2, stride=2), ConvNeXtBlock(dim))
                                        for _ in range(2)])
        self.pre_module = WindowLimitedTransformer(dim, t_inter, t_layers, 128)
        self.post_module = WindowLimitedTransformer(dim, t_inter, t_layers, 128)

    def encode(self, z):
        z = self.pre_module(self.downsample(z))
        sz, sc = self.semantic_quantizer.encode(z)
        _, rc = self.quantizer.encode(z - sz)
        return torch.stack(sc + rc)

    def decode(self, codes):
        z = self.semantic_quantizer.decode([codes[0]]) + self.quantizer.decode(list(codes[1:]))
        return self.upsample(self.post_module(z))


ENC_DIM, ENC_RATES, DEC_DIM, DEC_RATES = 4, [2, 2, 2, 2], 32, [2, 2, 2, 2]
LATENT = ENC_DIM * 2 ** len(ENC_RATES)                         # 64
HOP = 2 ** len(ENC_RATES) * 4                                  # 64 samples per frame


class Codec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(ENC_DIM, ENC_RATES, LATENT, [0, 0, 0, 2])
        self.quantizer = Quantizer(LATENT, NUM_CODEBOOKS - 1, RESIDUAL, CODEBOOK, 4, 2 * LATENT, 2)
        self.decoder = Decoder(LATENT, DEC_DIM, DEC_RATES)

    def encode(self, audio):
        length = audio.shape[-1]
        x = F.pad(audio[None, None], (0, math.ceil(length / HOP) * HOP - length))
        return self.quantizer.encode(self.encoder(x))[:, : math.ceil(length / HOP)]

    def decode(self, codes):
        return self.decoder(self.quantizer.decode(codes))[0, 0]


# ------------------------------------------------------------------ main

def main():
    d = f"{out}/fish"
    import os
    os.makedirs(d, exist_ok=True)

    # a byte-level BPE tokenizer in the layout of Qwen3's, with Fish's special tokens
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    qwen = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+"
            r"|\s+(?!\S)|\s+")
    tk = Tokenizer(models.BPE())
    tk.normalizer = normalizers.NFC()
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(qwen), behavior="isolated"),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    corpus = ["convert the provided text to speech reference to the following", "Text Speech user assistant system",
              "Hello world, this is a test of the speech engine.", "Guten Morgen! Wie geht es dir?"] * 20
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, show_progress=False,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>", "<|phoneme_start|>", "<|phoneme_end|>",
                "<|text|>", "<|voice|>", "<|interleave|>", "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>"]
    specials += [f"<|semantic:{i}|>" for i in range(CODEBOOK)]
    tk.add_special_tokens(specials)
    tk.save(f"{d}/tokenizer.json")
    sem_begin = tk.token_to_id("<|semantic:0|>")
    sem_end = tk.token_to_id(f"<|semantic:{CODEBOOK - 1}|>")
    im_end = tk.token_to_id("<|im_end|>")
    vocab = tk.get_vocab_size()

    model = DualAR(vocab, sem_begin, sem_end)
    for n, p in model.named_parameters():
        if "norm" in n:
            p.copy_(1 + 0.1 * torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) * (0.6 if "embeddings" in n else 1 / math.sqrt(p.shape[-1])))
    for p in model.parameters():
        p.copy_(p.half().float())
    state = {}
    for k, v in model.state_dict().items():
        if "freqs" in k:
            continue
        if k.startswith("fast_"):
            # the published layout: the audio decoder's weights under audio_decoder., without fast_
            state["audio_decoder." + k[len("fast_"):]] = v
        elif k.startswith("codebook_embeddings."):
            state["audio_decoder." + k] = v
        else:
            state["text_model.model." + k] = v
    save_file({k: v.half().contiguous() for k, v in state.items()}, f"{d}/model.safetensors")
    config = {
        "model_type": "fish_qwen3_omni", "dtype": "bfloat16", "eos_token_id": im_end,
        "pad_token_id": tk.token_to_id("<|pad|>"), "audio_pad_token_id": tk.token_to_id("<|audio_pad|>"),
        "semantic_start_token_id": sem_begin, "semantic_end_token_id": sem_end,
        "text_config": {"model_type": "fish_qwen3", "vocab_size": vocab, "n_layer": N_LAYER, "n_head": N_HEAD,
                        "n_local_heads": N_LOCAL, "head_dim": HEAD_DIM, "dim": D, "intermediate_size": INTER,
                        "rope_base": ROPE, "norm_eps": EPS, "max_seq_len": 512, "tie_word_embeddings": True,
                        "attention_qkv_bias": False, "attention_o_bias": False, "attention_qk_norm": True},
        "audio_decoder_config": {"model_type": "fish_qwen3_audio_decoder", "vocab_size": CODEBOOK,
                                 "num_codebooks": NUM_CODEBOOKS, "n_layer": FAST_LAYERS, "n_head": N_HEAD,
                                 "n_local_heads": N_LOCAL, "head_dim": HEAD_DIM, "dim": D, "text_dim": D,
                                 "intermediate_size": INTER, "rope_base": ROPE, "norm_eps": EPS,
                                 "max_seq_len": NUM_CODEBOOKS + 1, "tie_word_embeddings": False,
                                 "attention_qkv_bias": False, "attention_o_bias": False,
                                 "attention_qk_norm": False}}
    json.dump(config, open(f"{d}/config.json", "w"), indent=1)

    codec = Codec()
    for n, p in codec.named_parameters():
        if n.endswith("alpha"):
            p.copy_(1 + 0.5 * torch.rand_like(p))
        elif n.endswith("gamma"):
            p.copy_(0.3 + 0.2 * torch.rand_like(p))
        elif "norm" in n and "weight" in n and p.dim() == 1:
            p.copy_(1 + 0.1 * torch.randn_like(p))
        elif "original0" in n or "weight_g" in n:
            p.copy_(0.5 + 0.5 * torch.rand_like(p))
        elif p.dim() == 1:
            p.copy_(0.05 * torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) / math.sqrt(max(1, p[0].numel())))
    # a gentle output layer keeps the waveform off tanh's flat ends
    codec.decoder.model[-2].conv.parametrizations.weight.original0.mul_(0.03)
    for p in codec.parameters():
        p.copy_(p.half().float())
    codec.eval()
    torch.save(codec.state_dict(), f"{d}/codec.pth")

    exp = {"vocab": vocab, "semantic_begin": sem_begin, "im_end": im_end}

    # codec: encoding a sound long enough for both attention windows, decoding random codes
    n = 12000
    t = torch.arange(n) / 44100
    wave = (0.4 * torch.sin(2 * math.pi * 220 * t) * torch.sin(2 * math.pi * 3 * t)
            + 0.05 * torch.randn(n)).float()
    wave = torch.from_numpy(np.round(wave.numpy(), 5))
    codes = codec.encode(wave)
    exp["codec_wave"] = tolist(wave, 5)
    exp["codec_codes"] = codes.tolist()
    frames = 150
    rand = torch.cat([torch.randint(0, CODEBOOK, (1, frames)), torch.randint(0, RESIDUAL, (NUM_CODEBOOKS - 1, frames))])
    exp["decode_codes"] = rand.tolist()
    exp["decode_audio"] = tolist(codec.decode(rand), 6)

    # sampling: the filtered, tempered distribution
    logits = torch.randn(40) * 2
    exp["sample_logits"] = tolist(logits)
    exp["sample_probs"] = [tolist(logits_to_probs(logits, tmp, tp, tk_), 6)
                           for tmp, tp, tk_ in [(0.7, 0.7, 30), (1.0, 0.9, 30), (0.7, 1.0, 5)]]

    # prompts and greedy frames
    ref_codes = codes[:, :6]
    cases = {
        "plain": (None, None, [("Hello world.", None)]),
        "clone": ("Guten Morgen!", ref_codes, [("Hello world.", None)]),
    }
    for name, (rt, rc, turns) in cases.items():
        prompt = encode_parts(tk, conversation(rt, rc, turns), sem_begin)
        exp[f"{name}_prompt"] = prompt.tolist()
        exp[f"{name}_frames"] = generate_greedy(model, prompt, 10, im_end)
    # the next chunk reads the first one's frames
    first = torch.tensor(exp["clone_frames"]).T
    prompt = encode_parts(tk, conversation("Guten Morgen!", ref_codes,
                                           [("Hello world.", first), ("Second part.", None)]), sem_begin)
    exp["next_prompt"] = prompt.tolist()
    exp["next_frames"] = generate_greedy(model, prompt, 10, im_end)
    exp["ref_codes"] = ref_codes.tolist()
    json.dump(exp, open(f"{d}/expected.json", "w"))
    print("frames", exp["plain_frames"][:3], "codes", codes[:, :5].tolist())


main()
