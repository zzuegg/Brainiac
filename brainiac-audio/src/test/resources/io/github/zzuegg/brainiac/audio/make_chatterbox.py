# Tiny random Chatterbox models (ResembleAI/chatterbox layout) and the reference outputs the
# Java port is checked against. The modules are transcribed from chatterbox-tts 0.1.7 and
# s3tokenizer 0.3.0 with the same parameter names, so the tensors are named as in the real
# ve.safetensors, t3_cfg.safetensors and s3gen.safetensors; only the sizes are small.
#
#   python make_chatterbox.py <resources dir>      (torch, transformers, librosa, tokenizers)
import json
import math
import sys

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaModel

out = sys.argv[1] if len(sys.argv) > 1 else "."
torch.manual_seed(0)
np.random.seed(0)
torch.set_grad_enabled(False)


def rnd(*shape, s=0.3):
    return torch.randn(*shape) * s


def init(module, s=None):
    for n, p in module.named_parameters():
        fan = p[0].numel() if p.dim() > 1 else 1
        p.copy_(torch.randn_like(p) * (s if s is not None else 1 / math.sqrt(max(1, fan))) if p.dim() > 1
                else torch.randn_like(p) * 0.1 + (1.0 if n.endswith("weight") and "norm" in n.lower() else 0.0))


def bn_init(bn):
    bn.weight.copy_(1 + 0.1 * torch.randn_like(bn.weight)) if bn.weight is not None else None
    bn.bias.copy_(0.1 * torch.randn_like(bn.bias)) if bn.bias is not None else None
    bn.running_mean.copy_(0.1 * torch.randn_like(bn.running_mean))
    bn.running_var.copy_(1 + 0.2 * torch.rand_like(bn.running_var))


def tolist(t, digits=6):
    return np.round(t.detach().float().numpy(), digits).tolist()


# ------------------------------------------------------------------ voice encoder (ve.safetensors)
VE_MELS, VE_HIDDEN, VE_EMBED = 40, 16, 16


class VoiceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(VE_MELS, VE_HIDDEN, num_layers=3, batch_first=True)
        self.proj = nn.Linear(VE_HIDDEN, VE_EMBED)
        self.similarity_weight = nn.Parameter(torch.tensor([10.]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]))

    def forward(self, mels):
        _, (hidden, _) = self.lstm(mels)
        raw = F.relu(self.proj(hidden[-1]))
        return raw / torch.linalg.norm(raw, dim=1, keepdim=True)


def ve_trim(wav, top_db=20, frame_length=2048, hop=512):
    return librosa.effects.trim(wav, top_db=top_db, frame_length=frame_length, hop_length=hop)[0]


def ve_mel(wav):
    spec = np.abs(librosa.stft(wav, n_fft=400, hop_length=160, win_length=400, center=True, pad_mode="reflect")) ** 2
    basis = librosa.filters.mel(sr=16000, n_fft=400, n_mels=VE_MELS, fmin=0, fmax=8000)
    return np.dot(basis, spec)  # (M, T)


def ve_embed(ve, wav16):
    wav = ve_trim(wav16)
    mel = ve_mel(wav).T.astype(np.float32)  # (T, M)
    frame_step = int(np.round((16000 / 1.3) / 160))
    win = 160
    n_wins, rem = divmod(max(len(mel) - win + frame_step, 0), frame_step)
    if n_wins == 0 or (rem + (win - frame_step)) / win >= 0.8:
        n_wins += 1
    target = win + frame_step * (n_wins - 1)
    if target > len(mel):
        mel = np.concatenate((mel, np.zeros((target - len(mel), VE_MELS), np.float32)))
    partials = torch.stack([torch.from_numpy(mel[i * frame_step:i * frame_step + win]) for i in range(n_wins)])
    e = ve(partials).mean(0)
    return (e / torch.linalg.norm(e)), len(wav), mel


# ------------------------------------------------------------------ T3 (t3_cfg.safetensors)
D = 64      # 1024 in the real model; heads are 64 wide as there
HEADS = 1
TEXT_VOCAB = 40
SPEECH_VOCAB = 6563   # 8194 in the real model; only the codes, start and stop are used
START_TEXT, STOP_TEXT = 255 % TEXT_VOCAB, 0   # the real model: 255 and 0 in a 704-token vocabulary
START_SPEECH, STOP_SPEECH = 6561, 6562
QUERY_TOKENS = 4
PROMPT_LEN = 12


class AttentionBlock2(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x1, x2):
        b, t, c = x1.shape
        q = self.to_q(self.norm(x1))
        k = self.to_k(self.norm(x2))
        v = self.to_v(self.norm(x2))
        sh = lambda x: x.view(x.shape[0], x.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        h = F.scaled_dot_product_attention(sh(q), sh(k), sh(v))
        h = h.permute(0, 2, 1, 3).reshape(b, t, c)
        return x1 + self.proj_out(h)


class Perceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attention_query = nn.Parameter(torch.empty(1, QUERY_TOKENS, D))
        self.attn = AttentionBlock2(D, 4)

    def forward(self, h):
        q = self.pre_attention_query.expand(h.shape[0], -1, -1)
        pre = self.attn(q, h)
        return self.attn(pre, pre)


class CondEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.spkr_enc = nn.Linear(VE_EMBED, D)
        self.emotion_adv_fc = nn.Linear(1, D, bias=False)
        self.perceiver = Perceiver()


class LearnedPos(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.emb = nn.Embedding(n, D)


class T3(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = LlamaConfig(vocab_size=8, hidden_size=D, intermediate_size=96, num_hidden_layers=2,
                               num_attention_heads=HEADS, num_key_value_heads=HEADS, head_dim=D // HEADS,
                               rms_norm_eps=1e-5, rope_theta=500000.0, max_position_embeddings=131072,
                               rope_scaling=dict(factor=8.0, high_freq_factor=4.0, low_freq_factor=1.0,
                                                 original_max_position_embeddings=8192, rope_type="llama3"),
                               attention_bias=False, mlp_bias=False, tie_word_embeddings=False)
        self.tfmr = LlamaModel(self.cfg)
        self.cond_enc = CondEnc()
        self.text_emb = nn.Embedding(TEXT_VOCAB, D)
        self.speech_emb = nn.Embedding(SPEECH_VOCAB, D)
        self.text_pos_emb = LearnedPos(64)
        self.speech_pos_emb = LearnedPos(128)
        self.text_head = nn.Linear(D, TEXT_VOCAB, bias=False)
        self.speech_head = nn.Linear(D, SPEECH_VOCAB, bias=False)

    def cond(self, speaker_emb, prompt_tokens, exaggeration):
        c = self.cond_enc
        spk = c.spkr_enc(speaker_emb.view(-1, VE_EMBED))[:, None]
        pe = self.speech_emb(prompt_tokens) + self.speech_pos_emb.emb(torch.arange(prompt_tokens.shape[1]))
        pe = c.perceiver(pe)
        emo = c.emotion_adv_fc(torch.tensor([[[exaggeration]]]).view(-1, 1, 1))
        return torch.cat([spk, pe, emo], dim=1)

    def inference(self, speaker_emb, prompt_tokens, exaggeration, text_tokens, max_new, cfg_weight=0.5,
                  repetition_penalty=1.2):
        """Greedy decoding of the reference T3.inference (the sampling warpers keep the arg max)."""
        cond = self.cond(speaker_emb, prompt_tokens, exaggeration)
        tt = torch.tensor([text_tokens, text_tokens])
        te = self.text_emb(tt)
        te[1].zero_()
        te = te + self.text_pos_emb.emb(torch.arange(tt.shape[1]))
        bos = self.speech_emb(torch.tensor([[START_SPEECH]])) + self.speech_pos_emb.emb(torch.tensor([[0]]))
        embeds = torch.cat([cond.expand(2, -1, -1), te, bos.expand(2, -1, -1), bos.expand(2, -1, -1)], dim=1)
        generated = [START_SPEECH]
        predicted = []
        for i in range(max_new):
            h = self.tfmr(inputs_embeds=embeds).last_hidden_state
            logits = self.speech_head(h[:, -1])
            cond_l, uncond_l = logits[0], logits[1]
            l = cond_l + cfg_weight * (cond_l - uncond_l)
            for t in set(generated):
                l[t] = l[t] * repetition_penalty if l[t] < 0 else l[t] / repetition_penalty
            nxt = int(l.argmax())
            predicted.append(nxt)
            generated.append(nxt)
            if nxt == STOP_SPEECH:
                break
            e = self.speech_emb(torch.tensor([[nxt]])) + self.speech_pos_emb.emb(torch.tensor([[i + 1]]))
            embeds = torch.cat([embeds, e.expand(2, -1, -1)], dim=1)
        return predicted, cond


# ------------------------------------------------------------------ S3Gen flow (s3gen.safetensors "flow.")
E = 32        # encoder width (512 in the real model)
E_HEADS = 4
MEL = 80
SPK = 24      # CAMPPlus embedding (192 in the real model)
DEC_CH = 32   # estimator channels (256)
DEC_HEADS, DEC_HEAD_DIM = 2, 64
TOKEN_VOCAB = 6561


class RelPosAttention(nn.Module):
    def __init__(self, n_head, n_feat):
        super().__init__()
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))

    @staticmethod
    def rel_shift(x):
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1), dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        return x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]

    def forward(self, x, pos_emb):
        b = x.size(0)
        q = self.linear_q(x).view(b, -1, self.h, self.d_k)
        k = self.linear_k(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k).transpose(1, 2)
        ac = torch.matmul((q + self.pos_bias_u).transpose(1, 2), k.transpose(-2, -1))
        bd = torch.matmul((q + self.pos_bias_v).transpose(1, 2), p.transpose(-2, -1))
        if ac.shape != bd.shape:
            bd = self.rel_shift(bd)
        scores = (ac + bd) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        return self.linear_out(y)


class FF(nn.Module):
    def __init__(self, d, units):
        super().__init__()
        self.w_1 = nn.Linear(d, units)
        self.w_2 = nn.Linear(units, d)

    def forward(self, x):
        return self.w_2(F.silu(self.w_1(x)))


class ConformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = RelPosAttention(E_HEADS, E)
        self.feed_forward = FF(E, 64)
        self.norm_ff = nn.LayerNorm(E, eps=1e-12)
        self.norm_mha = nn.LayerNorm(E, eps=1e-12)

    def forward(self, x, pos):
        x = x + self.self_attn(self.norm_mha(x), pos)
        return x + self.feed_forward(self.norm_ff(x))


def espnet_pos(T, d):
    pe_pos = torch.zeros(T, d)
    pe_neg = torch.zeros(T, d)
    position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
    pe_pos[:, 0::2] = torch.sin(position * div)
    pe_pos[:, 1::2] = torch.cos(position * div)
    pe_neg[:, 0::2] = torch.sin(-1 * position * div)
    pe_neg[:, 1::2] = torch.cos(-1 * position * div)
    pe = torch.cat([torch.flip(pe_pos, [0]).unsqueeze(0), pe_neg[1:].unsqueeze(0)], dim=1)
    return pe   # [1, 2T-1, d]


class LinearEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(E, E), nn.LayerNorm(E, eps=1e-5))

    def forward(self, x):
        x = self.out(x) * math.sqrt(E)
        return x, espnet_pos(x.size(1), E)


class PreLookahead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(E, E, 4)
        self.conv2 = nn.Conv1d(E, E, 3)

    def forward(self, inputs):
        o = F.pad(inputs.transpose(1, 2), (0, 3))
        o = F.leaky_relu(self.conv1(o))
        o = self.conv2(F.pad(o, (2, 0)))
        return o.transpose(1, 2) + inputs


class UpLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(E, E, 5)

    def forward(self, x):
        return self.conv(F.pad(F.interpolate(x, scale_factor=2.0, mode="nearest"), (4, 0)))


class UpsampleConformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = LinearEmbed()
        self.pre_lookahead_layer = PreLookahead()
        self.encoders = nn.ModuleList([ConformerLayer() for _ in range(2)])
        self.up_layer = UpLayer()
        self.up_embed = LinearEmbed()
        self.up_encoders = nn.ModuleList([ConformerLayer() for _ in range(2)])
        self.after_norm = nn.LayerNorm(E, eps=1e-5)

    def forward(self, xs):
        xs, pos = self.embed(xs)
        xs = self.pre_lookahead_layer(xs)
        for l in self.encoders:
            xs = l(xs, pos)
        xs = self.up_layer(xs.transpose(1, 2)).transpose(1, 2)
        xs, pos = self.up_embed(xs)
        for l in self.up_encoders:
            xs = l(xs, pos)
        return self.after_norm(xs)


class CausalConv1d(nn.Conv1d):
    def __init__(self, i, o, k):
        super().__init__(i, o, k)
        self.p = k - 1

    def forward(self, x):
        return super().forward(F.pad(x, (self.p, 0)))


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class CausalBlock1D(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(CausalConv1d(dim, dim_out, 3), Transpose(), nn.LayerNorm(dim_out), Transpose(),
                                   nn.Mish())

    def forward(self, x, mask):
        return self.block(x * mask) * mask


class CausalResnetBlock1D(nn.Module):
    def __init__(self, dim, dim_out, temb):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(temb, dim_out))
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, t):
        h = self.block1(x, mask)
        h = h + self.mlp(t).unsqueeze(-1)
        h = self.block2(h, mask)
        return h + self.res_conv(x * mask)


class DiffAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inner = DEC_HEADS * DEC_HEAD_DIM
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim), nn.Dropout(0.0)])

    def forward(self, x):
        b, t, _ = x.shape
        sh = lambda y: y.view(b, t, DEC_HEADS, DEC_HEAD_DIM).transpose(1, 2)
        h = F.scaled_dot_product_attention(sh(self.to_q(x)), sh(self.to_k(x)), sh(self.to_v(x)))
        return self.to_out[0](h.transpose(1, 2).reshape(b, t, -1))


class GELUProj(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.proj = nn.Linear(i, o)

    def forward(self, x):
        return F.gelu(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.ModuleList([GELUProj(dim, dim * 4), nn.Dropout(0.0), nn.Linear(dim * 4, dim)])

    def forward(self, x):
        for m in self.net:
            x = m(x)
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = DiffAttention(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x):
        x = self.attn1(self.norm1(x)) + x
        return self.ff(self.norm3(x)) + x


class TimestepEmbedding(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.linear_1 = nn.Linear(i, o)
        self.linear_2 = nn.Linear(o, o)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


def sinusoidal(x, dim, scale=1000):
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half).float() * -emb)
    emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ConditionalDecoder(nn.Module):
    def __init__(self, in_channels=4 * MEL, n_blocks=1, num_mid_blocks=2):
        super().__init__()
        ch = DEC_CH
        self.in_channels = in_channels
        temb = ch * 4
        self.time_mlp = TimestepEmbedding(in_channels, temb)
        mk = lambda n: nn.ModuleList([BasicTransformerBlock(ch) for _ in range(n)])
        self.down_blocks = nn.ModuleList([nn.ModuleList([CausalResnetBlock1D(in_channels, ch, temb), mk(n_blocks),
                                                         CausalConv1d(ch, ch, 3)])])
        self.mid_blocks = nn.ModuleList([nn.ModuleList([CausalResnetBlock1D(ch, ch, temb), mk(n_blocks)])
                                         for _ in range(num_mid_blocks)])
        self.up_blocks = nn.ModuleList([nn.ModuleList([CausalResnetBlock1D(2 * ch, ch, temb), mk(n_blocks),
                                                       CausalConv1d(ch, ch, 3)])])
        self.final_block = CausalBlock1D(ch, ch)
        self.final_proj = nn.Conv1d(ch, MEL, 1)

    def forward(self, x, mask, mu, t, spks, cond):
        t = self.time_mlp(sinusoidal(t, self.in_channels))
        x = torch.cat([x, mu, spks[:, :, None].expand(-1, -1, x.shape[-1]), cond], dim=1)
        hiddens = []
        for resnet, blocks, down in self.down_blocks:
            x = resnet(x, mask, t).transpose(1, 2)
            for b in blocks:
                x = b(x)
            x = x.transpose(1, 2)
            hiddens.append(x)
            x = down(x * mask)
        for resnet, blocks in self.mid_blocks:
            x = resnet(x, mask, t).transpose(1, 2)
            for b in blocks:
                x = b(x)
            x = x.transpose(1, 2)
        for resnet, blocks, up in self.up_blocks:
            skip = hiddens.pop()
            x = torch.cat([x[:, :, :skip.shape[-1]], skip], dim=1)
            x = resnet(x, mask, t).transpose(1, 2)
            for b in blocks:
                x = b(x)
            x = x.transpose(1, 2)
            x = up(x * mask)
        x = self.final_block(x, mask)
        return self.final_proj(x * mask) * mask


class CFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.estimator = ConditionalDecoder()

    def forward(self, mu, mask, spks, cond, z, n_timesteps=10, cfg_rate=0.7):
        t_span = torch.linspace(0, 1, n_timesteps + 1)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        x = z
        B, T = mu.size(0), x.size(2)
        for t, r in zip(t_span[:-1], t_span[1:]):
            x_in = torch.cat([x, x])
            mask_in = torch.cat([mask, mask])
            mu_in = torch.cat([mu, torch.zeros_like(mu)])
            t_in = torch.stack([t, t])
            spks_in = torch.cat([spks, torch.zeros_like(spks)])
            cond_in = torch.cat([cond, torch.zeros_like(cond)])
            d = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            d, cfg_d = d[:B], d[B:]
            d = (1.0 + cfg_rate) * d - cfg_rate * cfg_d
            x = x + (r - t) * d
        return x


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embedding = nn.Embedding(TOKEN_VOCAB, E)
        self.spk_embed_affine_layer = nn.Linear(SPK, MEL)
        self.encoder = UpsampleConformerEncoder()
        self.encoder_proj = nn.Linear(E, MEL)
        self.decoder = CFM()

    def inference(self, token, prompt_token, prompt_feat, embedding, z):
        embedding = self.spk_embed_affine_layer(F.normalize(embedding[None], dim=1))
        tok = torch.cat([prompt_token, token])[None]
        h = self.encoder(self.input_embedding(tok))
        mel1 = prompt_feat.shape[0]
        h = self.encoder_proj(h)
        conds = torch.zeros(1, h.shape[1], MEL)
        conds[0, :mel1] = prompt_feat
        mask = torch.ones(1, 1, h.shape[1])
        feat = self.decoder(h.transpose(1, 2), mask, embedding, conds.transpose(1, 2), z)
        return feat[:, :, mel1:], h


# ------------------------------------------------------------------ HiFT vocoder ("mel2wav.")
BASE = 16
UP_RATES, UP_KERNELS = [8, 5, 3], [16, 11, 7]
RES_KERNELS, RES_DIL = [3, 7, 11], [[1, 3, 5]] * 3
SRC_KERNELS = [7, 7, 11]
N_FFT, HOP = 16, 4
HARMONICS = 8
F0_CH = 16
wn = torch.nn.utils.parametrizations.weight_norm


class Snake(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(c))

    def forward(self, x):
        a = self.alpha[None, :, None]
        return x + (1.0 / (a + 1e-9)) * torch.sin(x * a) ** 2


def get_padding(k, d=1):
    return int((k * d - d) / 2)


class ResBlock(nn.Module):
    def __init__(self, c, k, dil):
        super().__init__()
        self.convs1 = nn.ModuleList([wn(nn.Conv1d(c, c, k, 1, dilation=d, padding=get_padding(k, d))) for d in dil])
        self.convs2 = nn.ModuleList([wn(nn.Conv1d(c, c, k, 1, dilation=1, padding=get_padding(k, 1))) for _ in dil])
        self.activations1 = nn.ModuleList([Snake(c) for _ in dil])
        self.activations2 = nn.ModuleList([Snake(c) for _ in dil])

    def forward(self, x):
        for i in range(len(self.convs1)):
            xt = self.convs2[i](self.activations2[i](self.convs1[i](self.activations1[i](x))))
            x = xt + x
        return x


class F0Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for i in range(5):
            layers += [wn(nn.Conv1d(MEL if i == 0 else F0_CH, F0_CH, 3, padding=1)), nn.ELU()]
        self.condnet = nn.Sequential(*layers)
        self.classifier = nn.Linear(F0_CH, 1)

    def forward(self, x):
        return torch.abs(self.classifier(self.condnet(x).transpose(1, 2)).squeeze(-1))


class SourceModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_linear = nn.Linear(HARMONICS + 1, 1)

    def forward(self, f0_up, phases, noise):
        """f0_up [L]; phases [H+1] (the first 0); noise [H+1, L] standard normal."""
        sr = 24000
        Fm = torch.stack([f0_up * (i + 1) / sr for i in range(HARMONICS + 1)])   # [H+1, L]
        theta = 2 * np.pi * (torch.cumsum(Fm, dim=-1) % 1)
        sine = 0.1 * torch.sin(theta + phases[:, None])
        uv = (f0_up > 10).float()
        amp = uv * 0.003 + (1 - uv) * 0.1 / 3
        sine = sine * uv + amp * noise
        return torch.tanh(self.l_linear(sine.T)).T   # [1, L]


class HiFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_source = SourceModule()
        self.conv_pre = wn(nn.Conv1d(MEL, BASE, 7, 1, padding=3))
        self.ups = nn.ModuleList([wn(nn.ConvTranspose1d(BASE // 2 ** i, BASE // 2 ** (i + 1), k, u, padding=(k - u) // 2))
                                  for i, (u, k) in enumerate(zip(UP_RATES, UP_KERNELS))])
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        cum = np.cumprod([1] + UP_RATES[::-1][:-1])
        for i, (u, k) in enumerate(zip(cum[::-1], SRC_KERNELS)):
            c = BASE // 2 ** (i + 1)
            self.source_downs.append(nn.Conv1d(N_FFT + 2, c, 1, 1) if u == 1
                                     else nn.Conv1d(N_FFT + 2, c, int(u) * 2, int(u), padding=int(u) // 2))
            self.source_resblocks.append(ResBlock(c, k, [1, 3, 5]))
        self.resblocks = nn.ModuleList()
        for i in range(len(UP_RATES)):
            c = BASE // 2 ** (i + 1)
            for k, d in zip(RES_KERNELS, RES_DIL):
                self.resblocks.append(ResBlock(c, k, d))
        self.conv_post = wn(nn.Conv1d(c, N_FFT + 2, 7, 1, padding=3))
        self.f0_predictor = F0Predictor()
        self.window = torch.hann_window(N_FFT)

    def decode(self, x, s):
        spec = torch.view_as_real(torch.stft(s, N_FFT, HOP, N_FFT, window=self.window, return_complex=True))
        s_stft = torch.cat([spec[..., 0], spec[..., 1]], dim=1)
        x = self.conv_pre(x)
        for i in range(len(UP_RATES)):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            if i == len(UP_RATES) - 1:
                x = F.pad(x, (1, 0), mode="reflect")
            x = x + self.source_resblocks[i](self.source_downs[i](s_stft))
            xs = None
            for j in range(3):
                r = self.resblocks[i * 3 + j](x)
                xs = r if xs is None else xs + r
            x = xs / 3
        x = self.conv_post(F.leaky_relu(x))
        mag = torch.clip(torch.exp(x[:, :N_FFT // 2 + 1]), max=1e2)
        ph = torch.sin(x[:, N_FFT // 2 + 1:])
        y = torch.istft(torch.complex(mag * torch.cos(ph), mag * torch.sin(ph)), N_FFT, HOP, N_FFT, window=self.window)
        return torch.clamp(y, -0.99, 0.99)


# ------------------------------------------------------------------ S3 tokenizer ("tokenizer.")
S3_STATE, S3_HEADS, S3_MELS = 128, 2, 128


class S3Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(S3_STATE, S3_STATE)
        self.key = nn.Linear(S3_STATE, S3_STATE, bias=False)
        self.value = nn.Linear(S3_STATE, S3_STATE)
        self.out = nn.Linear(S3_STATE, S3_STATE)
        self.fsmn_block = nn.Conv1d(S3_STATE, S3_STATE, 31, groups=S3_STATE, bias=False)

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        q = self.query(x).view(b, t, S3_HEADS, -1)
        k = self.key(x).view(b, t, S3_HEADS, -1)
        v = self.value(x).view(b, t, S3_HEADS, -1)
        rot = lambda y: torch.cat((-y[..., y.shape[-1] // 2:], y[..., :y.shape[-1] // 2]), dim=-1)
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
        vv = v.reshape(b, t, -1)
        fsm = self.fsmn_block(F.pad(vv.transpose(1, 2), (15, 15))).transpose(1, 2) + vv
        scale = (d // S3_HEADS) ** -0.25
        qk = (q.permute(0, 2, 1, 3) * scale) @ (k.permute(0, 2, 3, 1) * scale)
        w = torch.softmax(qk, dim=-1)
        o = (w @ v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.out(o) + fsm


class S3Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = S3Attn()
        self.attn_ln = nn.LayerNorm(S3_STATE, eps=1e-5)
        self.mlp = nn.Sequential(nn.Linear(S3_STATE, 4 * S3_STATE), nn.GELU(), nn.Linear(4 * S3_STATE, S3_STATE))
        self.mlp_ln = nn.LayerNorm(S3_STATE)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_ln(x), cos, sin)
        return x + self.mlp(self.mlp_ln(x))


class S3Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(S3_MELS, S3_STATE, 3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(S3_STATE, S3_STATE, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList([S3Block()])

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x)).permute(0, 2, 1)
        T = x.shape[1]
        freqs = 1.0 / (10000 ** (torch.arange(0, 64, 2)[:32].float() / 64))
        ang = torch.outer(torch.arange(T).float(), freqs)
        cos = torch.cat([ang.cos(), ang.cos()], -1)[None, :, None]
        sin = torch.cat([ang.sin(), ang.sin()], -1)[None, :, None]
        for b in self.blocks:
            x = b(x, cos, sin)
        return x


class Codebook(nn.Module):
    def __init__(self):
        super().__init__()
        self.project_down = nn.Linear(S3_STATE, 8)


class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self._codebook = Codebook()


class S3Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = S3Encoder()
        self.quantizer = Quantizer()
        self.mel_filters = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=S3_MELS)).float()

    def log_mel(self, audio):
        stft = torch.stft(audio, 400, 160, window=torch.hann_window(400), return_complex=True)
        mag = stft[..., :-1].abs() ** 2
        log_spec = torch.clamp(self.mel_filters @ mag, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0

    def forward(self, wav, max_len=None):
        mel = self.log_mel(wav)
        if max_len is not None:
            mel = mel[..., :max_len * 4]
        h = self.encoder(mel[None])
        h = self.quantizer._codebook.project_down(h).tanh() * 0.9990000128746033
        h = h.round() + 1
        powers = torch.pow(3, torch.arange(8).float())
        return (h * powers).sum(-1)[0].int(), mel


# ------------------------------------------------------------------ CAMPPlus ("speaker_encoder.")
M_CH = 8
GROWTH, BN_SIZE, INIT_CH = 8, 2, 16
LAYERS = (2, 2, 2)


def bnr(c):
    s = nn.Sequential()
    s.add_module("batchnorm", nn.BatchNorm1d(c))
    s.add_module("relu", nn.ReLU())
    return s


class BasicResBlock(nn.Module):
    def __init__(self, i, p, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(i, p, 3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(p)
        self.conv2 = nn.Conv2d(p, p, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(p)
        self.shortcut = nn.Sequential()
        if stride != 1 or i != p:
            self.shortcut = nn.Sequential(nn.Conv2d(i, p, 1, stride=(stride, 1), bias=False), nn.BatchNorm2d(p))

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o)) + self.shortcut(x)
        return F.relu(o)


class FCM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, M_CH, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(M_CH)
        self.layer1 = nn.Sequential(BasicResBlock(M_CH, M_CH, 2), BasicResBlock(M_CH, M_CH, 1))
        self.layer2 = nn.Sequential(BasicResBlock(M_CH, M_CH, 2), BasicResBlock(M_CH, M_CH, 1))
        self.conv2 = nn.Conv2d(M_CH, M_CH, 3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(M_CH)

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
        o = self.layer2(self.layer1(o))
        o = F.relu(self.bn2(self.conv2(o)))
        return o.reshape(o.shape[0], o.shape[1] * o.shape[2], o.shape[3])


class CAMLayer(nn.Module):
    def __init__(self, bn_ch, out_ch, k, dil):
        super().__init__()
        self.linear_local = nn.Conv1d(bn_ch, out_ch, k, padding=(k - 1) // 2 * dil, dilation=dil, bias=False)
        self.linear1 = nn.Conv1d(bn_ch, bn_ch // 2, 1)
        self.linear2 = nn.Conv1d(bn_ch // 2, out_ch, 1)

    def forward(self, x):
        y = self.linear_local(x)
        seg = F.avg_pool1d(x, kernel_size=100, stride=100, ceil_mode=True)
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, 100).reshape(*shape[:-1], -1)[..., :x.shape[-1]]
        ctx = F.relu(self.linear1(x.mean(-1, keepdim=True) + seg))
        return y * torch.sigmoid(self.linear2(ctx))


class CAMDenseLayer(nn.Module):
    def __init__(self, i, o, bn_ch, k, dil):
        super().__init__()
        self.nonlinear1 = bnr(i)
        self.linear1 = nn.Conv1d(i, bn_ch, 1, bias=False)
        self.nonlinear2 = bnr(bn_ch)
        self.cam_layer = CAMLayer(bn_ch, o, k, dil)

    def forward(self, x):
        return self.cam_layer(self.nonlinear2(self.linear1(self.nonlinear1(x))))


class CAMBlock(nn.ModuleList):
    def __init__(self, n, i, o, bn_ch, k, dil):
        super().__init__()
        for j in range(n):
            self.add_module("tdnnd%d" % (j + 1), CAMDenseLayer(i + j * o, o, bn_ch, k, dil))

    def forward(self, x):
        for l in self:
            x = torch.cat([x, l(x)], dim=1)
        return x


class TDNN(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.linear = nn.Conv1d(i, o, 5, stride=2, padding=2, bias=False)
        self.nonlinear = bnr(o)

    def forward(self, x):
        return self.nonlinear(self.linear(x))


class Transit(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.nonlinear = bnr(i)
        self.linear = nn.Conv1d(i, o, 1, bias=False)

    def forward(self, x):
        return self.linear(self.nonlinear(x))


class Dense(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.linear = nn.Conv1d(i, o, 1, bias=False)
        self.nonlinear = nn.Sequential()
        self.nonlinear.add_module("batchnorm", nn.BatchNorm1d(o, affine=False))

    def forward(self, x):
        return self.nonlinear(self.linear(x.unsqueeze(-1)).squeeze(-1))


class CAMPPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = FCM()
        ch = M_CH * (80 // 8)
        self.xvector = nn.Sequential()
        self.xvector.add_module("tdnn", TDNN(ch, INIT_CH))
        ch = INIT_CH
        for i, (n, k, d) in enumerate(zip(LAYERS, (3, 3, 3), (1, 2, 2))):
            self.xvector.add_module("block%d" % (i + 1), CAMBlock(n, ch, GROWTH, BN_SIZE * GROWTH, k, d))
            ch += n * GROWTH
            self.xvector.add_module("transit%d" % (i + 1), Transit(ch, ch // 2))
            ch //= 2
        self.xvector.add_module("out_nonlinear", bnr(ch))
        self.dense = Dense(ch * 2, SPK)

    def forward(self, feats):
        x = self.xvector(self.head(feats.permute(0, 2, 1)))
        stats = torch.cat([x.mean(-1), x.std(-1, unbiased=True)], dim=-1)
        return self.dense(stats)


def kaldi_fbank(wave, num_mel_bins=80, sr=16000):
    """torchaudio.compliance.kaldi.fbank with its defaults (no dither), transcribed."""
    wave = wave.double()
    win, shift, padded = 400, 160, 512
    n = 1 + (len(wave) - win) // shift
    frames = torch.stack([wave[i * shift:i * shift + win] for i in range(n)])
    frames = frames - frames.mean(dim=1, keepdim=True)
    prev = torch.cat([frames[:, :1], frames[:, :-1]], dim=1)
    frames = frames - 0.97 * prev
    i = torch.arange(win, dtype=torch.float64)
    window = (0.5 - 0.5 * torch.cos(2 * math.pi * i / (win - 1))) ** 0.85
    frames = F.pad(frames * window, (0, padded - win))
    power = torch.fft.rfft(frames).abs() ** 2
    mel = lambda f: 1127.0 * math.log(1.0 + f / 700.0)
    low, high = mel(20.0), mel(sr / 2)
    delta = (high - low) / (num_mel_bins + 1)
    bins = torch.zeros(num_mel_bins, padded // 2 + 1, dtype=torch.float64)
    for m in range(num_mel_bins):
        left, center, right = low + m * delta, low + (m + 1) * delta, low + (m + 2) * delta
        for k in range(padded // 2):
            f = mel(k * sr / padded)
            if left < f < right:
                bins[m, k] = (f - left) / (center - left) if f <= center else (right - f) / (right - center)
    e = power @ bins.T
    e = torch.clamp(e, min=float(torch.finfo(torch.float32).eps)).log()
    return e.float()


# ------------------------------------------------------------------ S3Gen 24 kHz mel
def s3gen_mel(y):
    basis = torch.from_numpy(librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=8000)).float()
    y = F.pad(y[None, None], (720, 720), mode="reflect")[0]
    spec = torch.view_as_real(torch.stft(y, 1920, hop_length=480, win_length=1920, window=torch.hann_window(1920),
                                         center=False, return_complex=True))
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(basis @ spec, min=1e-5))[0]   # [80, T]


# ================================================================== build and save
def half(v):
    return (v.half() if v.is_floating_point() else v).contiguous()


def main():
    d = f"{out}/chatterbox"
    import os
    os.makedirs(d, exist_ok=True)

    ve = VoiceEncoder()
    init(ve, 0.4)
    t3 = T3()
    init(t3, 0.2)
    flow = Flow()
    init(flow)
    hift = HiFT()
    init(hift)
    tok = S3Tokenizer()
    init(tok)
    camp = CAMPPlus()
    init(camp)
    for m in camp.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                bn_init(m)
            else:
                m.running_mean.copy_(0.1 * torch.randn_like(m.running_mean))
                m.running_var.copy_(1 + 0.2 * torch.rand_like(m.running_var))
    for m in hift.modules():
        if isinstance(m, Snake):
            m.alpha.copy_(1 + 0.3 * torch.rand_like(m.alpha))
    # a gentle output layer keeps the waveform inside the +-0.99 clamp
    hift.conv_post.parametrizations.weight.original0.mul_(0.2)
    # weights are stored in float16 (small files); the reference computes with those values
    for m in (ve, t3, flow, hift, tok, camp):
        m.eval()
        for t in list(m.parameters()) + list(m.buffers()):
            if t.is_floating_point():
                t.copy_(t.half().float())

    save_file({k: half(v) for k, v in ve.state_dict().items()}, f"{d}/ve.safetensors")
    save_file({k: half(v) for k, v in t3.state_dict().items()}, f"{d}/t3_cfg.safetensors")
    s3 = {}
    s3.update({"flow." + k: v for k, v in flow.state_dict().items()})
    s3.update({"mel2wav." + k: v for k, v in hift.state_dict().items() if k != "window"})
    s3.update({"tokenizer." + k: v for k, v in tok.state_dict().items()})
    # the real model keeps the dense layer inside xvector (after its stats pooling)
    s3.update({"speaker_encoder." + ("xvector." + k if k.startswith("dense.") else k): v
               for k, v in camp.state_dict().items()})
    save_file({k: half(v) for k, v in s3.items()}, f"{d}/s3gen.safetensors")

    # a reference voice: 1.3 seconds of a buzzy vowel with a pause, at 24 kHz and 16 kHz
    sr24 = 24000
    tt = np.arange(int(1.3 * sr24)) / sr24
    f0 = 140 + 20 * np.sin(2 * np.pi * 1.5 * tt)
    ph = 2 * np.pi * np.cumsum(f0) / sr24
    voice = (0.3 * np.sin(ph) + 0.15 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * tt))
    voice[int(0.6 * sr24):int(0.75 * sr24)] *= 0.01
    voice[:int(0.15 * sr24)] *= 0.001
    voice = np.round(voice + 0.002 * np.random.randn(len(voice)), 5).astype(np.float32)
    # rounded as stored, so the reference reads what the test reads (the log features amplify any change)
    voice16 = np.round(librosa.resample(voice, orig_sr=sr24, target_sr=16000), 5).astype(np.float32)

    exp = {"voice24": tolist(torch.from_numpy(voice), 5), "voice16": tolist(torch.from_numpy(voice16), 5)}

    # voice encoder
    emb, trimmed, vmel = ve_embed(ve, voice16)
    exp["ve_trimmed"] = trimmed
    exp["ve_mel"] = tolist(torch.from_numpy(vmel), 5)
    exp["ve_embedding"] = tolist(emb)

    # S3 tokenizer, CAMPPlus, 24 kHz mel
    w16 = torch.from_numpy(voice16)
    codes, s3mel = tok(w16)
    exp["s3_mel"] = tolist(s3mel, 5)
    exp["s3_codes"] = codes.tolist()
    codes_t3, _ = tok(w16[:6 * 16000], max_len=PROMPT_LEN)
    exp["s3_codes_prompt"] = codes_t3.tolist()
    fb = kaldi_fbank(w16)
    exp["fbank"] = tolist(fb, 4)
    xv = camp((fb - fb.mean(dim=0, keepdim=True))[None])[0]
    exp["xvector"] = tolist(xv)
    pmel = s3gen_mel(torch.from_numpy(voice))
    exp["prompt_mel"] = tolist(pmel, 5)

    # T3: greedy speech tokens for "hello world" token ids
    text_ids = [START_TEXT, 7, 8, 9, 2, 10, 11, 12, STOP_TEXT]
    prompt = torch.tensor([codes_t3.tolist()[:PROMPT_LEN]]).long()
    speech, cond = t3.inference(emb, prompt, 0.5, text_ids, max_new=20)
    exp["t3_text_ids"] = text_ids
    exp["t3_prompt_tokens"] = prompt[0].tolist()
    exp["t3_cond"] = tolist(cond[0])
    exp["t3_speech"] = speech
    speech2, _ = t3.inference(emb, prompt, 1.2, text_ids, max_new=20, cfg_weight=0.0, repetition_penalty=1.0)
    exp["t3_speech_exaggerated_nocfg"] = speech2

    # flow: tokens -> mel with given noise
    ptok = torch.tensor(codes.tolist()[:pmel.shape[1] // 2]).long()
    pfeat = pmel.T[:2 * len(ptok)]
    token = torch.tensor([5, 900, 2000, 6000, 31, 77, 4000, 123, 3]).long()
    T2 = 2 * (len(ptok) + len(token))
    z = torch.randn(1, MEL, T2)
    mel, h = flow.inference(token, ptok, pfeat, xv, z)
    exp["flow_prompt_tokens"] = ptok.tolist()
    exp["flow_tokens"] = token.tolist()
    exp["flow_noise"] = tolist(z[0], 5)
    exp["flow_mu"] = tolist(h[0], 5)
    exp["flow_mel"] = tolist(mel[0], 5)

    # HiFT: f0, the source (with given phases and noise) and the waveform
    f0p = hift.f0_predictor(mel)[0]
    L = mel.shape[2] * 480
    f0_up = f0p.repeat_interleave(480)
    phases = torch.cat([torch.zeros(1), (torch.rand(HARMONICS) * 2 - 1) * math.pi])
    noise = torch.randn(HARMONICS + 1, L)
    s = hift.m_source(f0_up, phases, noise)
    s_zero = hift.m_source(f0_up, torch.zeros(HARMONICS + 1), torch.zeros(HARMONICS + 1, L))
    wav = hift.decode(mel, s)[0]
    n_trim = 480
    fade = torch.zeros(2 * n_trim)
    fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
    wav[:len(fade)] *= fade
    exp["f0"] = tolist(f0p)
    exp["source_phases"] = tolist(phases)
    exp["source_quiet"] = tolist(s_zero[0][::7], 6)
    exp["hift_source"] = tolist(s[0], 5)
    exp["hift_audio"] = tolist(wav, 5)
    json.dump(exp, open(f"{d}/expected.json", "w"))

    # the built-in voice, as conds.pt
    conds = {"t3": {"speaker_emb": emb[None].clone(), "clap_emb": None,
                    "cond_prompt_speech_tokens": prompt.clone(), "cond_prompt_speech_emb": None,
                    "emotion_adv": 0.5 * torch.ones(1, 1, 1)},
             "gen": {"prompt_token": ptok[None].clone(), "prompt_token_len": torch.tensor([len(ptok)]),
                     "prompt_feat": pfeat[None].clone(), "prompt_feat_len": None, "embedding": xv[None].clone()}}
    torch.save(conds, f"{d}/conds.pt")

    # a grapheme tokenizer.json in the layout of the real one
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {"[STOP]": 0, "[UNK]": 1, "[SPACE]": 2}
    ids = (i for i in range(3, TEXT_VOCAB) if i != START_TEXT)
    for c in "abcdefghijklmnopqrstuvwxyz.,!?'":
        vocab[c] = next(ids)
    vocab["[START]"] = START_TEXT                  # the id T3 starts text with, as in the real vocabulary
    merges = [("h", "e"), ("l", "l"), ("he", "ll")]
    for a, b in merges:
        vocab.setdefault(a + b, next(ids))
    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.add_special_tokens(["[START]", "[STOP]", "[UNK]", "[SPACE]"])
    tk.save(f"{d}/tokenizer.json")
    samples = ["Hello world.", "Hi, hello!"]
    json.dump({"texts": samples, "ids": [tk.encode(t.replace(" ", "[SPACE]")).ids for t in samples]},
              open(f"{d}/tokenizer_expected.json", "w"))
    print("T3", speech, speech2)
    print("tokens", codes.tolist()[:10], "prompt", codes_t3.tolist())


main()
