# A tiny random Kokoro (StyleTTS2 + iSTFTNet) saved like hexgrad/Kokoro-82M (config.json,
# a .pth of per-module state dicts with "module." prefixes, a voice pack), with reference
# audio from the kokoro package with deterministic source noise.
# Run: PYTHONPATH=<kokoro and misaki sources> python make_kokoro.py <this folder>/kokoro
import json, os, sys, torch, numpy as np
import torch.nn as nn
from torch.nn.utils import weight_norm
import types
# load the model code without the package's pipeline (and its G2P dependencies)
sys.modules['kokoro'] = types.ModuleType('kokoro')
sys.modules['kokoro'].__path__ = [p + '/kokoro' for p in sys.path if os.path.isfile(p + '/kokoro/istftnet.py')][:1]
import kokoro.istftnet as ist
import kokoro.model as km

out = sys.argv[1]
os.makedirs(f"{out}/voices", exist_ok=True)
D, R, O = 24, 6, 16   # stand-ins for the decoder's 1024, 64 and 512 channels

class TinyDecoder(nn.Module):
    def __init__(self, dim_in, style_dim, dim_out, resblock_kernel_sizes, upsample_rates, upsample_initial_channel,
                 resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size):
        super().__init__()
        self.encode = ist.AdainResBlk1d(dim_in + 2, D, style_dim)
        self.decode = nn.ModuleList()
        for _ in range(3):
            self.decode.append(ist.AdainResBlk1d(D + 2 + R, D, style_dim))
        self.decode.append(ist.AdainResBlk1d(D + 2 + R, O, style_dim, upsample=True))
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(dim_in, R, kernel_size=1)))
        self.generator = ist.Generator(style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel,
                                       resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft,
                                       gen_istft_hop_size)
    forward = ist.Decoder.forward

km.Decoder = TinyDecoder
# the source noise from a formula instead of a random generator, so the reference is
# deterministic (KokoroTest draws the same values): value k is frac(sin(k) * 43758.5453) * 2 - 1
def pseudo_noise(x, *a, **k):
    i = np.arange(x.numel(), dtype=np.float64)
    v = np.sin(i) * 43758.5453
    return torch.from_numpy(((v - np.floor(v)) * 2 - 1).astype(np.float32)).reshape(x.shape)
ist.torch.randn_like = pseudo_noise

phonemes = list(" .,!?ˈˌəɪʊæɑɔɛʌθðŋʃʒʧʤabdefhijklmnoprstuvwz")
vocab = {p: i + 1 for i, p in enumerate(phonemes)}
config = {
    "istftnet": {"upsample_kernel_sizes": [4, 4], "upsample_rates": [2, 2], "gen_istft_hop_size": 2,
                 "gen_istft_n_fft": 8, "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]],
                 "resblock_kernel_sizes": [3, 5], "upsample_initial_channel": O},
    "dim_in": 64, "dropout": 0.2, "hidden_dim": 16, "max_conv_dim": 512, "max_dur": 6, "multispeaker": True,
    "n_layer": 2, "n_mels": 80, "n_token": len(phonemes) + 1, "style_dim": 128, "text_encoder_kernel_size": 5,
    "plbert": {"hidden_size": 20, "num_attention_heads": 2, "intermediate_size": 28,
               "max_position_embeddings": 40, "num_hidden_layers": 2, "dropout": 0.1, "embedding_size": 12},
    "vocab": vocab,
}
json.dump(config, open(f"{out}/config.json", "w"), ensure_ascii=False, indent=1)

torch.manual_seed(0)
# build without loading weights
orig = torch.load
km.torch.load = lambda *a, **k: {}
model = km.KModel(config=config, model="none").eval()
km.torch.load = orig
with torch.no_grad():
    for n, p in model.named_parameters():
        if n.endswith("weight_g"):
            p.uniform_(0.5, 1.5)
        elif "LayerNorm" in n or "layer_norm" in n:
            p.uniform_(0.8, 1.2) if n.endswith("weight") else p.normal_(0, 0.1)
        elif "alpha" in n:
            p.uniform_(0.5, 1.5)
        else:
            p.normal_(0, 0.15)
    model.predictor.F0_proj.bias.fill_(120.0)
    model.predictor.duration_proj.linear_layer.weight.normal_(0, 4.0)
    model.predictor.duration_proj.linear_layer.bias.fill_(0.0)
    model.decoder.generator.conv_post.weight_g.fill_(0.2)
state = {k: {"module." + n: t for n, t in getattr(model, k).state_dict().items()}
         for k in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]}
torch.save(state, f"{out}/kokoro.pth")

voice = torch.randn(12, 1, 256) * 0.5  # KModel splits the style at 128
torch.save(voice, f"{out}/voices/test_voice.pt")

text = "hˈɛloʊ wˈɜld, ðɪs ɪz ə tˈɛst."
ps = "".join(c for c in text if c in vocab)
ids = [vocab[c] for c in ps]
ref_s = voice[len(ps) - 1] if len(ps) <= 12 else voice[11]
with torch.no_grad():
    o = model(ps, ref_s, speed=1, return_output=True)
    fast = model(ps, ref_s, speed=1.5, return_output=True)
json.dump({"phonemes": ps, "ids": ids, "durations": o.pred_dur.tolist(), "audio": o.audio.numpy().round(6).tolist(),
           "fast_durations": fast.pred_dur.tolist()}, open(f"{out}/expected.json", "w"), ensure_ascii=False)
print(len(ids), o.pred_dur.tolist(), o.audio.shape, float(o.audio.abs().max()))
