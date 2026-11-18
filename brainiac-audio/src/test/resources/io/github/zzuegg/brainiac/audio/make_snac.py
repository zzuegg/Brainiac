# Tiny random SNAC decoders saved like hubertsiuzdak/snac_24khz (config.json,
# pytorch_model.bin), with reference audio from the snac package. The decoder's noise comes
# from a formula so the reference is deterministic (SnacTest draws the same values): in each
# noise block, value k is frac(sin(k) * 43758.5453) * 2 - 1.
# Run: PYTHONPATH=<snac sources> python make_snac.py <this folder>
import json, os, sys
import numpy as np
import torch
import snac.layers as layers
from snac import SNAC

out = sys.argv[1]
torch.manual_seed(1)

def pseudo(shape, **kw):
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.float64)
    v = np.sin(i) * 43758.5453
    return torch.from_numpy(((v - np.floor(v)) * 2 - 1).astype(np.float32)).reshape(shape)
layers.torch.randn = lambda shape, **kw: pseudo(tuple(shape))

def make(name, attn):
    cfg = {"sampling_rate": 24000, "encoder_dim": 4, "encoder_rates": [2, 2], "decoder_dim": 128,
           "decoder_rates": [4, 2], "attn_window_size": attn, "codebook_size": 16, "codebook_dim": 4,
           "vq_strides": [4, 2, 1], "noise": True, "depthwise": True}
    d = f"{out}/{name}"
    os.makedirs(d, exist_ok=True)
    json.dump(cfg, open(f"{d}/config.json", "w"), indent=1)
    model = SNAC(**cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("alpha"):
                p.copy_(torch.rand_like(p) + 0.5)
            elif n.endswith("original0"):          # weight-norm magnitudes
                p.copy_(torch.rand_like(p) * 0.6 + 0.4)
            else:
                p.copy_(torch.randn_like(p) * 0.25)
    # the encoder is left out: only decoding is tested
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}, f"{d}/pytorch_model.bin")
    model.eval()
    frames = 4
    codes = [torch.tensor([[(5 * i + 3) % 16 for i in range(frames)]]),
             torch.tensor([[(7 * i + 1) % 16 for i in range(2 * frames)]]),
             torch.tensor([[(3 * i + 11) % 16 for i in range(4 * frames)]])]
    with torch.no_grad():
        audio = model.decode(codes)[0, 0]
    json.dump({"codes": [c[0].tolist() for c in codes], "audio": audio.numpy().round(6).tolist()},
              open(f"{d}/expected.json", "w"))
    print(name, audio.shape, float(audio.abs().max()))

make("snac", None)
make("snac-attention", 4)
