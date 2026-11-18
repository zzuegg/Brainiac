# A tiny random F5-TTS (DiT + flow matching) and Vocos vocoder, saved like SWivid/F5-TTS
# (model.safetensors with "ema_model." keys, vocab.txt) and charactr/vocos-mel-24khz
# (config.yaml, pytorch_model.bin), with reference outputs from the f5_tts and vocos packages.
# The flow's start noise comes from a formula, so the reference is deterministic (F5Test
# draws the same values): value k is frac(sin(k) * 43758.5453) * 2 - 1.
# Run: PYTHONPATH=<f5_tts and vocos sources>:<folder with stubs.py> python make_f5.py <this folder>
# (stubs.py replaces the training-only imports of f5_tts with mocks)
import json, os, sys, tempfile
import numpy as np
import torch
import soundfile as sf
import yaml
from safetensors.torch import save_file
import stubs

u = stubs.load('f5_tts.infer.utils_infer')
from f5_tts.model import CFM, DiT
from f5_tts.model.utils import convert_char_to_pinyin
from vocos import Vocos
from vocos.feature_extractors import MelSpectrogramFeatures
from vocos.models import VocosBackbone
from vocos.heads import ISTFTHead

out = sys.argv[1]
f5_dir, vocos_dir = f"{out}/f5", f"{out}/vocos"
os.makedirs(f5_dir, exist_ok=True)
os.makedirs(vocos_dir, exist_ok=True)
torch.manual_seed(0)

SR, NFFT, HOP, MELS = 24000, 256, 64, 8
u.hop_length, u.win_length, u.n_fft, u.n_mel_channels = HOP, NFFT, NFFT, MELS   # infer_process reads these

def pseudo(n):
    i = np.arange(n, dtype=np.float64)
    v = np.sin(i) * 43758.5453
    return ((v - np.floor(v)) * 2 - 1).astype(np.float32)

def randomize(module, scale):
    with torch.no_grad():
        for name, p in module.named_parameters():
            fan = p.shape[1] * (p[0][0].numel() if p.dim() > 2 else 1) if p.dim() > 1 else 4
            p.copy_(torch.randn_like(p) * scale / np.sqrt(fan))

# --- Vocos ---------------------------------------------------------------------------------
vocos_cfg = {
    "feature_extractor": {"class_path": "vocos.feature_extractors.MelSpectrogramFeatures",
                          "init_args": {"sample_rate": SR, "n_fft": NFFT, "hop_length": HOP, "n_mels": MELS,
                                        "padding": "center"}},
    "backbone": {"class_path": "vocos.models.VocosBackbone",
                 "init_args": {"input_channels": MELS, "dim": 16, "intermediate_dim": 48, "num_layers": 2}},
    "head": {"class_path": "vocos.heads.ISTFTHead",
             "init_args": {"dim": 16, "n_fft": NFFT, "hop_length": HOP, "padding": "same"}},
}
yaml.safe_dump(vocos_cfg, open(f"{vocos_dir}/config.yaml", "w"), sort_keys=False)
vocos = Vocos.from_hparams(f"{vocos_dir}/config.yaml")
randomize(vocos, 1.0)
with torch.no_grad():
    for n, p in vocos.named_parameters():
        if n.endswith("gamma"):
            p.copy_(torch.rand_like(p) * 0.5 + 0.1)
        if "norm" in n and n.endswith("weight"):
            p.copy_(1 + torch.randn_like(p) * 0.1)
torch.save(vocos.state_dict(), f"{vocos_dir}/pytorch_model.bin")
vocos = u.load_vocoder("vocos", is_local=True, local_path=vocos_dir, device="cpu")

# --- F5-TTS --------------------------------------------------------------------------------
chars = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-:")
open(f"{f5_dir}/vocab.txt", "w").write("".join(c + "\n" for c in chars))
vocab = {c: i for i, c in enumerate(chars)}
json.dump({"dim_head": 8}, open(f"{f5_dir}/config.json", "w"))
dit = DiT(dim=32, depth=2, heads=4, dim_head=8, ff_mult=2, mel_dim=MELS, text_num_embeds=len(chars), text_dim=16,
          conv_layers=2, text_mask_padding=True, attn_mask_enabled=False)
model = CFM(transformer=dit, mel_spec_kwargs=dict(n_fft=NFFT, hop_length=HOP, win_length=NFFT, n_mel_channels=MELS,
                                                   target_sample_rate=SR, mel_spec_type="vocos"),
            odeint_kwargs=dict(method="euler"), vocab_char_map=vocab)
randomize(model, 1.0)
state = {"ema_model." + k: v.contiguous() for k, v in model.state_dict().items()}
state["ema_model.step"] = torch.tensor(1250000)
save_file(state, f"{f5_dir}/model.safetensors")
model.eval()

# reference audio: a gliding tone with a little hiss, RMS below 0.1 so it is scaled up and back
n = 2400
t = np.arange(n) / SR
ref = (0.06 * np.sin(2 * np.pi * (180 + 900 * t) * t) + 0.01 * pseudo(n)).astype(np.float32)
ref_text = "Some call me nature. "
gen_text = "Hello,world! It's 3.14."

expected = {"sampleRate": SR, "reference": ref.tolist(), "refText": ref_text, "genText": gen_text}

# the mel features of the reference
with torch.no_grad():
    expected["referenceMel"] = model.mel_spec(torch.from_numpy(ref)[None])[0].numpy().round(5).tolist()
# the text as F5 spells it
expected["chars"] = convert_char_to_pinyin([ref_text + "  " + gen_text])[0]
cases = ["Hello,world! It's 3.14 and well-known U.S.A. e-mail abc_def (paren)test \"quote\"word 100%sure a+b",
         "Café naïve – em—dash ; “curly” ‘quotes’ a__b 1.2.3 50%off x- -x a..b"]
expected["textCases"] = [{"text": c, "chars": convert_char_to_pinyin([c])[0]} for c in cases]
expected["chunkCases"] = [{"text": c, "max": m, "chunks": u.chunk_text(c, max_chars=m)} for c, m in [
    ("One sentence. Two sentences, with a comma; and more! Is it done? Yes.", 30),
    ("Short. Also short. Another short one. And the last one here.", 20),
    ("A long sentence without any breaks at all that goes on and on.", 15)]]

# one network evaluation, with and without the conditions (classifier-free guidance packs both)
with torch.no_grad():
    L = 20
    x = torch.from_numpy(pseudo(L * MELS).reshape(1, L, MELS))
    cond = torch.zeros(1, L, MELS)
    cond[:, :8] = torch.from_numpy(np.cos(np.arange(8 * MELS)).astype(np.float32).reshape(1, 8, MELS))
    ids = [vocab.get(c, 0) for c in "hi there"]
    text = torch.tensor([ids])
    pred = dit(x=x, cond=cond, text=text, time=torch.tensor(0.3), cfg_infer=True, cache=False)
    expected["forward"] = {"x": x[0].numpy().tolist(), "cond": cond[0].numpy().tolist(), "ids": ids, "time": 0.3,
                           "pred": pred[0].numpy().round(5).tolist(), "nullPred": pred[1].numpy().round(5).tolist()}

# vocos on a mel
with torch.no_grad():
    mel = torch.from_numpy(np.cos(np.arange(MELS * 12) * 0.37).astype(np.float32).reshape(1, MELS, 12) * 2 - 3)
    expected["vocos"] = {"mel": mel[0].numpy().tolist(), "audio": vocos.decode(mel)[0].numpy().round(6).tolist()}

# the whole pipeline: chunk the text, estimate the length, sample the flow (4 Euler steps), vocode
real_randn = torch.randn
def fake_randn(*shape, **kw):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
        shape = tuple(shape[0])
    shape = tuple(int(s) for s in shape)
    return torch.from_numpy(pseudo(int(np.prod(shape))).reshape(shape)).to(kw.get("dtype") or torch.float32)
torch.randn = fake_randn
with tempfile.TemporaryDirectory() as d:
    wav = f"{d}/ref.wav"
    sf.write(wav, ref, SR, subtype="FLOAT")
    u.torchaudio.load = lambda f: (torch.from_numpy(sf.read(f, dtype="float32")[0])[None], sf.info(f).samplerate)
    audio, sr = u.torchaudio.load(wav)
    expected["referenceLoaded"] = audio[0].numpy().tolist()
    wave, _, spec = u.infer_process(wav, ref_text, gen_text, model, vocos, nfe_step=4, cfg_strength=2.0,
                                    sway_sampling_coef=-1.0, speed=1.0, device="cpu", progress=None)
torch.randn = real_randn
expected["steps"] = 4
expected["mel"] = spec.round(5).tolist()
expected["audio"] = wave.round(6).tolist()
json.dump(expected, open(f"{out}/f5/expected.json", "w"), ensure_ascii=False)
print("mel frames", spec.shape, "audio", wave.shape, "range", float(np.abs(wave).max()))
