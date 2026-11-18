# Tiny random VITS models saved as Hugging Face would (weight-normalized convolutions as
# parametrizations), with reference waveforms from transformers with the noise off.
# Run: python make_vits.py <this folder>
import json, os, sys, torch
from transformers import VitsConfig, VitsModel, VitsTokenizer
out = sys.argv[1]
chars = ["<pad>", " ", "a", "b", "c", "d", "e", "h", "i", "l", "o", "r", "s", "t", "w", "'", "-"]
def make(name, **kw):
    torch.manual_seed(0)
    cfg = VitsConfig(vocab_size=len(chars), hidden_size=16, num_hidden_layers=2, num_attention_heads=2, window_size=2,
                     ffn_dim=24, ffn_kernel_size=3, flow_size=8, spectrogram_bins=9, upsample_initial_channel=16,
                     upsample_rates=[4, 2], upsample_kernel_sizes=[8, 4], resblock_kernel_sizes=[3, 5],
                     resblock_dilation_sizes=[[1, 3], [1, 2]], duration_predictor_filter_channels=12,
                     duration_predictor_flow_bins=4, depth_separable_num_layers=2, duration_predictor_num_flows=2,
                     prior_encoder_num_flows=2, prior_encoder_num_wavenet_layers=2, posterior_encoder_num_wavenet_layers=1,
                     wavenet_kernel_size=3, noise_scale=0.0, noise_scale_duration=0.0, sampling_rate=8000, **kw)
    model = VitsModel(cfg).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n and n.endswith("weight") and "parametrizations" not in n:
                p.uniform_(0.5, 1.5)
            elif "duration_predictor" in n:
                p.normal_(0, 0.15)
            else:
                p.normal_(0, 0.3)
    d = f"{out}/{name}"
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    vocab = {c: i for i, c in enumerate(chars)}
    json.dump(vocab, open(f"{d}/vocab.json", "w"))
    json.dump({"add_blank": True, "normalize": True, "phonemize": False, "is_uroman": False, "language": "eng",
               "tokenizer_class": "VitsTokenizer", "pad_token": "<pad>", "unk_token": "<unk>"},
              open(f"{d}/tokenizer_config.json", "w"))
    for f in os.listdir(d):
        if f not in ("config.json", "model.safetensors", "vocab.json", "tokenizer_config.json"):
            os.remove(f"{d}/{f}")
    tok = VitsTokenizer(f"{d}/vocab.json", add_blank=True, normalize=True, phonemize=False, language="eng")
    text = "Hello, World! Is it 'red-white'?"
    ids = tok(text).input_ids
    with torch.no_grad():
        spk = {"speaker_id": 1} if cfg.num_speakers > 1 else {}
        o = model(torch.tensor([ids]), **spk)
        fast = model(torch.tensor([ids]), speaking_rate=1.5, **spk)
    json.dump({"text": text, "ids": ids, "waveform": o.waveform[0].numpy().round(6).tolist(),
               "fast_length": int(fast.waveform.shape[1])}, open(f"{d}/expected.json", "w"))
make("vits")
make("vits-deterministic", use_stochastic_duration_prediction=False)
make("vits-speakers", num_speakers=3, speaker_embedding_size=6)
