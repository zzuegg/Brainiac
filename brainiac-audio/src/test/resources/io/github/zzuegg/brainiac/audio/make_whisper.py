# A tiny random Whisper saved as Hugging Face would, with reference encoder outputs and
# decoder logits from transformers, and log-mel features computed as OpenAI's whisper does.
# Run: python make_whisper.py <this folder>
import json, os, sys, torch, numpy as np
from transformers import WhisperConfig, WhisperForConditionalGeneration
from transformers.audio_utils import mel_filter_bank
out = sys.argv[1]
torch.manual_seed(0)

# log-mel as whisper.audio.log_mel_spectrogram (filters as librosa.filters.mel, Slaney)
sr = 16000
t = np.arange(int(0.9 * sr)) / sr
audio = (0.4 * np.sin(2 * np.pi * 440 * t) * np.exp(-t) + 0.1 * np.sin(2 * np.pi * 1234 * t)
         + 0.05 * np.random.RandomState(1).randn(len(t))).astype(np.float32)
def log_mel(x, n_mels):
    filters = torch.tensor(mel_filter_bank(num_frequency_bins=201, num_mel_filters=n_mels, min_frequency=0.0,
                                           max_frequency=8000.0, sampling_rate=sr, norm="slaney",
                                           mel_scale="slaney").T, dtype=torch.float32)
    stft = torch.stft(torch.tensor(x), 400, 160, window=torch.hann_window(400), return_complex=True)
    mag = stft[..., :-1].abs() ** 2
    spec = (filters @ mag).clamp(min=1e-10).log10()
    spec = torch.maximum(spec, spec.max() - 8.0)
    return ((spec + 4.0) / 4.0).numpy()
mel80 = log_mel(audio, 80)
mel128 = log_mel(audio, 128)
os.makedirs(f"{out}/whisper", exist_ok=True)
json.dump({"audio": audio.round(7).tolist(), "mel80": mel80.round(6).tolist(), "mel128_first": mel128[:, :5].round(6).tolist()},
          open(f"{out}/mel.json", "w"))

# the model: 2 + 2 layers, 32 wide, 48 tokens, 40 encoder positions (80 frames)
cfg = WhisperConfig(vocab_size=48, num_mel_bins=80, d_model=32, encoder_layers=2, encoder_attention_heads=4,
                    decoder_layers=2, decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64,
                    max_source_positions=40, max_target_positions=32, decoder_start_token_id=1, pad_token_id=0,
                    eos_token_id=2, bos_token_id=2)
model = WhisperForConditionalGeneration(cfg).eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if "layer_norm" in n:
            p.uniform_(0.5, 1.5) if n.endswith("weight") else p.normal_(0, 0.1)
        else:
            p.normal_(0, 0.2)
d = f"{out}/whisper"
model.save_pretrained(d, safe_serialization=True)
for f in os.listdir(d):
    if f not in ("config.json", "model.safetensors"):
        os.remove(f"{d}/{f}")
feats = torch.tensor(mel80[:, :80]).unsqueeze(0)
ids = [1, 5, 9, 3, 7, 11]
with torch.no_grad():
    enc = model.model.encoder(feats).last_hidden_state
    logits = model(input_features=feats, decoder_input_ids=torch.tensor([ids])).logits[0]
json.dump({"ids": ids, "encoder": enc[0].numpy().round(6).tolist(), "logits": logits.numpy().round(6).tolist()},
          open(f"{d}/expected.json", "w"))
