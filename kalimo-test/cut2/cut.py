"""Cut the first sound out of whole word takes and build "<sound> ... wie <Wort>." clips.

usage: python -I cut.py <work dir with gen/judged tsv> <de|en> <out dir>
Inputs: <lang>-<sound>-word-seedN.wav and <lang>-<sound>-line-seedN.wav (whole Qwen3 takes) + judged.tsv (Whisper).
Phone timings: facebook/wav2vec2-lv-60-espeak-cv-ft CTC frames (20 ms), refined on the energy curve.
No resampling of the voice, no pitch change: continuants are lengthened by crossfading steady frames of the same take.
"""
import json
import os
import re
import sys
import unicodedata
import wave

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

sys.stdout.reconfigure(encoding='utf-8')
work, lang, out = sys.argv[1].rstrip('/\\') + '/', sys.argv[2], sys.argv[3].rstrip('/\\') + '/'
os.makedirs(out + 'candidates', exist_ok=True)
MODEL = 'D:/_Programming/Brainiac-models-extra/wav2vec2-lv-60-espeak-cv-ft'
fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
vocab = {v: k for k, v in json.load(open(MODEL + '/vocab.json', encoding='utf-8')).items()}
BLANK = model.config.pad_token_id
SPECIAL = {'<pad>', '<s>', '</s>', '<unk>'}

ACCEPT = {
    'm': {'m'}, 'l': {'l', 'ɫ'}, 's': {'s', 'z'}, 'r': {'ʁ', 'r', 'ʀ', 'ɹ', 'ɾ', 'χ'}, 'n': {'n'}, 'f': {'f'},
    'h': {'h'}, 'w': {'v', 'w', 'ʋ'}, 't': {'t', 'tʰ'}, 'k': {'k', 'kʰ'}, 'p': {'p', 'pʰ', 'ph'}, 'b': {'b', 'p'},
    'g': {'ɡ', 'g', 'k'}, 'd': {'d', 't'},
}
VOWEL_ACCEPT = {
    'de': {'a': {'a', 'aː', 'ɑ', 'ɑː'}, 'o': {'o', 'oː', 'ɔ'}, 'i': {'i', 'iː', 'ɪ'}, 'e': {'e', 'eː', 'ɛ', 'ɛː', 'iː', 'eɪ'},
           'u': {'u', 'uː', 'ʊ'}},
    'en': {'a': {'æ', 'a', 'ɐ'}, 'o': {'ɒ', 'ɑ', 'ɔ', 'ɑː', 'ɒː'}, 'e': {'ɛ', 'e', 'a', 'æ'}, 'u': {'ʌ', 'ɐ', 'ə', 'a'},
           'i': {'ɪ', 'i'}},
}[lang]
STOPS = set('tkpbdg')
VOWEL_CHARS = set('aeiouɑɛɪɔʊæəɐøœyʏɒʌɜɚɨ')
SR16 = 16000


def accept(sound):
    return VOWEL_ACCEPT[sound] if sound in 'aeiou' else ACCEPT[sound]


def load(path):
    w = wave.open(path)
    sr, n = w.getframerate(), w.getnframes()
    x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    return x, sr


def save(path, x, sr):
    y = np.clip(x, -1, 1)
    w = wave.open(path, 'wb')
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes((y * 32767).astype(np.int16).tobytes())
    w.close()


def to16(x, sr):
    if sr == SR16:
        return x
    t = np.arange(0, len(x) / sr, 1 / SR16)
    return np.interp(t, np.arange(len(x)) / sr, x).astype(np.float32)


def ctc(x, sr):
    """Phone spans [(phone, start s, end s)] from greedy CTC frames."""
    inp = fe(to16(x, sr), sampling_rate=SR16, return_tensors='pt')
    with torch.no_grad():
        ids = model(inp.input_values).logits.argmax(-1)[0].tolist()
    spans, prev = [], None
    for i, t in enumerate(ids):
        if t == BLANK or vocab.get(t, '?') in SPECIAL:
            prev = None
            continue
        if t == prev and spans and vocab.get(t) != 'ʔ':
            spans[-1][2] = (i + 1) * 0.02
        else:
            tok = re.sub(r'\d', '', vocab.get(t, '?'))
            tok = {'th': 't', 'kh': 'k', 'ph': 'p', 'tʰ': 't', 'kʰ': 'k', 'pʰ': 'p'}.get(tok, tok)
            if tok == 'ʔ' and not spans:
                prev = t
                continue
            spans.append([tok, i * 0.02, (i + 1) * 0.02])
        prev = t
    return spans


def phones(x, sr):
    pad = np.zeros(int(0.2 * sr), dtype=np.float32)
    return [s[0] for s in ctc(np.concatenate([pad, x, pad]), sr)]


def envelope(x, sr, hop_s=0.005):
    hop = int(hop_s * sr)
    n = len(x) // hop
    e = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    return 20 * np.log10(e + 1e-9), hop


def crossfade_join(a, b, n):
    if n <= 0 or len(a) < n or len(b) < n:
        return np.concatenate([a, b])
    w = np.linspace(0, 1, n, dtype=np.float32)
    mid = a[-n:] * (1 - w) + b[:n] * w
    return np.concatenate([a[:-n], mid, b[n:]])


def fade(x, sr, fin, fout):
    y = x.copy()
    ni, no = int(fin * sr), int(fout * sr)
    if ni:
        y[:ni] *= np.linspace(0, 1, ni)
    if no:
        y[-no:] *= np.linspace(1, 0, no)
    return y


def cut_sound(x, sr, sound):
    """Returns (bare sound, info dict) or (None, info) when the first phone is not the target."""
    spans = ctc(x, sr)
    info = {'phones': ' '.join(s[0] for s in spans)}
    if len(spans) < 2:
        info['why'] = 'too few phones'
        return None, info
    first, second = spans[0], spans[1]
    info['first'] = first[0]
    if first[0] not in accept(sound):
        info['why'] = 'first phone %s' % first[0]
        return None, info
    env, hop = envelope(x, sr)
    peak = env.max()
    floor = np.percentile(env[:max(4, int(0.15 / 0.005))], 50)
    above = np.nonzero(env > max(floor + 10, peak - 50))[0]
    onset = max(0, above[0] - 4) * hop if len(above) else 0
    b = second[1]
    lo, hi = int(max(onset / sr, b - 0.05) / 0.005), int(min(len(env) - 2, (b + 0.05) / 0.005))
    sm = np.convolve(env, np.ones(3) / 3, mode='same')
    d = np.diff(sm)
    if hi > lo + 1:
        if sound in 'aeiou':
            k = lo + int(np.argmin(d[lo:hi]))  # vowel -> consonant: steepest fall
        else:
            k = lo + int(np.argmax(d[lo:hi]))  # consonant -> vowel: steepest rise
        boundary = k * hop
    else:
        boundary = int(b * sr)
    if sound in STOPS:
        seg_lo, seg_hi = onset, max(onset + int(0.01 * sr), boundary)
        a0, a1 = int(seg_lo / hop), max(int(seg_lo / hop) + 1, int(seg_hi / hop))
        burst = (a0 + int(np.argmax(d[a0:a1]))) * hop if a1 - 1 > a0 else seg_lo
        end = min(max(boundary, burst + int(0.04 * sr)), burst + int(0.12 * sr))
        start = max(0, burst - int(0.03 * sr))
        bare = fade(x[start:end], sr, 0.003, 0.015)
        info.update(kind='stop', start=start / sr, burst=burst / sr, end=end / sr)
    else:
        start, end = onset, boundary
        seg = x[start:end]
        dur = len(seg) / sr
        info.update(kind='continuant', start=start / sr, end=end / sr, natural=dur)
        if dur < 0.04:
            info['why'] = 'onset too short (%.0f ms)' % (dur * 1000)
            return None, info
        if dur > 0.70:
            seg = seg[:int(0.65 * sr)]
        target = 0.6
        if len(seg) / sr < 0.55:
            # steady chunk: the frames with the flattest energy in the middle of the sound
            n = len(seg)
            clen = int(min(0.12, 0.6 * dur) * sr)
            c0 = max(0, n // 2 - clen // 2)
            chunk = seg[c0:c0 + clen]
            xf = int(0.02 * sr) if clen > int(0.04 * sr) else clen // 3
            y = seg[:c0 + clen]
            while len(y) / sr < target:
                y = crossfade_join(y, chunk, xf)
            seg = y[:int(target * sr)]
        bare = fade(seg, sr, 0.02, 0.025)
    return bare, info


def fold(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s.lower()) if unicodedata.category(c) != 'Mn')


def rms(x):
    a = np.abs(x)
    act = x[a > 0.02 * a.max()] if a.max() > 0 else x
    return float(np.sqrt(np.mean(act ** 2) + 1e-12))


judged = {}
for l in open(work + 'judged.tsv', encoding='utf-8'):
    if l.strip():
        c = l.rstrip('\n').split('\t')
        judged[c[0]] = c
sounds = []
for f in judged:
    m = re.match(r'(de|en)-(\w)-word-seed(\d+)\.wav', f)
    if m and m.group(2) not in sounds:
        sounds.append(m.group(2))

report = ['# Kalimo %s letter sounds cut from whole word takes' % lang, '',
          'Source: whole Qwen3-TTS 1.7B Base takes of the word alone (5 seeds) and of the carrier line (5 seeds), same cloned voice.',
          'Boundary: wav2vec2 espeak CTC phone timings, refined to the steepest energy rise (consonant to vowel) or fall (vowel to consonant).',
          'Continuants and vowels: the real onset sound, lengthened to about 600 ms by crossfading up to 120 ms of its own steady middle (no resampling, no pitch change), 20/25 ms fades. Stops: from 30 ms before the burst up to the start of the vowel (40 to 120 ms after the burst), said once. Onset found 10 dB above the noise floor, with 20 ms lead-in.',
          'Clip = bare sound, 350 ms pause, whole carrier take; parts matched in loudness (RMS for continuants, peak for stops), clip normalised to about -20 dBFS RMS.',
          'Check: wav2vec2 on the whole clip; the sound part is every phone before the carrier ("v i" / "l aɪ"). **clean** = only the target phone there, no vowel. The recogniser hears nothing in many bare stops and some fricatives (**silent**), so an ear decides.', '',
          '## Chosen per sound', '', '| sound | word | clip | bare | bare ms | sound part heard | verdict | source take | carrier take |',
          '|---|---|---|---|---|---|---|---|---|']
details = ['', '## Every candidate', '', '| sound | take | Whisper heard (word) | first phone | bare ms | sound part heard | verdict |',
           '|---|---|---|---|---|---|---|']


def carrier_split(ph):
    for i in range(len(ph)):
        if lang == 'de' and ph[i] in ('v', 'f', 'w', 'ð', 'b', 'ʋ') and i + 1 < len(ph) and ph[i + 1].startswith('i'):
            return ph[:i], True
        if lang == 'en' and ph[i] in ('l', 'ɫ') and i + 1 < len(ph) and ph[i + 1].startswith('a'):
            return ph[:i], True
    return ph, False


def build(bare, line, sound, sr):
    if sound in STOPS:
        b = bare * (0.8 * np.abs(line).max() / (np.abs(bare).max() + 1e-9))
    else:
        b = bare * (0.9 * rms(line) / rms(bare))
    clip = np.concatenate([b, np.zeros(int(0.35 * sr), dtype=np.float32), line])
    g = min(0.1 / rms(clip), 0.89 / (np.abs(clip).max() + 1e-9))
    return clip * g, b * g


RANK = {'clean': 3, 'silent': 2, 'not clean': 0}
summary = []
for sound in sounds:
    word_text = None
    words = [f for f in sorted(judged) if re.match(r'%s-%s-word-seed\d+\.wav' % (lang, sound), f)]
    lines = [f for f in sorted(judged) if re.match(r'%s-%s-line-seed\d+\.wav' % (lang, sound), f)]
    word_text = judged[words[0]][2].rstrip('.')
    carrier = next((f for f in lines if fold(word_text) in fold(judged[f][4])), None)
    carrier_note = ''
    if carrier is None:
        carrier, carrier_note = lines[0], ' (Whisper missed the word in every carrier take)'
    line, lsr = load(work + carrier)
    cands = []
    for f in words:
        heard = judged[f][4]
        word_ok = fold(word_text) in fold(heard)
        x, sr = load(work + f)
        bare, info = cut_sound(x, sr, sound)
        if bare is None:
            details.append('| %s | %s | %s | %s | - | - | %s |' % (sound, f, heard, info.get('first', '-'), info.get('why')))
            continue
        clip, b = build(bare, line, sound, sr)
        part, found = carrier_split(phones(clip, sr))
        acc = accept(sound)
        if not part:
            verdict = 'silent'
        elif sound in 'aeiou':
            verdict = 'clean' if len(part) == 1 and part[0] in acc else 'not clean'
        else:
            verdict = 'clean' if all(p in acc for p in part) else 'not clean'
        score = (RANK[verdict], word_ok, -abs(len(bare) / sr - (0.1 if sound in STOPS else 0.6)))
        cands.append((score, f, clip, b, sr, part, verdict, heard))
        save(out + 'candidates/' + f.replace('-word-', '-bare-'), b, sr)
        save(out + 'candidates/' + f.replace('-word-', '-clip-'), clip, sr)
        details.append('| %s | %s | %s | %s | %.0f | %s | %s |' % (sound, f, heard, info['first'], len(bare) / sr * 1000,
                                                                 ' '.join(part), verdict + ('' if word_ok else ', word missed')))
    if not cands:
        report.append('| %s | %s | - | - | - | - | no usable cut | - | %s |' % (sound, word_text, carrier))
        summary.append('%s %s: no usable cut' % (sound, word_text))
        continue
    cands.sort(key=lambda c: c[0], reverse=True)
    score, f, clip, b, sr, part, verdict, heard = cands[0]
    save(out + '%s-%s.wav' % (lang, sound), clip, sr)
    save(out + '%s-%s-bare.wav' % (lang, sound), b, sr)
    report.append('| %s | %s | %s-%s.wav | %s-%s-bare.wav | %.0f | %s | %s | %s | %s%s |' % (
        sound, word_text, lang, sound, lang, sound, len(b) / sr * 1000, ' '.join(part) or '-', verdict, f, carrier, carrier_note))
    summary.append('%s %s: %s [%s] %.0f ms' % (sound, word_text, verdict, ' '.join(part), len(b) / sr * 1000))
    print(summary[-1])
open(out + 'report.md', 'w', encoding='utf-8').write('\n'.join(report + details) + '\n')
