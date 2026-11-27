"""Trim Kokoro sources and Chatterbox conversions, build clips with the Qwen3 carrier, check phones; write report."""
import os
import re
import sys

import numpy as np

sys.argv = ['cut.py', 'x', 'de', 'C:/Users/micha/AppData/Local/Temp/kalimo-test/vc-out']
src = open('C:/Users/micha/AppData/Local/Temp/kalimo-test/src/cut.py', encoding='utf-8').read()
g = {}
exec(compile(src[:src.index("judged = {}")], 'cut', 'exec'), g)
load, save, envelope, fade, rms, phones, ctc = (g[k] for k in ('load', 'save', 'envelope', 'fade', 'rms', 'phones', 'ctc'))
BASE = 'C:/Users/micha/AppData/Local/Temp/kalimo-test/'
VC = BASE + 'vc/'
OUT = BASE + 'vc-out/'
os.makedirs(OUT, exist_ok=True)


def carrier(lang, sound):
    for l in open(BASE + 'cut2/report-%s.md' % lang, encoding='utf-8'):
        c = [x.strip() for x in l.split('|')]
        if len(c) > 9 and c[1] == sound and c[3].endswith('.wav'):
            return BASE + 'cut-%s/' % lang + c[9].split(' ')[0]


def trim(x, sr, stop):
    env, hop = envelope(x, sr, 0.005)
    floor = np.percentile(env, 10)
    on = np.nonzero(env > max(floor + 15, env.max() - 45))[0]
    if not len(on):
        return x
    a = max(0, on[0] - 4) * hop
    b = min(len(x), (on[-1] + 4) * hop)
    y = x[a:b]
    return fade(y, sr, 0.003 if stop else 0.03, 0.015 if stop else 0.04)


def build(bare, line, stop, sr):
    b = bare * (0.8 * np.abs(line).max() / (np.abs(bare).max() + 1e-9)) if stop else bare * (0.9 * rms(line) / rms(bare))
    clip = np.concatenate([b, np.zeros(int(0.35 * sr), dtype=np.float32), line])
    k = min(0.1 / rms(clip), 0.89 / (np.abs(clip).max() + 1e-9))
    return clip * k, b * k


def sound_part(ph, lang):
    for i in range(len(ph)):
        if lang == 'de' and ph[i] in ('v', 'f', 'w', 'ð', 'b', 'ʋ') and i + 1 < len(ph) and ph[i + 1].startswith('i'):
            return ph[:i]
        if lang == 'en' and ph[i] in ('l', 'ɫ') and i + 1 < len(ph) and ph[i + 1].startswith('a'):
            return ph[:i]
    return ph


meta = {}
for l in open(VC + 'source.tsv', encoding='utf-8'):
    c = l.rstrip('\n').split('\t')
    meta[c[0]] = c
rows = ['# Kalimo letter sounds: exact source (Kokoro, raw IPA) + voice conversion (Chatterbox S3Gen)', '',
        'Source: Kokoro-82M (Apache-2.0), phonemes passed straight to the model (no text front end), voice af_heart (de) / bf_emma (en); Kokoro is deterministic, so one take per setting.',
        'Conversion: Chatterbox S3Gen (Resemble AI, MIT): S3 speech tokens of the source (with 240 ms silence padding) -> flow matching conditioned on the narrator reference -> HiFT vocoder.',
        'Targets: the female German Qwen3 narrator reference (de), the new British Qwen3 narrator reference (en).',
        'Clip = converted sound (silence trimmed), 350 ms pause, the whole Qwen3 take of the carrier line (same as cut2). Phones: wav2vec2 espeak on the bare sound (padded) and on the clip (part before "wie"/"like").', '',
        '| sound | setting (IPA, speed) | source bare | source ms | source phones | converted bare | converted ms | converted phones | clip | sound part in clip |',
        '|---|---|---|---|---|---|---|---|---|---|']
for f in sorted(os.listdir(VC)):
    m = re.match(r'source-(de|en)-(\w)-(.+)-seed0\.wav', f)
    if not m:
        continue
    lang, sound, setting = m.groups()
    stop = sound in 'tkpbdg'
    name = '%s-%s-%s' % (lang, sound, setting)
    s, ssr = load(VC + f)
    v, vsr = load(VC + f.replace('source-', 'vc-'))
    s, v = trim(s, ssr, stop), trim(v, vsr, stop)
    line, lsr = load(carrier(lang, sound))
    assert vsr == lsr == 24000 and ssr == 24000
    clip, vb = build(v, line, stop, vsr)
    _, sb = build(s, line, stop, ssr)
    save(OUT + name + '-source-bare.wav', sb, ssr)
    save(OUT + name + '-vc-bare.wav', vb, vsr)
    save(OUT + name + '.wav', clip, vsr)
    sclip, _ = build(s, line, stop, ssr)
    save(OUT + name + '-source-clip.wav', sclip, ssr)
    sp, vp = ' '.join(phones(s, ssr)), ' '.join(phones(v, vsr))
    part = ' '.join(sound_part(phones(clip, vsr), lang))
    ipa, speed = meta[f][1], meta[f][2]
    rows.append('| %s | %s ×%s | %s | %.0f | %s | %s | %.0f | %s | %s | %s |' % (
        lang + ' ' + sound, ipa, speed, name + '-source-bare.wav', len(s) / ssr * 1000, sp or '-',
        name + '-vc-bare.wav', len(v) / vsr * 1000, vp or '-', name + '.wav', part or '-'))
    print(rows[-1], flush=True)
open(OUT + 'report.md', 'w', encoding='utf-8').write('\n'.join(rows) + '\n')
