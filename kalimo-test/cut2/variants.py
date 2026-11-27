"""Cut variants for a few sounds: true acoustic onset, full stop release, held sounds at 450/600/750 ms."""
import re
import sys

import numpy as np

sys.argv = ['cut.py', 'x', 'de', 'C:/Users/micha/AppData/Local/Temp/kalimo-test/cut2v']
src = open('C:/Users/micha/AppData/Local/Temp/kalimo-test/src/cut.py', encoding='utf-8').read()
g = {}
exec(compile(src[:src.index("judged = {}")], 'cut', 'exec'), g)
load, save, ctc, envelope, fade, crossfade_join, rms, phones = (g[k] for k in (
    'load', 'save', 'ctc', 'envelope', 'fade', 'crossfade_join', 'rms', 'phones'))
BASE = 'C:/Users/micha/AppData/Local/Temp/kalimo-test/'
OUT = BASE + 'cut2v/'
import os
os.makedirs(OUT, exist_ok=True)
HOP = 0.005


def chosen(lang, sound):
    """Source and carrier take chosen in the cut2 run."""
    for l in open(BASE + 'cut2/report-%s.md' % lang, encoding='utf-8'):
        c = [x.strip() for x in l.split('|')]
        if len(c) > 9 and c[1] == sound and c[3].endswith('.wav'):
            return c[8], c[9].split(' ')[0]
    raise KeyError(sound)


def true_onset(x, sr, aligner_start):
    env, hop = envelope(x, sr, HOP)
    floor = np.percentile(env[:int(0.12 / HOP)], 50)
    i = min(len(env) - 1, int(aligner_start / HOP))
    while i > 0 and env[i - 1] > floor + 6:
        i -= 1
    return max(0, i * hop - int(0.02 * sr)), env, hop


def voicing_onset(x, sr, after, before):
    """First frame after the burst where low-frequency periodic energy rises: steepest rise of the
    0-500 Hz band energy between after and before (seconds)."""
    n = int(HOP * sr)
    k0, k1 = int(after * sr / n), int(before * sr / n)
    e = []
    for k in range(k0, k1):
        f = x[k * n:(k + 2) * n]
        if len(f) < n:
            break
        spec = np.abs(np.fft.rfft(f * np.hanning(len(f))))
        freqs = np.fft.rfftfreq(len(f), 1 / sr)
        e.append(10 * np.log10(np.sum(spec[(freqs > 80) & (freqs < 500)] ** 2) + 1e-10))
    if len(e) < 3:
        return before
    d = np.diff(np.convolve(e, np.ones(3) / 3, mode='same'))
    return (k0 + int(np.argmax(d)) + 1) * n / sr


def hold(seg, sr, target, fade_s=0.04):
    n = len(seg)
    dur = n / sr
    clen = int(min(0.12, 0.6 * dur) * sr)
    c0 = max(0, n // 2 - clen // 2)
    chunk = seg[c0:c0 + clen]
    xf = min(int(0.02 * sr), clen // 3)
    y = seg[:c0 + clen]
    while len(y) / sr < target:
        y = crossfade_join(y, chunk, xf)
    return fade(y[:int(target * sr)], sr, fade_s, fade_s)


def build(bare, line, stop, sr):
    b = bare * (0.8 * np.abs(line).max() / (np.abs(bare).max() + 1e-9)) if stop else bare * (0.9 * rms(line) / rms(bare))
    clip = np.concatenate([b, np.zeros(int(0.35 * sr), dtype=np.float32), line])
    k = min(0.1 / rms(clip), 0.89 / (np.abs(clip).max() + 1e-9))
    return clip * k, b * k


rows = ['# Cut variants (round 2): de m, f, t, k, p and en t', '',
        'Start: walked back from the aligner start to the first frame above the noise floor (+6 dB), then 20 ms earlier.',
        'Stops: from that start through the burst and the whole aspiration up to the voicing onset of the vowel (rise of 80-500 Hz energy), end at voicing minus 0 / 10 / 25 ms (v1 / v2 / v3), 3 ms fade in, 10 ms fade out.',
        'Held sounds: the real onset sound up to the next phone, lengthened by crossfading its steady middle to 450 / 600 / 750 ms (v1 / v2 / v3), 40 ms fades.',
        'Clip = bare sound, 350 ms pause, whole carrier take.', '',
        '| variant | bare | source take | onset s | sound end s | bare ms | natural ms | phones heard in clip |', '|---|---|---|---|---|---|---|---|']
for lang, sound in [('de', 'm'), ('de', 'f'), ('de', 't'), ('de', 'k'), ('de', 'p'), ('en', 't')]:
    src_f, car_f = chosen(lang, sound)
    work = BASE + 'cut-%s/' % lang
    x, sr = load(work + src_f)
    line, _ = load(work + car_f)
    spans = ctc(x, sr)
    first, second = spans[0], spans[1]
    start, env, hop = true_onset(x, sr, first[1])
    stop = sound in 'tkpbdg'
    if stop:
        env_s = env[int(start / hop):int(second[2] / HOP) + 1]
        burst = (start / sr) + HOP * int(np.argmax(np.diff(env_s))) if len(env_s) > 2 else start / sr
        voice = voicing_onset(x, sr, burst + 0.02, second[2] + 0.06)
        natural = voice - start / sr
        variants = [(v, voice - cut) for v, cut in (('v1', 0.0), ('v2', 0.01), ('v3', 0.025))]
        for v, end in variants:
            bare = fade(x[start:int(end * sr)], sr, 0.003, 0.01)
            clip, b = build(bare, line, True, sr)
            ph = phones(clip, sr)
            save(OUT + '%s-%s-%s.wav' % (lang, sound, v), clip, sr)
            save(OUT + '%s-%s-%s-bare.wav' % (lang, sound, v), b, sr)
            rows.append('| %s-%s-%s | %s-%s-%s-bare.wav | %s | %.3f | %.3f | %.0f | %.0f | %s |' % (
                lang, sound, v, lang, sound, v, src_f, start / sr, end, len(b) / sr * 1000, natural * 1000, ' '.join(ph)))
    else:
        # held: real sound from the true onset to the steepest rise near the next phone
        lo, hi = int(max(start / hop, second[1] / HOP - 10)), int(second[1] / HOP + 10)
        sm = np.convolve(env, np.ones(3) / 3, mode='same')
        end = (lo + int(np.argmax(np.diff(sm)[lo:hi]))) * hop
        seg = x[start:end]
        natural = len(seg) / sr
        for v, target in (('v1', 0.45), ('v2', 0.6), ('v3', 0.75)):
            bare = hold(seg, sr, target)
            clip, b = build(bare, line, False, sr)
            ph = phones(clip, sr)
            save(OUT + '%s-%s-%s.wav' % (lang, sound, v), clip, sr)
            save(OUT + '%s-%s-%s-bare.wav' % (lang, sound, v), b, sr)
            rows.append('| %s-%s-%s | %s-%s-%s-bare.wav | %s | %.3f | %.3f | %.0f | %.0f | %s |' % (
                lang, sound, v, lang, sound, v, src_f, start / sr, end / sr, len(b) / sr * 1000, natural * 1000, ' '.join(ph)))
    print(rows[-1])
open(OUT + 'report.md', 'w', encoding='utf-8').write('\n'.join(rows) + '\n')
