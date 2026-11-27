"""Select one take per line: Whisper exact, >=150 ms quiet at the end, not faster than the line's median, best speaker cosine.
usage: python game2_pick.py <round> ; writes game2/retry.tsv (lines without a valid take) and, on the final round, the files + report."""
import os
import re
import shutil
import sys
import unicodedata
import wave
from collections import OrderedDict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
D = 'C:/Users/micha/AppData/Local/Temp/kalimo-test/game2/'
final = len(sys.argv) > 1 and sys.argv[1] == 'final'


def fold(s):
    s = ''.join(c for c in unicodedata.normalize('NFD', s.lower()) if unicodedata.category(c) != 'Mn')
    return re.sub(r"[^a-z0-9]", '', s.replace('ß', 'ss'))


def measure(path):
    w = wave.open(path)
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768
    sr = w.getframerate()
    env = np.convolve(np.abs(x), np.ones(240) / 240, 'same')
    idx = np.nonzero(env > 0.03 * env.max())[0]
    return (len(x) - idx[-1]) / sr, (idx[-1] - idx[0]) / sr


lines = OrderedDict()
for l in open(D + 'lines.tsv', encoding='utf-8'):
    stem, lang, text = l.rstrip('\n').split('\t')
    lines[stem.replace('/', '_')] = (stem, lang, text)
takes = {}
for l in open(D + 'takes/judged.tsv', encoding='utf-8'):
    c = l.rstrip('\n').split('\t')
    if len(c) < 7:
        continue
    f, lang, text, sec, cos, spoken, heard = c[:7]
    key = f.rsplit('-seed', 1)[0]
    gap, speech = measure(D + 'takes/' + f)
    takes.setdefault(key, []).append(dict(f=f, sec=float(sec), cos=float(cos), spoken=spoken, heard=heard,
                                          ok=fold(heard) == fold(text), gap=gap, speech=speech))
retry, report, flagged = [], [], []
for key, (stem, lang, text) in lines.items():
    ts = takes.get(key, [])
    if not ts:
        retry.append('%s\t%s\t%s\n' % (stem, lang, text))
        continue
    med = float(np.median([t['speech'] for t in ts]))
    for t in ts:
        t['fast'] = t['speech'] < 0.85 * med
        t['valid'] = t['ok'] and t['gap'] >= 0.15 and not t['fast']
    valid = [t for t in ts if t['valid']]
    if valid:
        top = max(t['cos'] for t in valid)
        valid = [t for t in valid if t['cos'] >= top - 0.03]
        best = max(valid, key=lambda t: t['cos'])
        note = ''
    else:
        retry.append('%s\t%s\t%s\n' % (stem, lang, text))
        best = max(ts, key=lambda t: (t['ok'], min(t['gap'], 0.15), t['cos']))
        note = 'no take met every rule: ' + ', '.join(
            x for x, bad in (('Whisper', not best['ok']), ('end quiet %.0f ms' % (best['gap'] * 1000), best['gap'] < 0.15),
                             ('fast', best['fast'])) if bad)
        flagged.append('%s: %s' % (stem, note))
    report.append((stem, text, best, len(ts), note))
open(D + 'retry.tsv', 'w', encoding='utf-8').write(''.join(retry))
print('lines', len(lines), 'need retry', len(retry))
if final:
    out = ['# Sound Safari lines v2: one cloned narrator per language, whole takes', '',
           'Qwen3-TTS 1.7B Base clone (German: competitive-learn female narrator; English: British narrator), temperature 0.6 (was 0.9), 24 kHz mono WAV. Nothing cut, sped up or processed.',
           'German text is spoken with " …" added at the end (in a test it gave quiet at the end in 9 of 20 takes against 0 of 20 without, and slightly slower speech; Whisper still heard every word). English: plain text first, " …" only for retries.',
           'Per line 4 takes (up to 12 when needed). A take counts only if Whisper large-v3 hears the text exactly, at least 150 ms of quiet follow the speech, and it is not more than 15 % faster than the median take of that line. Among those, takes more than 0.03 below the best speaker similarity are dropped, and the most similar one is kept.',
           'Speaker similarity: cosine between the Qwen3 speaker encoder embedding of the take and of the narrator reference.', '',
           '| file | text | spoken as | Whisper | end quiet ms | speech s | speaker cos | takes | note |', '|---|---|---|---|---|---|---|---|---|']
    for stem, text, b, n, note in report:
        lang, rid = stem.split('/')
        os.makedirs(D + lang, exist_ok=True)
        shutil.copy(D + 'takes/' + b['f'], D + lang + '/' + rid + '.wav')
        out.append('| %s.wav | %s | %s | %s | %.0f | %.2f | %.3f | %d | %s |' % (stem, text, b['spoken'], b['heard'], b['gap'] * 1000, b['speech'], b['cos'], n, note))
    cos = [b['cos'] for _, _, b, _, _ in report]
    out += ['', 'Speaker cosine of the chosen takes: min %.3f, median %.3f, max %.3f.' % (min(cos), np.median(cos), max(cos)), '',
            '## Lines without a take that met every rule (best available kept)', ''] + ['- ' + x for x in flagged or ['none']]
    open(D + 'report.md', 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    print('flagged', len(flagged))
    print('\n'.join(flagged))
