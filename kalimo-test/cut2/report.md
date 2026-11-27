# Cut variants (round 2): de m, f, t, k, p and en t

Start: walked back from the aligner start to the first frame above the noise floor (+6 dB), then 20 ms earlier.
Stops: from that start through the burst and the whole aspiration up to the voicing onset of the vowel (rise of 80-500 Hz energy), end at voicing minus 0 / 10 / 25 ms (v1 / v2 / v3), 3 ms fade in, 10 ms fade out.
Held sounds: the real onset sound up to the next phone, lengthened by crossfading its steady middle to 450 / 600 / 750 ms (v1 / v2 / v3), 40 ms fades.
Clip = bare sound, 350 ms pause, whole carrier take.

| variant | bare | source take | onset s | sound end s | bare ms | natural ms | phones heard in clip |
|---|---|---|---|---|---|---|---|
| de-m-v1 | de-m-v1-bare.wav | de-m-word-seed0.wav | 0.385 | 0.540 | 450 | 155 | v iː m aʊ z |
| de-m-v2 | de-m-v2-bare.wav | de-m-word-seed0.wav | 0.385 | 0.540 | 600 | 155 | v iː m aʊ z |
| de-m-v3 | de-m-v3-bare.wav | de-m-word-seed0.wav | 0.385 | 0.540 | 750 | 155 | v iː m aʊ z |
| de-f-v1 | de-f-v1-bare.wav | de-f-word-seed1.wav | 0.195 | 0.370 | 450 | 175 | v iː f ɪ ʃ |
| de-f-v2 | de-f-v2-bare.wav | de-f-word-seed1.wav | 0.195 | 0.370 | 600 | 175 | v iː f ɪ ʃ |
| de-f-v3 | de-f-v3-bare.wav | de-f-word-seed1.wav | 0.195 | 0.370 | 750 | 175 | v iː f ɪ ʃ |
| de-t-v1 | de-t-v1-bare.wav | de-t-word-seed0.wav | 0.285 | 0.395 | 110 | 110 | t v iː t iː ɡ a |
| de-t-v2 | de-t-v2-bare.wav | de-t-word-seed0.wav | 0.285 | 0.385 | 100 | 110 | d v iː t iː ɡ a |
| de-t-v3 | de-t-v3-bare.wav | de-t-word-seed0.wav | 0.285 | 0.370 | 85 | 110 | d v iː t iː ɡ a |
| de-k-v1 | de-k-v1-bare.wav | de-k-word-seed3.wav | 0.145 | 0.275 | 130 | 130 | k v iː k uː |
| de-k-v2 | de-k-v2-bare.wav | de-k-word-seed3.wav | 0.145 | 0.265 | 120 | 130 | v iː k uː |
| de-k-v3 | de-k-v3-bare.wav | de-k-word-seed3.wav | 0.145 | 0.250 | 105 | 130 | v iː k uː |
| de-p-v1 | de-p-v1-bare.wav | de-p-word-seed3.wav | 0.160 | 0.260 | 100 | 100 | p v iː p o p ə |
| de-p-v2 | de-p-v2-bare.wav | de-p-word-seed3.wav | 0.160 | 0.250 | 90 | 100 | v iː p o b |
| de-p-v3 | de-p-v3-bare.wav | de-p-word-seed3.wav | 0.160 | 0.235 | 75 | 100 | v iː p o p ə |
| en-t-v1 | en-t-v1-bare.wav | en-t-word-seed1.wav | 0.140 | 0.235 | 95 | 95 | t l aɪ k t aɪ ɡ ɚ |
| en-t-v2 | en-t-v2-bare.wav | en-t-word-seed1.wav | 0.140 | 0.225 | 85 | 95 | t l aɪ k t aɪ ɡ ɚ |
| en-t-v3 | en-t-v3-bare.wav | en-t-word-seed1.wav | 0.140 | 0.210 | 70 | 95 | l aɪ k t aɪ ɡ ɚ |
