# Kalimo de letter sounds cut from whole word takes

Source: whole Qwen3-TTS 1.7B Base takes of the word alone (5 seeds) and of the carrier line (5 seeds), same cloned voice.
Boundary: wav2vec2 espeak CTC phone timings, refined to the steepest energy rise (consonant to vowel) or fall (vowel to consonant).
Continuants and vowels: the real onset sound, lengthened to about 350 ms by crossfading 80 ms of its own steady middle (no resampling, no pitch change), 20/25 ms fades. Stops: burst plus at most 30 ms, said once.
Clip = bare sound, 350 ms pause, whole carrier take; parts matched in loudness (RMS for continuants, peak for stops), clip normalised to about -20 dBFS RMS.
Check: wav2vec2 on the whole clip; the sound part is every phone before the carrier ("v i" / "l aɪ"). **clean** = only the target phone there, no vowel. The recogniser hears nothing in many bare stops and some fricatives (**silent**), so an ear decides.

## Chosen per sound

| sound | word | clip | bare | bare ms | sound part heard | verdict | source take | carrier take |
|---|---|---|---|---|---|---|---|---|
| m | Maus | de-m.wav | de-m-bare.wav | 350 | - | silent | de-m-word-seed0.wav | de-m-line-seed0.wav |
| a | Ameise | de-a.wav | de-a-bare.wav | 350 | a | clean | de-a-word-seed0.wav | de-a-line-seed0.wav |
| l | Löffel | de-l.wav | de-l-bare.wav | 350 | - | silent | de-l-word-seed0.wav | de-l-line-seed0.wav |
| o | Oma | de-o.wav | de-o-bare.wav | 350 | o | clean | de-o-word-seed2.wav | de-o-line-seed0.wav |
| i | Insel | de-i.wav | de-i-bare.wav | 350 | iː n n i n n i n | not clean | de-i-word-seed0.wav | de-i-line-seed0.wav |
| e | Esel | de-e.wav | de-e-bare.wav | 350 | eː | clean | de-e-word-seed3.wav | de-e-line-seed0.wav |
| s | Sofa | de-s.wav | de-s-bare.wav | 350 | z | clean | de-s-word-seed0.wav | de-s-line-seed0.wav |
| u | Ufo | de-u.wav | de-u-bare.wav | 350 | uː | clean | de-u-word-seed1.wav | de-u-line-seed0.wav |
| r | Rose | de-r.wav | de-r-bare.wav | 350 | r r | clean | de-r-word-seed3.wav | de-r-line-seed0.wav |
| t | Tiger | de-t.wav | de-t-bare.wav | 45 | - | silent | de-t-word-seed0.wav | de-t-line-seed0.wav |
| n | Nase | de-n.wav | de-n-bare.wav | 350 | - | silent | de-n-word-seed0.wav | de-n-line-seed0.wav |
| f | Fisch | de-f.wav | de-f-bare.wav | 350 | - | silent | de-f-word-seed1.wav | de-f-line-seed0.wav |
| h | Hund | de-h.wav | de-h-bare.wav | 350 | - | silent | de-h-word-seed0.wav | de-h-line-seed0.wav |
| b | Baum | de-b.wav | de-b-bare.wav | 45 | b | clean | de-b-word-seed3.wav | de-b-line-seed0.wav |
| k | Kuh | de-k.wav | de-k-bare.wav | 45 | - | silent | de-k-word-seed0.wav | de-k-line-seed0.wav (Whisper missed the word in every carrier take) |
| p | Puppe | de-p.wav | de-p-bare.wav | 45 | - | silent | de-p-word-seed0.wav | de-p-line-seed0.wav |
| g | Gans | de-g.wav | de-g-bare.wav | 45 | - | silent | de-g-word-seed2.wav | de-g-line-seed1.wav |
| w | Wurm | de-w.wav | de-w-bare.wav | 350 | - | silent | de-w-word-seed1.wav | de-w-line-seed0.wav |
| d | Dach | de-d.wav | de-d-bare.wav | 45 | - | silent | de-d-word-seed2.wav | de-d-line-seed0.wav |

## Every candidate

| sound | take | Whisper heard (word) | first phone | bare ms | sound part heard | verdict |
|---|---|---|---|---|---|---|
| m | de-m-word-seed0.wav | Maus | m | 350 |  | silent |
| m | de-m-word-seed1.wav | Maus | m | 350 |  | silent |
| m | de-m-word-seed2.wav | Maus | m | 350 |  | silent |
| m | de-m-word-seed3.wav | Maus | m | 350 |  | silent |
| m | de-m-word-seed4.wav | Maus | m | 350 |  | silent |
| a | de-a-word-seed0.wav | Ameise | a | 350 | a | clean |
| a | de-a-word-seed1.wav | Ameise | ɑː | 350 | a | clean |
| a | de-a-word-seed2.wav | Ameise | ɑː | 350 | a | clean |
| a | de-a-word-seed3.wav | Ameise | a | 350 | a | clean |
| a | de-a-word-seed4.wav | Ameise | a | 350 | a | clean |
| l | de-l-word-seed0.wav | Löffel | l | 350 |  | silent |
| l | de-l-word-seed1.wav | Löffel | l | 350 |  | silent |
| l | de-l-word-seed2.wav | Löffel | l | 350 |  | silent |
| l | de-l-word-seed3.wav | Löffel | l | 350 |  | silent |
| l | de-l-word-seed4.wav | Löffel | l | 350 |  | silent |
| o | de-o-word-seed0.wav | Oma | uː | - | - | first phone uː |
| o | de-o-word-seed1.wav | Oma | u | - | - | first phone u |
| o | de-o-word-seed2.wav | Oma | o | 350 | o | clean |
| o | de-o-word-seed3.wav | Oma | u | - | - | first phone u |
| o | de-o-word-seed4.wav | Oma | o | 350 | o | clean |
| i | de-i-word-seed0.wav | Insel | ɪ | 350 | iː n n i n n i n | not clean |
| i | de-i-word-seed1.wav | Insel | ɪ | 350 | r r r | not clean |
| i | de-i-word-seed2.wav | Insel | ɪ | 350 | i n n i n i n | not clean |
| i | de-i-word-seed3.wav | Insel | ɪ | 350 | eː | not clean |
| i | de-i-word-seed4.wav | Insel | i | 350 | i l l l l ð | not clean |
| e | de-e-word-seed0.wav | Esel | i | - | - | first phone i |
| e | de-e-word-seed1.wav | Esel | iː | 350 | i | not clean |
| e | de-e-word-seed2.wav | Esel | iː | 350 | i | not clean |
| e | de-e-word-seed3.wav | Esel | iː | 350 | eː | clean |
| e | de-e-word-seed4.wav | Ease | iː | 350 | iː | clean, word missed |
| s | de-s-word-seed0.wav | Sofa | z | 350 | z | clean |
| s | de-s-word-seed1.wav | Sofa | z | 350 |  | silent |
| s | de-s-word-seed2.wav | Sofa | z | 350 | z | clean |
| s | de-s-word-seed3.wav | Sofa | s | 350 |  | silent |
| s | de-s-word-seed4.wav | Sofa | z | 350 |  | silent |
| u | de-u-word-seed0.wav | UFO | uː | 350 | u r | not clean |
| u | de-u-word-seed1.wav | UFO | u | 350 | uː | clean |
| u | de-u-word-seed2.wav | UFO | uː | 350 | u r r | not clean |
| u | de-u-word-seed3.wav | Uff. | u | 350 |  | silent, word missed |
| u | de-u-word-seed4.wav | UFO | u | 350 | uː | clean |
| r | de-r-word-seed0.wav | Rose | ɹ | 350 |  | silent |
| r | de-r-word-seed1.wav | Rose | ɹ | 350 |  | silent |
| r | de-r-word-seed2.wav | Rose | ɹ | 350 |  | silent |
| r | de-r-word-seed3.wav | Rose | ɹ | 350 | r r | clean |
| r | de-r-word-seed4.wav | Rose | r | 350 |  | silent |
| t | de-t-word-seed0.wav | Tiger | t | 45 |  | silent |
| t | de-t-word-seed1.wav | Tiger | t | 45 |  | silent |
| t | de-t-word-seed2.wav | TIGA | t | 45 |  | silent, word missed |
| t | de-t-word-seed3.wav | Tiger | t | 45 |  | silent |
| t | de-t-word-seed4.wav | Tiger | t | 45 |  | silent |
| n | de-n-word-seed0.wav | Nase | n | 350 |  | silent |
| n | de-n-word-seed1.wav | Nase | n | 350 |  | silent |
| n | de-n-word-seed2.wav | Nase | n | 350 |  | silent |
| n | de-n-word-seed3.wav | Nase. | n | 350 | n ə n n | not clean |
| n | de-n-word-seed4.wav | Nase | n | 350 | n n ə n n ə | not clean |
| f | de-f-word-seed0.wav | Fisch | s | - | - | first phone s |
| f | de-f-word-seed1.wav | Fisch | f | 350 |  | silent |
| f | de-f-word-seed2.wav | Fisch | f | 350 |  | silent |
| f | de-f-word-seed3.wav | Fisch | s | - | - | first phone s |
| f | de-f-word-seed4.wav | Fisch | f | 350 | f r r r | not clean |
| h | de-h-word-seed0.wav | Hund | h | 350 |  | silent |
| h | de-h-word-seed1.wav | Hund | x | - | - | first phone x |
| h | de-h-word-seed2.wav | Hund | h | 350 |  | silent |
| h | de-h-word-seed3.wav | Hund | x | - | - | first phone x |
| h | de-h-word-seed4.wav | Hund | h | 350 |  | silent |
| b | de-b-word-seed0.wav | Baum | b | 20 |  | silent |
| b | de-b-word-seed1.wav | Baum | b | 45 |  | silent |
| b | de-b-word-seed2.wav | Baum | b | 45 |  | silent |
| b | de-b-word-seed3.wav | Baum | b | 45 | b | clean |
| b | de-b-word-seed4.wav | Baum. | p | 15 |  | silent |
| k | de-k-word-seed0.wav | Cool. | k | 45 |  | silent, word missed |
| k | de-k-word-seed1.wav | Cool. | k | 45 |  | silent, word missed |
| k | de-k-word-seed2.wav | Cool. | k | 45 |  | silent, word missed |
| k | de-k-word-seed3.wav | Cool. | k | 45 |  | silent, word missed |
| k | de-k-word-seed4.wav | Cool. | t | - | - | first phone t |
| p | de-p-word-seed0.wav | Puppe | p | 45 |  | silent |
| p | de-p-word-seed1.wav | Puppe | p | 25 |  | silent |
| p | de-p-word-seed2.wav | Puppe | p | 15 |  | silent |
| p | de-p-word-seed3.wav | Puppe | p | 45 |  | silent |
| p | de-p-word-seed4.wav | Puppe | p | 45 |  | silent |
| g | de-g-word-seed0.wav | Gänse. | ɡ | 40 |  | silent |
| g | de-g-word-seed1.wav | Ganz. | ɡ | 45 |  | silent, word missed |
| g | de-g-word-seed2.wav | Gans | ɡ | 45 |  | silent |
| g | de-g-word-seed3.wav | Ganz. | ɡ | 45 |  | silent, word missed |
| g | de-g-word-seed4.wav | Gänz | ɡ | 15 |  | silent, word missed |
| w | de-w-word-seed0.wav | Woh. | v | 350 |  | silent, word missed |
| w | de-w-word-seed1.wav | Wurm | v | 350 |  | silent |
| w | de-w-word-seed2.wav | Wohnen | v | 350 | r r | not clean, word missed |
| w | de-w-word-seed3.wav | Wurm | v | 350 |  | silent |
| w | de-w-word-seed4.wav | Wurm | v | 350 |  | silent |
| d | de-d-word-seed0.wav | Dach | d | 20 |  | silent |
| d | de-d-word-seed1.wav | Dach | d | 20 |  | silent |
| d | de-d-word-seed2.wav | Dach | d | 45 |  | silent |
| d | de-d-word-seed3.wav | Dach | d | 35 |  | silent |
| d | de-d-word-seed4.wav | Dach | d | 15 |  | silent |
