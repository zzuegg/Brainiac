# Kalimo de letter sounds cut from whole word takes

Source: whole Qwen3-TTS 1.7B Base takes of the word alone (5 seeds) and of the carrier line (5 seeds), same cloned voice.
Boundary: wav2vec2 espeak CTC phone timings, refined to the steepest energy rise (consonant to vowel) or fall (vowel to consonant).
Continuants and vowels: the real onset sound, lengthened to about 600 ms by crossfading up to 120 ms of its own steady middle (no resampling, no pitch change), 20/25 ms fades. Stops: from 30 ms before the burst up to the start of the vowel (40 to 120 ms after the burst), said once. Onset found 10 dB above the noise floor, with 20 ms lead-in.
Clip = bare sound, 350 ms pause, whole carrier take; parts matched in loudness (RMS for continuants, peak for stops), clip normalised to about -20 dBFS RMS.
Check: wav2vec2 on the whole clip; the sound part is every phone before the carrier ("v i" / "l aɪ"). **clean** = only the target phone there, no vowel. The recogniser hears nothing in many bare stops and some fricatives (**silent**), so an ear decides.

## Chosen per sound

| sound | word | clip | bare | bare ms | sound part heard | verdict | source take | carrier take |
|---|---|---|---|---|---|---|---|---|
| m | Maus | de-m.wav | de-m-bare.wav | 600 | - | silent | de-m-word-seed0.wav | de-m-line-seed0.wav |
| a | Ameise | de-a.wav | de-a-bare.wav | 600 | a | clean | de-a-word-seed0.wav | de-a-line-seed0.wav |
| l | Löffel | de-l.wav | de-l-bare.wav | 600 | l l l l | clean | de-l-word-seed3.wav | de-l-line-seed0.wav |
| o | Oma | de-o.wav | de-o-bare.wav | 600 | o b o b o b o b o | not clean | de-o-word-seed2.wav | de-o-line-seed0.wav |
| i | Insel | de-i.wav | de-i-bare.wav | 600 | i n i n i n i n n i n | not clean | de-i-word-seed0.wav | de-i-line-seed0.wav |
| e | Esel | de-e.wav | de-e-bare.wav | 600 | iː | clean | de-e-word-seed4.wav | de-e-line-seed0.wav |
| s | Sofa | de-s.wav | de-s-bare.wav | 600 | z | clean | de-s-word-seed0.wav | de-s-line-seed0.wav |
| u | Ufo | de-u.wav | de-u-bare.wav | 600 | uː | clean | de-u-word-seed1.wav | de-u-line-seed0.wav |
| r | Rose | de-r.wav | de-r-bare.wav | 600 | - | silent | de-r-word-seed0.wav | de-r-line-seed0.wav |
| t | Tiger | de-t.wav | de-t-bare.wav | 100 | t | clean | de-t-word-seed0.wav | de-t-line-seed0.wav |
| n | Nase | de-n.wav | de-n-bare.wav | 600 | - | silent | de-n-word-seed0.wav | de-n-line-seed0.wav |
| f | Fisch | de-f.wav | de-f-bare.wav | 600 | - | silent | de-f-word-seed1.wav | de-f-line-seed0.wav |
| h | Hund | de-h.wav | de-h-bare.wav | 600 | - | silent | de-h-word-seed0.wav | de-h-line-seed0.wav |
| b | Baum | de-b.wav | de-b-bare.wav | 70 | b | clean | de-b-word-seed0.wav | de-b-line-seed0.wav |
| k | Kuh | de-k.wav | de-k-bare.wav | 100 | k | clean | de-k-word-seed3.wav | de-k-line-seed0.wav (Whisper missed the word in every carrier take) |
| p | Puppe | de-p.wav | de-p-bare.wav | 110 | p | clean | de-p-word-seed3.wav | de-p-line-seed0.wav |
| g | Gans | de-g.wav | de-g-bare.wav | 75 | - | silent | de-g-word-seed2.wav | de-g-line-seed1.wav |
| w | Wurm | de-w.wav | de-w-bare.wav | 600 | - | silent | de-w-word-seed1.wav | de-w-line-seed0.wav |
| d | Dach | de-d.wav | de-d-bare.wav | 70 | d | clean | de-d-word-seed0.wav | de-d-line-seed0.wav |

## Every candidate

| sound | take | Whisper heard (word) | first phone | bare ms | sound part heard | verdict |
|---|---|---|---|---|---|---|
| m | de-m-word-seed0.wav | Maus | m | 600 |  | silent |
| m | de-m-word-seed1.wav | Maus | m | 600 |  | silent |
| m | de-m-word-seed2.wav | Maus | m | 600 |  | silent |
| m | de-m-word-seed3.wav | Maus | m | 600 |  | silent |
| m | de-m-word-seed4.wav | Maus | m | 600 |  | silent |
| a | de-a-word-seed0.wav | Ameise | a | 600 | a | clean |
| a | de-a-word-seed1.wav | Ameise | ɑː | 600 | a | clean |
| a | de-a-word-seed2.wav | Ameise | ɑː | 600 | a | clean |
| a | de-a-word-seed3.wav | Ameise | a | 600 | a | clean |
| a | de-a-word-seed4.wav | Ameise | a | 600 | a | clean |
| l | de-l-word-seed0.wav | Löffel | l | 600 |  | silent |
| l | de-l-word-seed1.wav | Löffel | l | 600 |  | silent |
| l | de-l-word-seed2.wav | Löffel | l | 600 | h ø | not clean |
| l | de-l-word-seed3.wav | Löffel | l | 600 | l l l l | clean |
| l | de-l-word-seed4.wav | Löffel | l | 600 |  | silent |
| o | de-o-word-seed0.wav | Oma | uː | - | - | first phone uː |
| o | de-o-word-seed1.wav | Oma | u | - | - | first phone u |
| o | de-o-word-seed2.wav | Oma | o | 600 | o b o b o b o b o | not clean |
| o | de-o-word-seed3.wav | Oma | u | - | - | first phone u |
| o | de-o-word-seed4.wav | Oma | o | 640 | o m | not clean |
| i | de-i-word-seed0.wav | Insel | ɪ | 600 | i n i n i n i n n i n | not clean |
| i | de-i-word-seed1.wav | Insel | ɪ | 600 | i n i n i n n i n n n i n | not clean |
| i | de-i-word-seed2.wav | Insel | ɪ | 600 | i n n i n i n i n l l | not clean |
| i | de-i-word-seed3.wav | Insel | ɪ | 600 | eː | not clean |
| i | de-i-word-seed4.wav | Insel | i | 600 | i l l l l l l ð | not clean |
| e | de-e-word-seed0.wav | Esel | i | - | - | first phone i |
| e | de-e-word-seed1.wav | Esel | iː | 600 | i | not clean |
| e | de-e-word-seed2.wav | Esel | iː | 600 | i | not clean |
| e | de-e-word-seed3.wav | Esel | iː | 600 | i | not clean |
| e | de-e-word-seed4.wav | Ease | iː | 600 | iː | clean, word missed |
| s | de-s-word-seed0.wav | Sofa | z | 600 | z | clean |
| s | de-s-word-seed1.wav | Sofa | z | 600 | z | clean |
| s | de-s-word-seed2.wav | Sofa | z | 600 | z | clean |
| s | de-s-word-seed3.wav | Sofa | s | 600 |  | silent |
| s | de-s-word-seed4.wav | Sofa | z | 600 |  | silent |
| u | de-u-word-seed0.wav | UFO | uː | 600 | uː ɹ ɹ ɹ ɹ | not clean |
| u | de-u-word-seed1.wav | UFO | u | 600 | uː | clean |
| u | de-u-word-seed2.wav | UFO | uː | 600 | uː ɹ ɹ ɹ ɹ ɹ | not clean |
| u | de-u-word-seed3.wav | Uff. | u | 600 |  | silent, word missed |
| u | de-u-word-seed4.wav | UFO | u | 600 | uː | clean |
| r | de-r-word-seed0.wav | Rose | ɹ | 600 |  | silent |
| r | de-r-word-seed1.wav | Rose | ɹ | 600 |  | silent |
| r | de-r-word-seed2.wav | Rose | ɹ | 600 |  | silent |
| r | de-r-word-seed3.wav | Rose | ɹ | 600 | v o v v o v o v u v u | not clean |
| r | de-r-word-seed4.wav | Rose | r | 600 | uː | not clean |
| t | de-t-word-seed0.wav | Tiger | t | 100 | t | clean |
| t | de-t-word-seed1.wav | Tiger | t | 105 | t | clean |
| t | de-t-word-seed2.wav | TIGA | t | 100 | t | clean, word missed |
| t | de-t-word-seed3.wav | Tiger | t | 125 | t | clean |
| t | de-t-word-seed4.wav | Tiger | t | 100 | t | clean |
| n | de-n-word-seed0.wav | Nase | n | 600 |  | silent |
| n | de-n-word-seed1.wav | Nase | n | 600 |  | silent |
| n | de-n-word-seed2.wav | Nase | n | 600 |  | silent |
| n | de-n-word-seed3.wav | Nase. | n | 600 | ə n n ə n ə n ə n ə n | not clean |
| n | de-n-word-seed4.wav | Nase | n | 600 | n ə n ə n n n ə n ə n | not clean |
| f | de-f-word-seed0.wav | Fisch | s | - | - | first phone s |
| f | de-f-word-seed1.wav | Fisch | f | 600 |  | silent |
| f | de-f-word-seed2.wav | Fisch | f | 600 |  | silent |
| f | de-f-word-seed3.wav | Fisch | s | - | - | first phone s |
| f | de-f-word-seed4.wav | Fisch | f | 600 | s iː t ð t ð | not clean |
| h | de-h-word-seed0.wav | Hund | h | 600 |  | silent |
| h | de-h-word-seed1.wav | Hund | x | - | - | first phone x |
| h | de-h-word-seed2.wav | Hund | h | 600 |  | silent |
| h | de-h-word-seed3.wav | Hund | x | - | - | first phone x |
| h | de-h-word-seed4.wav | Hund | h | 600 |  | silent |
| b | de-b-word-seed0.wav | Baum | b | 70 | b | clean |
| b | de-b-word-seed1.wav | Baum | b | 115 |  | silent |
| b | de-b-word-seed2.wav | Baum | b | 130 |  | silent |
| b | de-b-word-seed3.wav | Baum | b | 150 | b a | not clean |
| b | de-b-word-seed4.wav | Baum. | p | 70 |  | silent |
| k | de-k-word-seed0.wav | Cool. | k | 115 | k | clean, word missed |
| k | de-k-word-seed1.wav | Cool. | k | 110 |  | silent, word missed |
| k | de-k-word-seed2.wav | Cool. | k | 135 | k | clean, word missed |
| k | de-k-word-seed3.wav | Cool. | k | 100 | k | clean, word missed |
| k | de-k-word-seed4.wav | Cool. | t | - | - | first phone t |
| p | de-p-word-seed0.wav | Puppe | p | 85 |  | silent |
| p | de-p-word-seed1.wav | Puppe | p | 70 |  | silent |
| p | de-p-word-seed2.wav | Puppe | p | 70 |  | silent |
| p | de-p-word-seed3.wav | Puppe | p | 110 | p | clean |
| p | de-p-word-seed4.wav | Puppe | p | 110 |  | silent |
| g | de-g-word-seed0.wav | Gänse. | ɡ | 70 |  | silent |
| g | de-g-word-seed1.wav | Ganz. | ɡ | 105 |  | silent, word missed |
| g | de-g-word-seed2.wav | Gans | ɡ | 75 |  | silent |
| g | de-g-word-seed3.wav | Ganz. | ɡ | 70 |  | silent, word missed |
| g | de-g-word-seed4.wav | Gänz | ɡ | 70 |  | silent, word missed |
| w | de-w-word-seed0.wav | Woh. | v | 600 |  | silent, word missed |
| w | de-w-word-seed1.wav | Wurm | v | 600 |  | silent |
| w | de-w-word-seed2.wav | Wohnen | v | 600 | r r r a r r r | not clean, word missed |
| w | de-w-word-seed3.wav | Wurm | v | 600 | r | not clean |
| w | de-w-word-seed4.wav | Wurm | v | 600 |  | silent |
| d | de-d-word-seed0.wav | Dach | d | 70 | d | clean |
| d | de-d-word-seed1.wav | Dach | d | 70 | d | clean |
| d | de-d-word-seed2.wav | Dach | d | 100 |  | silent |
| d | de-d-word-seed3.wav | Dach | d | 70 | d | clean |
| d | de-d-word-seed4.wav | Dach | d | 70 | t | clean |
