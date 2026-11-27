# Kalimo en letter sounds cut from whole word takes

Source: whole Qwen3-TTS 1.7B Base takes of the word alone (5 seeds) and of the carrier line (5 seeds), same cloned voice.
Boundary: wav2vec2 espeak CTC phone timings, refined to the steepest energy rise (consonant to vowel) or fall (vowel to consonant).
Continuants and vowels: the real onset sound, lengthened to about 350 ms by crossfading 80 ms of its own steady middle (no resampling, no pitch change), 20/25 ms fades. Stops: burst plus at most 30 ms, said once.
Clip = bare sound, 350 ms pause, whole carrier take; parts matched in loudness (RMS for continuants, peak for stops), clip normalised to about -20 dBFS RMS.
Check: wav2vec2 on the whole clip; the sound part is every phone before the carrier ("v i" / "l aɪ"). **clean** = only the target phone there, no vowel. The recogniser hears nothing in many bare stops and some fricatives (**silent**), so an ear decides.

## Chosen per sound

| sound | word | clip | bare | bare ms | sound part heard | verdict | source take | carrier take |
|---|---|---|---|---|---|---|---|---|
| s | Snake | en-s.wav | en-s-bare.wav | 350 | - | silent | en-s-word-seed0.wav | en-s-line-seed0.wav |
| a | Ant | en-a.wav | en-a-bare.wav | 350 | æ | clean | en-a-word-seed2.wav | en-a-line-seed0.wav |
| t | Tiger | en-t.wav | en-t-bare.wav | 45 | - | silent | en-t-word-seed0.wav | en-t-line-seed0.wav |
| p | Pig | en-p.wav | en-p-bare.wav | 45 | - | silent | en-p-word-seed0.wav | en-p-line-seed0.wav |
| m | Mouse | en-m.wav | en-m-bare.wav | 350 | - | silent | en-m-word-seed0.wav | en-m-line-seed0.wav |
| d | Dog | en-d.wav | en-d-bare.wav | 45 | - | silent | en-d-word-seed0.wav | en-d-line-seed0.wav |
| g | Goat | en-g.wav | en-g-bare.wav | 45 | - | silent | en-g-word-seed2.wav | en-g-line-seed0.wav |
| o | Otter | en-o.wav | en-o-bare.wav | 350 | ɑː | clean | en-o-word-seed1.wav | en-o-line-seed0.wav |
| k | Kangaroo | en-k.wav | en-k-bare.wav | 45 | - | silent | en-k-word-seed0.wav | en-k-line-seed0.wav |
| e | Egg | en-e.wav | en-e-bare.wav | 350 | ɛ | clean | en-e-word-seed4.wav | en-e-line-seed0.wav |
| u | Up | en-u.wav | en-u-bare.wav | 350 | a | clean | en-u-word-seed1.wav | en-u-line-seed0.wav |
| r | Rain | en-r.wav | en-r-bare.wav | 350 | - | silent | en-r-word-seed0.wav | en-r-line-seed0.wav |
| h | Horse | en-h.wav | en-h-bare.wav | 350 | h | clean | en-h-word-seed0.wav | en-h-line-seed0.wav |
| b | Bird | en-b.wav | en-b-bare.wav | 45 | - | silent | en-b-word-seed0.wav | en-b-line-seed0.wav |
| l | Leaf | en-l.wav | en-l-bare.wav | 350 | - | silent | en-l-word-seed1.wav | en-l-line-seed0.wav |
| w | Web | en-w.wav | en-w-bare.wav | 350 | - | silent | en-w-word-seed0.wav | en-w-line-seed0.wav |
| n | Net | en-n.wav | en-n-bare.wav | 350 | - | silent | en-n-word-seed0.wav | en-n-line-seed0.wav |

## Every candidate

| sound | take | Whisper heard (word) | first phone | bare ms | sound part heard | verdict |
|---|---|---|---|---|---|---|
| s | en-s-word-seed0.wav | Snake! | s | 350 |  | silent |
| s | en-s-word-seed1.wav | Snake | s | 350 |  | silent |
| s | en-s-word-seed2.wav | Snake | s | 350 |  | silent |
| s | en-s-word-seed3.wav | Snake! | s | 350 |  | silent |
| s | en-s-word-seed4.wav | Snake | s | 350 |  | silent |
| a | en-a-word-seed0.wav | Ant | æ | 350 | eɪ n n | not clean |
| a | en-a-word-seed1.wav | And | a | 350 | h | not clean, word missed |
| a | en-a-word-seed2.wav | Ant | æ | 350 | æ | clean |
| a | en-a-word-seed3.wav | Aunt | æ | 315 | æ n | not clean, word missed |
| a | en-a-word-seed4.wav | Ant | aː | - | - | first phone aː |
| t | en-t-word-seed0.wav | Tiger | t | 45 |  | silent |
| t | en-t-word-seed1.wav | Tiger! | t | 45 |  | silent |
| t | en-t-word-seed2.wav | Tiger | t | 45 |  | silent |
| t | en-t-word-seed3.wav | Tiger | t | 45 |  | silent |
| t | en-t-word-seed4.wav | Tiger | t | 45 |  | silent |
| p | en-p-word-seed0.wav | Pig | p | 45 |  | silent |
| p | en-p-word-seed1.wav | Pig! | p | 45 |  | silent |
| p | en-p-word-seed2.wav | Pig | p | 45 |  | silent |
| p | en-p-word-seed3.wav | Pig! | p | 45 |  | silent |
| p | en-p-word-seed4.wav | Pig | p | 45 |  | silent |
| m | en-m-word-seed0.wav | Mouse. | m | 350 |  | silent |
| m | en-m-word-seed1.wav | Mouse | m | 350 | m a n a n a n a | not clean |
| m | en-m-word-seed2.wav | Mouse | m | 350 |  | silent |
| m | en-m-word-seed3.wav | Mouse. | m | 350 |  | silent |
| m | en-m-word-seed4.wav | Mouse | m | 350 |  | silent |
| d | en-d-word-seed0.wav | Dog. | d | 45 |  | silent |
| d | en-d-word-seed1.wav | Dog! | d | 45 |  | silent |
| d | en-d-word-seed2.wav | Dog | d | 45 |  | silent |
| d | en-d-word-seed3.wav | Dog | d | 35 |  | silent |
| d | en-d-word-seed4.wav | Dog | d | 40 |  | silent |
| g | en-g-word-seed0.wav | Goat | ɡ | 20 |  | silent |
| g | en-g-word-seed1.wav | Goat. | ɡ | 15 |  | silent |
| g | en-g-word-seed2.wav | Goat | ɡ | 45 |  | silent |
| g | en-g-word-seed3.wav | Goat. | ɡ | 20 |  | silent |
| g | en-g-word-seed4.wav | Goat | ɡ | 25 |  | silent |
| o | en-o-word-seed0.wav | Otter | a | - | - | first phone a |
| o | en-o-word-seed1.wav | Otter? | ɑ | 350 | ɑː | clean |
| o | en-o-word-seed2.wav | Otter! | aː | - | - | first phone aː |
| o | en-o-word-seed3.wav | Otter | ɑː | 350 | ɑː | clean |
| o | en-o-word-seed4.wav | Otter? | aː | - | - | first phone aː |
| k | en-k-word-seed0.wav | Kangaroo? | k | 45 |  | silent |
| k | en-k-word-seed1.wav | Kangaroo? | k | 45 |  | silent |
| k | en-k-word-seed2.wav | Kangaroo | k | 45 |  | silent |
| k | en-k-word-seed3.wav | Kangaroo | k | 45 |  | silent |
| k | en-k-word-seed4.wav | Kangaroo! | k | 45 |  | silent |
| e | en-e-word-seed0.wav | Egg | a | 325 | ɛ ɡ | not clean |
| e | en-e-word-seed1.wav | Egg. | a | 350 | aː | not clean |
| e | en-e-word-seed2.wav | Egg | a | 350 | aʊ | not clean |
| e | en-e-word-seed3.wav | Egg | - | - | - | too few phones |
| e | en-e-word-seed4.wav | Egg? | a | 350 | ɛ | clean |
| u | en-u-word-seed0.wav |  | - | - | - | too few phones |
| u | en-u-word-seed1.wav | Up! | a | 350 | a | clean |
| u | en-u-word-seed2.wav | Up! | - | - | - | too few phones |
| u | en-u-word-seed3.wav | Up. | a | 350 | a r ɹ | not clean |
| u | en-u-word-seed4.wav | Up! | a | 350 |  | silent |
| r | en-r-word-seed0.wav | Rain | ɹ | 350 |  | silent |
| r | en-r-word-seed1.wav | Rain | ɹ | 350 |  | silent |
| r | en-r-word-seed2.wav | Rain | ɹ | 350 |  | silent |
| r | en-r-word-seed3.wav | Rain | ɹ | - | - | onset too short (35 ms) |
| r | en-r-word-seed4.wav | Rain | ɹ | 350 |  | silent |
| h | en-h-word-seed0.wav | Horse | h | 350 | h | clean |
| h | en-h-word-seed1.wav | Horse | h | 350 |  | silent |
| h | en-h-word-seed2.wav | Horse | h | 350 |  | silent |
| h | en-h-word-seed3.wav | Horse | h | 350 |  | silent |
| h | en-h-word-seed4.wav | Horse | uɨ | - | - | first phone uɨ |
| b | en-b-word-seed0.wav | Bird. | b | 45 |  | silent |
| b | en-b-word-seed1.wav | Bird | b | 20 |  | silent |
| b | en-b-word-seed2.wav | Bird | b | 15 |  | silent |
| b | en-b-word-seed3.wav | Bird. | b | 15 |  | silent |
| b | en-b-word-seed4.wav | Third, | ð | - | - | first phone ð |
| l | en-l-word-seed0.wav | Leaf | l | 350 | uː | not clean |
| l | en-l-word-seed1.wav | Leaf | l | 350 |  | silent |
| l | en-l-word-seed2.wav | Leaf | l | 350 |  | silent |
| l | en-l-word-seed3.wav | Leaf | l | 350 |  | silent |
| l | en-l-word-seed4.wav | Leaf | l | - | - | onset too short (25 ms) |
| w | en-w-word-seed0.wav | Web | w | 350 |  | silent |
| w | en-w-word-seed1.wav | web | w | 350 | uː | not clean |
| w | en-w-word-seed2.wav | Web | w | 350 | uː | not clean |
| w | en-w-word-seed3.wav | Web | w | 350 |  | silent |
| w | en-w-word-seed4.wav | web | w | - | - | onset too short (30 ms) |
| n | en-n-word-seed0.wav | net | n | 350 |  | silent |
| n | en-n-word-seed1.wav | net | n | 350 |  | silent |
| n | en-n-word-seed2.wav | net | n | 350 |  | silent |
| n | en-n-word-seed3.wav | net | m | - | - | first phone m |
| n | en-n-word-seed4.wav | net | n | 350 |  | silent |
