# Kalimo en letter sounds cut from whole word takes

Source: whole Qwen3-TTS 1.7B Base takes of the word alone (5 seeds) and of the carrier line (5 seeds), same cloned voice.
Boundary: wav2vec2 espeak CTC phone timings, refined to the steepest energy rise (consonant to vowel) or fall (vowel to consonant).
Continuants and vowels: the real onset sound, lengthened to about 600 ms by crossfading up to 120 ms of its own steady middle (no resampling, no pitch change), 20/25 ms fades. Stops: from 30 ms before the burst up to the start of the vowel (40 to 120 ms after the burst), said once. Onset found 10 dB above the noise floor, with 20 ms lead-in.
Clip = bare sound, 350 ms pause, whole carrier take; parts matched in loudness (RMS for continuants, peak for stops), clip normalised to about -20 dBFS RMS.
Check: wav2vec2 on the whole clip; the sound part is every phone before the carrier ("v i" / "l aɪ"). **clean** = only the target phone there, no vowel. The recogniser hears nothing in many bare stops and some fricatives (**silent**), so an ear decides.

## Chosen per sound

| sound | word | clip | bare | bare ms | sound part heard | verdict | source take | carrier take |
|---|---|---|---|---|---|---|---|---|
| s | Snake | en-s.wav | en-s-bare.wav | 600 | - | silent | en-s-word-seed0.wav | en-s-line-seed0.wav |
| a | Ant | en-a.wav | en-a-bare.wav | 600 | æ | clean | en-a-word-seed2.wav | en-a-line-seed0.wav |
| t | Tiger | en-t.wav | en-t-bare.wav | 120 | t | clean | en-t-word-seed1.wav | en-t-line-seed0.wav |
| p | Pig | en-p.wav | en-p-bare.wav | 85 | - | silent | en-p-word-seed1.wav | en-p-line-seed0.wav |
| m | Mouse | en-m.wav | en-m-bare.wav | 600 | - | silent | en-m-word-seed0.wav | en-m-line-seed0.wav |
| d | Dog | en-d.wav | en-d-bare.wav | 75 | d | clean | en-d-word-seed0.wav | en-d-line-seed0.wav |
| g | Goat | en-g.wav | en-g-bare.wav | 70 | - | silent | en-g-word-seed0.wav | en-g-line-seed0.wav |
| o | Otter | en-o.wav | en-o-bare.wav | 600 | ɑː | clean | en-o-word-seed1.wav | en-o-line-seed0.wav |
| k | Kangaroo | en-k.wav | en-k-bare.wav | 115 | k | clean | en-k-word-seed3.wav | en-k-line-seed0.wav |
| e | Egg | en-e.wav | en-e-bare.wav | 600 | a | clean | en-e-word-seed4.wav | en-e-line-seed0.wav |
| u | Up | en-u.wav | en-u-bare.wav | 600 | - | silent | en-u-word-seed4.wav | en-u-line-seed0.wav |
| r | Rain | en-r.wav | en-r-bare.wav | 600 | ɹ | clean | en-r-word-seed2.wav | en-r-line-seed0.wav |
| h | Horse | en-h.wav | en-h-bare.wav | 600 | h | clean | en-h-word-seed0.wav | en-h-line-seed0.wav |
| b | Bird | en-b.wav | en-b-bare.wav | 70 | - | silent | en-b-word-seed0.wav | en-b-line-seed0.wav |
| l | Leaf | en-l.wav | en-l-bare.wav | 600 | - | silent | en-l-word-seed1.wav | en-l-line-seed0.wav |
| w | Web | en-w.wav | en-w-bare.wav | 600 | - | silent | en-w-word-seed0.wav | en-w-line-seed0.wav |
| n | Net | en-n.wav | en-n-bare.wav | 600 | - | silent | en-n-word-seed0.wav | en-n-line-seed0.wav |

## Every candidate

| sound | take | Whisper heard (word) | first phone | bare ms | sound part heard | verdict |
|---|---|---|---|---|---|---|
| s | en-s-word-seed0.wav | Snake! | s | 600 |  | silent |
| s | en-s-word-seed1.wav | Snake | s | 600 |  | silent |
| s | en-s-word-seed2.wav | Snake | s | 600 |  | silent |
| s | en-s-word-seed3.wav | Snake! | s | 600 |  | silent |
| s | en-s-word-seed4.wav | Snake | s | 600 |  | silent |
| a | en-a-word-seed0.wav | Ant | æ | 600 | n a n a n a a | not clean |
| a | en-a-word-seed1.wav | And | a | 600 | h h | not clean, word missed |
| a | en-a-word-seed2.wav | Ant | æ | 600 | æ | clean |
| a | en-a-word-seed3.wav | Aunt | æ | 600 | eɪ | not clean, word missed |
| a | en-a-word-seed4.wav | Ant | aː | - | - | first phone aː |
| t | en-t-word-seed0.wav | Tiger | t | 70 |  | silent |
| t | en-t-word-seed1.wav | Tiger! | t | 120 | t | clean |
| t | en-t-word-seed2.wav | Tiger | t | 150 | t | clean |
| t | en-t-word-seed3.wav | Tiger | t | 150 | t a | not clean |
| t | en-t-word-seed4.wav | Tiger | t | 70 |  | silent |
| p | en-p-word-seed0.wav | Pig | p | 70 |  | silent |
| p | en-p-word-seed1.wav | Pig! | p | 85 |  | silent |
| p | en-p-word-seed2.wav | Pig | p | 85 |  | silent |
| p | en-p-word-seed3.wav | Pig! | p | 85 |  | silent |
| p | en-p-word-seed4.wav | Pig | p | 80 |  | silent |
| m | en-m-word-seed0.wav | Mouse. | m | 600 |  | silent |
| m | en-m-word-seed1.wav | Mouse | m | 600 | m a n a n a n a n a | not clean |
| m | en-m-word-seed2.wav | Mouse | m | 600 |  | silent |
| m | en-m-word-seed3.wav | Mouse. | m | 600 |  | silent |
| m | en-m-word-seed4.wav | Mouse | m | 600 |  | silent |
| d | en-d-word-seed0.wav | Dog. | d | 75 | d | clean |
| d | en-d-word-seed1.wav | Dog! | d | 75 | d | clean |
| d | en-d-word-seed2.wav | Dog | d | 70 | d | clean |
| d | en-d-word-seed3.wav | Dog | d | 70 | d | clean |
| d | en-d-word-seed4.wav | Dog | d | 70 |  | silent |
| g | en-g-word-seed0.wav | Goat | ɡ | 70 |  | silent |
| g | en-g-word-seed1.wav | Goat. | ɡ | 70 |  | silent |
| g | en-g-word-seed2.wav | Goat | ɡ | 70 |  | silent |
| g | en-g-word-seed3.wav | Goat. | ɡ | 70 |  | silent |
| g | en-g-word-seed4.wav | Goat | ɡ | 70 |  | silent |
| o | en-o-word-seed0.wav | Otter | a | - | - | first phone a |
| o | en-o-word-seed1.wav | Otter? | ɑ | 600 | ɑː | clean |
| o | en-o-word-seed2.wav | Otter! | aː | - | - | first phone aː |
| o | en-o-word-seed3.wav | Otter | ɑː | 600 | ɑː | clean |
| o | en-o-word-seed4.wav | Otter? | aː | - | - | first phone aː |
| k | en-k-word-seed0.wav | Kangaroo? | k | 75 |  | silent |
| k | en-k-word-seed1.wav | Kangaroo? | k | 115 |  | silent |
| k | en-k-word-seed2.wav | Kangaroo | k | 90 |  | silent |
| k | en-k-word-seed3.wav | Kangaroo | k | 115 | k | clean |
| k | en-k-word-seed4.wav | Kangaroo! | k | 75 |  | silent |
| e | en-e-word-seed0.wav | Egg | a | 600 | a r r | not clean |
| e | en-e-word-seed1.wav | Egg. | a | 650 | a | clean |
| e | en-e-word-seed2.wav | Egg | a | 600 | a h | not clean |
| e | en-e-word-seed3.wav | Egg | - | - | - | too few phones |
| e | en-e-word-seed4.wav | Egg? | a | 600 | a | clean |
| u | en-u-word-seed0.wav |  | - | - | - | too few phones |
| u | en-u-word-seed1.wav | Up! | a | 600 | a a a a | not clean |
| u | en-u-word-seed2.wav | Up! | - | - | - | too few phones |
| u | en-u-word-seed3.wav | Up. | a | 600 | a b a b a b a b a d | not clean |
| u | en-u-word-seed4.wav | Up! | a | 600 |  | silent |
| r | en-r-word-seed0.wav | Rain | ɹ | 600 |  | silent |
| r | en-r-word-seed1.wav | Rain | ɹ | 600 |  | silent |
| r | en-r-word-seed2.wav | Rain | ɹ | 600 | ɹ | clean |
| r | en-r-word-seed3.wav | Rain | ɹ | 600 |  | silent |
| r | en-r-word-seed4.wav | Rain | ɹ | 600 |  | silent |
| h | en-h-word-seed0.wav | Horse | h | 600 | h | clean |
| h | en-h-word-seed1.wav | Horse | h | 600 | h | clean |
| h | en-h-word-seed2.wav | Horse | h | 600 |  | silent |
| h | en-h-word-seed3.wav | Horse | h | 600 |  | silent |
| h | en-h-word-seed4.wav | Horse | uɨ | - | - | first phone uɨ |
| b | en-b-word-seed0.wav | Bird. | b | 70 |  | silent |
| b | en-b-word-seed1.wav | Bird | b | 70 |  | silent |
| b | en-b-word-seed2.wav | Bird | b | 70 |  | silent |
| b | en-b-word-seed3.wav | Bird. | b | 70 |  | silent |
| b | en-b-word-seed4.wav | Third, | ð | - | - | first phone ð |
| l | en-l-word-seed0.wav | Leaf | l | 600 | uː | not clean |
| l | en-l-word-seed1.wav | Leaf | l | 600 |  | silent |
| l | en-l-word-seed2.wav | Leaf | l | 600 |  | silent |
| l | en-l-word-seed3.wav | Leaf | l | 600 | uː | not clean |
| l | en-l-word-seed4.wav | Leaf | l | - | - | onset too short (35 ms) |
| w | en-w-word-seed0.wav | Web | w | 600 |  | silent |
| w | en-w-word-seed1.wav | web | w | 600 | uː | not clean |
| w | en-w-word-seed2.wav | Web | w | 600 | uː | not clean |
| w | en-w-word-seed3.wav | Web | w | 600 |  | silent |
| w | en-w-word-seed4.wav | web | w | 600 |  | silent |
| n | en-n-word-seed0.wav | net | n | 600 |  | silent |
| n | en-n-word-seed1.wav | net | n | 600 |  | silent |
| n | en-n-word-seed2.wav | net | n | 600 |  | silent |
| n | en-n-word-seed3.wav | net | m | - | - | first phone m |
| n | en-n-word-seed4.wav | net | n | 600 |  | silent |
