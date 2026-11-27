# Kalimo en letter sounds: whole Qwen3 takes, checked by phones and Whisper

Whole takes only: no cutting, no processing. Seeds 777+N (file suffix -seedN).
Phones: facebook/wav2vec2-lv-60-espeak-cv-ft (Apache-2.0) on the whole take; the sound part is every phone before "wie"/"like".
**clean** = the sound part holds only the target sound (no vowel after a consonant, no other consonant), and Whisper large-v3 heard the model word.
Top takes: clean takes of the best spelling, the single-phone ones first. An ear still decides.

## Best spelling per sound

| sound | spelling | clean takes | top 3 | other spellings |
|---|---|---|---|---|
| t | T | 0/15 | - | T! 0/15, Tt 0/15 |
| s | Sss | 6/15 | en-S-Sss-seed0.wav en-S-Sss-seed1.wav en-S-Sss-seed10.wav | - |
| m | M | 1/15 | en-M-cap-seed3.wav | - |

## Every take

| file | text | length s | sound part | all phones | Whisper | verdict |
|---|---|---|---|---|---|---|
| en-T-bang-seed0.wav | T! like tiger. | 1.84 | p l eɪ | p l eɪ l aɪ k t aɪ ɡ ɚ | Pleat like tiger. | wrong sound |
| en-T-bang-seed1.wav | T! like tiger. | 1.60 | t l i | t l i l aɪ k t aɪ ɡ ɚ | Plate like tiger | vowel added |
| en-T-bang-seed2.wav | T! like tiger. | 1.20 | b l ɛ ɡ t aɪ ɡ ɚ | b l ɛ ɡ t aɪ ɡ ɚ | Like Tiger | no carrier |
| en-T-bang-seed3.wav | T! like tiger. | 2.40 | ɡ l iː | ɡ l iː l aɪ k t aɪ ɡ ɚ | Glee, like tiger. | wrong sound |
| en-T-bang-seed4.wav | T! like tiger. | 1.36 | t uː | t uː l aɪ k t aɪ ɡ ɚ | 2. Like Tiger | vowel added |
| en-T-bang-seed5.wav | T! like tiger. | 1.60 | p l ɪ ɡ t aɪ ɡ ɚ | p l ɪ ɡ t aɪ ɡ ɚ | Click tiger! | no carrier |
| en-T-bang-seed6.wav | T! like tiger. | 2.08 | f l iː | f l iː l aɪ k t aɪ ɡ ɚ | Flea like tiger | wrong sound |
| en-T-bang-seed7.wav | T! like tiger. | 1.76 | t l ɛ k t aɪ ɡ ɚ | t l ɛ k t aɪ ɡ ɚ | Lake Tiger | no carrier |
| en-T-bang-seed8.wav | T! like tiger. | 1.68 | s | s l aɪ k t aɪ ɡ ɚ | Like Tiger? | wrong sound |
| en-T-bang-seed9.wav | T! like tiger. | 1.36 | p l i | p l i l aɪ k t aɪ ɡ ɚ | Plea, like tiger! | wrong sound |
| en-T-bang-seed10.wav | T! like tiger. | 2.40 | p l iː | p l iː l aɪ k t aɪ ɡ ɚ | Please, Lake Tiger! | wrong sound |
| en-T-bang-seed11.wav | T! like tiger. | 2.08 |  | l aɪ k t aɪ ɡ ɚ | Like tiger. | no sound |
| en-T-bang-seed12.wav | T! like tiger. | 1.44 | p l iː t | p l iː t l aɪ k t aɪ ɡ ɚ | Tleet like tiger. | vowel added |
| en-T-bang-seed13.wav | T! like tiger. | 1.92 | t l ɪ | t l ɪ l aɪ k t aɪ ɡ ɚ | T-like tiger! | vowel added |
| en-T-bang-seed14.wav | T! like tiger. | 1.36 | l iː k | l iː k l aɪ k t aɪ ɡ ɚ | Leet like tiger | wrong sound |
| en-T-Tt-seed0.wav | Tt, like tiger. | 2.56 | t iː t i | t iː t i l aɪ k t aɪ ɡ ɚ | T, T, like tiger. | vowel added |
| en-T-Tt-seed1.wav | Tt, like tiger. | 2.08 | t iː t iː | t iː t iː l aɪ k t aɪ ɡ ɚ | TT like tiger | vowel added |
| en-T-Tt-seed2.wav | Tt, like tiger. | 1.84 | t iː t i | t iː t i l aɪ k t aɪ ɡ ɚ | TT like tiger | vowel added |
| en-T-Tt-seed3.wav | Tt, like tiger. | 2.80 | t iː t iː | t iː t iː l aɪ k t aɪ ɡ ɚ | T, T, like tiger. | vowel added |
| en-T-Tt-seed4.wav | Tt, like tiger. | 1.52 | t iː t iː | t iː t iː l aɪ k t aɪ ɡ ɚ | TT like tiger | vowel added |
| en-T-Tt-seed5.wav | Tt, like tiger. | 2.96 | t iː t iː | t iː t iː l aɪ ɡ t aɪ ɡ ɚ | T.T. like tiger | vowel added |
| en-T-Tt-seed6.wav | Tt, like tiger. | 2.40 | t iː t i | t iː t i l aɪ k t aɪ ɡ ɚ | TT? Like Tiger? | vowel added |
| en-T-Tt-seed7.wav | Tt, like tiger. | 2.56 | t iː t iː | t iː t iː l aɪ k t aɪ ɡ ɚ | Tee-Tee, like Tiger. | vowel added |
| en-T-Tt-seed8.wav | Tt, like tiger. | 1.92 | t iː t iː | t iː t iː l aɪ k t aɪ ɡ ɚ | TT, like tiger. | vowel added |
| en-T-Tt-seed9.wav | Tt, like tiger. | 1.84 | t i t i | t i t i l aɪ k t aɪ ɡ ɚ | TT, like tiger. | vowel added |
| en-T-Tt-seed10.wav | Tt, like tiger. | 3.04 | t iː t | t iː t l aɪ k t aɪ ɡ ɚ | T.T. like tiger. | vowel added |
| en-T-Tt-seed11.wav | Tt, like tiger. | 2.32 | t iː t iː | t iː t iː l aɪ k t aɪ ɡ ɚ | T.T. like tiger | vowel added |
| en-T-Tt-seed12.wav | Tt, like tiger. | 1.84 | t i t i | t i t i l aɪ k t aɪ ɡ ɚ | Tee, tee, like tiger. | vowel added |
| en-T-Tt-seed13.wav | Tt, like tiger. | 2.32 | t iː t iː | t iː t iː l aɪ k t aɪ ɡ ɚ | TT, like tiger. | vowel added |
| en-T-Tt-seed14.wav | Tt, like tiger. | 1.92 | t iː t i | t iː t i l aɪ k t aɪ ɡ ɚ | TT, like tiger? | vowel added |
| en-T-cap-seed0.wav | T, like tiger. | 1.36 | b l ɪ k t aɪ ɡ ɚ | b l ɪ k t aɪ ɡ ɚ | Flick Tiger | no carrier |
| en-T-cap-seed1.wav | T, like tiger. | 2.48 |  | l aɪ k l ɪ t l ɪ k l ɪ k p l aɪ k t aɪ ɡ ɚ | Like, like, like, like, like Tiger! | no sound |
| en-T-cap-seed2.wav | T, like tiger. | 1.76 | f l iː | f l iː l aɪ k t aɪ ɡ ɚ | Flea like tiger? | wrong sound |
| en-T-cap-seed3.wav | T, like tiger. | 50.08 | l iː l iː k | l iː l iː k l aɪ k l aɪ ɡ l aɪ t l aɪ t l aɪ k t aɪ ɡ ɚ l aɪ k t aɪ ɡ ɚ l aɪ k t aɪ ɡ ɚ t aɪ ɡ ɚ l aɪ t aɪ ɡ ɚ l iː ɡ l iː l k l aɪ k l k l iː k l k w iː l iː l iː k l iː k k l l k l iː ɡ l iː ɡ l iː k l k iː l ɛ l iː ɛ n d iː k k w ɛ k s l iː k l iː l l iː k l iː l iː k h ʌ l ɛ l l ɛ k l ɛ k l l l iː k l eɪ l l eɪ k i d l iː k l eɪ ʌ n k | like to like, like, like, like, like like tiger. Like tiger. Like tiger...tiger. Like tiger. Leek leek.. Leek, leek, leek. Leek, leek. Well, leek.. leek...leek', leek, leek. Leek, leek... And... Leek... Lick lick lick Leg leg leg Lick lick lick Lick lick lick | wrong sound |
| en-T-cap-seed4.wav | T, like tiger. | 1.28 | p l iː ð | p l iː ð l aɪ k t aɪ ɡ ɚ | Pleat like tiger! | wrong sound |
| en-T-cap-seed5.wav | T, like tiger. | 4.16 | k l ɪ | k l ɪ l aɪ ɡ l ɛ ɡ p l aɪ k p l aɪ k t aɪ ɡ ɚ l aɪ ɡ l aɪ ɡ l aɪ | click like like like like tiger like like | wrong sound |
| en-T-cap-seed6.wav | T, like tiger. | 2.00 | f l iː | f l iː l aɪ k t aɪ ɡ ɚ | Flee, like Tiger! | wrong sound |
| en-T-cap-seed7.wav | T, like tiger. | 1.28 | f | f l aɪ k t aɪ ɡ ɚ | Blake Tiger | wrong sound |
| en-T-cap-seed8.wav | T, like tiger. | 1.68 | p l iː d | p l iː d l aɪ k t aɪ ɡ ɚ | Plea like tiger. | wrong sound |
| en-T-cap-seed9.wav | T, like tiger. | 1.76 | p l iː | p l iː l aɪ k t aɪ ɡ ɚ | Plea, like tiger! | wrong sound |
| en-T-cap-seed10.wav | T, like tiger. | 1.76 | p l ɪ k | p l ɪ k l aɪ k t aɪ ɡ ɚ | Flick like tiger. | wrong sound |
| en-T-cap-seed11.wav | T, like tiger. | 2.72 | l iː t | l iː t l aɪ k t aɪ ɡ ɚ | Bleat like tiger. | vowel added |
| en-T-cap-seed12.wav | T, like tiger. | 1.28 | l iː l ɛ k t aɪ ɡ ɚ | l iː l ɛ k t aɪ ɡ ɚ | Le- Lake Tiger! | no carrier |
| en-T-cap-seed13.wav | T, like tiger. | 1.52 |  | l aɪ k t aɪ ɡ ɚ | Like tiger. | no sound |
| en-T-cap-seed14.wav | T, like tiger. | 2.16 |  | l aɪ k t aɪ ɡ ɚ | Like Tiger | no sound |
| en-S-Sss-seed0.wav | Sss, like snake. | 2.48 | s | s l aɪ k s n eɪ k | Ssss, like snake! | clean |
| en-S-Sss-seed1.wav | Sss, like snake. | 1.76 | s | s l aɪ k s n eɪ k | Sssss, like snake. | clean |
| en-S-Sss-seed2.wav | Sss, like snake. | 2.32 | s | s l aɪ k s n eɪ k | Ssss, like snake. | clean |
| en-S-Sss-seed3.wav | Sss, like snake. | 1.68 |  | l aɪ k ə s n eɪ k | Ssssss, like snake! | no sound |
| en-S-Sss-seed4.wav | Sss, like snake. | 1.28 |  | l aɪ k s n iː k | Like snake | no sound |
| en-S-Sss-seed5.wav | Sss, like snake. | 1.68 |  | l aɪ k ə s n eɪ k | Ssssss, like snake. | no sound |
| en-S-Sss-seed6.wav | Sss, like snake. | 1.20 |  | l aɪ k s n eɪ | Ssssss, like snake. | no sound |
| en-S-Sss-seed7.wav | Sss, like snake. | 1.68 |  | l aɪ k s n eɪ k | Ssss... Like snake? | no sound |
| en-S-Sss-seed8.wav | Sss, like snake. | 1.92 | l ɪ k s n eɪ k | l ɪ k s n eɪ k | Like snake. | no carrier |
| en-S-Sss-seed9.wav | Sss, like snake. | 1.68 | l ɛ ɡ s n eɪ k | l ɛ ɡ s n eɪ k | Ssssss, like snake. | no carrier |
| en-S-Sss-seed10.wav | Sss, like snake. | 2.08 | s | s l aɪ k s n eɪ k | Ssss, like snake. | clean |
| en-S-Sss-seed11.wav | Sss, like snake. | 2.64 | s | s l aɪ k s n eɪ k | Ssssss, like snake. | clean |
| en-S-Sss-seed12.wav | Sss, like snake. | 1.36 | s | s l aɪ k ə s n eɪ k | Ssss, like snake. | clean |
| en-S-Sss-seed13.wav | Sss, like snake. | 1.84 |  | l aɪ k ə s n eɪ k | Ssssss... like snake? | no sound |
| en-S-Sss-seed14.wav | Sss, like snake. | 2.40 |  | l aɪ k s n iː k | Like snake. | no sound |
| en-M-cap-seed0.wav | M, like mouse. | 2.24 | m ɛ | m ɛ l aɪ k m aʊ s | Mm-hmm like mouse | vowel added |
| en-M-cap-seed1.wav | M, like mouse. | 3.76 | h uː | h uː l aɪ k m aʊ s | Hmm, like mouse. | wrong sound |
| en-M-cap-seed2.wav | M, like mouse. | 2.64 |  | l aɪ k m aʊ s | Hmm, like mouse. | no sound |
| en-M-cap-seed3.wav | M, like mouse. | 1.60 | m | m l aɪ k m aʊ s | Mmm, like mouse. | clean |
| en-M-cap-seed4.wav | M, like mouse. | 1.60 | n ɪ m | n ɪ m l aɪ k m aʊ s | Like mouse. | vowel added |
| en-M-cap-seed5.wav | M, like mouse. | 2.56 |  | l aɪ k m aʊ s | Mmm, like mouse. | no sound |
| en-M-cap-seed6.wav | M, like mouse. | 2.24 |  | l aɪ k m aʊ s | Mmm, like mouse. | no sound |
| en-M-cap-seed7.wav | M, like mouse. | 2.40 |  | l aɪ k m aʊ s | Mmm, like mouse. | no sound |
| en-M-cap-seed8.wav | M, like mouse. | 2.08 |  | l aɪ k m aʊ s | Mmm, like mouse. | no sound |
| en-M-cap-seed9.wav | M, like mouse. | 2.56 |  | l aɪ k m aʊ s | Mm, mm, like mouse. | no sound |
| en-M-cap-seed10.wav | M, like mouse. | 1.52 |  | l aɪ k n aʊ s | Mm, like mouse. | no sound |
| en-M-cap-seed11.wav | M, like mouse. | 1.92 |  | l aɪ k m aʊ s | Mmm, like mouse. | no sound |
| en-M-cap-seed12.wav | M, like mouse. | 1.84 | uː | uː l aɪ k m aʊ s | Mm, like mouse. | wrong sound |
| en-M-cap-seed13.wav | M, like mouse. | 3.04 |  | l aɪ k m aʊ s | mmm like mouse | no sound |
| en-M-cap-seed14.wav | M, like mouse. | 2.00 |  | l aɪ k m aʊ s | Hmm, like mouse. | no sound |
