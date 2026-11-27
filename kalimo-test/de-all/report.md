# Kalimo de letter sounds: whole Qwen3 takes, checked by phones and Whisper

Whole takes only: no cutting, no processing. Seeds 777+N (file suffix -seedN).
Phones: facebook/wav2vec2-lv-60-espeak-cv-ft (Apache-2.0) on the whole take; the sound part is every phone before "wie"/"like".
**clean** = the sound part holds only the target sound (no vowel after a consonant, no other consonant), and Whisper large-v3 heard the model word.
Top takes: clean takes of the best spelling, the single-phone ones first. An ear still decides.

## Best spelling per sound

| sound | spelling | clean takes | top 3 | other spellings |
|---|---|---|---|---|
| m | M | 0/15 | - | Mmm 0/15 |
| a | A | 15/15 | de-a-cap-seed0.wav de-a-cap-seed1.wav de-a-cap-seed10.wav | - |
| l | Lll | 1/15 | de-l-triple-seed4.wav | L 0/15 |
| o | O | 12/15 | de-o-cap-seed0.wav de-o-cap-seed10.wav de-o-cap-seed11.wav | - |
| i | I | 12/15 | de-i-cap-seed0.wav de-i-cap-seed1.wav de-i-cap-seed10.wav | - |
| e | E | 7/15 | de-e-cap-seed0.wav de-e-cap-seed1.wav de-e-cap-seed11.wav | - |
| s | Sss | 12/15 | de-s-triple-seed0.wav de-s-triple-seed1.wav de-s-triple-seed10.wav | S 4/15 |
| u | U | 14/15 | de-u-cap-seed0.wav de-u-cap-seed1.wav de-u-cap-seed10.wav | - |
| r | Rrr | 2/15 | de-r-triple-seed4.wav de-r-triple-seed8.wav | R 0/15 |
| t | T | 0/15 | - | T! 0/15 |
| n | Nnn | 2/15 | de-n-triple-seed11.wav de-n-triple-seed2.wav | N 0/15 |
| f | F | 0/15 | - | Fff 0/15 |
| h | H | 0/15 | - | Hh 0/15 |
| b | B | 0/15 | - | B! 0/15 |
| k | K! | 1/15 | de-k-bang-seed0.wav | K 0/15 |
| p | P | 1/15 | de-p-cap-seed14.wav | P! 0/15 |
| g | G | 0/15 | - | G! 0/15 |
| w | W | 0/15 | - | Www 0/15 |
| d | D | 0/15 | - | D! 0/15 |

## Every take

| file | text | length s | sound part | all phones | Whisper | verdict |
|---|---|---|---|---|---|---|
| de-m-cap-seed0.wav | M, wie Maus. | 1.60 | e m | e m v i m aʊ s | M wie Maus | vowel added |
| de-m-cap-seed1.wav | M, wie Maus. | 1.20 | e m | e m ð i m aʊ s | M wie Maus | vowel added |
| de-m-cap-seed2.wav | M, wie Maus. | 1.52 | e m | e m v iː m aʊ | M, M wie Maus. | vowel added |
| de-m-cap-seed3.wav | M, wie Maus. | 1.52 | e m | e m v iː m aʊ z | M. Wimaus | vowel added |
| de-m-cap-seed4.wav | M, wie Maus. | 1.60 | e m | e m v i m aʊ s | M wie Maus | vowel added |
| de-m-cap-seed5.wav | M, wie Maus. | 1.76 | e m | e m ð iː m aʊ s | M wie Maus | vowel added |
| de-m-cap-seed6.wav | M, wie Maus. | 1.76 | e m | e m v iː m aʊ s | M wie Maus | vowel added |
| de-m-cap-seed7.wav | M, wie Maus. | 1.44 | ɛ m | ɛ m ð iː m aʊ s | M wie Maus | vowel added |
| de-m-cap-seed8.wav | M, wie Maus. | 1.52 | e m | e m v iː m aʊ s | M wie Maus | vowel added |
| de-m-cap-seed9.wav | M, wie Maus. | 1.52 | e m | e m v i m aʊ s | M wie Maus | vowel added |
| de-m-cap-seed10.wav | M, wie Maus. | 2.00 |  | v iː m aʊ s | mh, wie Maus. | no sound |
| de-m-cap-seed11.wav | M, wie Maus. | 2.08 | ɛ m | ɛ m v iː m aʊ s | M wie Maus | vowel added |
| de-m-cap-seed12.wav | M, wie Maus. | 1.44 | ɛ m | ɛ m ð iː m aʊ s | M wie Maus | vowel added |
| de-m-cap-seed13.wav | M, wie Maus. | 1.68 | ɛ m | ɛ m v iː m aʊ s | M wie Maus | vowel added |
| de-m-cap-seed14.wav | M, wie Maus. | 2.24 | e m | e m v i m aʊ s | M wie Maus | vowel added |
| de-m-triple-seed0.wav | Mmm, wie Maus. | 1.84 |  | v iː m aʊ s | Mh, wie Maus. | no sound |
| de-m-triple-seed1.wav | Mmm, wie Maus. | 1.92 |  | v iː m aʊ z | Mhh, wie Maus. | no sound |
| de-m-triple-seed2.wav | Mmm, wie Maus. | 2.16 |  | v iː m aʊ s | Mh, wie Maus. | no sound |
| de-m-triple-seed3.wav | Mmm, wie Maus. | 1.76 |  | v iː m aʊ s | Wie Maus. | no sound |
| de-m-triple-seed4.wav | Mmm, wie Maus. | 1.68 |  | v iː m aʊ s | Wie Maus. | no sound |
| de-m-triple-seed5.wav | Mmm, wie Maus. | 1.76 |  | f iː m aʊ s | Wie Maus. | no sound |
| de-m-triple-seed6.wav | Mmm, wie Maus. | 1.76 |  | v iː m aʊ s | Mh, wie Maus. | no sound |
| de-m-triple-seed7.wav | Mmm, wie Maus. | 2.08 |  | v iː m aʊ s | Mh, wie Maus. | no sound |
| de-m-triple-seed8.wav | Mmm, wie Maus. | 1.84 |  | ð iː m aʊ s | Mh, wie Maus. | no sound |
| de-m-triple-seed9.wav | Mmm, wie Maus. | 1.68 |  | v iː m aʊ s | Wie Maus. | no sound |
| de-m-triple-seed10.wav | Mmm, wie Maus. | 2.08 |  | v iː m aʊ s | Mh, wie Maus. | no sound |
| de-m-triple-seed11.wav | Mmm, wie Maus. | 2.72 |  | ð iː m aʊ s | Wie Maus. | no sound |
| de-m-triple-seed12.wav | Mmm, wie Maus. | 1.28 |  | v iː m aʊ z | Mh, wie Maus. | no sound |
| de-m-triple-seed13.wav | Mmm, wie Maus. | 1.60 |  | v i m aʊ s | Mh, wie Maus. | no sound |
| de-m-triple-seed14.wav | Mmm, wie Maus. | 1.68 |  | v i m aʊ s | Mhh, wie Maus. | no sound |
| de-a-cap-seed0.wav | A, wie Ameise. | 1.68 | ɑː | ɑː v iː a m aɪ z ə | A wie Ameise | clean |
| de-a-cap-seed1.wav | A, wie Ameise. | 1.68 | ɑː | ɑː v iː a m aɪ z ə | A wie Ameise | clean |
| de-a-cap-seed2.wav | A, wie Ameise. | 1.92 | a | a v i a m aɪ z ə | A wie Ameise | clean |
| de-a-cap-seed3.wav | A, wie Ameise. | 1.68 | a | a v iː a m aɪ z ə | A wie Ameise | clean |
| de-a-cap-seed4.wav | A, wie Ameise. | 1.60 | a | a v iː a m aɪ z ə | A wie Ameise | clean |
| de-a-cap-seed5.wav | A, wie Ameise. | 1.52 | ɑː | ɑː v iː a m aɪ z ə | A wie Ameise | clean |
| de-a-cap-seed6.wav | A, wie Ameise. | 2.08 | ɑː | ɑː v iː a m aɪ z ə | A wie Ameise | clean |
| de-a-cap-seed7.wav | A, wie Ameise. | 1.84 | a | a v iː a m aɪ z ə | A wie Ameise | clean |
| de-a-cap-seed8.wav | A, wie Ameise. | 1.84 | a | a v iː a m aɪ z ə | A wie Ameise | clean |
| de-a-cap-seed9.wav | A, wie Ameise. | 1.60 | a | a v iː a m aɪ z ə | A wie Ameise | clean |
| de-a-cap-seed10.wav | A, wie Ameise. | 2.08 | a | a v iː a m aɪ z ə | A wie Ameise | clean |
| de-a-cap-seed11.wav | A, wie Ameise. | 2.00 | a | a v iː a m aɪ z ə | A wie Ameise | clean |
| de-a-cap-seed12.wav | A, wie Ameise. | 1.76 | a | a v iː a m aɪ z ə | A wie Ameise | clean |
| de-a-cap-seed13.wav | A, wie Ameise. | 1.76 | a | a ð iː a m aɪ z ə | A wie Ameise | clean |
| de-a-cap-seed14.wav | A, wie Ameise. | 1.76 | a | a v iː a m aɪ z ə | A wie Ameise | clean |
| de-l-cap-seed0.wav | L, wie Löffel. | 1.76 | ɛ l | ɛ l v i l a f | L wie Löffel | vowel added |
| de-l-cap-seed1.wav | L, wie Löffel. | 1.44 | ɛ l | ɛ l v i l a f ə | L wie Löffel | vowel added |
| de-l-cap-seed2.wav | L, wie Löffel. | 1.60 | ɛ l | ɛ l v i l a f | L wie Löffel | vowel added |
| de-l-cap-seed3.wav | L, wie Löffel. | 1.52 | ɛ l | ɛ l v i l a f ə | L wie Löffel | vowel added |
| de-l-cap-seed4.wav | L, wie Löffel. | 1.52 | ɛ l | ɛ l v i l a f ə | L. Willöffel | vowel added |
| de-l-cap-seed5.wav | L, wie Löffel. | 1.68 | ɛ l | ɛ l ð iː l ɛ f ə | L wie Löffel | vowel added |
| de-l-cap-seed6.wav | L, wie Löffel. | 1.76 | ɛ l | ɛ l v i l a f l | L wie Löffel | vowel added |
| de-l-cap-seed7.wav | L, wie Löffel. | 1.92 | e l | e l v i l a f l | L wie Löffel | vowel added |
| de-l-cap-seed8.wav | L, wie Löffel. | 2.24 | ɛ l | ɛ l v i l a f ə | L wie Löffel | vowel added |
| de-l-cap-seed9.wav | L, wie Löffel. | 2.24 | e l | e l v i l ʌ f ʌ l | L wie Löffel | vowel added |
| de-l-cap-seed10.wav | L, wie Löffel. | 2.00 | ɛ l | ɛ l v i l a f ə | L wie Löffel | vowel added |
| de-l-cap-seed11.wav | L, wie Löffel. | 2.32 | ɛ l | ɛ l v i l ə f | L. Wielöffel | vowel added |
| de-l-cap-seed12.wav | L, wie Löffel. | 1.52 | ɛ l | ɛ l v i l a f ø | L wie Löffel | vowel added |
| de-l-cap-seed13.wav | L, wie Löffel. | 1.52 | ɛ l | ɛ l v iː l ɛ f ə | L wie Löffel | vowel added |
| de-l-cap-seed14.wav | L, wie Löffel. | 1.44 | e l | e l v i l a f ə | L wie Löffel | vowel added |
| de-l-triple-seed0.wav | Lll, wie Löffel. | 2.00 | l ɛ l | l ɛ l v i l a f | Lell, wie löffel. | vowel added |
| de-l-triple-seed1.wav | Lll, wie Löffel. | 1.84 | l i l | l i l v iː l a f l | Lill, wie Löffel. | vowel added |
| de-l-triple-seed2.wav | Lll, wie Löffel. | 2.08 | l eː l | l eː l f iː l aː f ʉ | Lill, wie Löffel. | vowel added |
| de-l-triple-seed3.wav | Lll, wie Löffel. | 1.92 |  | v iː l a f l | L wie Löffel | no sound |
| de-l-triple-seed4.wav | Lll, wie Löffel. | 1.76 | l l | l l v iː l l a f l | Lil wie Löffel | clean |
| de-l-triple-seed5.wav | Lll, wie Löffel. | 2.16 | l a l | l a l v i l a f | Lelvi-Löffel | vowel added |
| de-l-triple-seed6.wav | Lll, wie Löffel. | 2.48 | l e l | l e l v iː l ɛ f ə | Lill, wie Löffel. | vowel added |
| de-l-triple-seed7.wav | Lll, wie Löffel. | 2.24 | n e l | n e l v i l a f l | Lell wie Löffel | vowel added |
| de-l-triple-seed8.wav | Lll, wie Löffel. | 2.16 | l i l | l i l v iː l a f | Lill, wie Löffel | vowel added |
| de-l-triple-seed9.wav | Lll, wie Löffel. | 1.36 | l ɛ l | l ɛ l v i l a f | Lil' wie Löffel | vowel added |
| de-l-triple-seed10.wav | Lll, wie Löffel. | 2.40 |  | v iː l a f | L wie Löffel | no sound |
| de-l-triple-seed11.wav | Lll, wie Löffel. | 1.76 | l ɛ l | l ɛ l v iː l a f l | Lell wie Löffel | vowel added |
| de-l-triple-seed12.wav | Lll, wie Löffel. | 2.32 | n ɛ n | n ɛ n v iː l ʌ f ʌ | Lell wie Löffel | wrong sound |
| de-l-triple-seed13.wav | Lll, wie Löffel. | 2.40 |  | v i l a f ø | O. Wie leffe. | no sound, word missed |
| de-l-triple-seed14.wav | Lll, wie Löffel. | 2.16 | l ɛ l | l ɛ l v i l a f ɛ | Lell wie Löffel | vowel added |
| de-o-cap-seed0.wav | O, wie Oma. | 1.68 | o | o v i uː m a | O wie Oma | clean |
| de-o-cap-seed1.wav | O, wie Oma. | 1.12 | uː | uː v i ʔ u m a | Oh, wie Oma. | wrong sound |
| de-o-cap-seed2.wav | O, wie Oma. | 1.12 | uː | uː v i u m a | O wie Oma. | wrong sound |
| de-o-cap-seed3.wav | O, wie Oma. | 1.20 | uː | uː v i ʔ uː m a | Oh, wie Oma. | wrong sound |
| de-o-cap-seed4.wav | O, wie Oma. | 1.68 | o | o v i o m a | O wie Oma | clean |
| de-o-cap-seed5.wav | O, wie Oma. | 1.36 | o | o v i o m a | Ovioma | clean |
| de-o-cap-seed6.wav | O, wie Oma. | 1.60 | o | o v iː o m a | O wie Oma | clean |
| de-o-cap-seed7.wav | O, wie Oma. | 1.52 | o | o v i u m a | O wie Oma | clean |
| de-o-cap-seed8.wav | O, wie Oma. | 1.36 | o | o v i o m a | O wie Oma | clean |
| de-o-cap-seed9.wav | O, wie Oma. | 1.36 | o | o v i u m a | O wie Oma | clean |
| de-o-cap-seed10.wav | O, wie Oma. | 2.00 | o | o v i o m a | O wie Oma | clean |
| de-o-cap-seed11.wav | O, wie Oma. | 1.84 | o | o v i o m a | Ovioma | clean |
| de-o-cap-seed12.wav | O, wie Oma. | 1.52 | oː | oː v iː uː m a | O wie Oma | clean |
| de-o-cap-seed13.wav | O, wie Oma. | 1.60 | o | o v i o m a | Ovioma | clean |
| de-o-cap-seed14.wav | O, wie Oma. | 1.44 | o | o v i u m a | Oh, wie Oma! | clean |
| de-i-cap-seed0.wav | I, wie Insel. | 1.68 | iː | iː v iː ɪ n z ə | I wie Insel | clean |
| de-i-cap-seed1.wav | I, wie Insel. | 1.28 | iː | iː v iː ʔ ɪ n z ə | I wie Insel | clean |
| de-i-cap-seed2.wav | I, wie Insel. | 1.92 | iː | iː v iː ɪ n z ə | I wie Insel | clean |
| de-i-cap-seed3.wav | I, wie Insel. | 1.44 |  | v iː ʔ i n z | I wie Insel | no sound |
| de-i-cap-seed4.wav | I, wie Insel. | 1.76 | iː | iː v iː ɪ n z ə l | I wie Insel | clean |
| de-i-cap-seed5.wav | I, wie Insel. | 1.60 | iː | iː v iː ɪ n z ə | E wie Insel | clean |
| de-i-cap-seed6.wav | I, wie Insel. | 1.92 | iː | iː v iː ɪ n z ə | I wie Insel | clean |
| de-i-cap-seed7.wav | I, wie Insel. | 1.84 | iː | iː v iː ɪ n z ə | I wie Insel | clean |
| de-i-cap-seed8.wav | I, wie Insel. | 1.92 | iː | iː v iː ɪ n z ə | I wie Insel | clean |
| de-i-cap-seed9.wav | I, wie Insel. | 2.08 | i v ɪ n z e l | i v ɪ n z e l | I wie Insel | no carrier |
| de-i-cap-seed10.wav | I, wie Insel. | 2.08 | iː | iː v iː ɪ n z ə | E wie Insel | clean |
| de-i-cap-seed11.wav | I, wie Insel. | 2.88 | iː ð ɪ ɪ n z ə | iː ð ɪ ɪ n z ə | I wie Insel | no carrier |
| de-i-cap-seed12.wav | I, wie Insel. | 1.68 | iː | iː v iː ɪ n z ə | I wie Insel | clean |
| de-i-cap-seed13.wav | I, wie Insel. | 1.44 | iː | iː v iː ʔ ɪ n z ə | I wie Insel | clean |
| de-i-cap-seed14.wav | I, wie Insel. | 1.92 | iː | iː v iː ɪ n z ə | I wie Insel | clean |
| de-e-cap-seed0.wav | E, wie Esel. | 2.00 | e | e v i e z ə | E wie Esel | clean |
| de-e-cap-seed1.wav | E, wie Esel. | 1.28 | e | e v i e z ɐ | E wie Esel | clean |
| de-e-cap-seed2.wav | E, wie Esel. | 1.76 | e | e v iː e z ə | E wie Esel | clean |
| de-e-cap-seed3.wav | E, wie Esel. | 1.52 | iː | iː v iː eː z ə | E. Wie Esel | wrong sound |
| de-e-cap-seed4.wav | E, wie Esel. | 1.84 | e | e v i e z a | E wie Esel | clean |
| de-e-cap-seed5.wav | E, wie Esel. | 1.52 | eɪ | eɪ v i eɪ z ə | E wie Esel | wrong sound |
| de-e-cap-seed6.wav | E, wie Esel. | 1.84 | i | i v i eː z ə l | E wie Esel | wrong sound |
| de-e-cap-seed7.wav | E, wie Esel. | 2.24 | e | e v i e z ə | E wie Ese | clean, word missed |
| de-e-cap-seed8.wav | E, wie Esel. | 1.76 | eɪ | eɪ v i eɪ z ə | E wie Esel | wrong sound |
| de-e-cap-seed9.wav | E, wie Esel. | 1.36 | i | i v i e z ə | E wie Esel | wrong sound |
| de-e-cap-seed10.wav | E, wie Esel. | 1.92 | eɪ | eɪ v iː eɪ z ə | E wie Esel | wrong sound |
| de-e-cap-seed11.wav | E, wie Esel. | 3.12 | e | e v i e z ə | E wie Esel | clean |
| de-e-cap-seed12.wav | E, wie Esel. | 1.60 | e | e v i ʔ eɪ z ə | E wie Esel | clean |
| de-e-cap-seed13.wav | E, wie Esel. | 2.00 | e | e v i ʔ e z ə | E wie Esel | clean |
| de-e-cap-seed14.wav | E, wie Esel. | 1.28 | e | e v iː e z a | E wie Ese | clean, word missed |
| de-s-cap-seed0.wav | S, wie Sofa. | 1.92 | z e s | z e s v i z o f a | SES wie Sofa | vowel added |
| de-s-cap-seed1.wav | S, wie Sofa. | 1.44 | s | s v iː z oː f a | ist wie Sofa. | clean |
| de-s-cap-seed2.wav | S, wie Sofa. | 2.16 | z e s | z e s v i z o f a | SES wie Sofa | vowel added |
| de-s-cap-seed3.wav | S, wie Sofa. | 1.36 | s | s v i z ɔ f ɐ | S wie Sofa | clean |
| de-s-cap-seed4.wav | S, wie Sofa. | 1.76 | z e s | z e s v i z o f a | Sess wie Sofa | vowel added |
| de-s-cap-seed5.wav | S, wie Sofa. | 1.52 | z e s | z e s v i z o f a | SES wie Sofa | vowel added |
| de-s-cap-seed6.wav | S, wie Sofa. | 2.00 | z e s | z e s v iː z oʊ f a | SES wie Sofa | vowel added |
| de-s-cap-seed7.wav | S, wie Sofa. | 2.24 | z e s | z e s v i z uː f ɐ | Sess wie Sofa | vowel added |
| de-s-cap-seed8.wav | S, wie Sofa. | 1.44 | z | z v iː z o f a | S wie Sofa. | clean |
| de-s-cap-seed9.wav | S, wie Sofa. | 1.52 | z e s | z e s v iː z oː f a | SES wie Sofa | vowel added |
| de-s-cap-seed10.wav | S, wie Sofa. | 1.68 | z e s | z e s v i z oː f a | SES wie Sofa | vowel added |
| de-s-cap-seed11.wav | S, wie Sofa. | 3.04 | z i s | z i s v i z o f a | SIS wie Sofa | vowel added |
| de-s-cap-seed12.wav | S, wie Sofa. | 1.92 | z e s | z e s v i z o f a | Sess wie Sofa | vowel added |
| de-s-cap-seed13.wav | S, wie Sofa. | 1.60 | z i | z i v iː z o f a | Seh wie Sofa | vowel added |
| de-s-cap-seed14.wav | S, wie Sofa. | 1.92 | s | s v iː z oː v a | S wie Sofa | clean |
| de-s-triple-seed0.wav | Sss, wie Sofa. | 2.40 | z | z v iː z o f a | Ssss, wie Sofa. | clean |
| de-s-triple-seed1.wav | Sss, wie Sofa. | 1.92 | z | z v iː z oː f a | wie Sofa. | clean |
| de-s-triple-seed2.wav | Sss, wie Sofa. | 1.52 | s | s v iː z oː f a | S wie Sofa | clean |
| de-s-triple-seed3.wav | Sss, wie Sofa. | 1.92 | s | s v iː z oː f a | S wie Sofa. | clean |
| de-s-triple-seed4.wav | Sss, wie Sofa. | 1.76 |  | v iː z oː f a | Ssss, wie Sofa. | no sound |
| de-s-triple-seed5.wav | Sss, wie Sofa. | 2.00 | s | s v iː z oː f a | S wie Sofa. | clean |
| de-s-triple-seed6.wav | Sss, wie Sofa. | 2.08 | z | z v i z o f a | Zzzz, wie Sofa. | clean |
| de-s-triple-seed7.wav | Sss, wie Sofa. | 2.24 | s | s v i z o f a | S wie Sofa. | clean |
| de-s-triple-seed8.wav | Sss, wie Sofa. | 1.76 | s | s v iː z oː f a | S wie Sofa | clean |
| de-s-triple-seed9.wav | Sss, wie Sofa. | 1.60 | z i s | z i s v i z oː f ɐ | wie Sofa. | vowel added |
| de-s-triple-seed10.wav | Sss, wie Sofa. | 2.40 | s | s v iː z oː f a | S wie Sofa | clean |
| de-s-triple-seed11.wav | Sss, wie Sofa. | 3.12 | s | s v i z o f a | S wie Sofa | clean |
| de-s-triple-seed12.wav | Sss, wie Sofa. | 1.76 |  | v iː z o f a | S wie Sofa | no sound |
| de-s-triple-seed13.wav | Sss, wie Sofa. | 1.68 | s | s v i z o f a | S wie Sofa | clean |
| de-s-triple-seed14.wav | Sss, wie Sofa. | 1.60 | z | z v i z o f ɐ | S wie Sofa | clean |
| de-u-cap-seed0.wav | U, wie Ufo. | 1.68 | u | u v i uː f o | OVUFO | clean |
| de-u-cap-seed1.wav | U, wie Ufo. | 1.36 | uː | uː v i uː f o | UV-UFO | clean |
| de-u-cap-seed2.wav | U, wie Ufo. | 1.60 | uː | uː v iː uː f o | O, wie UFO. | clean |
| de-u-cap-seed3.wav | U, wie Ufo. | 1.44 | uː | uː v i uː f o | U wie UFO | clean |
| de-u-cap-seed4.wav | U, wie Ufo. | 2.32 | u | u v i u f o | U wie UFO | clean |
| de-u-cap-seed5.wav | U, wie Ufo. | 1.52 | uː | uː ð i uː f o | U, the UFO. | clean |
| de-u-cap-seed6.wav | U, wie Ufo. | 2.16 | u | u v i u f o | U wie UFO | clean |
| de-u-cap-seed7.wav | U, wie Ufo. | 1.68 | u5 | u5 f iː uː f ɔ | U wie UFO | wrong sound |
| de-u-cap-seed8.wav | U, wie Ufo. | 1.76 | uː | uː v i u f o | U wie UFO | clean |
| de-u-cap-seed9.wav | U, wie Ufo. | 1.92 | uː | uː v iː uː f o | U wie UFO | clean |
| de-u-cap-seed10.wav | U, wie Ufo. | 1.76 | uː | uː v i u f o | OVUFO | clean |
| de-u-cap-seed11.wav | U, wie Ufo. | 2.32 | u | u v i uː f o | U wie UFO | clean |
| de-u-cap-seed12.wav | U, wie Ufo. | 1.68 | uː | uː v i uː f o | U wie UFO | clean |
| de-u-cap-seed13.wav | U, wie Ufo. | 1.20 | uː | uː v i ʔ u ŋ f o | Uwe Ufo | clean |
| de-u-cap-seed14.wav | U, wie Ufo. | 1.36 | uː | uː v i ʔ uː f o | U wie UFO | clean |
| de-r-cap-seed0.wav | R, wie Rose. | 1.84 | ɛ | ɛ v iː r ɔ z | R wie Rose | wrong sound |
| de-r-cap-seed1.wav | R, wie Rose. | 2.00 | ɛ | ɛ v iː r oː s | R. V. Rose | wrong sound |
| de-r-cap-seed2.wav | R, wie Rose. | 2.24 | ɛ r | ɛ r v iː r ɹ oʊ z | R.V. Rose | vowel added |
| de-r-cap-seed3.wav | R, wie Rose. | 1.36 | ɛ | ɛ v iː ɹ o z | R wie Rose | wrong sound |
| de-r-cap-seed4.wav | R, wie Rose. | 1.60 | r e | r e v i r o z e | Reh wie Rose | vowel added |
| de-r-cap-seed5.wav | R, wie Rose. | 1.76 | ɛ | ɛ v iː ɹ oʊ z | R.V. Rose | wrong sound |
| de-r-cap-seed6.wav | R, wie Rose. | 1.68 | ɛ | ɛ v iː ɹ oʊ z | R wie Rose | wrong sound |
| de-r-cap-seed7.wav | R, wie Rose. | 1.60 | ɛ | ɛ v iː ɹ oʊ z | L.V. Rose | wrong sound |
| de-r-cap-seed8.wav | R, wie Rose. | 1.76 | ɛ | ɛ v iː ɹ oʊ z | R wie Rose | wrong sound |
| de-r-cap-seed9.wav | R, wie Rose. | 1.28 | ɛ | ɛ v iː ɹ oʊ z | Air V Rose | wrong sound |
| de-r-cap-seed10.wav | R, wie Rose. | 1.84 | a | a v i ɹ o s | RV-ROS | wrong sound, word missed |
| de-r-cap-seed11.wav | R, wie Rose. | 2.88 | ɛ | ɛ v iː ɹ oʊ z | R.V. Rose | wrong sound |
| de-r-cap-seed12.wav | R, wie Rose. | 2.00 | ɛ | ɛ v iː ɹ oʊ z | R wie Rose | wrong sound |
| de-r-cap-seed13.wav | R, wie Rose. | 1.60 | ɛ | ɛ v iː ɹ oʊ z | R wie Rose | wrong sound |
| de-r-cap-seed14.wav | R, wie Rose. | 1.28 | ɛ | ɛ v iː ɹ oʊ z | L.V. Rose | wrong sound |
| de-r-triple-seed0.wav | Rrr, wie Rose. | 2.16 | h ə | h ə v iː ɹ oʊ z | R wie Rose | wrong sound |
| de-r-triple-seed1.wav | Rrr, wie Rose. | 1.84 |  | v iː ɹ oʊ z | R wie Rose | no sound |
| de-r-triple-seed2.wav | Rrr, wie Rose. | 2.24 | ɹ a | ɹ a v iː ɹ oʊ z | R. V. Rose | vowel added |
| de-r-triple-seed3.wav | Rrr, wie Rose. | 1.44 | r ə | r ə v iː r oː z | Rrrr, wie Rose. | vowel added |
| de-r-triple-seed4.wav | Rrr, wie Rose. | 2.24 | r | r v iː ɹ oʊ z | Rhe, V-Rose | clean |
| de-r-triple-seed5.wav | Rrr, wie Rose. | 1.52 | x ə | x ə v iː r oː z ə | Grrr, wie Rose. | wrong sound |
| de-r-triple-seed6.wav | Rrr, wie Rose. | 2.32 | r ə | r ə v iː ɹ oʊ z | R. V. Rose | vowel added |
| de-r-triple-seed7.wav | Rrr, wie Rose. | 2.00 |  | v iː w oː z | Wiros. | no sound, word missed |
| de-r-triple-seed8.wav | Rrr, wie Rose. | 1.68 | r | r v iː ɹ oʊ z | R wie Rose | clean |
| de-r-triple-seed9.wav | Rrr, wie Rose. | 1.20 | r ə | r ə v i r o z | Rheviruse | vowel added, word missed |
| de-r-triple-seed10.wav | Rrr, wie Rose. | 1.76 |  | v iː r oː z ə | Wie Rose. | no sound |
| de-r-triple-seed11.wav | Rrr, wie Rose. | 2.40 | ə | ə v iː ɹ oʊ z | Wie Rose | wrong sound |
| de-r-triple-seed12.wav | Rrr, wie Rose. | 1.28 | r ə | r ə v iː r oʊ z | R wie Rose | vowel added |
| de-r-triple-seed13.wav | Rrr, wie Rose. | 1.92 | r ʌ | r ʌ v iː ɹ oʊ z | Rhe wie Rose | vowel added |
| de-r-triple-seed14.wav | Rrr, wie Rose. | 1.68 | r ə | r ə v iː ɹ oʊ z | R wie Rose | vowel added |
| de-t-cap-seed0.wav | T, wie Tiger. | 1.44 | t iː | t iː v iː t iː ɡ ə | Die V-Tiger | vowel added |
| de-t-cap-seed1.wav | T, wie Tiger. | 1.68 | d i | d i v i t i ɡ a | Die V-Tiger | wrong sound |
| de-t-cap-seed2.wav | T, wie Tiger. | 1.52 | d iː | d iː v iː t i ɡ ə | Die W-Tiger | wrong sound |
| de-t-cap-seed3.wav | T, wie Tiger. | 2.24 | t i | t i v i t i ɡ a | T wie Tiger | vowel added |
| de-t-cap-seed4.wav | T, wie Tiger. | 1.60 | d iː | d iː v iː t i ɡ a | TV-Tiger | wrong sound |
| de-t-cap-seed5.wav | T, wie Tiger. | 1.20 | d iː | d iː v iː t iː ɡ a | D wie Tiger | wrong sound |
| de-t-cap-seed6.wav | T, wie Tiger. | 1.84 | d iː | d iː v iː t i ɡ ə | D wie Tiger | wrong sound |
| de-t-cap-seed7.wav | T, wie Tiger. | 1.76 | d iː | d iː v iː t iː ɡ a | D wie Tiger | wrong sound |
| de-t-cap-seed8.wav | T, wie Tiger. | 1.44 | d iː | d iː v iː t iː ɡ ə | Die V-Tiger | wrong sound |
| de-t-cap-seed9.wav | T, wie Tiger. | 1.20 | d iː | d iː v i t i ɡ ə | D wie Tiger | wrong sound |
| de-t-cap-seed10.wav | T, wie Tiger. | 2.32 | d iː | d iː v iː t i ɡ ə | Die VTiger | wrong sound |
| de-t-cap-seed11.wav | T, wie Tiger. | 2.96 | d i | d i v i t i ɡ a | die wie Tiger | wrong sound |
| de-t-cap-seed12.wav | T, wie Tiger. | 1.36 | d | d v iː t iː ɡ ə | Die V-Tiger | wrong sound |
| de-t-cap-seed13.wav | T, wie Tiger. | 1.68 | d iː | d iː v iː th iː ɡ ə | Die V-Tiger | wrong sound |
| de-t-cap-seed14.wav | T, wie Tiger. | 1.36 | d iː | d iː v iː t iː ɡ a | Die V-Tiger | wrong sound |
| de-t-bang-seed0.wav | T! wie Tiger. | 1.60 | t iː | t iː v iː t iː ɡ a | T wie Tiger | vowel added |
| de-t-bang-seed1.wav | T! wie Tiger. | 1.60 | t iː | t iː v iː t i ɡ ə | T wie Tiger | vowel added |
| de-t-bang-seed2.wav | T! wie Tiger. | 1.92 | t iː | t iː v iː t iː ɡ ə | T wie Tiger | vowel added |
| de-t-bang-seed3.wav | T! wie Tiger. | 1.68 | t iː | t iː v iː t iː ɡ ə | T wie Tiger | vowel added |
| de-t-bang-seed4.wav | T! wie Tiger. | 1.28 | t iː | t iː ð iː t iː ɡ ə | T wie Tiger | vowel added |
| de-t-bang-seed5.wav | T! wie Tiger. | 1.44 | t iː | t iː v iː t iː ɡ ə | T wie Tiger | vowel added |
| de-t-bang-seed6.wav | T! wie Tiger. | 2.80 | t iː | t iː v i t i ɡ a | T wie Tiger | vowel added |
| de-t-bang-seed7.wav | T! wie Tiger. | 1.68 | t iː | t iː v iː t iː ɡ a | T wie Tiger | vowel added |
| de-t-bang-seed8.wav | T! wie Tiger. | 2.00 | t iː | t iː v iː t iː ɡ a | T wie Tiger | vowel added |
| de-t-bang-seed9.wav | T! wie Tiger. | 1.44 | t i | t i v iː t iː ɡ ə | T wie Tiger | vowel added |
| de-t-bang-seed10.wav | T! wie Tiger. | 3.20 | t iː | t iː v i t i ɡ a | T wie Tiger | vowel added |
| de-t-bang-seed11.wav | T! wie Tiger. | 2.72 | t iː | t iː v iː t i ɡ a | T wie Tiger | vowel added |
| de-t-bang-seed12.wav | T! wie Tiger. | 1.60 | t iː | t iː v iː t iː ɡ ə | T wie Tiger | vowel added |
| de-t-bang-seed13.wav | T! wie Tiger. | 1.68 | t i5 | t i5 v i5 th i5 ɡ ə1 | T wie Tiger | vowel added |
| de-t-bang-seed14.wav | T! wie Tiger. | 1.60 | t iː | t iː v iː t iː ɡ ə | T wie Tiger | vowel added |
| de-n-cap-seed0.wav | N, wie Nase. | 1.60 | e n | e n v i n a z e | N wie Nase | vowel added |
| de-n-cap-seed1.wav | N, wie Nase. | 1.36 | e n | e n v iː n a z ə | N wie Nase | vowel added |
| de-n-cap-seed2.wav | N, wie Nase. | 1.52 | e n | e n v i n a z | N wie Nase | vowel added |
| de-n-cap-seed3.wav | N, wie Nase. | 1.60 | e n | e n v iː n a z e | N wie Nase | vowel added |
| de-n-cap-seed4.wav | N, wie Nase. | 1.68 | e n | e n v iː n aː z | N wie Nase | vowel added |
| de-n-cap-seed5.wav | N, wie Nase. | 1.84 | e n | e n v i n a z | N wie Nase | vowel added |
| de-n-cap-seed6.wav | N, wie Nase. | 1.60 | e n | e n v i n a z ə | N wie Nase | vowel added |
| de-n-cap-seed7.wav | N, wie Nase. | 1.68 | e n | e n v i n aː z | N wie Nase | vowel added |
| de-n-cap-seed8.wav | N, wie Nase. | 2.08 | e n | e n v iː n aː z e | N wie Nase | vowel added |
| de-n-cap-seed9.wav | N, wie Nase. | 1.68 | e n | e n v iː n a z ə | N wie Nase | vowel added |
| de-n-cap-seed10.wav | N, wie Nase. | 2.24 | m e n | m e n v iː n aː z | N wie Nase | vowel added |
| de-n-cap-seed11.wav | N, wie Nase. | 1.68 | e n | e n v iː n a z | N wie Nase | vowel added |
| de-n-cap-seed12.wav | N, wie Nase. | 1.76 | e n | e n v i n a z ə | N wie Nase | vowel added |
| de-n-cap-seed13.wav | N, wie Nase. | 1.84 | e n | e n v i n aː z | N wie Nase | vowel added |
| de-n-cap-seed14.wav | N, wie Nase. | 1.60 | e n | e n v i n a z | N wie Nase | vowel added |
| de-n-triple-seed0.wav | Nnn, wie Nase. | 1.92 |  | v iː n aː z | Wie Nase. | no sound |
| de-n-triple-seed1.wav | Nnn, wie Nase. | 2.32 |  | v iː n aː z | n wie Nase | no sound |
| de-n-triple-seed2.wav | Nnn, wie Nase. | 2.32 | n | n v iː n aː z ə | N, wie Nase. | clean |
| de-n-triple-seed3.wav | Nnn, wie Nase. | 1.92 |  | v iː n aː z e | Wie Nase. | no sound |
| de-n-triple-seed4.wav | Nnn, wie Nase. | 1.36 |  | v iː n aː z | Wie Nase. | no sound |
| de-n-triple-seed5.wav | Nnn, wie Nase. | 2.24 |  | f iː n aː z | N, wie Nase. | no sound |
| de-n-triple-seed6.wav | Nnn, wie Nase. | 1.60 |  | v i n aː z | Wie Nase. | no sound |
| de-n-triple-seed7.wav | Nnn, wie Nase. | 2.16 |  | v i n aː z | Mh? Wie Nase. | no sound |
| de-n-triple-seed8.wav | Nnn, wie Nase. | 2.00 |  | v iː n aː z | N, wie Nase. | no sound |
| de-n-triple-seed9.wav | Nnn, wie Nase. | 1.60 |  | v iː n aː z | Wie Nase. | no sound |
| de-n-triple-seed10.wav | Nnn, wie Nase. | 2.40 |  | v iː n a z | N, wie Nase. | no sound |
| de-n-triple-seed11.wav | Nnn, wie Nase. | 2.56 | n | n v iː n aː z ə | N wie Nase | clean |
| de-n-triple-seed12.wav | Nnn, wie Nase. | 1.84 |  | v iː n aː z | N wie Nase | no sound |
| de-n-triple-seed13.wav | Nnn, wie Nase. | 2.32 | ə | ə v iː n aː z | N wie Nase. | wrong sound |
| de-n-triple-seed14.wav | Nnn, wie Nase. | 1.36 |  | v iː n aː z | N wie Nase | no sound |
| de-f-cap-seed0.wav | F, wie Fisch. | 2.48 | ɛ s | ɛ s v iː f ɪ ʃ | F wie Fisch | wrong sound |
| de-f-cap-seed1.wav | F, wie Fisch. | 1.68 | ɛ f | ɛ f v iː f ɪ ʃ | F wie Fisch | vowel added |
| de-f-cap-seed2.wav | F, wie Fisch. | 1.68 | ð ɛ θ | ð ɛ θ ð iː f ɪ ʃ | Weth wie Fisch | wrong sound |
| de-f-cap-seed3.wav | F, wie Fisch. | 1.52 | ɛ s | ɛ s v iː f ɪ ʃ | F wie Fisch | wrong sound |
| de-f-cap-seed4.wav | F, wie Fisch. | 1.84 | ɛ θ | ɛ θ v iː f ɪ ʃ | F wie Fisch | wrong sound |
| de-f-cap-seed5.wav | F, wie Fisch. | 1.44 | ɛ s | ɛ s v iː θ ɪ ʃ | F wie Fisch | wrong sound |
| de-f-cap-seed6.wav | F, wie Fisch. | 1.92 | ɛ s | ɛ s v iː f ɪ ʃ | F wie Fisch | wrong sound |
| de-f-cap-seed7.wav | F, wie Fisch. | 1.92 | v ɪ s | v ɪ s v iː f ɪ ʃ | Wiff wie Fisch | wrong sound |
| de-f-cap-seed8.wav | F, wie Fisch. | 1.36 | v ɛ | v ɛ v iː f ɪ ʃ | Wie Fisch | wrong sound |
| de-f-cap-seed9.wav | F, wie Fisch. | 1.60 | e s | e s v iː f i ʃ | F wie Fisch | wrong sound |
| de-f-cap-seed10.wav | F, wie Fisch. | 2.24 | v ɛ s | v ɛ s v iː s ɪ ʃ | Wiff wie Fisch | wrong sound |
| de-f-cap-seed11.wav | F, wie Fisch. | 2.24 |  | v iː f ɪ ʃ | Wie Fisch | no sound |
| de-f-cap-seed12.wav | F, wie Fisch. | 1.44 | ɛ s | ɛ s v iː θ ɪ ʃ | F wie Fisch | wrong sound |
| de-f-cap-seed13.wav | F, wie Fisch. | 1.60 | ɛ θ | ɛ θ v iː θ ɪ ʃ | F wie Fisch | wrong sound |
| de-f-cap-seed14.wav | F, wie Fisch. | 1.44 | ɛ f | ɛ f v iː f ɪ ʃ | F wie Fisch | vowel added |
| de-f-triple-seed0.wav | Fff, wie Fisch. | 1.84 |  | v iː f ɪ ʃ | f wie Fisch | no sound |
| de-f-triple-seed1.wav | Fff, wie Fisch. | 1.52 |  | ð iː f ɪ ʃ | Wie Fisch. | no sound |
| de-f-triple-seed2.wav | Fff, wie Fisch. | 1.84 |  | v iː f ɪ ʃ | wie Fisch. | no sound |
| de-f-triple-seed3.wav | Fff, wie Fisch. | 1.68 |  | v iː f ɪ ʃ | Wie Fisch | no sound |
| de-f-triple-seed4.wav | Fff, wie Fisch. | 1.44 |  | v iː f ɪ ʃ | Wie Fisch | no sound |
| de-f-triple-seed5.wav | Fff, wie Fisch. | 1.52 |  | v iː f ɪ ʃ | F wie Fisch. | no sound |
| de-f-triple-seed6.wav | Fff, wie Fisch. | 1.60 |  | ð iː f ɪ ʃ | Wie Fisch. | no sound |
| de-f-triple-seed7.wav | Fff, wie Fisch. | 1.68 |  | ð iː f ɪ ʃ | Wie Fisch | no sound |
| de-f-triple-seed8.wav | Fff, wie Fisch. | 1.76 | s | s v iː f ɪ ʃ | wie fisch | wrong sound |
| de-f-triple-seed9.wav | Fff, wie Fisch. | 1.36 |  | v iː f ɪ ʃ | Wie Fisch | no sound |
| de-f-triple-seed10.wav | Fff, wie Fisch. | 1.60 |  | v iː f ɪ ʃ | Wie Fisch | no sound |
| de-f-triple-seed11.wav | Fff, wie Fisch. | 2.16 |  | v iː f ɪ ʃ | f wie Fisch | no sound |
| de-f-triple-seed12.wav | Fff, wie Fisch. | 1.76 |  | ð iː θ ɪ ʃ | Wie Fisch. | no sound |
| de-f-triple-seed13.wav | Fff, wie Fisch. | 1.60 |  | v iː f ɪ ʃ | Wie Fisch | no sound |
| de-f-triple-seed14.wav | Fff, wie Fisch. | 1.52 |  | v iː f ɪ ʃ | F wie Fisch | no sound |
| de-h-cap-seed0.wav | H, wie Hund. | 1.60 | h iː | h iː v iː h u n | He wie Hund | vowel added |
| de-h-cap-seed1.wav | H, wie Hund. | 1.12 | h iː | h iː v iː h u n | Wie Hund. | vowel added |
| de-h-cap-seed2.wav | H, wie Hund. | 1.68 | h i | h i v i h o n | He wie Hund | vowel added |
| de-h-cap-seed3.wav | H, wie Hund. | 1.68 | h iː | h iː v iː h u n t | hie wie Hund | vowel added |
| de-h-cap-seed4.wav | H, wie Hund. | 1.36 | h iː | h iː v iː h ʊ n t | Wie Hund | vowel added |
| de-h-cap-seed5.wav | H, wie Hund. | 1.76 | h iː | h iː v iː h ʊ n t | Hi wie Hund | vowel added |
| de-h-cap-seed6.wav | H, wie Hund. | 1.60 | h i | h i v iː h u n | He wie Hund | vowel added |
| de-h-cap-seed7.wav | H, wie Hund. | 1.60 | h i | h i v iː h u n | He wie Hund | vowel added |
| de-h-cap-seed8.wav | H, wie Hund. | 2.24 | h iː | h iː v iː h ʊ n t | Wie Hund | vowel added |
| de-h-cap-seed9.wav | H, wie Hund. | 1.28 | h iː | h iː v iː h u n t | Hi wie Hund | vowel added |
| de-h-cap-seed10.wav | H, wie Hund. | 2.16 | h iː | h iː v iː h u n | Hie wie Hund | vowel added |
| de-h-cap-seed11.wav | H, wie Hund. | 1.76 | h iː | h iː v iː h u n t | hie wie hund | vowel added |
| de-h-cap-seed12.wav | H, wie Hund. | 1.68 | h iː | h iː v iː h u n t | Hi, wie Hund! | vowel added |
| de-h-cap-seed13.wav | H, wie Hund. | 1.76 | h iː | h iː v iː h u n t | Hi, wie Hund | vowel added |
| de-h-cap-seed14.wav | H, wie Hund. | 1.44 | h iː | h iː v iː h ʊ n t | He wie Hund | vowel added |
| de-h-double-seed0.wav | Hh, wie Hund. | 2.08 | h a | h a v iː h u n t | Hä, wie Hund | vowel added |
| de-h-double-seed1.wav | Hh, wie Hund. | 2.08 | h iː | h iː v i h u n | He wie Hund | vowel added |
| de-h-double-seed2.wav | Hh, wie Hund. | 1.84 | h ɛ | h ɛ v iː h ʊ n t | Hä, wie Hund. | vowel added |
| de-h-double-seed3.wav | Hh, wie Hund. | 1.60 | h ɛ | h ɛ v iː h u n t | Hä, wie Hund. | vowel added |
| de-h-double-seed4.wav | Hh, wie Hund. | 1.60 | h a | h a v iː h u n t | Ha! Wie Hund! | vowel added |
| de-h-double-seed5.wav | Hh, wie Hund. | 1.28 | h a | h a v iː h u n | Ha! Wie Hund. | vowel added |
| de-h-double-seed6.wav | Hh, wie Hund. | 2.32 | h eː | h eː v iː h o n t | he wie hund | vowel added |
| de-h-double-seed7.wav | Hh, wie Hund. | 1.44 | h a | h a v iː h u n | Ha, wie Hund! | vowel added |
| de-h-double-seed8.wav | Hh, wie Hund. | 1.76 | h e | h e v iː h u n t | Hä, wie Hund. | vowel added |
| de-h-double-seed9.wav | Hh, wie Hund. | 1.36 | h a | h a v iː h u n d | H wie Hund | vowel added |
| de-h-double-seed10.wav | Hh, wie Hund. | 2.08 | h a | h a v i h u n | H wie Hund | vowel added |
| de-h-double-seed11.wav | Hh, wie Hund. | 1.68 | h eɪ | h eɪ v iː h ʊ n t | Hey, wie Hund. | vowel added |
| de-h-double-seed12.wav | Hh, wie Hund. | 1.20 | h a | h a v iː h u n | Ha, wie Hund! | vowel added |
| de-h-double-seed13.wav | Hh, wie Hund. | 1.76 | h iː | h iː v iː h ʊ n d | Hey, wie Hund! | vowel added |
| de-h-double-seed14.wav | Hh, wie Hund. | 1.28 | h eː | h eː v iː h ʊ n t | Hey, wie Hund! | vowel added |
| de-b-cap-seed0.wav | B, wie Baum. | 1.36 | b eː | b eː v iː b aʊ | B wie Baum | vowel added |
| de-b-cap-seed1.wav | B, wie Baum. | 1.12 | b eː | b eː v i b aʊ m | B wie Baum | vowel added |
| de-b-cap-seed2.wav | B, wie Baum. | 1.52 | b eː | b eː v iː b aʊ n | B wie Baum | vowel added |
| de-b-cap-seed3.wav | B, wie Baum. | 1.28 |  | ð i v iː b aʊ m | B wie Baum | no sound |
| de-b-cap-seed4.wav | B, wie Baum. | 1.52 | b e | b e v i b a m | b. Wie Baum | vowel added |
| de-b-cap-seed5.wav | B, wie Baum. | 1.68 | b eː | b eː v iː b aʊ m | B wie Baum | vowel added |
| de-b-cap-seed6.wav | B, wie Baum. | 1.76 | b ɨ | b ɨ v iː b aʊ m | B wie Baum | wrong sound |
| de-b-cap-seed7.wav | B, wie Baum. | 2.00 |  | v i v iː b aʊ | B wie Baum | no sound |
| de-b-cap-seed8.wav | B, wie Baum. | 1.76 | b eː | b eː v iː b aʊ m | b. Wie Baum | vowel added |
| de-b-cap-seed9.wav | B, wie Baum. | 2.24 | b eː | b eː ð iː b aʊ m | B wie Baum | vowel added |
| de-b-cap-seed10.wav | B, wie Baum. | 1.60 |  | b i v i b aʊ m | B wie Baum | no sound |
| de-b-cap-seed11.wav | B, wie Baum. | 1.60 | b eɪ | b eɪ v iː b aʊ m | B wie Baum | vowel added |
| de-b-cap-seed12.wav | B, wie Baum. | 1.52 | d eɪ | d eɪ v iː b aʊ m | D wie Baum | wrong sound |
| de-b-cap-seed13.wav | B, wie Baum. | 1.52 | b eː | b eː v iː b aʊ m | B wie Baum | vowel added |
| de-b-cap-seed14.wav | B, wie Baum. | 1.44 |  | b i v i b aʊ m | B wie Baum | no sound |
| de-b-bang-seed0.wav | B! wie Baum. | 1.84 | b eː | b eː v iː b aʊ m | b. Wie Baum | vowel added |
| de-b-bang-seed1.wav | B! wie Baum. | 1.52 | b eː | b eː v iː b aʊ m | b. Wie Baum | vowel added |
| de-b-bang-seed2.wav | B! wie Baum. | 1.68 | d eɪ | d eɪ v i b aʊ | b. Wiebaum | wrong sound |
| de-b-bang-seed3.wav | B! wie Baum. | 1.44 | b ɨ v ɨ b aʊ m | b ɨ v ɨ b aʊ m | b. Wie Baum | no carrier |
| de-b-bang-seed4.wav | B! wie Baum. | 2.00 |  | b i v iː b aʊ m | b. Wie Baum | no sound |
| de-b-bang-seed5.wav | B! wie Baum. | 1.20 | b eː v ɨ b aʊ | b eː v ɨ b aʊ | B wie Baum | no carrier |
| de-b-bang-seed6.wav | B! wie Baum. | 2.00 | b ɨ | b ɨ v iː b aʊ | b. Wie Baum | wrong sound |
| de-b-bang-seed7.wav | B! wie Baum. | 3.12 |  | b iː v i b aʊ m | wie Baum | no sound |
| de-b-bang-seed8.wav | B! wie Baum. | 1.92 |  | b iː v iː b aʊ m | B. Wie Baum | no sound |
| de-b-bang-seed9.wav | B! wie Baum. | 1.52 | b eɪ ð ɨ b aʊ m | b eɪ ð ɨ b aʊ m | B wie Baum | no carrier |
| de-b-bang-seed10.wav | B! wie Baum. | 1.76 | b əɪ | b əɪ ð iː b aʊ | B wie Baum | vowel added |
| de-b-bang-seed11.wav | B! wie Baum. | 2.32 | b eː | b eː v iː b aʊ | b wie Baum | vowel added |
| de-b-bang-seed12.wav | B! wie Baum. | 1.52 | b ɨ | b ɨ v iː b aʊ m | b. Wie Baum | wrong sound |
| de-b-bang-seed13.wav | B! wie Baum. | 1.68 |  | b i v iː b aʊ | B wie Baum | no sound |
| de-b-bang-seed14.wav | B! wie Baum. | 1.20 | b ɨ | b ɨ v iː b aʊ m | b. Wie Baum | wrong sound |
| de-k-cap-seed0.wav | K, wie Kuh. | 1.84 | k kh ei5 | k kh ei5 v iː k u5 | K wie Kuh | vowel added |
| de-k-cap-seed1.wav | K, wie Kuh. | 1.36 | k aɪ | k aɪ v i k u | KVQ | vowel added, word missed |
| de-k-cap-seed2.wav | K, wie Kuh. | 1.68 | k eɪ | k eɪ v iː k uː | K.V.Q. | vowel added, word missed |
| de-k-cap-seed3.wav | K, wie Kuh. | 1.44 | k aɪ | k aɪ v i k uː | KVQ | vowel added, word missed |
| de-k-cap-seed4.wav | K, wie Kuh. | 1.36 | k aɪ | k aɪ v iː k uː | K, wie cool! | vowel added, word missed |
| de-k-cap-seed5.wav | K, wie Kuh. | 2.32 | k eɪ | k eɪ v i k uː | K wie Kuh | vowel added |
| de-k-cap-seed6.wav | K, wie Kuh. | 1.76 | k eɪ | k eɪ v i k uː | K. Wiku | vowel added, word missed |
| de-k-cap-seed7.wav | K, wie Kuh. | 2.00 | k a | k a v i k uː | KWQ | vowel added, word missed |
| de-k-cap-seed8.wav | K, wie Kuh. | 1.84 | k eɪ | k eɪ v i k uː | K. Wie cool. | vowel added, word missed |
| de-k-cap-seed9.wav | K, wie Kuh. | 1.44 | k a | k a v i k uː | KWQ | vowel added, word missed |
| de-k-cap-seed10.wav | K, wie Kuh. | 2.08 | k eɪ | k eɪ v i k uː | KVQ | vowel added, word missed |
| de-k-cap-seed11.wav | K, wie Kuh. | 2.88 | k aɪ | k aɪ v i k uː | KVQ | vowel added, word missed |
| de-k-cap-seed12.wav | K, wie Kuh. | 1.44 | kh eɪ | kh eɪ v i k u5 | K wie Kuh | wrong sound |
| de-k-cap-seed13.wav | K, wie Kuh. | 1.76 | k ei5 | k ei5 v i5 k u5 | KWQ | vowel added, word missed |
| de-k-cap-seed14.wav | K, wie Kuh. | 2.32 | k eɪ | k eɪ v i k uː | KVQ | vowel added, word missed |
| de-k-bang-seed0.wav | K! wie Kuh. | 1.76 | k | k v i k uː | K wie Kuh | clean |
| de-k-bang-seed1.wav | K! wie Kuh. | 1.68 | k a | k a v i k uː | KWQ | vowel added, word missed |
| de-k-bang-seed2.wav | K! wie Kuh. | 1.84 | kh a | kh a v i5 k u5 | KWQ | wrong sound, word missed |
| de-k-bang-seed3.wav | K! wie Kuh. | 1.76 | k a | k a v i k uː | KWQ | vowel added, word missed |
| de-k-bang-seed4.wav | K! wie Kuh. | 1.76 | k | k v i k uː | Wie cool! | clean, word missed |
| de-k-bang-seed5.wav | K! wie Kuh. | 2.08 | k aɪ | k aɪ v i k uː | K wie Kuh | vowel added |
| de-k-bang-seed6.wav | K! wie Kuh. | 1.84 | k a | k a v i k uː | KWQ | vowel added, word missed |
| de-k-bang-seed7.wav | K! wie Kuh. | 1.52 | k eː | k eː v iː k uː | K, wie cool! | vowel added, word missed |
| de-k-bang-seed8.wav | K! wie Kuh. | 1.92 | k eɪ | k eɪ v i k uː | K.V.Q. | vowel added, word missed |
| de-k-bang-seed9.wav | K! wie Kuh. | 2.08 | k ɛ | k ɛ v iː k uː | K. Wiku | vowel added, word missed |
| de-k-bang-seed10.wav | K! wie Kuh. | 2.00 | k eɪ | k eɪ v i k uː | KWQ | vowel added, word missed |
| de-k-bang-seed11.wav | K! wie Kuh. | 2.72 | k a | k a v i k uː | KWQ | vowel added, word missed |
| de-k-bang-seed12.wav | K! wie Kuh. | 1.28 | k ɛ | k ɛ v i k u | KWQ | vowel added, word missed |
| de-k-bang-seed13.wav | K! wie Kuh. | 2.08 | k a | k a v iː k u5 | K. Wie cool! | vowel added, word missed |
| de-k-bang-seed14.wav | K! wie Kuh. | 1.60 | kh ɑ5 | kh ɑ5 v i5 kh u5 | KWQ | wrong sound, word missed |
| de-p-cap-seed0.wav | P, wie Puppe. | 2.16 | p iː | p iː v iː p ɔ p ə | P wie Puppe | vowel added |
| de-p-cap-seed1.wav | P, wie Puppe. | 1.60 | p i | p i v i p o p | Pi, wie Puppe. | vowel added |
| de-p-cap-seed2.wav | P, wie Puppe. | 1.68 | p iː | p iː v iː p ɔ p | Pi, wie Puppe. | vowel added |
| de-p-cap-seed3.wav | P, wie Puppe. | 1.36 | p i | p i v i p o b | Pi wie Puppe | vowel added |
| de-p-cap-seed4.wav | P, wie Puppe. | 1.44 | p eː | p eː v iː p ʊ b ə | P wie Puppe | vowel added |
| de-p-cap-seed5.wav | P, wie Puppe. | 1.76 | p i | p i v i p o p | Pi wie Puppe | vowel added |
| de-p-cap-seed6.wav | P, wie Puppe. | 1.92 | p iː | p iː ð iː p ɔ p ɚ | Pi wie Puppe | vowel added |
| de-p-cap-seed7.wav | P, wie Puppe. | 1.44 | p iː | p iː v iː p b ə | Pi wie Puppe | vowel added |
| de-p-cap-seed8.wav | P, wie Puppe. | 1.68 | p i | p i v iː p o p ə | Pi wie Puppe | vowel added |
| de-p-cap-seed9.wav | P, wie Puppe. | 1.52 | ph iː | ph iː v iː p ɔ p ə | P wie Puppe | wrong sound |
| de-p-cap-seed10.wav | P, wie Puppe. | 1.76 | p i | p i v i p o p | Pi wie Puppe | vowel added |
| de-p-cap-seed11.wav | P, wie Puppe. | 2.64 | p i | p i v iː p o p ə | P wie Puppe | vowel added |
| de-p-cap-seed12.wav | P, wie Puppe. | 1.60 | p i | p i v i p o p e | Pi wie Puppe | vowel added |
| de-p-cap-seed13.wav | P, wie Puppe. | 1.60 | p i | p i v iː p ɔ b ə | Pi wie Puppe | vowel added |
| de-p-cap-seed14.wav | P, wie Puppe. | 1.44 | p | p ð iː p ɔ p ə | Pi wie Puppe | clean |
| de-p-bang-seed0.wav | P! wie Puppe. | 1.68 | p iː | p iː v iː p ɔ p ə | Pi wie Puppe | vowel added |
| de-p-bang-seed1.wav | P! wie Puppe. | 1.44 | p iː | p iː v iː p ɔ p ə | Pi wie Puppe | vowel added |
| de-p-bang-seed2.wav | P! wie Puppe. | 1.92 | p iː | p iː v iː p ɔ p ə | P wie Puppe | vowel added |
| de-p-bang-seed3.wav | P! wie Puppe. | 2.08 | p iː | p iː v iː p ɔ p ə | Pi wie Puppe | vowel added |
| de-p-bang-seed4.wav | P! wie Puppe. | 1.76 | p iː | p iː v iː p o p a | Pi wie Puppe | vowel added |
| de-p-bang-seed5.wav | P! wie Puppe. | 1.92 | p iː | p iː v iː p ʊ b ʌ | P wie Puppe | vowel added |
| de-p-bang-seed6.wav | P! wie Puppe. | 2.08 | ph i5 | ph i5 v i5 ph o p ə1 | Pi wie Puppe | wrong sound |
| de-p-bang-seed7.wav | P! wie Puppe. | 1.68 | p iː | p iː v iː p ɔ b ə | P wie Puppe | vowel added |
| de-p-bang-seed8.wav | P! wie Puppe. | 1.76 | p i | p i v iː p o b | Pi wie Puppe | vowel added |
| de-p-bang-seed9.wav | P! wie Puppe. | 2.00 | p iː | p iː v iː p oː p ə | Pi wie Puppe | vowel added |
| de-p-bang-seed10.wav | P! wie Puppe. | 2.32 | p iː | p iː v iː p ɔ b ə | P wie Puppe | vowel added |
| de-p-bang-seed11.wav | P! wie Puppe. | 2.80 | p iː | p iː v iː p ɔ p ə | Pi wie Puppe | vowel added |
| de-p-bang-seed12.wav | P! wie Puppe. | 1.84 | ph i5 | ph i5 v i5 ph oː p ɚ | Piep, wie Puppe. | wrong sound |
| de-p-bang-seed13.wav | P! wie Puppe. | 2.08 | p iː | p iː v iː p ɔ p ə | P wie Puppe | vowel added |
| de-p-bang-seed14.wav | P! wie Puppe. | 1.92 | ph iː | ph iː v iː p uː p ə | Pi wie Puppe | wrong sound |
| de-g-cap-seed0.wav | G, wie Gans. | 1.76 | ɡ eː | ɡ eː v iː ɡ ɛ n s | G wie Gänse | vowel added |
| de-g-cap-seed1.wav | G, wie Gans. | 1.28 | ɡ iː | ɡ iː v iː ɡ a n t | Geh vegan! | vowel added, word missed |
| de-g-cap-seed2.wav | G, wie Gans. | 1.52 | ɡ i | ɡ i v i ɡ a n t | Geh vegan! | vowel added, word missed |
| de-g-cap-seed3.wav | G, wie Gans. | 1.20 | ɡ iː | ɡ iː v iː ɡ a n | Kiwi Gans | vowel added |
| de-g-cap-seed4.wav | G, wie Gans. | 1.76 | ɡ iː | ɡ iː v iː ɡ a n s | G wie ganz | vowel added, word missed |
| de-g-cap-seed5.wav | G, wie Gans. | 2.24 | ɡ e | ɡ e v i ɡ a n s | G-Vegans | vowel added |
| de-g-cap-seed6.wav | G, wie Gans. | 1.84 | ɡ iː | ɡ iː v iː ɡ a n s | Geh vegans! | vowel added |
| de-g-cap-seed7.wav | G, wie Gans. | 1.68 | ɡ i | ɡ i v i ɡ a n t | GEE Vegans | vowel added |
| de-g-cap-seed8.wav | G, wie Gans. | 1.92 | ɡ i | ɡ i v i ɡ a n | G wie ganz. | vowel added, word missed |
| de-g-cap-seed9.wav | G, wie Gans. | 1.76 | ɡ iː ɡ iː ɡ æ n s | ɡ iː ɡ iː ɡ æ n s | Gigi Gans | no carrier |
| de-g-cap-seed10.wav | G, wie Gans. | 1.52 | ɡ i | ɡ i v i ɡ a n s | Geh vegan! | vowel added, word missed |
| de-g-cap-seed11.wav | G, wie Gans. | 1.60 | ɡ iː | ɡ iː v iː ɡ a n t | GiVegans | vowel added |
| de-g-cap-seed12.wav | G, wie Gans. | 1.68 | d i | d i v i ɡ a n | Wie ganz. | wrong sound, word missed |
| de-g-cap-seed13.wav | G, wie Gans. | 1.36 | ɡ eː | ɡ eː v iː ɡ a n s | Geh vegan! | vowel added, word missed |
| de-g-cap-seed14.wav | G, wie Gans. | 1.28 | ɡ i | ɡ i v iː ɡ a n | Geh vegans! | vowel added |
| de-g-bang-seed0.wav | G! wie Gans. | 2.32 | ɡ iː | ɡ iː v iː ɡ a n s | Wie ganz. | vowel added, word missed |
| de-g-bang-seed1.wav | G! wie Gans. | 1.76 | ɡ i | ɡ i v i ɡ a n s | Geh wie ganz. | vowel added, word missed |
| de-g-bang-seed2.wav | G! wie Gans. | 1.76 | ɡ iː | ɡ iː v iː ɡ a n t | G wie ganz | vowel added, word missed |
| de-g-bang-seed3.wav | G! wie Gans. | 1.44 | ɡ iː | ɡ iː v iː ɡ a n s | Wie ganz? | vowel added, word missed |
| de-g-bang-seed4.wav | G! wie Gans. | 1.44 | ɡ iː | ɡ iː v iː ɡ a n s | Wie ganz? | vowel added, word missed |
| de-g-bang-seed5.wav | G! wie Gans. | 1.44 | ɡ iː | ɡ iː v iː ɡ a n s | Gi Vigans | vowel added |
| de-g-bang-seed6.wav | G! wie Gans. | 2.00 | ɡ iː | ɡ iː v iː ɡ a n t | GEE Vegans | vowel added |
| de-g-bang-seed7.wav | G! wie Gans. | 2.00 | ɡ iː | ɡ iː v iː ɡ a n s | Geh vegan! | vowel added, word missed |
| de-g-bang-seed8.wav | G! wie Gans. | 2.08 | ɡ iː | ɡ iː v iː ɡ a n s | Wie ganz? | vowel added, word missed |
| de-g-bang-seed9.wav | G! wie Gans. | 1.84 | ɡ i | ɡ i v iː ɡ a n s | Geh vegans! | vowel added |
| de-g-bang-seed10.wav | G! wie Gans. | 2.00 | ɡ eː | ɡ eː v i ɡ a n s | G wie ganz | vowel added, word missed |
| de-g-bang-seed11.wav | G! wie Gans. | 2.48 | ɡ iː | ɡ iː v iː ɡ a n s | G wie ganz | vowel added, word missed |
| de-g-bang-seed12.wav | G! wie Gans. | 2.08 | ɡ iː | ɡ iː v iː ɡ a n s | G. Vegans | vowel added |
| de-g-bang-seed13.wav | G! wie Gans. | 1.84 | ɡ iː | ɡ iː v iː ɡ a n t | GiVegans | vowel added |
| de-g-bang-seed14.wav | G! wie Gans. | 1.60 | ɡ iː | ɡ iː v iː ɡ a n s | GIE wie GANS | vowel added |
| de-w-cap-seed0.wav | W, wie Wurm. | 1.60 |  | v i v i v o | VIVO | no sound, word missed |
| de-w-cap-seed1.wav | W, wie Wurm. | 1.36 |  | v i v i v o | Wie? Wie wo? | no sound, word missed |
| de-w-cap-seed2.wav | W, wie Wurm. | 2.08 |  | v i v i v o m | Wie wie Wurm | no sound |
| de-w-cap-seed3.wav | W, wie Wurm. | 1.60 |  | v i v i v o m | Wie wie Wurm | no sound |
| de-w-cap-seed4.wav | W, wie Wurm. | 1.52 |  | v i v i v o | Wie? Wie Wurm. | no sound |
| de-w-cap-seed5.wav | W, wie Wurm. | 1.44 |  | v i v i v o | Wie, wie wo? | no sound, word missed |
| de-w-cap-seed6.wav | W, wie Wurm. | 1.92 |  | v iː v iː v oː | Wie? Wie Wurm. | no sound |
| de-w-cap-seed7.wav | W, wie Wurm. | 1.44 |  | v iː v iː v o m | Wie wie Wurm | no sound |
| de-w-cap-seed8.wav | W, wie Wurm. | 1.84 |  | ð iː ð iː v ʊ m | Wie? Wie wurm? | no sound |
| de-w-cap-seed9.wav | W, wie Wurm. | 1.60 |  | v i v i v o r m | Wie wie Wurm. | no sound |
| de-w-cap-seed10.wav | W, wie Wurm. | 2.32 |  | v iː v iː v o r m | Wie Wie-Wurm | no sound |
| de-w-cap-seed11.wav | W, wie Wurm. | 2.08 |  | v iː v iː v o m | Wie wie Wurm | no sound |
| de-w-cap-seed12.wav | W, wie Wurm. | 1.52 |  | v iː v iː v oː m | Wie wie Wurm | no sound |
| de-w-cap-seed13.wav | W, wie Wurm. | 1.68 |  | v iː v iː v o m | Wie wie Wurm | no sound |
| de-w-cap-seed14.wav | W, wie Wurm. | 1.76 |  | v i v i v o m | Wie? Wiewo? | no sound, word missed |
| de-w-triple-seed0.wav | Www, wie Wurm. | 1.84 |  | v iː eː v iː v uː m | www.vivom | no sound, word missed |
| de-w-triple-seed1.wav | Www, wie Wurm. | 2.24 |  | v i v i v ø m | Wie, wie Wurm | no sound |
| de-w-triple-seed2.wav | Www, wie Wurm. | 2.16 |  | v iː v i v iː v u r m | Wie, wie, wie Wurm. | no sound |
| de-w-triple-seed3.wav | Www, wie Wurm. | 1.84 | v e v eː | v e v eː v i v o m | www.wurm.de | vowel added |
| de-w-triple-seed4.wav | Www, wie Wurm. | 1.76 | v e v e | v e v e v i v i v o r m | www.vivorm | vowel added, word missed |
| de-w-triple-seed5.wav | Www, wie Wurm. | 1.36 | v e v e | v e v e v i v ɔ | www.vivom | vowel added, word missed |
| de-w-triple-seed6.wav | Www, wie Wurm. | 2.88 |  | v iː v iː v uː m | W wie Wurm | no sound |
| de-w-triple-seed7.wav | Www, wie Wurm. | 1.44 |  | v i v i v i v o | Wewe vivo. | no sound, word missed |
| de-w-triple-seed8.wav | Www, wie Wurm. | 1.84 | v e i | v e i v i v i o | www.wium.de | vowel added, word missed |
| de-w-triple-seed9.wav | Www, wie Wurm. | 2.24 |  | v i v i v o m | Wie wie Wurm | no sound |
| de-w-triple-seed10.wav | Www, wie Wurm. | 2.88 |  | v i v i v o | Wie, wie, wo? | no sound, word missed |
| de-w-triple-seed11.wav | Www, wie Wurm. | 1.92 |  | v i v i v i v u m | Wie, wie, wie Wurm | no sound |
| de-w-triple-seed12.wav | Www, wie Wurm. | 2.16 |  | v iː v i v v u m | Wie wie, wie Wurm | no sound |
| de-w-triple-seed13.wav | Www, wie Wurm. | 1.84 | v eː | v eː v i v iː v ʉ | Wewe vivo! | vowel added, word missed |
| de-w-triple-seed14.wav | Www, wie Wurm. | 1.84 |  | v iː v iː v uː m | Wie? Wie wom? | no sound, word missed |
| de-d-cap-seed0.wav | D, wie Dach. | 1.60 | d eː | d eː v iː d a x | D. Wie Dach | vowel added |
| de-d-cap-seed1.wav | D, wie Dach. | 1.36 | d iː | d iː v iː d a x | Die Wiedach | vowel added |
| de-d-cap-seed2.wav | D, wie Dach. | 1.84 | d iː | d iː v iː d a x | Die Wiedach | vowel added |
| de-d-cap-seed3.wav | D, wie Dach. | 1.36 | d iː | d iː v iː d a x | D wie Dach | vowel added |
| de-d-cap-seed4.wav | D, wie Dach. | 1.28 | d iː | d iː v iː d a | D wie Dach | vowel added |
| de-d-cap-seed5.wav | D, wie Dach. | 1.92 | d iː | d iː v iː d a x | Die wie Dach | vowel added |
| de-d-cap-seed6.wav | D, wie Dach. | 1.60 | d i | d i v i d a x | Die Wiedach | vowel added |
| de-d-cap-seed7.wav | D, wie Dach. | 1.60 | d iː | d iː v iː d a x | Die Wiedach | vowel added |
| de-d-cap-seed8.wav | D, wie Dach. | 1.76 | d iː | d iː v iː d a x | Die wie Dach. | vowel added |
| de-d-cap-seed9.wav | D, wie Dach. | 1.12 | d iː | d iː v i d a x | Die Widach | vowel added |
| de-d-cap-seed10.wav | D, wie Dach. | 2.00 | d əɪ | d əɪ v iː d a x | D. Wie Dach | vowel added |
| de-d-cap-seed11.wav | D, wie Dach. | 1.68 | d i | d i v i d a x | D. Wie Dach | vowel added |
| de-d-cap-seed12.wav | D, wie Dach. | 1.36 | d iː | d iː v iː d a x | d. Wie Dach | vowel added |
| de-d-cap-seed13.wav | D, wie Dach. | 1.60 | d iː | d iː v iː d a x | D. Wiedach | vowel added |
| de-d-cap-seed14.wav | D, wie Dach. | 1.12 | d əɪ | d əɪ v iː d a x | D. Wiedach | vowel added |
| de-d-bang-seed0.wav | D! wie Dach. | 2.16 | d iː | d iː v iː d a x | Die Wiedach | vowel added |
| de-d-bang-seed1.wav | D! wie Dach. | 1.92 | d i | d i v i d a x | Die wie Dach. | vowel added |
| de-d-bang-seed2.wav | D! wie Dach. | 1.84 | d i | d i v iː d a x | Die Wiedach | vowel added |
| de-d-bang-seed3.wav | D! wie Dach. | 1.52 | d iː | d iː v iː d a x | Die Wiedach | vowel added |
| de-d-bang-seed4.wav | D! wie Dach. | 1.28 | d ɨ | d ɨ v iː d a x | D. Wiedach | wrong sound |
| de-d-bang-seed5.wav | D! wie Dach. | 1.44 | d iː | d iː v iː d a x | d. Wie Dach | vowel added |
| de-d-bang-seed6.wav | D! wie Dach. | 1.52 | d iː | d iː v iː d a x | Die Wiedach | vowel added |
| de-d-bang-seed7.wav | D! wie Dach. | 2.32 | d eː d i | d eː d i v iː d a x | DTV Dach | vowel added |
| de-d-bang-seed8.wav | D! wie Dach. | 1.92 | d iː | d iː v iː d a x | Die Wiedach | vowel added |
| de-d-bang-seed9.wav | D! wie Dach. | 1.28 | d iː | d iː v iː d a x | Die Wiedach | vowel added |
| de-d-bang-seed10.wav | D! wie Dach. | 1.92 | d iː | d iː v iː d a x | Die Wiedach | vowel added |
| de-d-bang-seed11.wav | D! wie Dach. | 2.08 | d iː | d iː v iː d a x | Die Wiedach | vowel added |
| de-d-bang-seed12.wav | D! wie Dach. | 1.36 | d iː | d iː v iː d a x | Die Wiedach | vowel added |
| de-d-bang-seed13.wav | D! wie Dach. | 1.68 | d i | d i v i d a x | Die Wiedach | vowel added |
| de-d-bang-seed14.wav | D! wie Dach. | 1.60 | d iː | d iː v iː d a x | Die Wiedach | vowel added |
