# Kalimo letter-sound spelling sweep (Qwen3-TTS, whole takes, no cutting)

Voice: Qwen3-TTS 1.7B Base, cloned from the competitive-learn female narrator reference; seeds 777, 778, 779 (`-seed0..2`).
Judge: Whisper large-v3 (Vulkan, bf16), language forced to the line language.

**Caveat:** Whisper writes a bare hissed "sss" and the letter name "Es" both often as just "S", and "T" for both a clean t and "Tee".
So "bare sound" means only that Whisper did *not* write a letter name (Es, Ses, Die, The, ...); an ear has to confirm.

| file | text | length s | Whisper heard | verdict |
|---|---|---|---|---|
| de-S-Ss-seed0.wav | Ss, wie Sonne. | 1.52 | Es wie Sonne | name/other |
| de-S-Ss-seed1.wav | Ss, wie Sonne. | 1.36 | Es wie Sonne. | name/other |
| de-S-Ss-seed2.wav | Ss, wie Sonne. | 1.52 | Es wie Sonne | name/other |
| en-S-Ss-seed0.wav | Ss, like snake. | 2.08 | Esses, like snake? | name/other |
| en-S-Ss-seed1.wav | Ss, like snake. | 2.32 | S. Like snake. | bare sound |
| en-S-Ss-seed2.wav | Ss, like snake. | 2.00 | S, like snake. | bare sound |
| de-S-Sss-seed0.wav | Sss, wie Sonne. | 1.92 | S. Visone | bare sound |
| de-S-Sss-seed1.wav | Sss, wie Sonne. | 1.28 | S wie Sonne | bare sound |
| de-S-Sss-seed2.wav | Sss, wie Sonne. | 1.44 | S wie Sonne. | bare sound |
| en-S-Sss-seed0.wav | Sss, like snake. | 2.24 | ss like snake | bare sound |
| en-S-Sss-seed1.wav | Sss, like snake. | 1.92 | ss like snake | bare sound |
| en-S-Sss-seed2.wav | Sss, like snake. | 1.92 | S like snake | bare sound |
| de-S-Ssss-seed0.wav | Ssss, wie Sonne. | 2.40 | S wie Sonne | bare sound |
| de-S-Ssss-seed1.wav | Ssss, wie Sonne. | 1.92 | Ssss, wie Sonne. | bare sound |
| de-S-Ssss-seed2.wav | Ssss, wie Sonne. | 1.68 | S wie Sonne | bare sound |
| en-S-Ssss-seed0.wav | Ssss, like snake. | 1.68 | Ssss, like snake. | bare sound |
| en-S-Ssss-seed1.wav | Ssss, like snake. | 1.76 | Zzzz, like snake. | bare sound |
| en-S-Ssss-seed2.wav | Ssss, like snake. | 2.16 | Zzzzz, like snake. | bare sound |
| de-S-s-ellipsis-seed0.wav | s… wie Sonne. | 1.92 | S wie Sonne | bare sound |
| de-S-s-ellipsis-seed1.wav | s… wie Sonne. | 1.36 | Wie Sonne. | sound dropped |
| de-S-s-ellipsis-seed2.wav | s… wie Sonne. | 1.68 | S wie Sonne | bare sound |
| en-S-s-ellipsis-seed0.wav | s… like snake. | 1.68 | Like snake. | sound dropped |
| en-S-s-ellipsis-seed1.wav | s… like snake. | 1.68 | S like snake. | bare sound |
| en-S-s-ellipsis-seed2.wav | s… like snake. | 1.52 | Zlike Snake | name/other |
| de-S-ss-dash-seed0.wav | ss – wie Sonne. | 2.40 | SIS wie Sonne | name/other |
| de-S-ss-dash-seed1.wav | ss – wie Sonne. | 0.88 | Wie Sonne. | sound dropped |
| de-S-ss-dash-seed2.wav | ss – wie Sonne. | 1.76 | Wie Sonne | sound dropped |
| en-S-ss-dash-seed0.wav | ss – like snake. | 1.68 | S like snake. | bare sound |
| en-S-ss-dash-seed1.wav | ss – like snake. | 2.08 | S like snake. | bare sound |
| en-S-ss-dash-seed2.wav | ss – like snake. | 2.56 | S like snake. | bare sound |
| de-T-Tt-seed0.wav | Tt, wie Tiger. | 2.56 | T wie Tiger | bare sound |
| de-T-Tt-seed1.wav | Tt, wie Tiger. | 1.52 | Tietz wie Tiger | name/other |
| de-T-Tt-seed2.wav | Tt, wie Tiger. | 2.08 | T wie Tiger | bare sound |
| en-T-Tt-seed0.wav | Tt, like tiger. | 2.72 | Tate. Like Tiger. | name/other |
| en-T-Tt-seed1.wav | Tt, like tiger. | 1.92 | T like tiger | bare sound |
| en-T-Tt-seed2.wav | Tt, like tiger. | 2.00 | T like tiger | bare sound |
| de-T-T-bang-seed0.wav | T! wie Tiger. | 1.60 | T wie Tiger | bare sound |
| de-T-T-bang-seed1.wav | T! wie Tiger. | 1.60 | T wie Tiger | bare sound |
| de-T-T-bang-seed2.wav | T! wie Tiger. | 1.92 | T wie Tiger | bare sound |
| en-T-T-bang-seed0.wav | T! like tiger. | 1.60 | Like Tiger | sound dropped |
| en-T-T-bang-seed1.wav | T! like tiger. | 1.68 | t like tiger | bare sound |
| en-T-T-bang-seed2.wav | T! like tiger. | 1.28 | It's like Tiger. | sound dropped |
| de-T-Th-seed0.wav | Th, wie Tiger. | 1.84 | Wie Tiger | sound dropped |
| de-T-Th-seed1.wav | Th, wie Tiger. | 1.76 | Die wie Tiger | name/other |
| de-T-Th-seed2.wav | Th, wie Tiger. | 1.92 | The V-Tiger | name/other |
| en-T-Th-seed0.wav | Th, like tiger. | 2.64 | The, like tiger. | name/other |
| en-T-Th-seed1.wav | Th, like tiger. | 2.08 | The like, like tiger. | name/other |
| en-T-Th-seed2.wav | Th, like tiger. | 1.76 | The Lick Tiger | name/other |
| de-T-lower-tt-seed0.wav | tt, wie Tiger. | 2.56 | T.D. wie Tiger | name/other |
| de-T-lower-tt-seed1.wav | tt, wie Tiger. | 1.84 | T.T. wie Tiger | name/other |
| de-T-lower-tt-seed2.wav | tt, wie Tiger. | 1.52 | T wie Tiger | bare sound |
| en-T-lower-tt-seed0.wav | tt, like tiger. | 2.16 | Pleat like tiger. | sound dropped |
| en-T-lower-tt-seed1.wav | tt, like tiger. | 1.44 | Like Tiger | sound dropped |
| en-T-lower-tt-seed2.wav | tt, like tiger. | 1.52 | Tlaik Tiger | name/other |
| de-T-T-dash-seed0.wav | T – wie Tiger. | 1.60 | Die B-Tiger | name/other |
| de-T-T-dash-seed1.wav | T – wie Tiger. | 1.92 | D. Wie Tiger | name/other |
| de-T-T-dash-seed2.wav | T – wie Tiger. | 1.84 | Die V-Tiger | name/other |
| en-T-T-dash-seed0.wav | T – like tiger. | 2.00 | Like Tiger | sound dropped |
| en-T-T-dash-seed1.wav | T – like tiger. | 1.68 | T-L-I-K-E TIGER | name/other |
| en-T-T-dash-seed2.wav | T – like tiger. | 1.68 | Like Tiger. | sound dropped |
| de-S-prev-Hs-seed0.wav | Hs, wie Sonne. | 1.84 | Hass wie Sonne | name/other |
| de-S-prev-Hs-seed1.wav | Hs, wie Sonne. | 1.84 | Haze wie Sonne | name/other |
| de-S-prev-Hs-seed2.wav | Hs, wie Sonne. | 1.84 | Haas, Wiesone | name/other |
| de-S-prev-Sz-seed0.wav | Sz, wie Sonne. | 1.68 | S wie Sonne | bare sound |
| de-S-prev-Sz-seed1.wav | Sz, wie Sonne. | 1.20 | Zzzz, wie Sonne. | bare sound |
| de-S-prev-Sz-seed2.wav | Sz, wie Sonne. | 1.44 | Sbvi Sonne | name/other |
| de-S-prev-Zs-seed0.wav | Zs, wie Sonne. | 2.00 | Setz wie Sonne. | name/other |
| de-S-prev-Zs-seed1.wav | Zs, wie Sonne. | 1.44 | SETS wie Sonne | name/other |
| de-S-prev-Zs-seed2.wav | Zs, wie Sonne. | 1.60 | Z wie Sonne | bare sound |
| de-S-prev-period-seed0.wav | S. Wie Sonne. | 2.64 | Sess wie Sonne | name/other |
| de-S-prev-period-seed1.wav | S. Wie Sonne. | 1.60 | Es wie Sonne. | name/other |
| de-S-prev-period-seed2.wav | S. Wie Sonne. | 1.84 | SES wie Sonne | name/other |
| de-T-prev-Tch-seed0.wav | Tch, wie Tiger. | 2.32 | Tch, wie Tiger. | bare sound |
| de-T-prev-Tch-seed1.wav | Tch, wie Tiger. | 1.44 | T wie Tiger. | bare sound |
| de-T-prev-Tch-seed2.wav | Tch, wie Tiger. | 1.68 | Tch, wie Tiger. | bare sound |
| de-T-prev-Th-seed0.wav | Th, wie Tiger. | 1.84 | Wie Tiger | sound dropped |
| de-T-prev-Th-seed1.wav | Th, wie Tiger. | 1.76 | Die wie Tiger | name/other |
| de-T-prev-Th-seed2.wav | Th, wie Tiger. | 1.92 | The V-Tiger | name/other |
| de-T-prev-bang-seed0.wav | T! Wie Tiger. | 2.96 | D-D-D-D wie Tiger. | name/other |
| de-T-prev-bang-seed1.wav | T! Wie Tiger. | 1.52 | T wie Tiger | bare sound |
| de-T-prev-bang-seed2.wav | T! Wie Tiger. | 1.84 | T wie Tiger | bare sound |
| de-T-prev-period-seed0.wav | T. Wie Tiger. | 2.00 | Die VTiger | name/other |
| de-T-prev-period-seed1.wav | T. Wie Tiger. | 2.32 | T wie Tiger | bare sound |
| de-T-prev-period-seed2.wav | T. Wie Tiger. | 1.92 | T wie Tiger | bare sound |
| de-S-orig-wie-Sonne-seed0.wav | S, wie Sonne. | 1.68 | SES wie Sonne | name/other |
| de-S-orig-wie-Sonne-seed1.wav | S, wie Sonne. | 1.44 | Sess wie Sonne | name/other |
| de-S-orig-wie-Sonne-seed2.wav | S, wie Sonne. | 1.44 | Sess wie Sonne | name/other |
| de-T-orig-wie-Tiger-seed0.wav | T, wie Tiger. | 1.44 | Die V-Tiger | name/other |
| de-T-orig-wie-Tiger-seed1.wav | T, wie Tiger. | 1.68 | Die V-Tiger | name/other |
| de-T-orig-wie-Tiger-seed2.wav | T, wie Tiger. | 1.52 | Die W-Tiger | name/other |
| en-S-orig-like-snake-seed0.wav | S, like snake. | 2.16 | Zess like snake | name/other |
| en-S-orig-like-snake-seed1.wav | S, like snake. | 1.12 | S'like Snake. | name/other |
| en-S-orig-like-snake-seed2.wav | S, like snake. | 1.76 | The like snake | name/other |
| en-T-orig-like-tiger-seed0.wav | T, like tiger. | 1.60 | Like tiger. | sound dropped |
| en-T-orig-like-tiger-seed1.wav | T, like tiger. | 1.60 | Like Tiger. | sound dropped |
| en-T-orig-like-tiger-seed2.wav | T, like tiger. | 1.36 | Like tiger | sound dropped |

## Summary per spelling (bare sound out of 3 seeds)

| spelling | text | bare | dropped | name/other |
|---|---|---|---|---|
| de-S-Ss | Ss, wie Sonne. | 0 | 0 | 3 |
| en-S-Ss | Ss, like snake. | 2 | 0 | 1 |
| de-S-Sss | Sss, wie Sonne. | 3 | 0 | 0 |
| en-S-Sss | Sss, like snake. | 3 | 0 | 0 |
| de-S-Ssss | Ssss, wie Sonne. | 3 | 0 | 0 |
| en-S-Ssss | Ssss, like snake. | 3 | 0 | 0 |
| de-S-s-ellipsis | s… wie Sonne. | 2 | 1 | 0 |
| en-S-s-ellipsis | s… like snake. | 1 | 1 | 1 |
| de-S-ss-dash | ss – wie Sonne. | 0 | 2 | 1 |
| en-S-ss-dash | ss – like snake. | 3 | 0 | 0 |
| de-T-Tt | Tt, wie Tiger. | 2 | 0 | 1 |
| en-T-Tt | Tt, like tiger. | 2 | 0 | 1 |
| de-T-T-bang | T! wie Tiger. | 3 | 0 | 0 |
| en-T-T-bang | T! like tiger. | 1 | 2 | 0 |
| de-T-Th | Th, wie Tiger. | 0 | 1 | 2 |
| en-T-Th | Th, like tiger. | 0 | 0 | 3 |
| de-T-lower-tt | tt, wie Tiger. | 1 | 0 | 2 |
| en-T-lower-tt | tt, like tiger. | 0 | 2 | 1 |
| de-T-T-dash | T – wie Tiger. | 0 | 0 | 3 |
| en-T-T-dash | T – like tiger. | 0 | 2 | 1 |
| de-S-prev-Hs | Hs, wie Sonne. | 0 | 0 | 3 |
| de-S-prev-Sz | Sz, wie Sonne. | 2 | 0 | 1 |
| de-S-prev-Zs | Zs, wie Sonne. | 1 | 0 | 2 |
| de-S-prev-period | S. Wie Sonne. | 0 | 0 | 3 |
| de-T-prev-Tch | Tch, wie Tiger. | 3 | 0 | 0 |
| de-T-prev-Th | Th, wie Tiger. | 0 | 1 | 2 |
| de-T-prev-bang | T! Wie Tiger. | 2 | 0 | 1 |
| de-T-prev-period | T. Wie Tiger. | 2 | 0 | 1 |
| de-S-orig-wie-Sonne | S, wie Sonne. | 0 | 0 | 3 |
| de-T-orig-wie-Tiger | T, wie Tiger. | 0 | 0 | 3 |
| en-S-orig-like-snake | S, like snake. | 0 | 0 | 3 |
| en-T-orig-like-tiger | T, like tiger. | 0 | 3 | 0 |
