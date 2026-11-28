# German accent: old narrator reference vs native-German designs

Cause: the takes were made with language "german", but the German narrator reference (competitive-learn
00-reference.wav) was designed without saying the accent. Brainiac #230 found that VoiceDesign then tends to give
an English-sounding voice. The new references are designed with "She is a native German speaker from Germany with a
standard German (Hochdeutsch) accent, no English accent" (description in English, as #230 recommends).

Each voice: 4 test lines, 2 takes each (seeds 777, 778), temperature 0.6, " …" ending, Qwen3-TTS 1.7B Base clone.
Files: `<voice>-<line>-seed<N>.wav`; the voice's own reference is `reference-<voice>.wav`.

| voice | reference | Whisper detects German | vowel in "Gans" (wav2vec2 phones) | Whisper exact* |
|---|---|---|---|---|
| old | competitive-learn female (used in game/game2) | 8/8 | ɛ 4, a 4 | 6/8 |
| calm230 | #230 calm narrator (accent stated) | 8/8 | ɛ 8 | 5/8 |
| **new0** | new design, seed 2000 | 8/8 | **a 8** | 4/8 |
| new1 | new design, seed 2001 (broken take) | 1/8 | - | 0/8 |
| **new2** | new design, seed 2002 | 8/8 | **a 8** | 6/8 |
| new3 | new design, seed 2003 | 8/8 | a 7 | 3/8 |

*Most misses are "Ganz" for "Gans" (the same sound in German) or "genau so".
"Gɛns" (ɛ) is the English-accented vowel MiZu heard; new0 and new2 say a German "a" in every take.
Lines: ask-g = "Hör gut zu: Gans. Was fängt genauso an?", match-geschenk/giraffe/gitarre = "Geschenk/Giraffe/Gitarre, wie Gans!".
