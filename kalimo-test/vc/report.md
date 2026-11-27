# Kalimo letter sounds: exact source (Kokoro, raw IPA) + voice conversion (Chatterbox S3Gen)

Source: Kokoro-82M (Apache-2.0), phonemes passed straight to the model (no text front end), voice af_heart (de) / bf_emma (en); Kokoro is deterministic, so one take per setting.
Conversion: Chatterbox S3Gen (Resemble AI, MIT): S3 speech tokens of the source (with 240 ms silence padding) -> flow matching conditioned on the narrator reference -> HiFT vocoder.
Targets: the female German Qwen3 narrator reference (de), the new British Qwen3 narrator reference (en).
Clip = converted sound (silence trimmed), 350 ms pause, the whole Qwen3 take of the carrier line (same as cut2). Phones: wav2vec2 espeak on the bare sound (padded) and on the clip (part before "wie"/"like").

| sound | setting (IPA, speed) | source bare | source ms | source phones | converted bare | converted ms | converted phones | clip | sound part in clip |
|---|---|---|---|---|---|---|---|---|---|
| de f | fː ×0.6 | de-f-f-source-bare.wav | 615 | θ a f ɑɨ | de-f-f-vc-bare.wav | 1390 | s ɑː f ei | de-f-f.wav | s a f eː |
| de f | fːːː ×1.0 | de-f-flong-source-bare.wav | 490 | f ʌ ɹ | de-f-flong-vc-bare.wav | 570 | s uɨ r əɪ | de-f-flong.wav | s r eː |
| de f | fːːː ×0.6 | de-f-flong-slow-source-bare.wav | 835 | aɪ f ø r a | de-f-flong-slow-vc-bare.wav | 885 | h aɪ f ø r aɪ | de-f-flong-slow.wav | h eː v ø r aɪ |
| de m | mː ×0.6 | de-m-m-source-bare.wav | 470 | aː m a | de-m-m-vc-bare.wav | 645 | h a m a | de-m-m.wav | h a m a |
| de m | mːːː ×1.0 | de-m-mlong-source-bare.wav | 515 | m ø ɹ a | de-m-mlong-vc-bare.wav | 615 | m uɨ r ɛ | de-m-mlong.wav | m ʉ ɹ ɛ |
| de m | mːːː ×0.6 | de-m-mlong-slow-source-bare.wav | 890 | k a i m ʌ n ʌ ɡ ʌ r a | de-m-mlong-slow-vc-bare.wav | 2135 | b a h iː m l ɛ ɛ | de-m-mlong-slow.wav | h aɪ m l ɛ ɛ |
| de t | t ×1.0 | de-t-t-source-bare.wav | 185 | t l a | de-t-t-vc-bare.wav | 540 | t aː | de-t-t.wav | d ɛ |
| de t | tʰ ×1.0 | de-t-th-source-bare.wav | 235 | t a | de-t-th-vc-bare.wav | 325 | t a | de-t-th.wav | t a |
| de t | tʰ ×0.7 | de-t-th-slow-source-bare.wav | 455 | ʌ ð t a | de-t-th-slow-vc-bare.wav | 785 | əɜ n t ɑ | de-t-th-slow.wav | əɜ n t ɑ |
| en t | t ×1.0 | en-t-t-source-bare.wav | 215 | b a h a | en-t-t-vc-bare.wav | 255 | l a h a t | en-t-t.wav | v ɛ ð |
| en t | tʰ ×1.0 | en-t-th-source-bare.wav | 260 | t a | en-t-th-vc-bare.wav | 275 | l a tʃ a | en-t-th.wav | l ɛ t ʌ |
| en t | tʰ ×0.7 | en-t-th-slow-source-bare.wav | 385 | ɔɨ t a n | en-t-th-slow-vc-bare.wav | 400 | l o t ai | en-t-th-slow.wav | ð ə t ɛ |
