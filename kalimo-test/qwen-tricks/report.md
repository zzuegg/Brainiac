# Qwen3 tricks for bare letter sounds

1. **design**: VoiceDesign 1.7B, voice description + phonics instruction (English or German instruction text).
2. **custom**: CustomVoice 1.7B, preset voice "serena" + the same instruction.
3. **primed**: Base 1.7B clone whose reference already contains bare sounds: de = narrator "Ssss, wie Sofa." + cut2 de-m-v2 + de-f-v2 (transcript "Ssss, wie Sofa. Mmm, wie Maus. Fff, wie Fisch."); en = narrator "Ssss, like snake." + cut2 en-m (transcript "Ssss, like snake. Mmm, like mouse."). References saved as primed-reference-*.wav.
5 seeds each. Sound part = wav2vec2 espeak phones before "wie"/"like"; **clean** = only the target consonant there (no vowel). Whisper large-v3 checks the word.

## Clean takes per method

| line + method | clean / 5 | clean files |
|---|---|---|
| de-m-design-instr-en | 1 | de-m-design-instr-en-seed3.wav |
| de-m-design-instr-de | 0 | - |
| de-f-design-instr-en | 0 | - |
| de-f-design-instr-de | 0 | - |
| de-t-design-instr-en | 0 | - |
| de-t-design-instr-de | 0 | - |
| de-k-design-instr-en | 0 | - |
| de-k-design-instr-de | 0 | - |
| en-t-design-instr-en | 0 | - |
| en-t-design-instr-de | 0 | - |
| de-m-custom-serena-instr-en | 0 | - |
| de-m-custom-serena-instr-de | 0 | - |
| de-f-custom-serena-instr-en | 0 | - |
| de-f-custom-serena-instr-de | 0 | - |
| de-t-custom-serena-instr-en | 0 | - |
| de-t-custom-serena-instr-de | 1 | de-t-custom-serena-instr-de-seed4.wav |
| de-k-custom-serena-instr-en | 0 | - |
| de-k-custom-serena-instr-de | 0 | - |
| en-t-custom-serena-instr-en | 0 | - |
| en-t-custom-serena-instr-de | 0 | - |
| de-m-primed | 0 | - |
| de-m-cap-primed | 0 | - |
| de-f-primed | 0 | - |
| de-t-primed | 0 | - |
| de-k-primed | 0 | - |
| en-t-primed | 0 | - |
| en-m-cap-primed | 0 | - |

## Every take

| file | text | s | sound part | Whisper | verdict |
|---|---|---|---|---|---|
| de-m-design-instr-en-seed0.wav | Mmm, wie Maus. | 1.84 | - | Mhh, wie Maus. | no carrier |
| de-m-design-instr-en-seed1.wav | Mmm, wie Maus. | 1.68 | - | Mmm, die Maus. | no carrier |
| de-m-design-instr-en-seed2.wav | Mmm, wie Maus. | 2.08 | - | Mh, wie Maus? | no carrier |
| de-m-design-instr-en-seed3.wav | Mmm, wie Maus. | 1.68 | m | Mh, wie Maus. | clean |
| de-m-design-instr-en-seed4.wav | Mmm, wie Maus. | 1.36 | - | Hm? Wie Maus? | no carrier |
| de-m-design-instr-de-seed0.wav | Mmm, wie Maus. | 2.56 | - | Hmm, wie Maus. | no carrier |
| de-m-design-instr-de-seed1.wav | Mmm, wie Maus. | 1.60 | - | Wie Maus. | no carrier |
| de-m-design-instr-de-seed2.wav | Mmm, wie Maus. | 2.08 | - | Mh, wie Maus. | no carrier |
| de-m-design-instr-de-seed3.wav | Mmm, wie Maus. | 2.08 | - | Hmm, wie Maus? | no carrier |
| de-m-design-instr-de-seed4.wav | Mmm, wie Maus. | 1.84 | - | Mhh, wie Maus. | no carrier |
| de-f-design-instr-en-seed0.wav | Fff, wie Fisch. | 1.36 | h | Hm, wie fisch! | not clean |
| de-f-design-instr-en-seed1.wav | Fff, wie Fisch. | 2.16 | - | Wie Fisch. | no carrier |
| de-f-design-instr-en-seed2.wav | Fff, wie Fisch. | 2.00 | h | Huh, wie Fisch. | not clean |
| de-f-design-instr-en-seed3.wav | Fff, wie Fisch. | 2.16 | uː | Hmm, wie Fisch! | not clean |
| de-f-design-instr-en-seed4.wav | Fff, wie Fisch. | 1.60 | - | Pfff, wie Fisch. | no carrier |
| de-f-design-instr-de-seed0.wav | Fff, wie Fisch. | 2.32 | - | Fee Fish. | no carrier, word missed |
| de-f-design-instr-de-seed1.wav | Fff, wie Fisch. | 1.68 | h ɔ | Pfuh, wie Fisch! | not clean |
| de-f-design-instr-de-seed2.wav | Fff, wie Fisch. | 2.00 | - | The Fish | no carrier, word missed |
| de-f-design-instr-de-seed3.wav | Fff, wie Fisch. | 4.08 | h ø | Fö, wie Fisch. | not clean |
| de-f-design-instr-de-seed4.wav | Fff, wie Fisch. | 2.00 | h a | Hah, wie Fisch. | not clean |
| de-t-design-instr-en-seed0.wav | T, wie Tiger. | 1.92 | t iː | T wie Tiger | not clean |
| de-t-design-instr-en-seed1.wav | T, wie Tiger. | 1.60 | t iː | Tee, Bietega. | not clean, word missed |
| de-t-design-instr-en-seed2.wav | T, wie Tiger. | 1.68 | t iː | Tee, be Tigger. | not clean, word missed |
| de-t-design-instr-en-seed3.wav | T, wie Tiger. | 1.68 | t iː | T wie Tiger. | not clean |
| de-t-design-instr-en-seed4.wav | T, wie Tiger. | 1.60 | t ʌ | Tch, wie Tiger. | not clean |
| de-t-design-instr-de-seed0.wav | T, wie Tiger. | 1.60 | t i | T wie Tiger | not clean |
| de-t-design-instr-de-seed1.wav | T, wie Tiger. | 1.20 | - | T. B. Tiger | no carrier |
| de-t-design-instr-de-seed2.wav | T, wie Tiger. | 1.92 | t ɨ | T wie Tiger | not clean |
| de-t-design-instr-de-seed3.wav | T, wie Tiger. | 2.00 | t ø | Tee wie Tiger. | not clean |
| de-t-design-instr-de-seed4.wav | T, wie Tiger. | 1.20 | t aɪ | Tei wie Tiger | not clean |
| de-k-design-instr-en-seed0.wav | K, wie Kuh. | 1.36 | k a | K, wie cool. | not clean, word missed |
| de-k-design-instr-en-seed1.wav | K, wie Kuh. | 1.04 | a | K, wie cool! | not clean, word missed |
| de-k-design-instr-en-seed2.wav | K, wie Kuh. | 1.52 | a | K. B. Q. | not clean, word missed |
| de-k-design-instr-en-seed3.wav | K, wie Kuh. | 1.44 | k a | K. Wie cool! | not clean, word missed |
| de-k-design-instr-en-seed4.wav | K, wie Kuh. | 1.44 | kh ɑ | K.W.Q. | not clean, word missed |
| de-k-design-instr-de-seed0.wav | K, wie Kuh. | 1.28 | - | K.W.K. | no carrier, word missed |
| de-k-design-instr-de-seed1.wav | K, wie Kuh. | 3.36 | k a | K wie Kuh | not clean |
| de-k-design-instr-de-seed2.wav | K, wie Kuh. | 1.36 | t aː | Haviku! | not clean, word missed |
| de-k-design-instr-de-seed3.wav | K, wie Kuh. | 1.76 | a | K. Viku | not clean, word missed |
| de-k-design-instr-de-seed4.wav | K, wie Kuh. | 1.36 | k a | K. Be cool. | not clean, word missed |
| en-t-design-instr-en-seed0.wav | T, like tiger. | 1.68 | t iː | See? Like Tiger. | not clean |
| en-t-design-instr-en-seed1.wav | T, like tiger. | 1.68 | t iː | T like tiger. | not clean |
| en-t-design-instr-en-seed2.wav | T, like tiger. | 2.48 | t iː | T like tiger | not clean |
| en-t-design-instr-en-seed3.wav | T, like tiger. | 2.32 | t iː | T, like tiger. | not clean |
| en-t-design-instr-en-seed4.wav | T, like tiger. | 1.68 | t iː | T, like tiger. | not clean |
| en-t-design-instr-de-seed0.wav | T, like tiger. | 2.00 | iː | T like tiger | not clean |
| en-t-design-instr-de-seed1.wav | T, like tiger. | 1.92 | s iː | T like tiger. | not clean |
| en-t-design-instr-de-seed2.wav | T, like tiger. | 1.52 | t iː | Tea, like tiger? | not clean |
| en-t-design-instr-de-seed3.wav | T, like tiger. | 1.92 | t iː | T like tiger. | not clean |
| en-t-design-instr-de-seed4.wav | T, like tiger. | 1.84 | - | Tee, like tiger. | no carrier |
| de-m-custom-serena-instr-en-seed0.wav | Mmm, wie Maus. | 2.88 | - | Wie Maus. | no carrier |
| de-m-custom-serena-instr-en-seed1.wav | Mmm, wie Maus. | 1.92 | - | Wie Maus. | no carrier |
| de-m-custom-serena-instr-en-seed2.wav | Mmm, wie Maus. | 2.08 | - | Hm, wie Maus? | no carrier |
| de-m-custom-serena-instr-en-seed3.wav | Mmm, wie Maus. | 2.24 | - | Wie Maus. | no carrier |
| de-m-custom-serena-instr-en-seed4.wav | Mmm, wie Maus. | 1.60 | - | Wie Maus. | no carrier |
| de-m-custom-serena-instr-de-seed0.wav | Mmm, wie Maus. | 2.32 | - | Wie Maus. | no carrier |
| de-m-custom-serena-instr-de-seed1.wav | Mmm, wie Maus. | 1.84 | - | Mhh, wie Maus. | no carrier |
| de-m-custom-serena-instr-de-seed2.wav | Mmm, wie Maus. | 2.32 | - | Mmmh, die Maus. | no carrier |
| de-m-custom-serena-instr-de-seed3.wav | Mmm, wie Maus. | 1.84 | - | Mmmh, wie Maus. | no carrier |
| de-m-custom-serena-instr-de-seed4.wav | Mmm, wie Maus. | 1.84 | - | Hmm, wie Maus. | no carrier |
| de-f-custom-serena-instr-en-seed0.wav | Fff, wie Fisch. | 1.52 | - | Wie Fisch. | no carrier |
| de-f-custom-serena-instr-en-seed1.wav | Fff, wie Fisch. | 1.76 | - | Wie Fisch. | no carrier |
| de-f-custom-serena-instr-en-seed2.wav | Fff, wie Fisch. | 1.92 | - | Wie Fisch | no carrier |
| de-f-custom-serena-instr-en-seed3.wav | Fff, wie Fisch. | 1.44 | - | Wie Fisch! | no carrier |
| de-f-custom-serena-instr-en-seed4.wav | Fff, wie Fisch. | 2.16 | - | Wie Fisch. | no carrier |
| de-f-custom-serena-instr-de-seed0.wav | Fff, wie Fisch. | 1.44 | - | Wir fischen. | no carrier |
| de-f-custom-serena-instr-de-seed1.wav | Fff, wie Fisch. | 2.00 | t | Pffff, wie Fisch. | not clean |
| de-f-custom-serena-instr-de-seed2.wav | Fff, wie Fisch. | 2.48 | - | Wie Fisch! | no carrier |
| de-f-custom-serena-instr-de-seed3.wav | Fff, wie Fisch. | 4.16 | - | Wie Fisch | no carrier |
| de-f-custom-serena-instr-de-seed4.wav | Fff, wie Fisch. | 2.80 | - | Wie Fisch. | no carrier |
| de-t-custom-serena-instr-en-seed0.wav | T, wie Tiger. | 1.52 | th ʉ | T wie Tiger | not clean |
| de-t-custom-serena-instr-en-seed1.wav | T, wie Tiger. | 2.48 | t iː | Tee with Tiger | not clean |
| de-t-custom-serena-instr-en-seed2.wav | T, wie Tiger. | 2.16 | t iː | T. V. Tiger | not clean |
| de-t-custom-serena-instr-en-seed3.wav | T, wie Tiger. | 1.44 | tɕh iɛ | Hier, wie Tiger. | not clean |
| de-t-custom-serena-instr-en-seed4.wav | T, wie Tiger. | 1.68 | - | T. V. Tiger | no carrier |
| de-t-custom-serena-instr-de-seed0.wav | T, wie Tiger. | 1.60 | t iː | T. V. Tiger | not clean |
| de-t-custom-serena-instr-de-seed1.wav | T, wie Tiger. | 2.64 | - | T.V. Tiger | no carrier |
| de-t-custom-serena-instr-de-seed2.wav | T, wie Tiger. | 2.00 | t iː | T. We Tiger | not clean |
| de-t-custom-serena-instr-de-seed3.wav | T, wie Tiger. | 1.84 | t iː | T. V. Tager | not clean, word missed |
| de-t-custom-serena-instr-de-seed4.wav | T, wie Tiger. | 2.00 | t | Wie Tiger | clean |
| de-k-custom-serena-instr-en-seed0.wav | K, wie Kuh. | 1.36 | k a | Kaa, wie cool! | not clean, word missed |
| de-k-custom-serena-instr-en-seed1.wav | K, wie Kuh. | 1.36 | kh ɑ | K. Be cool. | not clean, word missed |
| de-k-custom-serena-instr-en-seed2.wav | K, wie Kuh. | 1.44 | k a | Ka-Mi-Ku | not clean, word missed |
| de-k-custom-serena-instr-en-seed3.wav | K, wie Kuh. | 1.36 | k a | K.V.Q. | not clean, word missed |
| de-k-custom-serena-instr-en-seed4.wav | K, wie Kuh. | 2.16 | - | Hey, wie cool! | no carrier, word missed |
| de-k-custom-serena-instr-de-seed0.wav | K, wie Kuh. | 1.44 | - | K.V.K.U. | no carrier, word missed |
| de-k-custom-serena-instr-de-seed1.wav | K, wie Kuh. | 2.16 | k a | K. V. Kuh | not clean |
| de-k-custom-serena-instr-de-seed2.wav | K, wie Kuh. | 1.28 | k a | K, wie cool! | not clean, word missed |
| de-k-custom-serena-instr-de-seed3.wav | K, wie Kuh. | 1.84 | k | Wie cool! | clean, word missed |
| de-k-custom-serena-instr-de-seed4.wav | K, wie Kuh. | 1.60 | - | Hör wie cool! | no carrier, word missed |
| en-t-custom-serena-instr-en-seed0.wav | T, like tiger. | 2.96 | t iː | T like tiger | not clean |
| en-t-custom-serena-instr-en-seed1.wav | T, like tiger. | 2.96 | t iː | T. Like tiger. | not clean |
| en-t-custom-serena-instr-en-seed2.wav | T, like tiger. | 2.24 | t iː | T like tiger | not clean |
| en-t-custom-serena-instr-en-seed3.wav | T, like tiger. | 1.68 | t iː | T like tiger | not clean |
| en-t-custom-serena-instr-en-seed4.wav | T, like tiger. | 1.92 | t iː | T like tiger | not clean |
| en-t-custom-serena-instr-de-seed0.wav | T, like tiger. | 2.72 | t i | T like tiger | not clean |
| en-t-custom-serena-instr-de-seed1.wav | T, like tiger. | 2.56 | t eɪ | Tay, like tiger | not clean |
| en-t-custom-serena-instr-de-seed2.wav | T, like tiger. | 2.24 | - | Tch, like tiger. | no carrier |
| en-t-custom-serena-instr-de-seed3.wav | T, like tiger. | 1.44 | t ɪ n | Teen Like Tiger | not clean |
| en-t-custom-serena-instr-de-seed4.wav | T, like tiger. | 2.80 | t oʊ | Oh, like tiger | not clean |
| de-m-primed-seed0.wav | Mmm, wie Maus. | 1.52 | - | Wie Maus | no carrier |
| de-m-primed-seed1.wav | Mmm, wie Maus. | 1.36 | - | Wie Maus. | no carrier |
| de-m-primed-seed2.wav | Mmm, wie Maus. | 1.92 | - | Mh, wie Maus. | no carrier |
| de-m-primed-seed3.wav | Mmm, wie Maus. | 1.60 | - | Wie Maus. | no carrier |
| de-m-primed-seed4.wav | Mmm, wie Maus. | 1.92 | - | Mh, wie Maus. | no carrier |
| de-m-cap-primed-seed0.wav | M, wie Maus. | 1.28 | e m | M wie Maus | not clean |
| de-m-cap-primed-seed1.wav | M, wie Maus. | 0.96 | ə n | Envy Mouse | not clean, word missed |
| de-m-cap-primed-seed2.wav | M, wie Maus. | 1.52 | e m | M wie Maus. | not clean |
| de-m-cap-primed-seed3.wav | M, wie Maus. | 1.76 | e m | M wie Maus | not clean |
| de-m-cap-primed-seed4.wav | M, wie Maus. | 2.40 | - | Wie Maus. | no carrier |
| de-f-primed-seed0.wav | Fff, wie Fisch. | 1.44 | - | Wie Fisch | no carrier |
| de-f-primed-seed1.wav | Fff, wie Fisch. | 1.36 | - | Wie Fisch. | no carrier |
| de-f-primed-seed2.wav | Fff, wie Fisch. | 1.92 | - | Wie Fisch | no carrier |
| de-f-primed-seed3.wav | Fff, wie Fisch. | 2.08 | f ɪ θ | Pfiff wie Fisch | not clean |
| de-f-primed-seed4.wav | Fff, wie Fisch. | 1.44 | - | Wie Fisch | no carrier |
| de-t-primed-seed0.wav | T, wie Tiger. | 2.00 | t iː | T wie Tiger | not clean |
| de-t-primed-seed1.wav | T, wie Tiger. | 1.76 | t i | T wie Tiger | not clean |
| de-t-primed-seed2.wav | T, wie Tiger. | 1.76 | t iː ɡ | Teak wie Tiger | not clean |
| de-t-primed-seed3.wav | T, wie Tiger. | 1.92 | t i | T wie Tiger | not clean |
| de-t-primed-seed4.wav | T, wie Tiger. | 1.92 | t iː | T wie Tiger | not clean |
| de-k-primed-seed0.wav | K, wie Kuh. | 1.84 | k a | K.W.Q. | not clean, word missed |
| de-k-primed-seed1.wav | K, wie Kuh. | 1.76 | k eɪ | KVQ | not clean, word missed |
| de-k-primed-seed2.wav | K, wie Kuh. | 1.52 | k eɪ | K.V.Q. | not clean, word missed |
| de-k-primed-seed3.wav | K, wie Kuh. | 1.92 | k eɪ | KVQ | not clean, word missed |
| de-k-primed-seed4.wav | K, wie Kuh. | 1.44 | k aɪ | KVQ | not clean, word missed |
| en-t-primed-seed0.wav | T, like tiger. | 1.92 | - | Like tiger | no carrier |
| en-t-primed-seed1.wav | T, like tiger. | 1.28 | - | Like Tiger. | no carrier |
| en-t-primed-seed2.wav | T, like tiger. | 1.44 | - | Like Tiger. | no carrier |
| en-t-primed-seed3.wav | T, like tiger. | 1.68 | - | Like tiger. | no carrier |
| en-t-primed-seed4.wav | T, like tiger. | 655.28 | l aɪ k | like like like like like tiger like like tiger like like catch like tiger like like like like so Like, like, like, like, like, like. The do I'm going to turn it over to you, and I'm going to ask you to come up here and give us a little bit of a shout out. We're going to be talking about the new technology that's coming out of the Google Cloud Platform. So, we're going to start with a little bit of a shout out to the Google Cloud Platform. So, we're going to start with a little bit of a shout out to the Google Cloud Platform. So, we're going to start with a little bit of a shout out to the Google Cloud Platform. So, we're going to start with a little bit of a shout out to the Google Cloud Platform. So, we're going to start with a little bit of a shout out to the Google Cloud Platform. So, we're going to start with a little bit of a shout out to the Google Cloud Platform. So, we're going to start with a little bit of a shout out to the Google Cloud Platform. So, we're going to start with a little bit of a shout out to the Google Cloud Platform. So, we're going to start with a little bit of a shout out to the Google Cloud Platform. So, we're going to start with a little bit of a shout out to the Google Cloud Platform. So, we're going to start with a little bit of a shout out to the Google Cloud Platform. So, we're going to start with a little bit of a shout out to the Google Cloud Platform. So, we're going to start with a little bit of a shout out to the Google Cloud Platform. So, we're going to start with a little bit of a shout out to the Google Cloud Platform. So, we're going to start with a little bit I'm going to take a few minutes to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I'm going to ask you a question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. I think it's a really good question. to be a part of that. I think that's one of the things that's really important to me, is that I'm not just a part of the project, I'm a part of the process. And I think that's one of the things that I'm really proud of, is that I'm not just a part of the process, but I'm also a part of the process. And I think that's one of the things that's really important to me, is that I'm not just a part of the process, I'm a part of the process. And I think that's one of the things that's really important to me, is that I'm not just a part of the process, but I'm also a part of the process. And I think that's one of the things that's really important to me, is that I'm not just a part of the process, but I'm also a part of the process. as you all and lots of others on this call. What's the relationship you've consistent the action that you've peaks don't accept at all? We'll go back to the Should I Do the Right Thing for a moment, which is just marionette I don't Patrick'sjesери and Mikehhh Confessini the cause and all for the vast majority of all the complex situations, when you said you'd hold their reassurance and also we称 of that aside, What's the relationship you've consistent the action All in hey up Lit envusia OH yes. Yes. Yes. Yes. Yes. Yes. Yes. I really think that I meet a partner of mine to do I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. F Yani Naobin, I think that's a great question. F Yani Naobin, I think that's a great question. I think that's a great question. F Yani Naobin, I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great question. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. I think that's a great point. People would be super appears People would be super appears People would be super appears All right. All right. All right. Yeah. Yeah. Yeah. Yeah. On the second day when it On the second day when it On the second day when it hit On the second day when it hit On the second day when it hit and I'm going to go ahead and put it on the screen. So, I'm going to put it on the screen. So, I'm going to put it on the screen. So, I'm going to put it on the screen. So, I'm going to put it on the screen. So, I'm going to put it on the screen. So, I'm going to put it on the screen. So, I'm going to put it on the screen. So, I'm going to put it on the screen. So, I'm going to put it on the screen. So, I'm going to put it on the screen. So, I'm going to put it on the screen. So, I'm going to put it on the screen. So, I'm going to put it on the screen. So, I'm going to put it on the screen. So, I'm going to put it on the screen. | not clean |
| en-m-cap-primed-seed0.wav | M, like mouse. | 2.08 | ɛ m | M, like mouse. | not clean |
| en-m-cap-primed-seed1.wav | M, like mouse. | 2.96 | - | Hmm, like mouse. | no carrier |
| en-m-cap-primed-seed2.wav | M, like mouse. | 2.08 | - | Like mouse. | no carrier |
| en-m-cap-primed-seed3.wav | M, like mouse. | 1.84 | ɛ m | M like mouse | not clean |
| en-m-cap-primed-seed4.wav | M, like mouse. | 2.16 | ɛ m | M like mouse. | not clean |
