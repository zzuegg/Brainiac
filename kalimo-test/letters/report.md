# Kalimo letter questions: whole Qwen3 takes

German: female narrator clone (competitive-learn reference). English: the new British narrator clone (en-voice/reference.wav).
3 seeds each (777, 778, 779 = -seed0..2). Best take = highest Whisper large-v3 word match, then the length closest to the median.
Letter name check (answers and "Das ist ein …" lines): the wav2vec2 espeak phones of the take must contain the letter name (e.g. "ɛ m" for M, "t eː" for T, "k eɪ" for English K).

| line | text | best take | Whisper heard | match | letter name | other takes |
|---|---|---|---|---|---|---|
| de-m-question | Mit welchem Buchstaben fängt Maus an? | de-m-question-seed0.wav | Mit welchem Buchstaben fängt Maus an? | 1.00 | - | seed2.wav (1.00), seed1.wav (0.83) |
| de-m-answer | Ja! Maus fängt mit M an. | de-m-answer-seed2.wav | Ja. Maus fängt mit M an. | 1.00 | ok | seed1.wav (1.00, name ok), seed0.wav (1.00, name ok) |
| de-a-question | Mit welchem Buchstaben fängt Ameise an? | de-a-question-seed2.wav | Mit welchem Buchstaben fängt Ameise an? | 1.00 | - | seed1.wav (1.00), seed0.wav (0.83) |
| de-a-answer | Ja! Ameise fängt mit A an. | de-a-answer-seed0.wav | Ja. Ameise fängt mit A an. | 1.00 | ok | seed2.wav (1.00, name ok), seed1.wav (1.00, name ok) |
| de-l-question | Mit welchem Buchstaben fängt Löffel an? | de-l-question-seed2.wav | Mit welchem Buchstaben fängt Löffel an? | 1.00 | - | seed0.wav (1.00), seed1.wav (1.00) |
| de-l-answer | Ja! Löffel fängt mit L an. | de-l-answer-seed2.wav | Ja, Löffel fängt mit L an. | 1.00 | ok | seed1.wav (1.00, name ok), seed0.wav (1.00, name ok) |
| de-o-question | Mit welchem Buchstaben fängt Oma an? | de-o-question-seed2.wav | Mit welchem Buchstaben fängt Oma an? | 1.00 | - | seed0.wav (1.00), seed1.wav (1.00) |
| de-o-answer | Ja! Oma fängt mit O an. | de-o-answer-seed0.wav | Ja. Oma fängt mit O an. | 1.00 | check | seed1.wav (1.00, name?), seed2.wav (1.00, name?) |
| de-i-question | Mit welchem Buchstaben fängt Insel an? | de-i-question-seed2.wav | Mit welchem Buchstaben fängt Insel an? | 1.00 | - | seed0.wav (1.00), seed1.wav (1.00) |
| de-i-answer | Ja! Insel fängt mit I an. | de-i-answer-seed1.wav | Ja, Insel fängt mit Ei an. | 0.83 | check | seed2.wav (0.83, name?), seed0.wav (0.83, name?) |
| de-e-question | Mit welchem Buchstaben fängt Esel an? | de-e-question-seed2.wav | Mit welchem Buchstaben fängt Esel an? | 1.00 | - | seed1.wav (1.00), seed0.wav (1.00) |
| de-e-answer | Ja! Esel fängt mit E an. | de-e-answer-seed2.wav | Ja, Esel fängt mit E an. | 1.00 | ok | seed1.wav (1.00, name ok), seed0.wav (1.00, name ok) |
| de-s-question | Mit welchem Buchstaben fängt Sofa an? | de-s-question-seed1.wav | Mit welchem Buchstaben fängt Sofa an? | 1.00 | - | seed0.wav (1.00), seed2.wav (0.83) |
| de-s-answer | Ja! Sofa fängt mit S an. | de-s-answer-seed2.wav | Ja. Sofa fängt mit S an. | 1.00 | ok | seed1.wav (1.00, name ok), seed0.wav (1.00, name ok) |
| de-u-question | Mit welchem Buchstaben fängt Ufo an? | de-u-question-seed2.wav | Mit welchem Buchstaben fängt UFO an? | 1.00 | - | seed0.wav (1.00), seed1.wav (1.00) |
| de-u-answer | Ja! Ufo fängt mit U an. | de-u-answer-seed1.wav | Ja, UFO fängt mit U an. | 1.00 | ok | seed2.wav (1.00, name ok), seed0.wav (1.00, name ok) |
| de-r-question | Mit welchem Buchstaben fängt Rose an? | de-r-question-seed2.wav | Mit welchem Buchstaben fängt Rose an? | 1.00 | - | seed1.wav (1.00), seed0.wav (1.00) |
| de-r-answer | Ja! Rose fängt mit R an. | de-r-answer-seed2.wav | Ja, Rose fängt mit R an. | 1.00 | check | seed1.wav (1.00, name?), seed0.wav (1.00, name?) |
| de-t-question | Mit welchem Buchstaben fängt Tiger an? | de-t-question-seed0.wav | Mit welchem Buchstaben fängt Tiger an? | 1.00 | - | seed1.wav (1.00), seed2.wav (0.83) |
| de-t-answer | Ja! Tiger fängt mit T an. | de-t-answer-seed0.wav | Ja, Tiger fängt mit Tee an. | 0.83 | ok | seed1.wav (1.00, name?), seed2.wav (1.00, name?) |
| de-n-question | Mit welchem Buchstaben fängt Nase an? | de-n-question-seed2.wav | Mit welchem Buchstaben fängt Nase an? | 1.00 | - | seed0.wav (1.00), seed1.wav (1.00) |
| de-n-answer | Ja! Nase fängt mit N an. | de-n-answer-seed1.wav | Ja, Nase fängt mit N an. | 1.00 | ok | seed0.wav (1.00, name ok), seed2.wav (1.00, name ok) |
| de-f-question | Mit welchem Buchstaben fängt Fisch an? | de-f-question-seed2.wav | Mit welchem Buchstaben fängt Fisch an? | 1.00 | - | seed0.wav (1.00), seed1.wav (1.00) |
| de-f-answer | Ja! Fisch fängt mit F an. | de-f-answer-seed2.wav | Ja, Fisch fängt mit F an. | 1.00 | ok | seed0.wav (1.00, name ok), seed1.wav (1.00, name ok) |
| de-h-question | Mit welchem Buchstaben fängt Hund an? | de-h-question-seed2.wav | Mit welchem Buchstaben fängt Hund an? | 1.00 | - | seed0.wav (1.00), seed1.wav (1.00) |
| de-h-answer | Ja! Hund fängt mit H an. | de-h-answer-seed2.wav | Ja. Hund fängt mit H an. | 1.00 | ok | seed0.wav (1.00, name ok), seed1.wav (1.00, name ok) |
| de-b-question | Mit welchem Buchstaben fängt Baum an? | de-b-question-seed0.wav | Mit welchem Buchstaben fängt Baum an? | 1.00 | - | seed2.wav (1.00), seed1.wav (1.00) |
| de-b-answer | Ja! Baum fängt mit B an. | de-b-answer-seed0.wav | Ja, Baum fängt mit B an. | 1.00 | ok | seed2.wav (1.00, name?), seed1.wav (1.00, name?) |
| de-k-question | Mit welchem Buchstaben fängt Kuh an? | de-k-question-seed2.wav | Mit welchem Buchstaben fängt Q an? | 0.83 | - | seed0.wav (0.83), seed1.wav (0.83) |
| de-k-answer | Ja! Kuh fängt mit K an. | de-k-answer-seed2.wav | Ja, Q fängt mit K an. | 0.83 | ok | seed0.wav (0.83, name ok), seed1.wav (0.83, name ok) |
| de-p-question | Mit welchem Buchstaben fängt Puppe an? | de-p-question-seed1.wav | Mit welchem Buchstaben fängt Puppe an? | 1.00 | - | seed0.wav (1.00), seed2.wav (0.83) |
| de-p-answer | Ja! Puppe fängt mit P an. | de-p-answer-seed2.wav | Ja, Puppe fängt mit P an. | 1.00 | ok | seed0.wav (1.00, name ok), seed1.wav (1.00, name ok) |
| de-g-question | Mit welchem Buchstaben fängt Gans an? | de-g-question-seed0.wav | Mit welchem Buchstaben fängt GANS an? | 1.00 | - | seed1.wav (1.00), seed2.wav (0.83) |
| de-g-answer | Ja! Gans fängt mit G an. | de-g-answer-seed2.wav | Ja, Gans fängt mit G an. | 1.00 | ok | seed0.wav (1.00, name ok), seed1.wav (0.83, name?) |
| de-w-question | Mit welchem Buchstaben fängt Wurm an? | de-w-question-seed1.wav | Mit welchem Buchstaben fängt Wurm an? | 1.00 | - | seed2.wav (1.00), seed0.wav (0.83) |
| de-w-answer | Ja! Wurm fängt mit W an. | de-w-answer-seed1.wav | Ja, Wurm fängt mit W an. | 1.00 | ok | seed0.wav (1.00, name ok), seed2.wav (1.00, name?) |
| de-d-question | Mit welchem Buchstaben fängt Dach an? | de-d-question-seed2.wav | Mit welchem Buchstaben fängt Dach an? | 1.00 | - | seed1.wav (1.00), seed0.wav (0.83) |
| de-d-answer | Ja! Dach fängt mit D an. | de-d-answer-seed2.wav | Ja, Dach fängt mit D an. | 1.00 | check | seed1.wav (0.83, name?), seed0.wav (0.83, name?) |
| de-x-das-ist-ein-S | Das ist ein S. | de-x-das-ist-ein-S-seed2.wav | Das ist ein S. | 1.00 | ok | seed0.wav (1.00, name ok), seed1.wav (1.00, name ok) |
| de-x-das-ist-ein-K | Das ist ein K. | de-x-das-ist-ein-K-seed2.wav | Das ist ein K. | 1.00 | check | seed0.wav (1.00, name?), seed1.wav (1.00, name?) |
| de-x-hoer-noch-mal-Maus | Hör noch mal: Maus. | de-x-hoer-noch-mal-Maus-seed0.wav | Hör nochmal, Maus. | 0.50 | - | seed2.wav (0.50), seed1.wav (0.50) |
| en-s-question | What letter does snake start with? | en-s-question-seed2.wav | What letter does snake start with? | 1.00 | - | seed1.wav (1.00), seed0.wav (1.00) |
| en-s-answer | Yes! Snake starts with S. | en-s-answer-seed2.wav | Yes, Snake starts with S. | 1.00 | ok | seed0.wav (1.00, name ok), seed1.wav (1.00, name ok) |
| en-a-question | What letter does ant start with? | en-a-question-seed1.wav | What letter does Ant start with? | 1.00 | - | seed0.wav (1.00), seed2.wav (1.00) |
| en-a-answer | Yes! Ant starts with A. | en-a-answer-seed2.wav | Yes. Ant starts with A. | 1.00 | ok | seed1.wav (1.00, name ok), seed0.wav (1.00, name ok) |
| en-t-question | What letter does tiger start with? | en-t-question-seed2.wav | What letter does Tiger start with? | 1.00 | - | seed0.wav (1.00), seed1.wav (1.00) |
| en-t-answer | Yes! Tiger starts with T. | en-t-answer-seed1.wav | Yes, Tiger starts with T. | 1.00 | ok | seed2.wav (1.00, name ok), seed0.wav (1.00, name ok) |
| en-p-question | What letter does pig start with? | en-p-question-seed2.wav | What letter does pig start with? | 1.00 | - | seed1.wav (1.00), seed0.wav (1.00) |
| en-p-answer | Yes! Pig starts with P. | en-p-answer-seed1.wav | Yes! Pig starts with P. | 1.00 | ok | seed2.wav (1.00, name ok), seed0.wav (1.00, name ok) |
| en-m-question | What letter does mouse start with? | en-m-question-seed2.wav | What letter does mouse start with? | 1.00 | - | seed0.wav (1.00), seed1.wav (1.00) |
| en-m-answer | Yes! Mouse starts with M. | en-m-answer-seed1.wav | Yes, mouse starts with M. | 1.00 | ok | seed0.wav (1.00, name ok), seed2.wav (1.00, name ok) |
| en-d-question | What letter does dog start with? | en-d-question-seed0.wav | What letter does dog start with? | 1.00 | - | seed2.wav (1.00), seed1.wav (1.00) |
| en-d-answer | Yes! Dog starts with D. | en-d-answer-seed2.wav | Yes. Dog starts with D. | 1.00 | ok | seed1.wav (1.00, name ok), seed0.wav (1.00, name ok) |
| en-g-question | What letter does goat start with? | en-g-question-seed2.wav | What letter does goat start with? | 1.00 | - | seed1.wav (1.00), seed0.wav (1.00) |
| en-g-answer | Yes! Goat starts with G. | en-g-answer-seed2.wav | Yes! Goat starts with G. | 1.00 | ok | seed0.wav (1.00, name ok), seed1.wav (1.00, name ok) |
| en-o-question | What letter does otter start with? | en-o-question-seed1.wav | What letter does otter start with? | 1.00 | - | seed2.wav (1.00), seed0.wav (1.00) |
| en-o-answer | Yes! Otter starts with O. | en-o-answer-seed1.wav | Yes! Otter starts with O. | 1.00 | ok | seed0.wav (1.00, name ok), seed2.wav (1.00, name ok) |
| en-k-question | What letter does kangaroo start with? | en-k-question-seed2.wav | What letter does kangaroo start with? | 1.00 | - | seed1.wav (1.00), seed0.wav (1.00) |
| en-k-answer | Yes! Kangaroo starts with K. | en-k-answer-seed1.wav | Yes! Kangaroo starts with K. | 1.00 | ok | seed2.wav (1.00, name ok), seed0.wav (1.00, name ok) |
| en-e-question | What letter does egg start with? | en-e-question-seed2.wav | What letter does egg start with? | 1.00 | - | seed0.wav (1.00), seed1.wav (1.00) |
| en-e-answer | Yes! Egg starts with E. | en-e-answer-seed1.wav | Yes! Egg starts with E. | 1.00 | ok | seed0.wav (1.00, name ok), seed2.wav (1.00, name ok) |
| en-u-question | What letter does up start with? | en-u-question-seed2.wav | What letter does up start with? | 1.00 | - | seed1.wav (1.00), seed0.wav (1.00) |
| en-u-answer | Yes! Up starts with U. | en-u-answer-seed1.wav | Yes! Up starts with you! | 0.80 | ok | seed0.wav (0.80, name ok), seed2.wav (0.40, name ok) |
| en-r-question | What letter does rain start with? | en-r-question-seed2.wav | What letter does rain start with? | 1.00 | - | seed1.wav (1.00), seed0.wav (1.00) |
| en-r-answer (best pick overridden to seed2: seed0 ends in "ow", seed2 says "ar") | Yes! Rain starts with R. | en-r-answer-seed1.wav | Yes, rain starts with R. | 1.00 | check | seed0.wav (1.00, name?), seed2.wav (1.00, name?) |
| en-h-question | What letter does horse start with? | en-h-question-seed0.wav | What letter does horse start with? | 1.00 | - | seed2.wav (1.00), seed1.wav (1.00) |
| en-h-answer | Yes! Horse starts with H. | en-h-answer-seed0.wav | Yes! Horse starts with H. | 1.00 | ok | seed1.wav (1.00, name ok), seed2.wav (1.00, name?) |
| en-b-question | What letter does bird start with? | en-b-question-seed2.wav | What letter does bird start with? | 1.00 | - | seed1.wav (1.00), seed0.wav (1.00) |
| en-b-answer | Yes! Bird starts with B. | en-b-answer-seed1.wav | Yes, Bird starts with B. | 1.00 | ok | seed2.wav (1.00, name ok), seed0.wav (1.00, name ok) |
| en-l-question | What letter does leaf start with? | en-l-question-seed1.wav | What letter does leaf start with? | 1.00 | - | seed2.wav (1.00), seed0.wav (1.00) |
| en-l-answer | Yes! Leaf starts with L. | en-l-answer-seed2.wav | Yes! Leaf starts with L. | 1.00 | ok | seed1.wav (1.00, name ok), seed0.wav (1.00, name?) |
| en-w-question | What letter does web start with? | en-w-question-seed2.wav | What letter does web start with? | 1.00 | - | seed0.wav (0.83), seed1.wav (0.83) |
| en-w-answer | Yes! Web starts with W. | en-w-answer-seed2.wav | Yes. Web starts with W. | 1.00 | ok | seed1.wav (1.00, name ok), seed0.wav (1.00, name ok) |
| en-n-question | What letter does net start with? | en-n-question-seed1.wav | What letter does net start with? | 1.00 | - | seed2.wav (1.00), seed0.wav (1.00) |
| en-n-answer | Yes! Net starts with N. | en-n-answer-seed0.wav | Yes! Net starts with N. | 1.00 | ok | seed2.wav (1.00, name ok), seed1.wav (1.00, name?) |
| en-x-thats-an-S | That's an S. | en-x-thats-an-S-seed0.wav | That's an S. | 1.00 | ok | seed1.wav (1.00, name ok), seed2.wav (1.00, name?) |
| en-x-thats-a-K | That's a K. | en-x-thats-a-K-seed1.wav | That's a K? | 1.00 | ok | seed0.wav (1.00, name ok), seed2.wav (1.00, name?) |
| en-x-listen-again-mouse | Listen again: mouse. | en-x-listen-again-mouse-seed1.wav | Listen again, mouse. | 1.00 | - | seed2.wav (1.00), seed0.wav (1.00) |

Lines whose best take is not a perfect Whisper match or where the letter name was not confirmed: de-o-answer, de-i-answer, de-r-answer, de-t-answer, de-k-question, de-k-answer, de-d-answer, de-x-das-ist-ein-K, de-x-hoer-noch-mal-Maus, en-u-answer, en-r-answer


## Listening notes

- **de-i-answer: wrong letter name.** In all 3 takes Qwen says the English "eye" (phones "aɪ") instead of the German "I" (iː). Will be re-recorded with a different spelling.
- de-k: Whisper writes "Kuh" as "Q" (they sound the same in German); the take is fine.
- de-o: the recogniser hears "uː" for "O"; Whisper hears "O". Needs an ear.
- de-r, de-d: the letter is merged with "mit" in the phones (ɛ, d ɨ); Whisper hears R and D.
- en-u: "you" is the right English name for U.
