# Sound Safari German lines v3: new native-German narrator (de-accent new0), whole takes

Qwen3-TTS 1.7B Base clone of reference-de.wav (VoiceDesign, native German speaker, Hochdeutsch, no English accent; see ../de-accent), temperature 0.6 (was 0.9), 24 kHz mono WAV. Nothing cut, sped up or processed.
German text is spoken with " …" added at the end (in a test it gave quiet at the end in 9 of 20 takes against 0 of 20 without, and slightly slower speech; Whisper still heard every word). English: plain text first, " …" only for retries.
Per line 4 takes (up to 12 when needed). A take counts only if Whisper large-v3 hears the text exactly, Whisper detects German when not told the language, at least 150 ms of quiet follow the speech, and it is not more than 15 % faster than the median take of that line. Among those, takes more than 0.03 below the best speaker similarity are dropped, and the most similar one is kept.
Speaker similarity: cosine between the Qwen3 speaker encoder embedding of the take and of the new reference.

| file | text | spoken as | Whisper | end quiet ms | speech s | speaker cos | takes | note |
|---|---|---|---|---|---|---|---|---|
| de/game.first-sounds.match.apple.0.wav | Apfel, wie Ameise! | Apfel, wie Ameise! … | Apfel wie Ameise | 775 | 1.96 | 0.985 | 12 |  |
| de/game.first-sounds.named.apple.0.wav | Das ist ein Apfel. | Das ist ein Apfel. … | Das ist ein Apfel. | 672 | 1.05 | 0.983 | 12 |  |
| de/game.first-sounds.match.ball.0.wav | Ball, wie Baum! | Ball, wie Baum! … | Ball wie Baum | 1062 | 1.26 | 0.980 | 12 |  |
| de/game.first-sounds.named.ball.0.wav | Das ist ein Ball. | Das ist ein Ball. … | Das ist ein Ball. | 588 | 1.01 | 0.986 | 12 |  |
| de/game.first-sounds.match.banana.0.wav | Banane, wie Baum! | Banane, wie Baum! … | Banane wie Baum | 186 | 1.84 | 0.982 | 12 |  |
| de/game.first-sounds.named.banana.0.wav | Das ist eine Banane. | Das ist eine Banane. … | Das ist eine Banane. | 288 | 1.08 | 0.986 | 12 |  |
| de/game.first-sounds.match.barrel.0.wav | Fass, wie Fisch! | Fass, wie Fisch! … | Fass wie Fisch | 231 | 1.59 | 0.974 | 12 |  |
| de/game.first-sounds.named.barrel.0.wav | Das ist ein Fass. | Das ist ein Fass. … | Das ist ein Fass. | 1170 | 0.92 | 0.982 | 12 |  |
| de/game.first-sounds.match.bear.0.wav | Bär, wie Baum! | Bär, wie Baum! … | Bär wie Baum | 288 | 1.41 | 0.977 | 12 |  |
| de/game.first-sounds.named.bear.0.wav | Das ist ein Bär. | Das ist ein Bär. … | Das ist ein Bär. | 252 | 0.88 | 0.984 | 12 |  |
| de/game.first-sounds.match.bee.0.wav | Biene, wie Baum! | Biene, wie Baum! … | Biene wie Baum | 246 | 1.46 | 0.978 | 12 |  |
| de/game.first-sounds.named.bee.0.wav | Das ist eine Biene. | Das ist eine Biene. … | Das ist eine Biene. | 552 | 0.97 | 0.987 | 12 |  |
| de/game.first-sounds.match.bicycle.0.wav | Fahrrad, wie Fisch! | Fahrrad, wie Fisch! … | Fahrrad wie Fisch | 246 | 1.51 | 0.979 | 12 |  |
| de/game.first-sounds.named.bicycle.0.wav | Das ist ein Fahrrad. | Das ist ein Fahrrad. … | Das ist ein Fahrrad. | 338 | 1.03 | 0.985 | 12 |  |
| de/game.first-sounds.match.boat.0.wav | Boot, wie Baum! | Boot, wie Baum! … | Boot, wie Baum. | 293 | 1.49 | 0.979 | 12 |  |
| de/game.first-sounds.named.boat.0.wav | Das ist ein Boot. | Das ist ein Boot. … | Das ist ein Boot. | 811 | 0.87 | 0.984 | 12 |  |
| de/game.first-sounds.match.book.0.wav | Buch, wie Baum! | Buch, wie Baum! … | Buch wie Baum | 274 | 1.43 | 0.976 | 12 |  |
| de/game.first-sounds.named.book.0.wav | Das ist ein Buch. | Das ist ein Buch. … | Das ist ein Buch. | 865 | 0.83 | 0.985 | 12 |  |
| de/game.first-sounds.match.cactus.0.wav | Kaktus, wie Kuh! | Kaktus, wie Kuh! … | Kaktus wie Kuh | 223 | 1.43 | 0.981 | 12 |  |
| de/game.first-sounds.named.cactus.0.wav | Das ist ein Kaktus. | Das ist ein Kaktus. … | Das ist ein Kaktus. | 409 | 1.15 | 0.986 | 12 |  |
| de/game.first-sounds.match.candle.0.wav | Kerze, wie Kuh! | Kerze, wie Kuh! … | Kerze wie Kuh | 309 | 1.28 | 0.981 | 12 |  |
| de/game.first-sounds.named.candle.0.wav | Das ist eine Kerze. | Das ist eine Kerze. … | Das ist eine Kerze. | 435 | 1.10 | 0.987 | 12 |  |
| de/game.first-sounds.match.carrot.0.wav | Möhre, wie Maus! | Möhre, wie Maus! … | Möhre wie Maus | 314 | 1.71 | 0.980 | 12 |  |
| de/game.first-sounds.named.carrot.0.wav | Das ist eine Möhre. | Das ist eine Möhre. … | Das ist eine Möhre. | 879 | 0.97 | 0.984 | 12 |  |
| de/game.first-sounds.match.cat.0.wav | Katze, wie Kuh! | Katze, wie Kuh! … | Katze wie Kuh | 502 | 1.12 | 0.983 | 12 |  |
| de/game.first-sounds.named.cat.0.wav | Das ist eine Katze. | Das ist eine Katze. … | Das ist eine Katze. | 492 | 1.12 | 0.987 | 12 |  |
| de/game.first-sounds.match.cheese.0.wav | Käse, wie Kuh! | Käse, wie Kuh! … | Käse wie Kuh | 594 | 1.28 | 0.978 | 12 |  |
| de/game.first-sounds.named.cheese.0.wav | Das ist Käse. | Das ist Käse. … | Das ist Käse. | 694 | 0.84 | 0.980 | 12 |  |
| de/game.first-sounds.match.clock.0.wav | Uhr, wie Ufo! | Uhr, wie Ufo! … | Uhr wie UFO | 677 | 1.44 | 0.973 | 12 |  |
| de/game.first-sounds.named.clock.0.wav | Das ist eine Uhr. | Das ist eine Uhr. … | Das ist eine Uhr. | 247 | 0.88 | 0.986 | 12 |  |
| de/game.first-sounds.match.cloud.0.wav | Wolke, wie Wurm! | Wolke, wie Wurm! … | Wolke wie Wurm | 331 | 1.56 | 0.978 | 12 |  |
| de/game.first-sounds.named.cloud.0.wav | Das ist eine Wolke. | Das ist eine Wolke. … | Das ist eine Wolke. | 287 | 1.07 | 0.987 | 12 |  |
| de/game.first-sounds.match.cookie.0.wav | Keks, wie Kuh! | Keks, wie Kuh! … | Keks wie Kuh | 493 | 1.39 | 0.977 | 12 |  |
| de/game.first-sounds.named.cookie.0.wav | Das ist ein Keks. | Das ist ein Keks. … | Das ist ein Keks. | 416 | 0.96 | 0.983 | 12 |  |
| de/game.first-sounds.match.corn.0.wav | Mais, wie Maus! | Mais, wie Maus! … | Mais, wie Maus. | 196 | 1.57 | 0.976 | 12 |  |
| de/game.first-sounds.named.corn.0.wav | Das ist Mais. | Das ist Mais. … | Das ist Mais. | 514 | 0.78 | 0.981 | 12 |  |
| de/game.first-sounds.match.cup.0.wav | Tasse, wie Tiger! | Tasse, wie Tiger! … | Tasse wie Tiger | 262 | 1.17 | 0.982 | 12 |  |
| de/game.first-sounds.named.cup.0.wav | Das ist eine Tasse. | Das ist eine Tasse. … | Das ist eine Tasse. | 455 | 1.08 | 0.985 | 12 |  |
| de/game.first-sounds.match.dino.0.wav | Dino, wie Dach! | Dino, wie Dach! … | Dino, wie Dach! | 851 | 1.24 | 0.982 | 12 |  |
| de/game.first-sounds.named.dino.0.wav | Das ist ein Dino. | Das ist ein Dino. … | Das ist ein Dino. | 777 | 0.92 | 0.986 | 12 |  |
| de/game.first-sounds.match.elephant.0.wav | Elefant, wie Esel! | Elefant, wie Esel! … | Elefant wie Esel | 266 | 1.83 | 0.986 | 12 |  |
| de/game.first-sounds.named.elephant.0.wav | Das ist ein Elefant. | Das ist ein Elefant. … | Das ist ein Elefant. | 367 | 1.12 | 0.987 | 12 |  |
| de/game.first-sounds.match.gift.0.wav | Geschenk, wie Gans! | Geschenk, wie Gans! … | Geschenk wie Gans | 305 | 1.36 | 0.988 | 12 |  |
| de/game.first-sounds.named.gift.0.wav | Das ist ein Geschenk. | Das ist ein Geschenk. … | Das ist ein Geschenk. | 349 | 0.96 | 0.986 | 12 |  |
| de/game.first-sounds.match.giraffe.0.wav | Giraffe, wie Gans! | Giraffe, wie Gans! … | Giraffe wie Gans | 288 | 1.50 | 0.986 | 12 |  |
| de/game.first-sounds.named.giraffe.0.wav | Das ist eine Giraffe. | Das ist eine Giraffe. … | Das ist eine Giraffe. | 1044 | 1.19 | 0.988 | 12 |  |
| de/game.first-sounds.match.guitar.0.wav | Gitarre, wie Gans! | Gitarre, wie Gans! … | Gitarre wie Gans | 285 | 1.09 | 0.985 | 12 |  |
| de/game.first-sounds.named.guitar.0.wav | Das ist eine Gitarre. | Das ist eine Gitarre. … | Das ist eine Gitarre. | 521 | 1.09 | 0.988 | 12 |  |
| de/game.first-sounds.match.hat.0.wav | Hut, wie Hund! | Hut, wie Hund! … | Hut wie Hund | 916 | 1.10 | 0.979 | 12 |  |
| de/game.first-sounds.named.hat.0.wav | Das ist ein Hut. | Das ist ein Hut. … | Das ist ein Hut. | 882 | 0.82 | 0.988 | 12 |  |
| de/game.first-sounds.match.heart.0.wav | Herz, wie Hund! | Herz, wie Hund! … | Herz wie Hund | 686 | 1.42 | 0.984 | 12 |  |
| de/game.first-sounds.named.heart.0.wav | Das ist ein Herz. | Das ist ein Herz. … | Das ist ein Herz. | 255 | 0.96 | 0.983 | 12 |  |
| de/game.first-sounds.match.hedgehog.0.wav | Igel, wie Insel! | Igel, wie Insel! … | Igel wie Insel | 379 | 1.48 | 0.980 | 12 |  |
| de/game.first-sounds.named.hedgehog.0.wav | Das ist ein Igel. | Das ist ein Igel. … | Das ist ein Igel. | 430 | 0.94 | 0.985 | 12 |  |
| de/game.first-sounds.match.house.0.wav | Haus, wie Hund! | Haus, wie Hund! … | Haus wie Hund | 551 | 0.94 | 0.981 | 12 |  |
| de/game.first-sounds.named.house.0.wav | Das ist ein Haus. | Das ist ein Haus. … | Das ist ein Haus. | 435 | 0.92 | 0.979 | 8 |  |
| de/game.first-sounds.match.ladder.0.wav | Leiter, wie Löffel! | Leiter, wie Löffel! … | Leiter wie Löffel | 1158 | 1.57 | 0.983 | 8 |  |
| de/game.first-sounds.named.ladder.0.wav | Das ist eine Leiter. | Das ist eine Leiter. … | Das ist eine Leiter. | 220 | 1.06 | 0.985 | 8 |  |
| de/game.first-sounds.match.lamp.0.wav | Lampe, wie Löffel! | Lampe, wie Löffel! … | Lampe wie Löffel | 617 | 1.55 | 0.982 | 8 |  |
| de/game.first-sounds.named.lamp.0.wav | Das ist eine Lampe. | Das ist eine Lampe. … | Das ist eine Lampe. | 432 | 1.10 | 0.986 | 8 |  |
| de/game.first-sounds.match.lantern.0.wav | Laterne, wie Löffel! | Laterne, wie Löffel! … | Laterne wie Löffel | 747 | 1.58 | 0.984 | 8 |  |
| de/game.first-sounds.named.lantern.0.wav | Das ist eine Laterne. | Das ist eine Laterne. … | Das ist eine Laterne. | 261 | 1.20 | 0.985 | 8 |  |
| de/game.first-sounds.match.lion.0.wav | Löwe, wie Löffel! | Löwe, wie Löffel! … | Löwe wie Löffel | 256 | 1.77 | 0.972 | 8 |  |
| de/game.first-sounds.named.lion.0.wav | Das ist ein Löwe. | Das ist ein Löwe. … | Das ist ein Löwe. | 394 | 0.97 | 0.986 | 8 |  |
| de/game.first-sounds.match.monkey.0.wav | Affe, wie Ameise! | Affe, wie Ameise! … | Affe wie Ameise | 494 | 1.28 | 0.982 | 8 |  |
| de/game.first-sounds.named.monkey.0.wav | Das ist ein Affe. | Das ist ein Affe. … | Das ist ein Affe. | 477 | 0.90 | 0.983 | 8 |  |
| de/game.first-sounds.match.moon.0.wav | Mond, wie Maus! | Mond, wie Maus! … | Mond wie Maus | 682 | 1.51 | 0.980 | 8 |  |
| de/game.first-sounds.named.moon.0.wav | Das ist der Mond. | Das ist der Mond. … | Das ist der Mond. | 523 | 0.92 | 0.986 | 8 |  |
| de/game.first-sounds.match.mushroom.0.wav | Pilz, wie Puppe! | Pilz, wie Puppe! … | Pilz wie Puppe | 674 | 1.14 | 0.980 | 8 |  |
| de/game.first-sounds.named.mushroom.0.wav | Das ist ein Pilz. | Das ist ein Pilz. … | Das ist ein Pilz. | 910 | 0.94 | 0.985 | 7 |  |
| de/game.first-sounds.match.nest.0.wav | Nest, wie Nase! | Nest, wie Nase! … | Nest wie Nase | 404 | 1.40 | 0.985 | 4 |  |
| de/game.first-sounds.named.nest.0.wav | Das ist ein Nest. | Das ist ein Nest. … | Das ist ein Nest. | 485 | 0.96 | 0.981 | 4 |  |
| de/game.first-sounds.match.octopus.0.wav | Oktopus, wie Oma! | Oktopus, wie Oma! … | Oktopus wie Oma | 365 | 1.84 | 0.984 | 4 |  |
| de/game.first-sounds.named.octopus.0.wav | Das ist ein Oktopus. | Das ist ein Oktopus. … | Das ist ein Oktopus. | 924 | 1.24 | 0.986 | 4 |  |
| de/game.first-sounds.match.orange.0.wav | Orange, wie Oma! | Orange, wie Oma! … | Orange wie Oma | 322 | 1.37 | 0.981 | 4 |  |
| de/game.first-sounds.named.orange.0.wav | Das ist eine Orange. | Das ist eine Orange. … | Das ist eine Orange. | 581 | 1.19 | 0.987 | 4 |  |
| de/game.first-sounds.match.pear.0.wav | Birne, wie Baum! | Birne, wie Baum! … | Birne wie Baum | 166 | 1.21 | 0.980 | 4 |  |
| de/game.first-sounds.named.pear.0.wav | Das ist eine Birne. | Das ist eine Birne. … | Das ist eine Birne. | 729 | 0.89 | 0.987 | 4 |  |
| de/game.first-sounds.match.penguin.0.wav | Pinguin, wie Puppe! | Pinguin, wie Puppe! … | Pinguin wie Puppe | 263 | 1.79 | 0.981 | 4 |  |
| de/game.first-sounds.named.penguin.0.wav | Das ist ein Pinguin. | Das ist ein Pinguin. … | Das ist ein Pinguin. | 502 | 1.10 | 0.985 | 4 |  |
| de/game.first-sounds.match.pizza.0.wav | Pizza, wie Puppe! | Pizza, wie Puppe! … | Pizza wie Puppe | 248 | 1.32 | 0.979 | 4 |  |
| de/game.first-sounds.named.pizza.0.wav | Das ist eine Pizza. | Das ist eine Pizza. … | Das ist eine Pizza. | 416 | 1.03 | 0.987 | 4 |  |
| de/game.first-sounds.match.rabbit.0.wav | Hase, wie Hund! | Hase, wie Hund! … | Hase wie Hund | 228 | 1.44 | 0.981 | 4 |  |
| de/game.first-sounds.named.rabbit.0.wav | Das ist ein Hase. | Das ist ein Hase. … | Das ist ein Hase. | 263 | 1.10 | 0.979 | 4 |  |
| de/game.first-sounds.match.rainbow.0.wav | Regenbogen, wie Rose! | Regenbogen, wie Rose! … | Regenbogen wie Rose | 371 | 1.78 | 0.982 | 4 |  |
| de/game.first-sounds.named.rainbow.0.wav | Das ist ein Regenbogen. | Das ist ein Regenbogen. … | Das ist ein Regenbogen. | 428 | 1.26 | 0.985 | 4 |  |
| de/game.first-sounds.match.robot.0.wav | Roboter, wie Rose! | Roboter, wie Rose! … | Roboter wie Rose | 566 | 1.52 | 0.982 | 4 |  |
| de/game.first-sounds.named.robot.0.wav | Das ist ein Roboter. | Das ist ein Roboter. … | Das ist ein Roboter. | 512 | 1.02 | 0.984 | 4 |  |
| de/game.first-sounds.match.rocket.0.wav | Rakete, wie Rose! | Rakete, wie Rose! … | Rakete wie Rose | 644 | 1.77 | 0.984 | 4 |  |
| de/game.first-sounds.named.rocket.0.wav | Das ist eine Rakete. | Das ist eine Rakete. … | Das ist eine Rakete. | 831 | 1.11 | 0.987 | 4 |  |
| de/game.first-sounds.match.soap.0.wav | Seife, wie Sofa! | Seife, wie Sofa! … | Seife wie Sofa | 644 | 1.87 | 0.981 | 4 |  |
| de/game.first-sounds.named.soap.0.wav | Das ist eine Seife. | Das ist eine Seife. … | Das ist eine Seife. | 752 | 1.10 | 0.985 | 4 |  |
| de/game.first-sounds.match.strawberry.0.wav | Erdbeere, wie Esel! | Erdbeere, wie Esel! … | Erdbeere wie Esel | 394 | 1.71 | 0.982 | 4 |  |
| de/game.first-sounds.named.strawberry.0.wav | Das ist eine Erdbeere. | Das ist eine Erdbeere. … | Das ist eine Erdbeere. | 1245 | 1.20 | 0.985 | 4 |  |
| de/game.first-sounds.match.submarine.0.wav | U-Boot, wie Ufo! | U-Boot, wie Ufo! … | U-Boot wie UFO | 550 | 1.58 | 0.974 | 4 |  |
| de/game.first-sounds.named.submarine.0.wav | Das ist ein U-Boot. | Das ist ein U-Boot. … | Das ist ein U-Boot. | 448 | 1.10 | 0.986 | 4 |  |
| de/game.first-sounds.match.sun.0.wav | Sonne, wie Sofa! | Sonne, wie Sofa! … | Sonne. Wie Sofa. | 216 | 1.81 | 0.982 | 4 |  |
| de/game.first-sounds.named.sun.0.wav | Das ist die Sonne. | Das ist die Sonne. … | Das ist die Sonne. | 1002 | 0.88 | 0.982 | 4 |  |
| de/game.first-sounds.match.table.0.wav | Tisch, wie Tiger! | Tisch, wie Tiger! … | Tisch wie Tiger | 154 | 1.22 | 0.980 | 4 |  |
| de/game.first-sounds.named.table.0.wav | Das ist ein Tisch. | Das ist ein Tisch. … | Das ist ein Tisch. | 622 | 1.06 | 0.983 | 4 |  |
| de/game.first-sounds.match.teddy.0.wav | Teddy, wie Tiger! | Teddy, wie Tiger! … | Teddy wie Tiger | 364 | 1.10 | 0.977 | 4 |  |
| de/game.first-sounds.named.teddy.0.wav | Das ist ein Teddy. | Das ist ein Teddy. … | Das ist ein Teddy. | 382 | 1.04 | 0.984 | 4 |  |
| de/game.first-sounds.match.tomato.0.wav | Tomate, wie Tiger! | Tomate, wie Tiger! … | Tomate wie Tiger | 587 | 1.42 | 0.985 | 4 |  |
| de/game.first-sounds.named.tomato.0.wav | Das ist eine Tomate. | Das ist eine Tomate. … | Das ist eine Tomate. | 777 | 1.16 | 0.988 | 4 |  |
| de/game.first-sounds.match.trophy.0.wav | Pokal, wie Puppe! | Pokal, wie Puppe! … | Pokal wie Puppe | 269 | 1.36 | 0.975 | 4 |  |
| de/game.first-sounds.named.trophy.0.wav | Das ist ein Pokal. | Das ist ein Pokal. … | Das ist ein Pokal. | 327 | 1.05 | 0.986 | 4 |  |
| de/game.first-sounds.match.watermelon.0.wav | Wassermelone, wie Wurm! | Wassermelone, wie Wurm! … | Wassermelone wie Wurm | 204 | 1.98 | 0.983 | 4 |  |
| de/game.first-sounds.named.watermelon.0.wav | Das ist eine Wassermelone. | Das ist eine Wassermelone. … | Das ist eine Wassermelone. | 568 | 1.39 | 0.987 | 4 |  |
| de/game.first-sounds.match.whale.0.wav | Wal, wie Wurm! | Wal, wie Wurm! … | Wal wie Wurm | 660 | 1.17 | 0.969 | 8 |  |
| de/game.first-sounds.named.whale.0.wav | Das ist ein Wal. | Das ist ein Wal. … | Das ist ein Wal. | 312 | 0.82 | 0.980 | 4 |  |
| de/game.first-sounds.again.0.wav | Fast! | Fast! … | Fast. | 206 | 0.46 | 0.961 | 4 |  |
| de/game.first-sounds.again.1.wav | Probier's noch mal. | Probier's noch mal. … | Probier's nochmal. | 468 | 1.05 | 0.984 | 4 |  |
| de/game.first-sounds.again.2.wav | Guter Versuch! | Guter Versuch! … | Guter Versuch! | 306 | 0.80 | 0.979 | 4 |  |
| de/game.first-sounds.again.3.wav | Noch ein Versuch! | Noch ein Versuch! … | Noch ein Versuch. | 769 | 1.06 | 0.984 | 4 |  |
| de/game.first-sounds.again.4.wav | Beinahe! | Beinahe! … | Beinahe. | 196 | 0.60 | 0.967 | 12 |  |
| de/game.first-sounds.ask.a.0.wav | Hör gut zu: Ameise. Was fängt genauso an? | Hör gut zu: Ameise. Was fängt genauso an? … | Hör gut zu. Ameise. Was fängt genauso an? | 1177 | 3.89 | 0.992 | 4 |  |
| de/game.first-sounds.ask.b.0.wav | Hör gut zu: Baum. Was fängt genauso an? | Hör gut zu: Baum. Was fängt genauso an? … | Hör gut zu. Baum. Was fängt genauso an? | 347 | 3.74 | 0.992 | 4 |  |
| de/game.first-sounds.ask.d.0.wav | Hör gut zu: Dach. Was fängt genauso an? | Hör gut zu: Dach. Was fängt genauso an? … | Hör gut zu. Dach. Was fängt genau so an? | 723 | 2.97 | 0.992 | 4 |  |
| de/game.first-sounds.ask.e.0.wav | Hör gut zu: Esel. Was fängt genauso an? | Hör gut zu: Esel. Was fängt genauso an? … | Hör gut zu. Esel. Was fängt genau so an? | 1128 | 3.53 | 0.993 | 4 |  |
| de/game.first-sounds.ask.f.0.wav | Hör gut zu: Fisch. Was fängt genauso an? | Hör gut zu: Fisch. Was fängt genauso an? … | Hör gut zu. Fisch. Was fängt genauso an? | 328 | 3.34 | 0.992 | 4 |  |
| de/game.first-sounds.ask.g.0.wav | Hör gut zu: Gans. Was fängt genauso an? | Hör gut zu: Gans. Was fängt genauso an? … | Hör gut zu, ganz. Was fängt genau so an? | 379 | 3.03 | 0.993 | 12 | no take met every rule: Whisper, fast |
| de/game.first-sounds.ask.h.0.wav | Hör gut zu: Hund. Was fängt genauso an? | Hör gut zu: Hund. Was fängt genauso an? … | Hör gut zu. Hund. Was fängt genau so an? | 222 | 3.38 | 0.992 | 4 |  |
| de/game.first-sounds.ask.i.0.wav | Hör gut zu: Insel. Was fängt genauso an? | Hör gut zu: Insel. Was fängt genauso an? … | Hör gut zu. Insel. Was fängt genau so an? | 1080 | 3.90 | 0.992 | 4 |  |
| de/game.first-sounds.ask.k.0.wav | Hör gut zu: Kuh. Was fängt genauso an? | Hör gut zu: Kuh. Was fängt genauso an? … | Hör gut zu. Kuh. Was fängt genau so an? | 364 | 3.29 | 0.991 | 4 |  |
| de/game.first-sounds.ask.l.0.wav | Hör gut zu: Löffel. Was fängt genauso an? | Hör gut zu: Löffel. Was fängt genauso an? … | Hör gut zu. Löffel. Was fängt genauso an? | 196 | 3.73 | 0.993 | 4 |  |
| de/game.first-sounds.ask.m.0.wav | Hör gut zu: Maus. Was fängt genauso an? | Hör gut zu: Maus. Was fängt genauso an? … | Hör gut zu. Maus, was fängt genauso an? | 649 | 3.69 | 0.992 | 4 |  |
| de/game.first-sounds.ask.n.0.wav | Hör gut zu: Nase. Was fängt genauso an? | Hör gut zu: Nase. Was fängt genauso an? … | Hör gut zu. Nase. Was fängt genau so an? | 727 | 3.37 | 0.992 | 4 |  |
| de/game.first-sounds.ask.o.0.wav | Hör gut zu: Oma. Was fängt genauso an? | Hör gut zu: Oma. Was fängt genauso an? … | Hör gut zu, Oma. Was fängt genau so an? | 1038 | 3.44 | 0.991 | 4 |  |
| de/game.first-sounds.ask.p.0.wav | Hör gut zu: Puppe. Was fängt genauso an? | Hör gut zu: Puppe. Was fängt genauso an? … | Hör gut zu. Puppe, was fängt genau so an? | 328 | 3.28 | 0.992 | 4 |  |
| de/game.first-sounds.ask.r.0.wav | Hör gut zu: Rose. Was fängt genauso an? | Hör gut zu: Rose. Was fängt genauso an? … | Hör gut zu, Rose. Was fängt genau so an? | 1196 | 3.21 | 0.992 | 4 |  |
| de/game.first-sounds.ask.s.0.wav | Hör gut zu: Sofa. Was fängt genauso an? | Hör gut zu: Sofa. Was fängt genauso an? … | Hör gut zu. Sofa. Was fängt genau so an? | 1132 | 3.80 | 0.991 | 4 |  |
| de/game.first-sounds.ask.t.0.wav | Hör gut zu: Tiger. Was fängt genauso an? | Hör gut zu: Tiger. Was fängt genauso an? … | Hör gut zu. Tiger. Was fängt genau so an? | 301 | 3.45 | 0.992 | 4 |  |
| de/game.first-sounds.ask.u.0.wav | Hör gut zu: Ufo. Was fängt genauso an? | Hör gut zu: Ufo. Was fängt genauso an? … | Hör gut zu, UFO. Was fängt genauso an? | 362 | 3.41 | 0.992 | 4 |  |
| de/game.first-sounds.ask.w.0.wav | Hör gut zu: Wurm. Was fängt genauso an? | Hör gut zu: Wurm. Was fängt genauso an? … | Hör gut zu, Wurm. Was fängt genauso an? | 414 | 3.92 | 0.991 | 4 |  |
| de/game.first-sounds.praise.0.wav | Ja, genau! | Ja, genau! … | Ja, genau. | 922 | 0.84 | 0.979 | 4 |  |
| de/game.first-sounds.praise.1.wav | Toll gehört! | Toll gehört! … | Toll gehört! | 1188 | 0.67 | 0.974 | 4 |  |
| de/game.first-sounds.praise.2.wav | Du hast es gehört! | Du hast es gehört! … | Du hast es gehört. | 1130 | 1.03 | 0.983 | 4 |  |
| de/game.first-sounds.praise.3.wav | Gut gemacht! | Gut gemacht! … | Gut gemacht! | 252 | 0.79 | 0.976 | 4 |  |
| de/game.first-sounds.praise.4.wav | Super Ohren! | Super Ohren! … | Superohren | 460 | 0.87 | 0.974 | 4 |  |
| de/game.first-sounds.praise.5.wav | Richtig! | Richtig! … | Richtig. | 345 | 0.54 | 0.967 | 4 |  |
| de/game.first-sounds.title.0.wav | Laut-Safari! | Laut-Safari! … | Laut Safari | 492 | 1.23 | 0.982 | 4 |  |

Speaker cosine of the chosen takes: min 0.961, median 0.984, max 0.993.

## Lines without a take that met every rule (best available kept)

- de/game.first-sounds.ask.g.0: no take met every rule: Whisper, fast
