# Sound Safari lines v2: one cloned narrator per language, whole takes

Qwen3-TTS 1.7B Base clone (German: competitive-learn female narrator; English: British narrator), temperature 0.6 (was 0.9), 24 kHz mono WAV. Nothing cut, sped up or processed.
German text is spoken with " …" added at the end (in a test it gave quiet at the end in 9 of 20 takes against 0 of 20 without, and slightly slower speech; Whisper still heard every word). English: plain text first, " …" only for retries.
Per line 4 takes (up to 12 when needed). A take counts only if Whisper large-v3 hears the text exactly, at least 150 ms of quiet follow the speech, and it is not more than 15 % faster than the median take of that line. Among those, takes more than 0.03 below the best speaker similarity are dropped, and the most similar one is kept.
Speaker similarity: cosine between the Qwen3 speaker encoder embedding of the take and of the narrator reference.

| file | text | spoken as | Whisper | end quiet ms | speech s | speaker cos | takes | note |
|---|---|---|---|---|---|---|---|---|
| de/game.first-sounds.match.apple.0.wav | Apfel, wie Ameise! | Apfel, wie Ameise! … | Apfel wie Ameise | 165 | 1.73 | 0.983 | 4 |  |
| de/game.first-sounds.named.apple.0.wav | Das ist ein Apfel. | Das ist ein Apfel. … | Das ist ein Apfel. | 151 | 1.16 | 0.978 | 4 |  |
| de/game.first-sounds.match.ball.0.wav | Ball, wie Baum! | Ball, wie Baum! … | Ball wie Baum | 419 | 1.42 | 0.971 | 4 |  |
| de/game.first-sounds.named.ball.0.wav | Das ist ein Ball. | Das ist ein Ball. … | Das ist ein Ball. | 631 | 0.82 | 0.981 | 4 |  |
| de/game.first-sounds.match.banana.0.wav | Banane, wie Baum! | Banane, wie Baum! … | Banane wie Baum | 369 | 1.49 | 0.978 | 4 |  |
| de/game.first-sounds.named.banana.0.wav | Das ist eine Banane. | Das ist eine Banane. … | Das ist eine Banane. | 347 | 1.31 | 0.983 | 8 |  |
| de/game.first-sounds.match.barrel.0.wav | Fass, wie Fisch! | Fass, wie Fisch! … | Fass wie Fisch | 313 | 1.43 | 0.973 | 4 |  |
| de/game.first-sounds.named.barrel.0.wav | Das ist ein Fass. | Das ist ein Fass. … | Das ist ein Fass. | 510 | 1.02 | 0.976 | 4 |  |
| de/game.first-sounds.match.bear.0.wav | Bär, wie Baum! | Bär, wie Baum! … | Bär wie Baum. | 180 | 2.22 | 0.974 | 4 |  |
| de/game.first-sounds.named.bear.0.wav | Das ist ein Bär. | Das ist ein Bär. … | Das ist ein Bär. | 506 | 0.82 | 0.981 | 4 |  |
| de/game.first-sounds.match.bee.0.wav | Biene, wie Baum! | Biene, wie Baum! … | Biene wie Baum. | 176 | 1.16 | 0.977 | 4 |  |
| de/game.first-sounds.named.bee.0.wav | Das ist eine Biene. | Das ist eine Biene. … | Das ist eine Biene. | 343 | 1.18 | 0.982 | 4 |  |
| de/game.first-sounds.match.bicycle.0.wav | Fahrrad, wie Fisch! | Fahrrad, wie Fisch! … | Fahrrad wie Fisch | 382 | 1.55 | 0.976 | 4 |  |
| de/game.first-sounds.named.bicycle.0.wav | Das ist ein Fahrrad. | Das ist ein Fahrrad. … | Das ist ein Fahrrad. | 194 | 1.17 | 0.977 | 8 |  |
| de/game.first-sounds.match.boat.0.wav | Boot, wie Baum! | Boot, wie Baum! … | Boot wie Baum | 220 | 1.45 | 0.969 | 4 |  |
| de/game.first-sounds.named.boat.0.wav | Das ist ein Boot. | Das ist ein Boot. … | Das ist ein Boot. | 160 | 1.12 | 0.981 | 4 |  |
| de/game.first-sounds.match.book.0.wav | Buch, wie Baum! | Buch, wie Baum! … | Buch wie Baum | 383 | 1.45 | 0.976 | 4 |  |
| de/game.first-sounds.named.book.0.wav | Das ist ein Buch. | Das ist ein Buch. … | Das ist ein Buch. | 692 | 0.99 | 0.980 | 4 |  |
| de/game.first-sounds.match.cactus.0.wav | Kaktus, wie Kuh! | Kaktus, wie Kuh! … | Kaktus wie Kuh | 212 | 1.66 | 0.978 | 8 |  |
| de/game.first-sounds.named.cactus.0.wav | Das ist ein Kaktus. | Das ist ein Kaktus. … | Das ist ein Kaktus. | 239 | 1.05 | 0.981 | 4 |  |
| de/game.first-sounds.match.candle.0.wav | Kerze, wie Kuh! | Kerze, wie Kuh! … | Kerze wie Kuh | 239 | 1.13 | 0.978 | 4 |  |
| de/game.first-sounds.named.candle.0.wav | Das ist eine Kerze. | Das ist eine Kerze. … | Das ist eine Kerze. | 375 | 1.22 | 0.983 | 4 |  |
| de/game.first-sounds.match.carrot.0.wav | Möhre, wie Maus! | Möhre, wie Maus! … | Möhre wie Maus | 276 | 1.13 | 0.976 | 4 |  |
| de/game.first-sounds.named.carrot.0.wav | Das ist eine Möhre. | Das ist eine Möhre. … | Das ist eine Möhre. | 215 | 1.01 | 0.984 | 8 |  |
| de/game.first-sounds.match.cat.0.wav | Katze, wie Kuh! | Katze, wie Kuh! … | Katze wie Kuh | 274 | 1.36 | 0.979 | 4 |  |
| de/game.first-sounds.named.cat.0.wav | Das ist eine Katze. | Das ist eine Katze. … | Das ist eine Katze. | 171 | 1.19 | 0.982 | 4 |  |
| de/game.first-sounds.match.cheese.0.wav | Käse, wie Kuh! | Käse, wie Kuh! … | Käse wie Kuh | 424 | 1.12 | 0.977 | 4 |  |
| de/game.first-sounds.named.cheese.0.wav | Das ist Käse. | Das ist Käse. … | Das ist Käse. | 169 | 1.08 | 0.974 | 4 |  |
| de/game.first-sounds.match.clock.0.wav | Uhr, wie Ufo! | Uhr, wie Ufo! … | Uhr wie UFO | 159 | 1.49 | 0.969 | 4 |  |
| de/game.first-sounds.named.clock.0.wav | Das ist eine Uhr. | Das ist eine Uhr. … | Das ist eine Uhr. | 394 | 0.91 | 0.982 | 4 |  |
| de/game.first-sounds.match.cloud.0.wav | Wolke, wie Wurm! | Wolke, wie Wurm! … | Wolke wie Wurm | 179 | 1.07 | 0.977 | 4 |  |
| de/game.first-sounds.named.cloud.0.wav | Das ist eine Wolke. | Das ist eine Wolke. … | Das ist eine Wolke. | 560 | 0.89 | 0.983 | 4 |  |
| de/game.first-sounds.match.cookie.0.wav | Keks, wie Kuh! | Keks, wie Kuh! … | Keks wie Kuh | 227 | 1.13 | 0.973 | 4 |  |
| de/game.first-sounds.named.cookie.0.wav | Das ist ein Keks. | Das ist ein Keks. … | Das ist ein Keks. | 231 | 0.90 | 0.981 | 4 |  |
| de/game.first-sounds.match.corn.0.wav | Mais, wie Maus! | Mais, wie Maus! … | Mais wie Maus. | 400 | 1.38 | 0.978 | 4 |  |
| de/game.first-sounds.named.corn.0.wav | Das ist Mais. | Das ist Mais. … | Das ist Mais. | 373 | 0.91 | 0.976 | 4 |  |
| de/game.first-sounds.match.cup.0.wav | Tasse, wie Tiger! | Tasse, wie Tiger! … | Tasse wie Tiger | 310 | 1.23 | 0.979 | 8 |  |
| de/game.first-sounds.named.cup.0.wav | Das ist eine Tasse. | Das ist eine Tasse. … | Das ist eine Tasse. | 366 | 0.93 | 0.975 | 4 |  |
| de/game.first-sounds.match.dino.0.wav | Dino, wie Dach! | Dino, wie Dach! … | Dino, wie dach? | 355 | 1.16 | 0.981 | 4 |  |
| de/game.first-sounds.named.dino.0.wav | Das ist ein Dino. | Das ist ein Dino. … | Das ist ein Dino. | 221 | 1.03 | 0.983 | 4 |  |
| de/game.first-sounds.match.elephant.0.wav | Elefant, wie Esel! | Elefant, wie Esel! … | Elefant wie Esel | 569 | 1.77 | 0.985 | 4 |  |
| de/game.first-sounds.named.elephant.0.wav | Das ist ein Elefant. | Das ist ein Elefant. … | Das ist ein Elefant. | 541 | 1.00 | 0.983 | 4 |  |
| de/game.first-sounds.match.gift.0.wav | Geschenk, wie Gans! | Geschenk, wie Gans! … | Geschenk wie Gäns | 198 | 1.45 | 0.981 | 4 |  |
| de/game.first-sounds.named.gift.0.wav | Das ist ein Geschenk. | Das ist ein Geschenk. … | Das ist ein Geschenk. | 223 | 1.00 | 0.981 | 4 |  |
| de/game.first-sounds.match.giraffe.0.wav | Giraffe, wie Gans! | Giraffe, wie Gans! … | Giraffe wie Gäns | 410 | 1.79 | 0.983 | 4 |  |
| de/game.first-sounds.named.giraffe.0.wav | Das ist eine Giraffe. | Das ist eine Giraffe. … | Das ist eine Giraffe. | 158 | 0.97 | 0.976 | 4 |  |
| de/game.first-sounds.match.guitar.0.wav | Gitarre, wie Gans! | Gitarre, wie Gans! … | Gitarre wie Gans | 260 | 1.34 | 0.983 | 4 |  |
| de/game.first-sounds.named.guitar.0.wav | Das ist eine Gitarre. | Das ist eine Gitarre. … | Das ist eine Gitarre. | 447 | 1.08 | 0.984 | 8 |  |
| de/game.first-sounds.match.hat.0.wav | Hut, wie Hund! | Hut, wie Hund! … | Hut wie Hund | 381 | 0.80 | 0.977 | 4 |  |
| de/game.first-sounds.named.hat.0.wav | Das ist ein Hut. | Das ist ein Hut. … | Das ist ein Hut. | 340 | 1.52 | 0.983 | 4 |  |
| de/game.first-sounds.match.heart.0.wav | Herz, wie Hund! | Herz, wie Hund! … | Herz wie Hund | 195 | 1.11 | 0.983 | 4 |  |
| de/game.first-sounds.named.heart.0.wav | Das ist ein Herz. | Das ist ein Herz. … | Das ist ein Herz. | 158 | 1.01 | 0.976 | 4 |  |
| de/game.first-sounds.match.hedgehog.0.wav | Igel, wie Insel! | Igel, wie Insel! … | Igel wie Insel | 188 | 1.86 | 0.978 | 8 |  |
| de/game.first-sounds.named.hedgehog.0.wav | Das ist ein Igel. | Das ist ein Igel. … | Das ist ein Igel. | 210 | 1.01 | 0.980 | 4 |  |
| de/game.first-sounds.match.house.0.wav | Haus, wie Hund! | Haus, wie Hund! … | Haus, wie Hund. | 266 | 1.19 | 0.974 | 4 |  |
| de/game.first-sounds.named.house.0.wav | Das ist ein Haus. | Das ist ein Haus. … | Das ist ein Haus. | 317 | 1.00 | 0.974 | 4 |  |
| de/game.first-sounds.match.ladder.0.wav | Leiter, wie Löffel! | Leiter, wie Löffel! … | Leiter wie Löffel | 246 | 1.38 | 0.981 | 8 |  |
| de/game.first-sounds.named.ladder.0.wav | Das ist eine Leiter. | Das ist eine Leiter. … | Das ist eine Leiter. | 262 | 1.13 | 0.981 | 4 |  |
| de/game.first-sounds.match.lamp.0.wav | Lampe, wie Löffel! | Lampe, wie Löffel! … | Lampe wie Löffel | 166 | 1.28 | 0.979 | 4 |  |
| de/game.first-sounds.named.lamp.0.wav | Das ist eine Lampe. | Das ist eine Lampe. … | Das ist eine Lampe. | 208 | 0.97 | 0.982 | 4 |  |
| de/game.first-sounds.match.lantern.0.wav | Laterne, wie Löffel! | Laterne, wie Löffel! … | Laterne wie Löffel | 164 | 1.56 | 0.984 | 4 |  |
| de/game.first-sounds.named.lantern.0.wav | Das ist eine Laterne. | Das ist eine Laterne. … | Das ist eine Laterne. | 181 | 1.11 | 0.977 | 4 |  |
| de/game.first-sounds.match.lion.0.wav | Löwe, wie Löffel! | Löwe, wie Löffel! … | Löwe wie Löffel | 177 | 1.20 | 0.969 | 4 |  |
| de/game.first-sounds.named.lion.0.wav | Das ist ein Löwe. | Das ist ein Löwe. … | Das ist ein Löwe. | 672 | 0.95 | 0.981 | 4 |  |
| de/game.first-sounds.match.monkey.0.wav | Affe, wie Ameise! | Affe, wie Ameise! … | Affe wie Ameise | 161 | 1.45 | 0.979 | 4 |  |
| de/game.first-sounds.named.monkey.0.wav | Das ist ein Affe. | Das ist ein Affe. … | Das ist ein Affe. | 185 | 0.87 | 0.982 | 4 |  |
| de/game.first-sounds.match.moon.0.wav | Mond, wie Maus! | Mond, wie Maus! … | Mond wie Maus | 182 | 1.51 | 0.975 | 8 |  |
| de/game.first-sounds.named.moon.0.wav | Das ist der Mond. | Das ist der Mond. … | Das ist der Mond. | 184 | 0.92 | 0.983 | 4 |  |
| de/game.first-sounds.match.mushroom.0.wav | Pilz, wie Puppe! | Pilz, wie Puppe! … | Pilz wie Puppe | 170 | 1.57 | 0.977 | 4 |  |
| de/game.first-sounds.named.mushroom.0.wav | Das ist ein Pilz. | Das ist ein Pilz. … | Das ist ein Pilz. | 399 | 1.05 | 0.974 | 4 |  |
| de/game.first-sounds.match.nest.0.wav | Nest, wie Nase! | Nest, wie Nase! … | Nest wie Nase | 208 | 1.41 | 0.982 | 4 |  |
| de/game.first-sounds.named.nest.0.wav | Das ist ein Nest. | Das ist ein Nest. … | Das ist ein Nest. | 177 | 1.12 | 0.978 | 4 |  |
| de/game.first-sounds.match.octopus.0.wav | Oktopus, wie Oma! | Oktopus, wie Oma! … | Oktopus wie Oma | 385 | 1.86 | 0.982 | 4 |  |
| de/game.first-sounds.named.octopus.0.wav | Das ist ein Oktopus. | Das ist ein Oktopus. … | Das ist ein Oktopus. | 159 | 1.15 | 0.983 | 4 |  |
| de/game.first-sounds.match.orange.0.wav | Orange, wie Oma! | Orange, wie Oma! … | Orange wie Oma. | 526 | 1.85 | 0.978 | 4 |  |
| de/game.first-sounds.named.orange.0.wav | Das ist eine Orange. | Das ist eine Orange. … | Das ist eine Orange. | 531 | 1.08 | 0.984 | 4 |  |
| de/game.first-sounds.match.pear.0.wav | Birne, wie Baum! | Birne, wie Baum! … | Birne wie Baum | 243 | 1.20 | 0.980 | 4 |  |
| de/game.first-sounds.named.pear.0.wav | Das ist eine Birne. | Das ist eine Birne. … | Das ist eine Birne. | 185 | 1.03 | 0.983 | 4 |  |
| de/game.first-sounds.match.penguin.0.wav | Pinguin, wie Puppe! | Pinguin, wie Puppe! … | Pinguin wie Puppe | 155 | 1.14 | 0.975 | 8 |  |
| de/game.first-sounds.named.penguin.0.wav | Das ist ein Pinguin. | Das ist ein Pinguin. … | Das ist ein Pinguin. | 705 | 1.07 | 0.984 | 4 |  |
| de/game.first-sounds.match.pizza.0.wav | Pizza, wie Puppe! | Pizza, wie Puppe! … | Pizza wie Puppe | 152 | 1.54 | 0.979 | 4 |  |
| de/game.first-sounds.named.pizza.0.wav | Das ist eine Pizza. | Das ist eine Pizza. … | Das ist eine Pizza. | 310 | 1.10 | 0.982 | 4 |  |
| de/game.first-sounds.match.rabbit.0.wav | Hase, wie Hund! | Hase, wie Hund! … | Hase wie Hund | 322 | 1.11 | 0.982 | 4 |  |
| de/game.first-sounds.named.rabbit.0.wav | Das ist ein Hase. | Das ist ein Hase. … | Das ist ein Hase. | 174 | 0.89 | 0.980 | 4 |  |
| de/game.first-sounds.match.rainbow.0.wav | Regenbogen, wie Rose! | Regenbogen, wie Rose! … | Regenbogen wie Rose | 173 | 1.46 | 0.978 | 4 |  |
| de/game.first-sounds.named.rainbow.0.wav | Das ist ein Regenbogen. | Das ist ein Regenbogen. … | Das ist ein Regenbogen. | 418 | 1.11 | 0.981 | 4 |  |
| de/game.first-sounds.match.robot.0.wav | Roboter, wie Rose! | Roboter, wie Rose! … | Roboter wie Rose. | 519 | 1.49 | 0.981 | 4 |  |
| de/game.first-sounds.named.robot.0.wav | Das ist ein Roboter. | Das ist ein Roboter. … | Das ist ein Roboter. | 186 | 1.17 | 0.983 | 4 |  |
| de/game.first-sounds.match.rocket.0.wav | Rakete, wie Rose! | Rakete, wie Rose! … | Rakete wie Rose | 414 | 1.50 | 0.981 | 4 |  |
| de/game.first-sounds.named.rocket.0.wav | Das ist eine Rakete. | Das ist eine Rakete. … | Das ist eine Rakete. | 334 | 1.12 | 0.981 | 4 |  |
| de/game.first-sounds.match.soap.0.wav | Seife, wie Sofa! | Seife, wie Sofa! … | Seife wie Sofa | 240 | 2.29 | 0.979 | 4 |  |
| de/game.first-sounds.named.soap.0.wav | Das ist eine Seife. | Das ist eine Seife. … | Das ist eine Seife. | 160 | 0.96 | 0.976 | 4 |  |
| de/game.first-sounds.match.strawberry.0.wav | Erdbeere, wie Esel! | Erdbeere, wie Esel! … | Erdbeere wie Esel | 152 | 1.89 | 0.981 | 4 |  |
| de/game.first-sounds.named.strawberry.0.wav | Das ist eine Erdbeere. | Das ist eine Erdbeere. … | Das ist eine Erdbeere. | 156 | 1.41 | 0.981 | 4 |  |
| de/game.first-sounds.match.submarine.0.wav | U-Boot, wie Ufo! | U-Boot, wie Ufo! … | U-Boot wie UFO | 199 | 1.64 | 0.974 | 4 |  |
| de/game.first-sounds.named.submarine.0.wav | Das ist ein U-Boot. | Das ist ein U-Boot. … | Das ist ein U-Boot. | 162 | 0.97 | 0.981 | 4 |  |
| de/game.first-sounds.match.sun.0.wav | Sonne, wie Sofa! | Sonne, wie Sofa! … | Sonne, wie Sofa. | 310 | 1.57 | 0.987 | 4 |  |
| de/game.first-sounds.named.sun.0.wav | Das ist die Sonne. | Das ist die Sonne. … | Das ist die Sonne. | 232 | 0.88 | 0.982 | 4 |  |
| de/game.first-sounds.match.table.0.wav | Tisch, wie Tiger! | Tisch, wie Tiger! … | Tisch wie Tiger | 498 | 1.51 | 0.976 | 4 |  |
| de/game.first-sounds.named.table.0.wav | Das ist ein Tisch. | Das ist ein Tisch. … | Das ist ein Tisch. | 385 | 0.98 | 0.976 | 4 |  |
| de/game.first-sounds.match.teddy.0.wav | Teddy, wie Tiger! | Teddy, wie Tiger! … | Teddy, wie Tiger | 198 | 1.65 | 0.979 | 4 |  |
| de/game.first-sounds.named.teddy.0.wav | Das ist ein Teddy. | Das ist ein Teddy. … | Das ist ein Teddy. | 387 | 1.51 | 0.980 | 4 |  |
| de/game.first-sounds.match.tomato.0.wav | Tomate, wie Tiger! | Tomate, wie Tiger! … | Tomate wie Tiger | 208 | 1.52 | 0.982 | 8 |  |
| de/game.first-sounds.named.tomato.0.wav | Das ist eine Tomate. | Das ist eine Tomate. … | Das ist eine Tomate. | 174 | 1.12 | 0.982 | 12 |  |
| de/game.first-sounds.match.trophy.0.wav | Pokal, wie Puppe! | Pokal, wie Puppe! … | Pokal wie Puppe | 403 | 1.20 | 0.971 | 4 |  |
| de/game.first-sounds.named.trophy.0.wav | Das ist ein Pokal. | Das ist ein Pokal. … | Das ist ein Pokal. | 292 | 1.17 | 0.982 | 4 |  |
| de/game.first-sounds.match.watermelon.0.wav | Wassermelone, wie Wurm! | Wassermelone, wie Wurm! … | Wassermelone wie Wurm | 192 | 1.85 | 0.985 | 4 |  |
| de/game.first-sounds.named.watermelon.0.wav | Das ist eine Wassermelone. | Das ist eine Wassermelone. … | Das ist eine Wassermelone. | 632 | 1.37 | 0.986 | 4 |  |
| de/game.first-sounds.match.whale.0.wav | Wal, wie Wurm! | Wal, wie Wurm! … | Wal wie Wurm | 246 | 1.08 | 0.969 | 12 |  |
| de/game.first-sounds.named.whale.0.wav | Das ist ein Wal. | Das ist ein Wal. … | Das ist ein Wal. | 331 | 0.88 | 0.980 | 4 |  |
| en/game.first-sounds.match.ball.0.wav | Ball, like bird! | Ball, like bird! | Ball like bird | 154 | 1.56 | 0.979 | 4 |  |
| en/game.first-sounds.match.banana.0.wav | Banana, like bird! | Banana, like bird! | Banana, like bird. | 186 | 1.35 | 0.984 | 4 |  |
| en/game.first-sounds.match.bear.0.wav | Bear, like bird! | Bear, like bird! … | Bear like bird. | 195 | 1.04 | 0.977 | 8 |  |
| en/game.first-sounds.match.bee.0.wav | Bee, like bird! | Bee, like bird! … | B like bird | 181 | 1.69 | 0.978 | 12 | no take met every rule: Whisper |
| en/game.first-sounds.match.bell.0.wav | Bell, like bird! | Bell, like bird! … | Bell like bird. | 181 | 1.29 | 0.972 | 8 |  |
| en/game.first-sounds.match.boat.0.wav | Boat, like bird! | Boat, like bird! | Boat, like bird. | 166 | 1.37 | 0.977 | 4 |  |
| en/game.first-sounds.match.book.0.wav | Book, like bird! | Book, like bird! | Book like bird. | 169 | 1.14 | 0.973 | 4 |  |
| en/game.first-sounds.match.bus.0.wav | Bus, like bird! | Bus, like bird! | Bus? Like bird? | 208 | 1.33 | 0.980 | 4 |  |
| en/game.first-sounds.match.candle.0.wav | Candle, like kangaroo! | Candle, like kangaroo! | Candle, like kangaroo? | 207 | 1.62 | 0.985 | 4 |  |
| en/game.first-sounds.match.car.0.wav | Car, like kangaroo! | Car, like kangaroo! | Car like kangaroo | 185 | 1.44 | 0.981 | 4 |  |
| en/game.first-sounds.match.carrot.0.wav | Carrot, like kangaroo! | Carrot, like kangaroo! | Carrot, like kangaroo. | 237 | 1.56 | 0.973 | 4 |  |
| en/game.first-sounds.match.cat.0.wav | Cat, like kangaroo! | Cat, like kangaroo! | Cat, like kangaroo. | 175 | 1.56 | 0.979 | 4 |  |
| en/game.first-sounds.match.cookie.0.wav | Cookie, like kangaroo! | Cookie, like kangaroo! | Cookie, like kangaroo? | 181 | 1.51 | 0.980 | 4 |  |
| en/game.first-sounds.match.cup.0.wav | Cup, like kangaroo! | Cup, like kangaroo! | Cup like kangaroo. | 162 | 1.55 | 0.977 | 4 |  |
| en/game.first-sounds.match.key.0.wav | Key, like kangaroo! | Key, like kangaroo! … | key like kangaroo | 178 | 1.61 | 0.981 | 8 |  |
| en/game.first-sounds.match.kite.0.wav | Kite, like kangaroo! | Kite, like kangaroo! | Kite? Like kangaroo! | 180 | 1.38 | 0.981 | 4 |  |
| en/game.first-sounds.match.dice.0.wav | Dice, like dog! | Dice, like dog! | Dice like dog | 173 | 1.63 | 0.977 | 4 |  |
| en/game.first-sounds.match.door.0.wav | Door, like dog! | Door, like dog! | Door. Like dog. | 180 | 1.90 | 0.970 | 4 |  |
| en/game.first-sounds.match.elephant.0.wav | Elephant, like egg! | Elephant, like egg! | Elephant, like egg! | 156 | 1.74 | 0.978 | 4 |  |
| en/game.first-sounds.match.gift.0.wav | Present, like pig! | Present, like pig! | Present, like pig! | 159 | 1.31 | 0.983 | 4 |  |
| en/game.first-sounds.match.pear.0.wav | Pear, like pig! | Pear, like pig! | Peer, like pig. | 173 | 1.55 | 0.977 | 12 | no take met every rule: Whisper |
| en/game.first-sounds.match.pencil.0.wav | Pencil, like pig! | Pencil, like pig! | Pencil, like pig. | 154 | 1.57 | 0.977 | 4 |  |
| en/game.first-sounds.match.penguin.0.wav | Penguin, like pig! | Penguin, like pig! | Penguin, like pig. | 162 | 1.57 | 0.981 | 4 |  |
| en/game.first-sounds.match.pizza.0.wav | Pizza, like pig! | Pizza, like pig! | Pizza, like pig. | 203 | 1.50 | 0.979 | 4 |  |
| en/game.first-sounds.match.pumpkin.0.wav | Pumpkin, like pig! | Pumpkin, like pig! | Pumpkin, like pig. | 159 | 1.38 | 0.979 | 4 |  |
| en/game.first-sounds.match.guitar.0.wav | Guitar, like goat! | Guitar, like goat! | Guitar, like goat. | 158 | 1.48 | 0.978 | 4 |  |
| en/game.first-sounds.match.hat.0.wav | Hat, like horse! | Hat, like horse! | Hat? Like horse! | 182 | 1.41 | 0.971 | 4 |  |
| en/game.first-sounds.match.heart.0.wav | Heart, like horse! | Heart, like horse! … | Heart, like horse. | 260 | 1.51 | 0.976 | 8 |  |
| en/game.first-sounds.match.hedgehog.0.wav | Hedgehog, like horse! | Hedgehog, like horse! … | Hedgehog, like horse. | 242 | 1.36 | 0.978 | 8 |  |
| en/game.first-sounds.match.house.0.wav | House, like horse! | House, like horse! … | House, like horse! | 260 | 1.42 | 0.970 | 8 |  |
| en/game.first-sounds.match.ladder.0.wav | Ladder, like leaf! | Ladder, like leaf! | Ladder? Like leaf? | 162 | 1.47 | 0.979 | 4 |  |
| en/game.first-sounds.match.lamp.0.wav | Lamp, like leaf! | Lamp, like leaf! | Lamp, like leaf! | 327 | 1.28 | 0.976 | 4 |  |
| en/game.first-sounds.match.lion.0.wav | Lion, like leaf! | Lion, like leaf! | Lion like leaf. | 989 | 1.52 | 0.981 | 4 |  |
| en/game.first-sounds.match.lollipop.0.wav | Lollipop, like leaf! | Lollipop, like leaf! | Lollipop, like leaf. | 277 | 1.51 | 0.978 | 4 |  |
| en/game.first-sounds.match.mittens.0.wav | Mittens, like mouse! | Mittens, like mouse! … | Mittens, like mouse. | 158 | 1.71 | 0.980 | 12 |  |
| en/game.first-sounds.match.monkey.0.wav | Monkey, like mouse! | Monkey, like mouse! … | Monkey, like mouse. | 184 | 1.77 | 0.971 | 8 |  |
| en/game.first-sounds.match.moon.0.wav | Moon, like mouse! | Moon, like mouse! | Moon like mouse. | 596 | 1.62 | 0.982 | 4 |  |
| en/game.first-sounds.match.mushroom.0.wav | Mushroom, like mouse! | Mushroom, like mouse! | Mushroom, like mouse. | 195 | 1.69 | 0.984 | 4 |  |
| en/game.first-sounds.match.nest.0.wav | Nest, like net! | Nest, like net! | Nest, like net. | 434 | 1.57 | 0.977 | 4 |  |
| en/game.first-sounds.match.octopus.0.wav | Octopus, like otter! | Octopus, like otter! | Octopus, like otter! | 181 | 1.91 | 0.980 | 4 |  |
| en/game.first-sounds.match.orange.0.wav | Orange, like otter! | Orange, like otter! … | Orange, like otter. | 158 | 1.28 | 0.982 | 8 |  |
| en/game.first-sounds.match.rabbit.0.wav | Rabbit, like rain! | Rabbit, like rain! | Rabbit, like rain. | 194 | 1.58 | 0.982 | 4 |  |
| en/game.first-sounds.match.rainbow.0.wav | Rainbow, like rain! | Rainbow, like rain! | Rainbow, like rain. | 203 | 1.86 | 0.977 | 4 |  |
| en/game.first-sounds.match.rocket.0.wav | Rocket, like rain! | Rocket, like rain! | Rocket like rain? | 211 | 1.30 | 0.981 | 4 |  |
| en/game.first-sounds.match.sandwich.0.wav | Sandwich, like snake! | Sandwich, like snake! | Sandwich, like snake. | 159 | 1.71 | 0.975 | 4 |  |
| en/game.first-sounds.match.soap.0.wav | Soap, like snake! | Soap, like snake! | Soap, like snake. | 179 | 1.37 | 0.979 | 4 |  |
| en/game.first-sounds.match.sun.0.wav | Sun, like snake! | Sun, like snake! | Sun! Like snake! | 170 | 1.46 | 0.980 | 4 |  |
| en/game.first-sounds.match.table.0.wav | Table, like tiger! | Table, like tiger! | Table like tiger. | 199 | 1.39 | 0.980 | 4 |  |
| en/game.first-sounds.match.teddy.0.wav | Teddy, like tiger! | Teddy, like tiger! | Teddy, like tiger. | 217 | 1.62 | 0.980 | 4 |  |
| en/game.first-sounds.match.tent.0.wav | Tent, like tiger! | Tent, like tiger! | Tent? Like Tiger? | 164 | 1.52 | 0.982 | 4 |  |
| en/game.first-sounds.match.tomato.0.wav | Tomato, like tiger! | Tomato, like tiger! | Tomato like tiger | 186 | 1.72 | 0.983 | 4 |  |
| en/game.first-sounds.match.turtle.0.wav | Turtle, like tiger! | Turtle, like tiger! | Turtle, like tiger. | 152 | 1.63 | 0.974 | 4 |  |
| en/game.first-sounds.match.umbrella.0.wav | Umbrella, like up! | Umbrella, like up! | Umbrella, like up! | 305 | 1.39 | 0.979 | 4 |  |
| en/game.first-sounds.match.watermelon.0.wav | Watermelon, like web! | Watermelon, like web! … | Watermelon like web. | 171 | 1.96 | 0.977 | 8 |  |
| en/game.first-sounds.match.whale.0.wav | Whale, like web! | Whale, like web! | Whale, like web. | 174 | 1.80 | 0.972 | 4 |  |
| en/game.first-sounds.match.windmill.0.wav | Windmill, like web! | Windmill, like web! | Windmill, like web. | 162 | 1.69 | 0.976 | 4 |  |
| en/game.first-sounds.match.apple.0.wav | Apple, like ant! | Apple, like ant! … | Apple like ant. | 187 | 2.06 | 0.973 | 8 |  |
| en/game.first-sounds.named.apple.0.wav | That's an apple. | That's an apple. | That's an apple. | 162 | 0.76 | 0.976 | 4 |  |
| en/game.first-sounds.named.ball.0.wav | That's a ball. | That's a ball. | That's a ball! | 262 | 0.90 | 0.971 | 4 |  |
| en/game.first-sounds.named.banana.0.wav | That's a banana. | That's a banana. | That's a banana! | 187 | 1.13 | 0.975 | 4 |  |
| en/game.first-sounds.named.bear.0.wav | That's a bear. | That's a bear. | That's a bear! | 190 | 0.90 | 0.974 | 4 |  |
| en/game.first-sounds.named.bee.0.wav | That's a bee. | That's a bee. | That's a bee. | 176 | 1.28 | 0.974 | 4 |  |
| en/game.first-sounds.named.bell.0.wav | That's a bell. | That's a bell. | That's a bell. | 181 | 0.58 | 0.970 | 4 |  |
| en/game.first-sounds.named.boat.0.wav | That's a boat. | That's a boat. | That's a boat! | 160 | 0.75 | 0.970 | 4 |  |
| en/game.first-sounds.named.book.0.wav | That's a book. | That's a book. | That's a book. | 180 | 0.71 | 0.968 | 4 |  |
| en/game.first-sounds.named.bus.0.wav | That's a bus. | That's a bus. | That's a bus. | 162 | 0.65 | 0.967 | 4 |  |
| en/game.first-sounds.named.candle.0.wav | That's a candle. | That's a candle. | That's a candle. | 157 | 0.73 | 0.981 | 4 |  |
| en/game.first-sounds.named.car.0.wav | That's a car. | That's a car. | That's a car. | 163 | 0.84 | 0.971 | 4 |  |
| en/game.first-sounds.named.carrot.0.wav | That's a carrot. | That's a carrot. | That's a carrot. | 169 | 0.66 | 0.972 | 4 |  |
| en/game.first-sounds.named.cat.0.wav | That's a cat. | That's a cat. | That's a cat. | 185 | 0.65 | 0.960 | 4 |  |
| en/game.first-sounds.named.cookie.0.wav | That's a cookie. | That's a cookie. | That's a cookie. | 213 | 0.68 | 0.971 | 4 |  |
| en/game.first-sounds.named.cup.0.wav | That's a cup. | That's a cup. … | That's a cup. | 200 | 0.55 | 0.965 | 8 |  |
| en/game.first-sounds.named.dice.0.wav | That's a dice. | That's a dice. … | That's a dice! | 293 | 0.79 | 0.972 | 8 |  |
| en/game.first-sounds.named.door.0.wav | That's a door. | That's a door. | That's a door. | 173 | 0.63 | 0.973 | 4 |  |
| en/game.first-sounds.named.egg.0.wav | That's an egg. | That's an egg. | That's an egg! | 177 | 0.78 | 0.978 | 4 |  |
| en/game.first-sounds.named.elephant.0.wav | That's an elephant. | That's an elephant. | That's an elephant. | 169 | 0.95 | 0.971 | 4 |  |
| en/game.first-sounds.named.gift.0.wav | That's a present. | That's a present. … | That's a present. | 198 | 0.89 | 0.971 | 12 |  |
| en/game.first-sounds.named.guitar.0.wav | That's a guitar. | That's a guitar. | That's a guitar! | 159 | 0.80 | 0.975 | 4 |  |
| en/game.first-sounds.named.hat.0.wav | That's a hat. | That's a hat. | That's a hat. | 160 | 0.73 | 0.962 | 4 |  |
| en/game.first-sounds.named.heart.0.wav | That's a heart. | That's a heart. | That's a heart. | 174 | 0.75 | 0.963 | 4 |  |
| en/game.first-sounds.named.hedgehog.0.wav | That's a hedgehog. | That's a hedgehog. | That's a hedgehog. | 183 | 1.23 | 0.972 | 4 |  |
| en/game.first-sounds.named.house.0.wav | That's a house. | That's a house. … | That's a house. | 358 | 0.90 | 0.971 | 8 |  |
| en/game.first-sounds.named.key.0.wav | That's a key. | That's a key. | That's a key. | 178 | 0.74 | 0.972 | 4 |  |
| en/game.first-sounds.named.kite.0.wav | That's a kite. | That's a kite. | That's a kite! | 183 | 0.63 | 0.964 | 4 |  |
| en/game.first-sounds.named.ladder.0.wav | That's a ladder. | That's a ladder. | That's a ladder! | 151 | 0.67 | 0.974 | 4 |  |
| en/game.first-sounds.named.lamp.0.wav | That's a lamp. | That's a lamp. … | That's a lamp. | 201 | 0.76 | 0.968 | 8 |  |
| en/game.first-sounds.named.lion.0.wav | That's a lion. | That's a lion. | That's a lion! | 194 | 0.94 | 0.979 | 4 |  |
| en/game.first-sounds.named.lollipop.0.wav | That's a lollipop. | That's a lollipop. | That's a lollipop? | 188 | 0.91 | 0.975 | 4 |  |
| en/game.first-sounds.named.mittens.0.wav | Those are mittens. | Those are mittens. | Those are mittens. | 181 | 0.91 | 0.978 | 4 |  |
| en/game.first-sounds.named.monkey.0.wav | That's a monkey. | That's a monkey. … | That's a monkey. | 167 | 0.99 | 0.974 | 8 |  |
| en/game.first-sounds.named.moon.0.wav | That's the moon. | That's the moon. | That's the moon. | 210 | 0.86 | 0.978 | 4 |  |
| en/game.first-sounds.named.mushroom.0.wav | That's a mushroom. | That's a mushroom. | That's a mushroom. | 154 | 0.85 | 0.972 | 4 |  |
| en/game.first-sounds.named.nest.0.wav | That's a nest. | That's a nest. … | That's a nest. | 178 | 0.80 | 0.973 | 8 |  |
| en/game.first-sounds.named.octopus.0.wav | That's an octopus. | That's an octopus. | That's an octopus. | 247 | 1.41 | 0.978 | 4 |  |
| en/game.first-sounds.named.orange.0.wav | That's an orange. | That's an orange. | That's an orange. | 200 | 0.92 | 0.979 | 4 |  |
| en/game.first-sounds.named.pear.0.wav | That's a pear. | That's a pear. | That's a pear. | 169 | 0.67 | 0.967 | 4 |  |
| en/game.first-sounds.named.pencil.0.wav | That's a pencil. | That's a pencil. | That's a pencil! | 154 | 0.78 | 0.968 | 4 |  |
| en/game.first-sounds.named.penguin.0.wav | That's a penguin. | That's a penguin. | That's a penguin | 157 | 0.89 | 0.979 | 4 |  |
| en/game.first-sounds.named.pizza.0.wav | That's a pizza. | That's a pizza. | That's a pizza. | 164 | 0.71 | 0.969 | 4 |  |
| en/game.first-sounds.named.pumpkin.0.wav | That's a pumpkin. | That's a pumpkin. … | That's a pumpkin. | 230 | 0.82 | 0.977 | 8 |  |
| en/game.first-sounds.named.rabbit.0.wav | That's a rabbit. | That's a rabbit. | That's a rabbit. | 256 | 0.92 | 0.966 | 4 |  |
| en/game.first-sounds.named.rainbow.0.wav | That's a rainbow. | That's a rainbow. | That's a rainbow! | 166 | 0.84 | 0.980 | 4 |  |
| en/game.first-sounds.named.rocket.0.wav | That's a rocket. | That's a rocket. … | That's a rocket. | 164 | 0.99 | 0.977 | 12 |  |
| en/game.first-sounds.named.sandwich.0.wav | That's a sandwich. | That's a sandwich. | That's a sandwich. | 162 | 1.00 | 0.970 | 4 |  |
| en/game.first-sounds.named.soap.0.wav | That's soap. | That's soap. | That's soap! | 168 | 0.56 | 0.967 | 4 |  |
| en/game.first-sounds.named.sun.0.wav | That's the sun. | That's the sun. | That's the sun. | 178 | 0.71 | 0.972 | 4 |  |
| en/game.first-sounds.named.table.0.wav | That's a table. | That's a table. | That's a table. | 309 | 0.78 | 0.973 | 4 |  |
| en/game.first-sounds.named.teddy.0.wav | That's a teddy. | That's a teddy. | That's a teddy! | 176 | 0.71 | 0.974 | 4 |  |
| en/game.first-sounds.named.tent.0.wav | That's a tent. | That's a tent. | That's a tent. | 193 | 0.62 | 0.972 | 4 |  |
| en/game.first-sounds.named.tomato.0.wav | That's a tomato. | That's a tomato. | That's a tomato. | 179 | 0.79 | 0.980 | 4 |  |
| en/game.first-sounds.named.turtle.0.wav | That's a turtle. | That's a turtle. | That's a turtle. | 192 | 0.70 | 0.964 | 4 |  |
| en/game.first-sounds.named.umbrella.0.wav | That's an umbrella. | That's an umbrella. | That's an umbrella! | 221 | 1.02 | 0.975 | 4 |  |
| en/game.first-sounds.named.watermelon.0.wav | That's a watermelon. | That's a watermelon. | That's a watermelon. | 171 | 1.04 | 0.976 | 4 |  |
| en/game.first-sounds.named.whale.0.wav | That's a whale. | That's a whale. | That's a whale. | 196 | 0.71 | 0.973 | 4 |  |
| en/game.first-sounds.named.windmill.0.wav | That's a windmill. | That's a windmill. | That's a windmill. | 184 | 0.93 | 0.982 | 4 |  |
| de/game.first-sounds.again.0.wav | Fast! | Fast! … | Fast. | 161 | 0.47 | 0.938 | 4 |  |
| de/game.first-sounds.again.1.wav | Probier's noch mal. | Probier's noch mal. … | Probier's nochmal. | 213 | 0.92 | 0.979 | 8 |  |
| de/game.first-sounds.again.2.wav | Guter Versuch! | Guter Versuch! … | Guter Versuch! | 240 | 0.82 | 0.975 | 4 |  |
| de/game.first-sounds.again.3.wav | Noch ein Versuch! | Noch ein Versuch! … | Noch ein Versuch. | 841 | 1.01 | 0.986 | 4 |  |
| de/game.first-sounds.again.4.wav | Beinahe! | Beinahe! … | Beinahe. | 243 | 0.56 | 0.948 | 4 |  |
| de/game.first-sounds.ask.a.0.wav | Hör gut zu: Ameise. Was fängt genauso an? | Hör gut zu: Ameise. Was fängt genauso an? … | Hör gut zu! Ameise. Was fängt genau so an? | 335 | 3.55 | 0.991 | 4 |  |
| de/game.first-sounds.ask.b.0.wav | Hör gut zu: Baum. Was fängt genauso an? | Hör gut zu: Baum. Was fängt genauso an? … | Hör gut zu. Baum. Was fängt genau so an? | 179 | 3.27 | 0.988 | 4 |  |
| de/game.first-sounds.ask.d.0.wav | Hör gut zu: Dach. Was fängt genauso an? | Hör gut zu: Dach. Was fängt genauso an? … | Hör gut zu! Dach, was fängt genau so an? | 196 | 3.13 | 0.989 | 4 |  |
| de/game.first-sounds.ask.e.0.wav | Hör gut zu: Esel. Was fängt genauso an? | Hör gut zu: Esel. Was fängt genauso an? … | Hör gut zu. Esel. Was fängt genau so an? | 165 | 2.95 | 0.989 | 4 |  |
| de/game.first-sounds.ask.f.0.wav | Hör gut zu: Fisch. Was fängt genauso an? | Hör gut zu: Fisch. Was fängt genauso an? … | Hör gut zu, Fisch. Was fängt genau so an? | 317 | 3.32 | 0.988 | 4 |  |
| de/game.first-sounds.ask.g.0.wav | Hör gut zu: Gans. Was fängt genauso an? | Hör gut zu: Gans. Was fängt genauso an? … | Hör gut zu. Gans, was fängt genau so an? | 424 | 3.65 | 0.991 | 4 |  |
| de/game.first-sounds.ask.h.0.wav | Hör gut zu: Hund. Was fängt genauso an? | Hör gut zu: Hund. Was fängt genauso an? … | Hör gut zu! Hund, was fängt genau so an? | 285 | 3.35 | 0.989 | 4 |  |
| de/game.first-sounds.ask.i.0.wav | Hör gut zu: Insel. Was fängt genauso an? | Hör gut zu: Insel. Was fängt genauso an? … | Hör gut zu! Insel. Was fängt genau so an? | 276 | 3.48 | 0.988 | 4 |  |
| de/game.first-sounds.ask.k.0.wav | Hör gut zu: Kuh. Was fängt genauso an? | Hör gut zu: Kuh. Was fängt genauso an? … | Hör gut zu! Kuh, was fängt genau so an? | 555 | 2.96 | 0.987 | 4 |  |
| de/game.first-sounds.ask.l.0.wav | Hör gut zu: Löffel. Was fängt genauso an? | Hör gut zu: Löffel. Was fängt genauso an? … | Hör gut zu! Löffel. Was fängt genau so an? | 162 | 3.26 | 0.990 | 4 |  |
| de/game.first-sounds.ask.m.0.wav | Hör gut zu: Maus. Was fängt genauso an? | Hör gut zu: Maus. Was fängt genauso an? … | Hör gut zu, Maus. Was fängt genau so an? | 163 | 2.95 | 0.990 | 4 |  |
| de/game.first-sounds.ask.n.0.wav | Hör gut zu: Nase. Was fängt genauso an? | Hör gut zu: Nase. Was fängt genauso an? … | Hör gut zu, Nase. Was fängt genau so an? | 209 | 3.60 | 0.990 | 4 |  |
| de/game.first-sounds.ask.o.0.wav | Hör gut zu: Oma. Was fängt genauso an? | Hör gut zu: Oma. Was fängt genauso an? … | Hör gut zu. Oma, was fängt genau so an? | 221 | 3.03 | 0.990 | 4 |  |
| de/game.first-sounds.ask.p.0.wav | Hör gut zu: Puppe. Was fängt genauso an? | Hör gut zu: Puppe. Was fängt genauso an? … | Hör gut zu! Puppe. Was fängt genau so an? | 180 | 3.22 | 0.989 | 4 |  |
| de/game.first-sounds.ask.r.0.wav | Hör gut zu: Rose. Was fängt genauso an? | Hör gut zu: Rose. Was fängt genauso an? … | Hör gut zu, Rose. Was fängt genau so an? | 319 | 3.30 | 0.987 | 4 |  |
| de/game.first-sounds.ask.s.0.wav | Hör gut zu: Sofa. Was fängt genauso an? | Hör gut zu: Sofa. Was fängt genauso an? … | Hör gut zu, Sofa. Was fängt genau so an? | 216 | 3.13 | 0.990 | 4 |  |
| de/game.first-sounds.ask.t.0.wav | Hör gut zu: Tiger. Was fängt genauso an? | Hör gut zu: Tiger. Was fängt genauso an? … | Hör gut zu. Tiger. Was fängt genau so an? | 166 | 2.98 | 0.989 | 4 |  |
| de/game.first-sounds.ask.u.0.wav | Hör gut zu: Ufo. Was fängt genauso an? | Hör gut zu: Ufo. Was fängt genauso an? … | Hör gut zu, UFO. Was fängt genau so an? | 165 | 2.98 | 0.987 | 4 |  |
| de/game.first-sounds.ask.w.0.wav | Hör gut zu: Wurm. Was fängt genauso an? | Hör gut zu: Wurm. Was fängt genauso an? … | Hör gut zu. Wurm. Was fängt genau so an? | 254 | 3.51 | 0.988 | 4 |  |
| de/game.first-sounds.praise.0.wav | Ja, genau! | Ja, genau! … | Ja, genau. | 702 | 1.02 | 0.971 | 4 |  |
| de/game.first-sounds.praise.1.wav | Toll gehört! | Toll gehört! … | Toll gehört! | 276 | 0.81 | 0.972 | 4 |  |
| de/game.first-sounds.praise.2.wav | Du hast es gehört! | Du hast es gehört! … | Du hast es gehört. | 266 | 0.94 | 0.975 | 4 |  |
| de/game.first-sounds.praise.3.wav | Gut gemacht! | Gut gemacht! … | Gut gemacht! | 165 | 0.71 | 0.974 | 4 |  |
| de/game.first-sounds.praise.4.wav | Super Ohren! | Super Ohren! … | Super Ohren | 182 | 0.91 | 0.978 | 4 |  |
| de/game.first-sounds.praise.5.wav | Richtig! | Richtig! … | Richtig. | 320 | 0.48 | 0.964 | 4 |  |
| de/game.first-sounds.title.0.wav | Laut-Safari! | Laut-Safari! … | Laut Safari | 424 | 1.22 | 0.972 | 4 |  |
| en/game.first-sounds.again.0.wav | Nearly! | Nearly! | Nearly! | 227 | 0.58 | 0.957 | 4 |  |
| en/game.first-sounds.again.1.wav | Try again. | Try again. | Try again. | 194 | 0.88 | 0.972 | 4 |  |
| en/game.first-sounds.again.2.wav | Good try! | Good try! | Good try! | 185 | 0.50 | 0.965 | 4 |  |
| en/game.first-sounds.again.3.wav | One more go! | One more go! | One more go! | 152 | 0.78 | 0.972 | 4 |  |
| en/game.first-sounds.again.4.wav | Almost! | Almost! … | Almost. | 159 | 0.59 | 0.963 | 8 |  |
| en/game.first-sounds.ask.a.0.wav | Listen: ant. Which one starts the same? | Listen: ant. Which one starts the same? | Listen, Ant, which one starts the same? | 204 | 2.56 | 0.985 | 4 |  |
| en/game.first-sounds.ask.b.0.wav | Listen: bird. Which one starts the same? | Listen: bird. Which one starts the same? | Listen, bird, which one starts the same? | 154 | 3.23 | 0.990 | 4 |  |
| en/game.first-sounds.ask.d.0.wav | Listen: dog. Which one starts the same? | Listen: dog. Which one starts the same? | Listen, dog, which one starts the same? | 188 | 2.78 | 0.989 | 4 |  |
| en/game.first-sounds.ask.e.0.wav | Listen: egg. Which one starts the same? | Listen: egg. Which one starts the same? | Listen. Egg. Which one starts the same? | 162 | 3.37 | 0.986 | 4 |  |
| en/game.first-sounds.ask.g.0.wav | Listen: goat. Which one starts the same? | Listen: goat. Which one starts the same? | Listen, goat, which one starts the same? | 170 | 2.58 | 0.986 | 4 |  |
| en/game.first-sounds.ask.h.0.wav | Listen: horse. Which one starts the same? | Listen: horse. Which one starts the same? | Listen, horse, which one starts the same? | 152 | 2.98 | 0.985 | 4 |  |
| en/game.first-sounds.ask.k.0.wav | Listen: kangaroo. Which one starts the same? | Listen: kangaroo. Which one starts the same? | Listen, kangaroo, which one starts the same? | 158 | 2.94 | 0.989 | 4 |  |
| en/game.first-sounds.ask.l.0.wav | Listen: leaf. Which one starts the same? | Listen: leaf. Which one starts the same? | Listen, Leaf! Which one starts the same? | 169 | 3.07 | 0.985 | 4 |  |
| en/game.first-sounds.ask.m.0.wav | Listen: mouse. Which one starts the same? | Listen: mouse. Which one starts the same? … | Listen, mouse, which one starts the same? | 224 | 3.15 | 0.986 | 8 |  |
| en/game.first-sounds.ask.n.0.wav | Listen: net. Which one starts the same? | Listen: net. Which one starts the same? | Listen, Net, which one starts the same? | 154 | 3.06 | 0.987 | 4 |  |
| en/game.first-sounds.ask.o.0.wav | Listen: otter. Which one starts the same? | Listen: otter. Which one starts the same? | Listen, Otter, which one starts the same? | 181 | 2.97 | 0.989 | 4 |  |
| en/game.first-sounds.ask.p.0.wav | Listen: pig. Which one starts the same? | Listen: pig. Which one starts the same? | Listen, Pig, which one starts the same? | 162 | 3.21 | 0.987 | 4 |  |
| en/game.first-sounds.ask.r.0.wav | Listen: rain. Which one starts the same? | Listen: rain. Which one starts the same? | Listen, Rain, which one starts the same? | 156 | 2.82 | 0.989 | 4 |  |
| en/game.first-sounds.ask.s.0.wav | Listen: snake. Which one starts the same? | Listen: snake. Which one starts the same? | Listen, snake! Which one starts the same? | 239 | 3.57 | 0.988 | 4 |  |
| en/game.first-sounds.ask.t.0.wav | Listen: tiger. Which one starts the same? | Listen: tiger. Which one starts the same? | Listen, Tiger, which one starts the same? | 186 | 2.98 | 0.987 | 4 |  |
| en/game.first-sounds.ask.u.0.wav | Listen: up. Which one starts the same? | Listen: up. Which one starts the same? | Listen, up. Which one starts the same? | 166 | 2.74 | 0.987 | 4 |  |
| en/game.first-sounds.ask.w.0.wav | Listen: web. Which one starts the same? | Listen: web. Which one starts the same? … | Listen, web, which one starts the same? | 175 | 3.06 | 0.984 | 12 |  |
| en/game.first-sounds.praise.0.wav | Yes, that's it! | Yes, that's it! | Yes, that's it! | 181 | 1.23 | 0.960 | 4 |  |
| en/game.first-sounds.praise.1.wav | Great listening! | Great listening! | Great listening. | 155 | 0.98 | 0.976 | 4 |  |
| en/game.first-sounds.praise.2.wav | You heard it! | You heard it! … | You heard it | 162 | 0.67 | 0.972 | 8 |  |
| en/game.first-sounds.praise.3.wav | Well done! | Well done! | Well done! | 161 | 1.20 | 0.974 | 4 |  |
| en/game.first-sounds.praise.4.wav | Super ears! | Super ears! … | Super ears. | 174 | 1.03 | 0.968 | 8 |  |
| en/game.first-sounds.praise.5.wav | That's right! | That's right! … | That's right. | 286 | 0.56 | 0.971 | 12 |  |
| en/game.first-sounds.title.0.wav | Sound Safari! | Sound Safari! | Sound Safari | 161 | 1.14 | 0.978 | 4 |  |

Speaker cosine of the chosen takes: min 0.938, median 0.978, max 0.991.

## Lines without a take that met every rule (best available kept)

- en/game.first-sounds.match.bee.0: Whisper writes "B"/"Be" for "Bee" (same sound); all 12 takes say it, the best one by end quiet and voice was kept.
- en/game.first-sounds.match.pear.0: Whisper writes "Peer" for "Pear" (near-homophone); all 12 takes the same, the best one by end quiet and voice was kept. Worth a listen.
