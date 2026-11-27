# Sound Safari lines, one Qwen3 narrator per language

German: female narrator clone (competitive-learn reference). English: British narrator clone (en-voice/reference.wav). Qwen3-TTS 1.7B Base, bf16 on Vulkan, 24 kHz mono 16-bit WAV.
3 takes per line (seeds 777-779); the file is the take with the best Whisper large-v3 word match, then the length closest to the median. All takes are in takes/.
Mood: the Base clone ignores style instructions, so the mood comes only from the text and punctuation and the narrator reference (calm, warm).

| file | mood | text | Whisper heard (best take) | match | s | take |
|---|---|---|---|---|---|---|
| de/game.first-sounds.again.0.wav | encouraging | Fast! | Fast. | 1.00 | 0.88 | seed2.wav |
| de/game.first-sounds.again.1.wav | encouraging | Probier's noch mal. | Probier's nochmal. | 0.33 | 1.12 | seed1.wav |
| de/game.first-sounds.again.2.wav | encouraging | Guter Versuch! | Guter Versuch! | 1.00 | 1.36 | seed2.wav |
| de/game.first-sounds.again.3.wav | encouraging | Noch ein Versuch! | Noch ein Versuch. | 1.00 | 1.52 | seed0.wav |
| de/game.first-sounds.again.4.wav | encouraging | Beinahe! | Beinahe. | 1.00 | - | fix take beinahe-a-seed3 (the first 3 takes were all heard as "Bye now"; 8 more takes tried, see fix/) |
| de/game.first-sounds.ask.a.0.wav | curious | Hör gut zu: Ameise. Was fängt genauso an? | Hör gut zu. Ameise. Was fängt genau so an? | 0.75 | 3.60 | seed2.wav |
| de/game.first-sounds.ask.b.0.wav | curious | Hör gut zu: Baum. Was fängt genauso an? | Hör gut zu. Baum. Was fängt genau so an? | 0.75 | 3.60 | seed0.wav |
| de/game.first-sounds.ask.d.0.wav | curious | Hör gut zu: Dach. Was fängt genauso an? | Hör gut zu. Dach. Was fängt genau so an? | 0.75 | 3.28 | seed0.wav |
| de/game.first-sounds.ask.e.0.wav | curious | Hör gut zu: Esel. Was fängt genauso an? | Hör gut zu! Esel. Was fängt genauso an? | 1.00 | 4.00 | seed0.wav |
| de/game.first-sounds.ask.f.0.wav | curious | Hör gut zu: Fisch. Was fängt genauso an? | Hör gut zu. Fisch. Was fängt genauso an? | 1.00 | 4.24 | seed0.wav |
| de/game.first-sounds.ask.g.0.wav | curious | Hör gut zu: Gans. Was fängt genauso an? | Hör gut zu, Gans. Was fängt genau so an? | 0.75 | 3.60 | seed2.wav |
| de/game.first-sounds.ask.h.0.wav | curious | Hör gut zu: Hund. Was fängt genauso an? | Hör gut zu, Hund. Was fängt genau so an? | 0.75 | 3.36 | seed2.wav |
| de/game.first-sounds.ask.i.0.wav | curious | Hör gut zu: Insel. Was fängt genauso an? | Hör gut zu. Insel. Was fängt genau so an? | 0.75 | 3.60 | seed2.wav |
| de/game.first-sounds.ask.k.0.wav | curious | Hör gut zu: Kuh. Was fängt genauso an? | Hör gut zu. Kuh, was fängt genau so an? | 0.75 | 3.44 | seed0.wav |
| de/game.first-sounds.ask.l.0.wav | curious | Hör gut zu: Löffel. Was fängt genauso an? | Hör gut zu! Löffel, was fängt genau so an? | 0.75 | 3.60 | seed1.wav |
| de/game.first-sounds.ask.m.0.wav | curious | Hör gut zu: Maus. Was fängt genauso an? | Hör gut zu, Maus. Was fängt genauso an? | 1.00 | 3.36 | seed1.wav |
| de/game.first-sounds.ask.n.0.wav | curious | Hör gut zu: Nase. Was fängt genauso an? | Hör gut zu. Nase. Was fängt genauso an? | 1.00 | 4.08 | seed2.wav |
| de/game.first-sounds.ask.o.0.wav | curious | Hör gut zu: Oma. Was fängt genauso an? | Hör gut zu. Oma, was fängt genau so an? | 0.75 | 3.52 | seed2.wav |
| de/game.first-sounds.ask.p.0.wav | curious | Hör gut zu: Puppe. Was fängt genauso an? | Hör gut zu, Puppe. Was fängt genauso an? | 1.00 | 4.08 | seed0.wav |
| de/game.first-sounds.ask.r.0.wav | curious | Hör gut zu: Rose. Was fängt genauso an? | Hör gut zu, Rose. Was fängt genau so an? | 0.75 | 3.44 | seed0.wav |
| de/game.first-sounds.ask.s.0.wav | curious | Hör gut zu: Sofa. Was fängt genauso an? | Hör gut zu. Sofa. Was fängt genau so an? | 0.75 | 3.68 | seed1.wav |
| de/game.first-sounds.ask.t.0.wav | curious | Hör gut zu: Tiger. Was fängt genauso an? | Hör gut zu. Tiger. Was fängt genauso an? | 1.00 | 3.52 | seed0.wav |
| de/game.first-sounds.ask.u.0.wav | curious | Hör gut zu: Ufo. Was fängt genauso an? | Hör gut zu. Ufo. Was fängt genauso an? | 1.00 | 3.36 | seed1.wav |
| de/game.first-sounds.ask.w.0.wav | curious | Hör gut zu: Wurm. Was fängt genauso an? | Hör gut zu. Wurm. Was fängt genau so an? | 0.75 | 3.44 | seed1.wav |
| de/game.first-sounds.praise.0.wav | excited | Ja, genau! | Ja, genau. | 1.00 | 1.28 | seed0.wav |
| de/game.first-sounds.praise.1.wav | excited | Toll gehört! | Toll gehört. | 1.00 | 1.12 | seed2.wav |
| de/game.first-sounds.praise.2.wav | excited | Du hast es gehört! | Du hast es gehört? | 1.00 | 1.44 | seed0.wav |
| de/game.first-sounds.praise.3.wav | excited | Gut gemacht! | Gut gemacht! | 1.00 | 1.04 | seed0.wav |
| de/game.first-sounds.praise.4.wav | excited | Super Ohren! | Super Ohren | 1.00 | 1.20 | seed2.wav |
| de/game.first-sounds.praise.5.wav | excited | Richtig! | Richtig. | 1.00 | 0.88 | seed2.wav |
| de/game.first-sounds.title.0.wav | cheerful | Laut-Safari! | Laut Safari | 1.00 | 1.28 | seed2.wav |
| de/game.first-sounds.word.apple.0.wav | neutral | Apfel. | Apfel | 1.00 | 1.04 | seed2.wav |
| de/game.first-sounds.word.ball.0.wav | neutral | Ball. | Ball | 1.00 | 0.96 | seed0.wav |
| de/game.first-sounds.word.banana.0.wav | neutral | Banane. | Banane | 1.00 | 0.96 | seed0.wav |
| de/game.first-sounds.word.barrel.0.wav | neutral | Fass. | Fass. | 1.00 | 0.80 | seed2.wav |
| de/game.first-sounds.word.bear.0.wav | neutral | Bär. | Bär | 1.00 | 0.80 | seed0.wav |
| de/game.first-sounds.word.bee.0.wav | neutral | Biene. | Biene | 1.00 | 0.80 | seed0.wav |
| de/game.first-sounds.word.bicycle.0.wav | neutral | Fahrrad. | Fahrrad | 1.00 | 1.04 | seed2.wav |
| de/game.first-sounds.word.boat.0.wav | neutral | Boot. | Boot | 1.00 | 0.56 | seed1.wav |
| de/game.first-sounds.word.book.0.wav | neutral | Buch. | Buch | 1.00 | 0.96 | seed2.wav |
| de/game.first-sounds.word.cactus.0.wav | neutral | Kaktus. | Kaktus | 1.00 | 0.96 | seed1.wav |
| de/game.first-sounds.word.candle.0.wav | neutral | Kerze. | Kerze | 1.00 | 0.80 | seed1.wav |
| de/game.first-sounds.word.carrot.0.wav | neutral | Möhre. | Möhre | 1.00 | 0.96 | seed2.wav |
| de/game.first-sounds.word.cat.0.wav | neutral | Katze. | Katze | 1.00 | 0.80 | seed2.wav |
| de/game.first-sounds.word.cheese.0.wav | neutral | Käse. | Käse | 1.00 | 0.96 | seed0.wav |
| de/game.first-sounds.word.clock.0.wav | neutral | Uhr. | Uhr | 1.00 | 0.72 | seed2.wav |
| de/game.first-sounds.word.cloud.0.wav | neutral | Wolke. | Wolke | 1.00 | 0.88 | seed2.wav |
| de/game.first-sounds.word.cookie.0.wav | neutral | Keks. | Keks | 1.00 | 0.80 | seed1.wav |
| de/game.first-sounds.word.corn.0.wav | neutral | Mais. | Mais. | 1.00 | 0.80 | seed0.wav |
| de/game.first-sounds.word.cup.0.wav | neutral | Tasse. | Tasse | 1.00 | 0.80 | seed2.wav |
| de/game.first-sounds.word.dino.0.wav | neutral | Dino. | Dino | 1.00 | 1.12 | seed2.wav |
| de/game.first-sounds.word.elephant.0.wav | neutral | Elefant. | Elefant | 1.00 | 1.12 | seed2.wav |
| de/game.first-sounds.word.gift.0.wav | neutral | Geschenk. | Geschenk | 1.00 | 0.96 | seed0.wav |
| de/game.first-sounds.word.giraffe.0.wav | neutral | Giraffe. | Giraffe | 1.00 | 0.96 | seed0.wav |
| de/game.first-sounds.word.guitar.0.wav | neutral | Gitarre. | Gitarre | 1.00 | 0.80 | seed2.wav |
| de/game.first-sounds.word.hat.0.wav | neutral | Hut. | Hut | 1.00 | 0.64 | seed1.wav |
| de/game.first-sounds.word.heart.0.wav | neutral | Herz. | Herz | 1.00 | 0.88 | seed2.wav |
| de/game.first-sounds.word.hedgehog.0.wav | neutral | Igel. | Igel | 1.00 | 0.88 | seed2.wav |
| de/game.first-sounds.word.house.0.wav | neutral | Haus. | Haus | 1.00 | 0.80 | seed2.wav |
| de/game.first-sounds.word.ladder.0.wav | neutral | Leiter. | Leiter | 1.00 | 0.88 | seed2.wav |
| de/game.first-sounds.word.lamp.0.wav | neutral | Lampe. | Lampe | 1.00 | 0.88 | seed2.wav |
| de/game.first-sounds.word.lantern.0.wav | neutral | Laterne. | Laterne | 1.00 | 0.96 | seed2.wav |
| de/game.first-sounds.word.lion.0.wav | neutral | Löwe. | Löwe | 1.00 | 0.96 | seed0.wav |
| de/game.first-sounds.word.monkey.0.wav | neutral | Affe. | Affe | 1.00 | 0.72 | seed2.wav |
| de/game.first-sounds.word.moon.0.wav | neutral | Mond. | Mond | 1.00 | 0.96 | seed2.wav |
| de/game.first-sounds.word.mushroom.0.wav | neutral | Pilz. | Pilz | 1.00 | 0.72 | seed1.wav |
| de/game.first-sounds.word.nest.0.wav | neutral | Nest. | Nest | 1.00 | 0.96 | seed0.wav |
| de/game.first-sounds.word.octopus.0.wav | neutral | Oktopus. | Oktopus | 1.00 | 1.28 | seed2.wav |
| de/game.first-sounds.word.orange.0.wav | neutral | Orange. | Orange | 1.00 | 0.96 | seed2.wav |
| de/game.first-sounds.word.pear.0.wav | neutral | Birne. | Birne. | 1.00 | 0.80 | seed0.wav |
| de/game.first-sounds.word.penguin.0.wav | neutral | Pinguin. | Pinguin | 1.00 | 0.88 | seed2.wav |
| de/game.first-sounds.word.pizza.0.wav | neutral | Pizza. | Pizza | 1.00 | 1.04 | seed0.wav |
| de/game.first-sounds.word.rabbit.0.wav | neutral | Hase. | Hase | 1.00 | 0.80 | seed0.wav |
| de/game.first-sounds.word.rainbow.0.wav | neutral | Regenbogen. | Regenbogen | 1.00 | 1.12 | seed0.wav |
| de/game.first-sounds.word.robot.0.wav | neutral | Roboter. | Roboter | 1.00 | 0.96 | seed0.wav |
| de/game.first-sounds.word.rocket.0.wav | neutral | Rakete. | Rakete | 1.00 | 0.96 | seed0.wav |
| de/game.first-sounds.word.soap.0.wav | neutral | Seife. | Seife | 1.00 | 0.88 | seed2.wav |
| de/game.first-sounds.word.strawberry.0.wav | neutral | Erdbeere. | Erdbeere | 1.00 | 1.04 | seed2.wav |
| de/game.first-sounds.word.submarine.0.wav | neutral | U-Boot. | U-Boot | 1.00 | 0.96 | seed2.wav |
| de/game.first-sounds.word.sun.0.wav | neutral | Sonne. | Sonne | 1.00 | 0.80 | seed2.wav |
| de/game.first-sounds.word.table.0.wav | neutral | Tisch. | Tisch | 1.00 | 0.80 | seed1.wav |
| de/game.first-sounds.word.teddy.0.wav | neutral | Teddy. | Teddy | 1.00 | 0.80 | seed2.wav |
| de/game.first-sounds.word.tomato.0.wav | neutral | Tomate. | Tomate | 1.00 | 1.04 | seed2.wav |
| de/game.first-sounds.word.trophy.0.wav | neutral | Pokal. | Pokal | 1.00 | 0.88 | seed2.wav |
| de/game.first-sounds.word.watermelon.0.wav | neutral | Wassermelone. | Wassermelone | 1.00 | 1.60 | seed0.wav |
| de/game.first-sounds.word.whale.0.wav | neutral | Wal. | Wahl | 0.00 | 0.72 | seed2.wav |
| en/game.first-sounds.again.0.wav | encouraging | Nearly! | Nearly. | 1.00 | 0.96 | seed0.wav |
| en/game.first-sounds.again.1.wav | encouraging | Try again. | Try again. | 1.00 | 1.20 | seed0.wav |
| en/game.first-sounds.again.2.wav | encouraging | Good try! | Good try! | 1.00 | 1.36 | seed1.wav |
| en/game.first-sounds.again.3.wav | encouraging | One more go! | One more go. | 1.00 | 1.20 | seed1.wav |
| en/game.first-sounds.again.4.wav | encouraging | Almost! | Almost? | 1.00 | 1.04 | seed0.wav |
| en/game.first-sounds.ask.a.0.wav | curious | Listen: ant. Which one starts the same? | Listen, Ant, which one starts the same? | 1.00 | 3.36 | seed2.wav |
| en/game.first-sounds.ask.b.0.wav | curious | Listen: bird. Which one starts the same? | Listen, bird, which one starts the same? | 1.00 | 3.52 | seed2.wav |
| en/game.first-sounds.ask.d.0.wav | curious | Listen: dog. Which one starts the same? | Listen, dog, which one starts the same? | 1.00 | 3.28 | seed2.wav |
| en/game.first-sounds.ask.e.0.wav | curious | Listen: egg. Which one starts the same? | Listen, Egg, which one starts the same? | 1.00 | 3.36 | seed2.wav |
| en/game.first-sounds.ask.g.0.wav | curious | Listen: goat. Which one starts the same? | Listen, goat, which one starts the same? | 1.00 | 3.28 | seed2.wav |
| en/game.first-sounds.ask.h.0.wav | curious | Listen: horse. Which one starts the same? | Listen, horse, which one starts the same? | 1.00 | 3.28 | seed1.wav |
| en/game.first-sounds.ask.k.0.wav | curious | Listen: kangaroo. Which one starts the same? | Listen, kangaroo, which one starts the same? | 1.00 | 3.92 | seed1.wav |
| en/game.first-sounds.ask.l.0.wav | curious | Listen: leaf. Which one starts the same? | Listen, Leaf! Which one starts the same? | 1.00 | 3.68 | seed2.wav |
| en/game.first-sounds.ask.m.0.wav | curious | Listen: mouse. Which one starts the same? | Listen. Mouse. Which one starts the same? | 1.00 | 3.52 | seed1.wav |
| en/game.first-sounds.ask.n.0.wav | curious | Listen: net. Which one starts the same? | Listen, net, which one starts the same? | 1.00 | 3.76 | seed1.wav |
| en/game.first-sounds.ask.o.0.wav | curious | Listen: otter. Which one starts the same? | Listen, Otter. Which one starts the same? | 1.00 | 3.52 | seed0.wav |
| en/game.first-sounds.ask.p.0.wav | curious | Listen: pig. Which one starts the same? | Listen, Pig, which one starts the same? | 1.00 | 3.36 | seed0.wav |
| en/game.first-sounds.ask.r.0.wav | curious | Listen: rain. Which one starts the same? | Listen, rain. Which one starts the same? | 1.00 | 3.36 | seed2.wav |
| en/game.first-sounds.ask.s.0.wav | curious | Listen: snake. Which one starts the same? | Listen, Snake, which one starts the same? | 1.00 | 3.60 | seed1.wav |
| en/game.first-sounds.ask.t.0.wav | curious | Listen: tiger. Which one starts the same? | Listen, Tiger, which one starts the same? | 1.00 | 3.20 | seed1.wav |
| en/game.first-sounds.ask.u.0.wav | curious | Listen: up. Which one starts the same? | Listen up! Which one starts the same? | 1.00 | 3.12 | seed0.wav |
| en/game.first-sounds.ask.w.0.wav | curious | Listen: web. Which one starts the same? | Listen, web! Which one starts the same? | 1.00 | 3.92 | seed0.wav |
| en/game.first-sounds.praise.0.wav | excited | Yes, that's it! | Yes, that's it. | 1.00 | 1.44 | seed2.wav |
| en/game.first-sounds.praise.1.wav | excited | Great listening! | Great listening. | 1.00 | 1.44 | seed2.wav |
| en/game.first-sounds.praise.2.wav | excited | You heard it! | You heard it. | 1.00 | 1.52 | seed1.wav |
| en/game.first-sounds.praise.3.wav | excited | Well done! | Well done! | 1.00 | 1.20 | seed2.wav |
| en/game.first-sounds.praise.4.wav | excited | Super ears! | Super Ears | 1.00 | 1.52 | seed0.wav |
| en/game.first-sounds.praise.5.wav | excited | That's right! | That's right. | 1.00 | 1.28 | seed0.wav |
| en/game.first-sounds.title.0.wav | cheerful | Sound Safari! | Sound Safari | 1.00 | 1.36 | seed2.wav |
| en/game.first-sounds.word.apple.0.wav | neutral | Apple. | Apple | 1.00 | 0.72 | seed1.wav |
| en/game.first-sounds.word.ball.0.wav | neutral | Ball. | Ball | 1.00 | 0.88 | seed2.wav |
| en/game.first-sounds.word.banana.0.wav | neutral | Banana. | Banana | 1.00 | 1.12 | seed0.wav |
| en/game.first-sounds.word.bear.0.wav | neutral | Bear. | Bear | 1.00 | 0.72 | seed1.wav |
| en/game.first-sounds.word.bee.0.wav | neutral | Bee. | B | 0.00 | 1.20 | seed1.wav |
| en/game.first-sounds.word.bell.0.wav | neutral | Bell. | Bell | 1.00 | 0.88 | seed2.wav |
| en/game.first-sounds.word.boat.0.wav | neutral | Boat. | Boat | 1.00 | 1.12 | seed0.wav |
| en/game.first-sounds.word.book.0.wav | neutral | Book. | book | 1.00 | 0.80 | seed2.wav |
| en/game.first-sounds.word.bus.0.wav | neutral | Bus. | BUS! | 1.00 | 0.80 | seed2.wav |
| en/game.first-sounds.word.candle.0.wav | neutral | Candle. | Candle! | 1.00 | 1.36 | seed1.wav |
| en/game.first-sounds.word.car.0.wav | neutral | Car. | Car. | 1.00 | 0.96 | seed2.wav |
| en/game.first-sounds.word.carrot.0.wav | neutral | Carrot. | Carrot | 1.00 | 0.96 | seed2.wav |
| en/game.first-sounds.word.cat.0.wav | neutral | Cat. | Cat | 1.00 | 0.72 | seed2.wav |
| en/game.first-sounds.word.cookie.0.wav | neutral | Cookie. | Cookie. | 1.00 | 1.04 | seed0.wav |
| en/game.first-sounds.word.cup.0.wav | neutral | Cup. | Cup | 1.00 | 0.72 | seed2.wav |
| en/game.first-sounds.word.dice.0.wav | neutral | Dice. | Dice | 1.00 | 0.80 | seed2.wav |
| en/game.first-sounds.word.door.0.wav | neutral | Door. | Door | 1.00 | 0.88 | seed2.wav |
| en/game.first-sounds.word.egg.0.wav | neutral | Egg. | Egg. | 1.00 | 1.20 | seed1.wav |
| en/game.first-sounds.word.elephant.0.wav | neutral | Elephant. | Elephant | 1.00 | 1.12 | seed2.wav |
| en/game.first-sounds.word.gift.0.wav | neutral | Present. | Present | 1.00 | 0.96 | seed2.wav |
| en/game.first-sounds.word.guitar.0.wav | neutral | Guitar. | Guitar | 1.00 | 1.12 | seed1.wav |
| en/game.first-sounds.word.hat.0.wav | neutral | Hat. | Hat! | 1.00 | 0.64 | seed2.wav |
| en/game.first-sounds.word.heart.0.wav | neutral | Heart. | Heart | 1.00 | 0.72 | seed1.wav |
| en/game.first-sounds.word.hedgehog.0.wav | neutral | Hedgehog. | Hedgehog? | 1.00 | 1.52 | seed2.wav |
| en/game.first-sounds.word.house.0.wav | neutral | House. | House? | 1.00 | 0.96 | seed2.wav |
| en/game.first-sounds.word.key.0.wav | neutral | Key. | Key | 1.00 | 0.96 | seed1.wav |
| en/game.first-sounds.word.kite.0.wav | neutral | Kite. | Kite! | 1.00 | 0.80 | seed1.wav |
| en/game.first-sounds.word.ladder.0.wav | neutral | Ladder. | Ladder | 1.00 | 0.96 | seed2.wav |
| en/game.first-sounds.word.lamp.0.wav | neutral | Lamp. | Lamp? | 1.00 | 0.96 | seed2.wav |
| en/game.first-sounds.word.lion.0.wav | neutral | Lion. | Lion | 1.00 | 1.20 | seed1.wav |
| en/game.first-sounds.word.lollipop.0.wav | neutral | Lollipop. | Lollipop | 1.00 | 1.20 | seed2.wav |
| en/game.first-sounds.word.mittens.0.wav | neutral | Mittens. | Mittens | 1.00 | 1.28 | seed2.wav |
| en/game.first-sounds.word.monkey.0.wav | neutral | Monkey. | Monkey | 1.00 | 0.88 | seed2.wav |
| en/game.first-sounds.word.moon.0.wav | neutral | Moon. | Moon | 1.00 | 1.12 | seed0.wav |
| en/game.first-sounds.word.mushroom.0.wav | neutral | Mushroom. | Mushroom. | 1.00 | 1.12 | seed0.wav |
| en/game.first-sounds.word.nest.0.wav | neutral | Nest. | Nest? | 1.00 | 1.04 | seed1.wav |
| en/game.first-sounds.word.octopus.0.wav | neutral | Octopus. | Octopus | 1.00 | 1.20 | seed2.wav |
| en/game.first-sounds.word.orange.0.wav | neutral | Orange. | Orange? | 1.00 | 1.04 | seed0.wav |
| en/game.first-sounds.word.pear.0.wav | neutral | Pear. | Peer | 0.00 | 0.88 | seed2.wav |
| en/game.first-sounds.word.pencil.0.wav | neutral | Pencil. | Pencil! | 1.00 | 1.44 | seed0.wav |
| en/game.first-sounds.word.penguin.0.wav | neutral | Penguin. | Penguin! | 1.00 | 1.12 | seed1.wav |
| en/game.first-sounds.word.pizza.0.wav | neutral | Pizza. | Pizza | 1.00 | 1.20 | seed2.wav |
| en/game.first-sounds.word.pumpkin.0.wav | neutral | Pumpkin. | Pumpkin? | 1.00 | 1.04 | seed1.wav |
| en/game.first-sounds.word.rabbit.0.wav | neutral | Rabbit. | Rabbit. | 1.00 | 1.20 | seed2.wav |
| en/game.first-sounds.word.rainbow.0.wav | neutral | Rainbow. | Rainbow | 1.00 | 1.12 | seed2.wav |
| en/game.first-sounds.word.rocket.0.wav | neutral | Rocket. | Rocket! | 1.00 | 0.72 | seed0.wav |
| en/game.first-sounds.word.sandwich.0.wav | neutral | Sandwich. | Sandwich. | 1.00 | 1.20 | seed2.wav |
| en/game.first-sounds.word.soap.0.wav | neutral | Soap. | soap | 1.00 | 1.04 | seed1.wav |
| en/game.first-sounds.word.sun.0.wav | neutral | Sun. | Sun | 1.00 | 1.28 | seed2.wav |
| en/game.first-sounds.word.table.0.wav | neutral | Table. | Table. | 1.00 | 1.20 | seed0.wav |
| en/game.first-sounds.word.teddy.0.wav | neutral | Teddy. | Teddy | 1.00 | 0.80 | seed2.wav |
| en/game.first-sounds.word.tent.0.wav | neutral | Tent. | Tent | 1.00 | 0.88 | seed2.wav |
| en/game.first-sounds.word.tomato.0.wav | neutral | Tomato. | Tomato | 1.00 | 0.88 | seed2.wav |
| en/game.first-sounds.word.turtle.0.wav | neutral | Turtle. | Turtle? | 1.00 | 0.88 | seed1.wav |
| en/game.first-sounds.word.umbrella.0.wav | neutral | Umbrella. | Umbrella | 1.00 | 1.12 | seed1.wav |
| en/game.first-sounds.word.watermelon.0.wav | neutral | Watermelon. | watermelon | 1.00 | 1.20 | seed2.wav |
| en/game.first-sounds.word.whale.0.wav | neutral | Whale. | whale | 1.00 | 0.80 | seed1.wav |
| en/game.first-sounds.word.windmill.0.wav | neutral | Windmill. | Windmill | 1.00 | 1.52 | seed0.wav |

## Not a perfect Whisper match (listen to these)

- de/game.first-sounds.again.1 (heard "Probier's nochmal.")
- de/game.first-sounds.ask.a.0 (heard "Hör gut zu. Ameise. Was fängt genau so an?")
- de/game.first-sounds.ask.b.0 (heard "Hör gut zu. Baum. Was fängt genau so an?")
- de/game.first-sounds.ask.d.0 (heard "Hör gut zu. Dach. Was fängt genau so an?")
- de/game.first-sounds.ask.g.0 (heard "Hör gut zu, Gans. Was fängt genau so an?")
- de/game.first-sounds.ask.h.0 (heard "Hör gut zu, Hund. Was fängt genau so an?")
- de/game.first-sounds.ask.i.0 (heard "Hör gut zu. Insel. Was fängt genau so an?")
- de/game.first-sounds.ask.k.0 (heard "Hör gut zu. Kuh, was fängt genau so an?")
- de/game.first-sounds.ask.l.0 (heard "Hör gut zu! Löffel, was fängt genau so an?")
- de/game.first-sounds.ask.o.0 (heard "Hör gut zu. Oma, was fängt genau so an?")
- de/game.first-sounds.ask.r.0 (heard "Hör gut zu, Rose. Was fängt genau so an?")
- de/game.first-sounds.ask.s.0 (heard "Hör gut zu. Sofa. Was fängt genau so an?")
- de/game.first-sounds.ask.w.0 (heard "Hör gut zu. Wurm. Was fängt genau so an?")
- de/game.first-sounds.word.whale.0 (heard "Wahl")
- en/game.first-sounds.word.bee.0 (heard "B")
- en/game.first-sounds.word.pear.0 (heard "Peer")

All other flags above are Whisper spelling only ("genau so" for "genauso", "nochmal", "Wahl" for "Wal", "B" for "bee", "Peer" for "pear"), not wrong speech.

Note for future German letter lines: write the letter I as lowercase "i" (or "Ih"). A capital "I" makes Qwen say the English "eye".
