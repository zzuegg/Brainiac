"""game2/lines.tsv: stem (lang/file stem), language, text. New match/named lines plus every old non-word line."""
import os

P = 'game.first-sounds.'
rows = []


def add(lang, key, text):
    rows.append('%s/%s%s.0\t%s\t%s\n' % (lang, P, key, 'german' if lang == 'de' else 'english', text))


de_match = ('apple = Apfel, wie Ameise! ; ball = Ball, wie Baum! ; banana = Banane, wie Baum! ; barrel = Fass, wie Fisch! ; '
            'bear = Bär, wie Baum! ; bee = Biene, wie Baum! ; bicycle = Fahrrad, wie Fisch! ; boat = Boot, wie Baum! ; '
            'book = Buch, wie Baum! ; cactus = Kaktus, wie Kuh! ; candle = Kerze, wie Kuh! ; carrot = Möhre, wie Maus! ; '
            'cat = Katze, wie Kuh! ; cheese = Käse, wie Kuh! ; clock = Uhr, wie Ufo! ; cloud = Wolke, wie Wurm! ; '
            'cookie = Keks, wie Kuh! ; corn = Mais, wie Maus! ; cup = Tasse, wie Tiger! ; dino = Dino, wie Dach! ; '
            'elephant = Elefant, wie Esel! ; gift = Geschenk, wie Gans! ; giraffe = Giraffe, wie Gans! ; '
            'guitar = Gitarre, wie Gans! ; hat = Hut, wie Hund! ; heart = Herz, wie Hund! ; hedgehog = Igel, wie Insel! ; '
            'house = Haus, wie Hund! ; ladder = Leiter, wie Löffel! ; lamp = Lampe, wie Löffel! ; '
            'lantern = Laterne, wie Löffel! ; lion = Löwe, wie Löffel! ; monkey = Affe, wie Ameise! ; moon = Mond, wie Maus! ; '
            'mushroom = Pilz, wie Puppe! ; nest = Nest, wie Nase! ; octopus = Oktopus, wie Oma! ; orange = Orange, wie Oma! ; '
            'pear = Birne, wie Baum! ; penguin = Pinguin, wie Puppe! ; pizza = Pizza, wie Puppe! ; rabbit = Hase, wie Hund! ; '
            'rainbow = Regenbogen, wie Rose! ; robot = Roboter, wie Rose! ; rocket = Rakete, wie Rose! ; soap = Seife, wie Sofa! ; '
            'strawberry = Erdbeere, wie Esel! ; submarine = U-Boot, wie Ufo! ; sun = Sonne, wie Sofa! ; table = Tisch, wie Tiger! ; '
            'teddy = Teddy, wie Tiger! ; tomato = Tomate, wie Tiger! ; trophy = Pokal, wie Puppe! ; '
            'watermelon = Wassermelone, wie Wurm! ; whale = Wal, wie Wurm!')
eine = set('banana bee candle carrot cat clock cloud cup giraffe guitar ladder lamp lantern orange pear pizza rocket soap strawberry tomato watermelon'.split())
de_named_exc = {'corn': 'Das ist Mais.', 'cheese': 'Das ist Käse.', 'sun': 'Das ist die Sonne.', 'moon': 'Das ist der Mond.'}
for item in de_match.split(' ; '):
    k, t = item.split(' = ')
    add('de', 'match.' + k, t)
    word = t.split(',')[0]
    add('de', 'named.' + k, de_named_exc.get(k, 'Das ist %s %s.' % ('eine' if k in eine else 'ein', word)))

groups = {'bird': 'ball banana bear bee bell boat book bus', 'kangaroo': 'candle car carrot cat cookie cup key kite',
          'dog': 'dice door', 'egg': 'elephant', 'pig': 'gift pear pencil penguin pizza pumpkin', 'goat': 'guitar',
          'horse': 'hat heart hedgehog house', 'leaf': 'ladder lamp lion lollipop', 'mouse': 'mittens monkey moon mushroom',
          'net': 'nest', 'otter': 'octopus orange', 'rain': 'rabbit rainbow rocket', 'snake': 'sandwich soap sun',
          'tiger': 'table teddy tent tomato turtle', 'up': 'umbrella', 'web': 'watermelon whale windmill', 'ant': 'apple'}
an = set('apple egg elephant octopus orange umbrella'.split())
en_named_exc = {'gift': "That's a present.", 'mittens': 'Those are mittens.', 'soap': "That's soap.",
                'sun': "That's the sun.", 'moon': "That's the moon."}
pics = []
for model, ps in groups.items():
    for p in ps.split():
        w = 'present' if p == 'gift' else p
        add('en', 'match.' + p, '%s, like %s!' % (w[0].upper() + w[1:], model))
        pics.append(p)
for p in sorted(pics + ['egg']):
    add('en', 'named.' + p, en_named_exc.get(p, "That's %s %s." % ('an' if p in an else 'a', p)))

# old non-word lines, same ids
for l in open('C:/Users/micha/AppData/Local/Temp/kalimo-test/game/lines.tsv', encoding='utf-8'):
    stem, lang, text, mood = l.rstrip('\n').split('\t')
    if '.word.' not in stem:
        rows.append('%s\t%s\t%s\n' % (stem, lang, text))
os.makedirs('C:/Users/micha/AppData/Local/Temp/kalimo-test/game2', exist_ok=True)
open('C:/Users/micha/AppData/Local/Temp/kalimo-test/game2/lines.tsv', 'w', encoding='utf-8').write(''.join(rows))
de = [r for r in rows if r.startswith('de/')]
en = [r for r in rows if r.startswith('en/')]
print('de', len(de), 'en', len(en), 'new de', sum('.match.' in r or '.named.' in r for r in de), 'new en', sum('.match.' in r or '.named.' in r for r in en))
