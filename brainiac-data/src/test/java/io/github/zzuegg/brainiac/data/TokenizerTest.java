package io.github.zzuegg.brainiac.data;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TokenizerTest {

    private static final String TEXT = """
            To be, or not to be, that is the question:
            Whether 'tis nobler in the mind to suffer
            The slings and arrows of outrageous fortune,
            Or to take arms against a sea of troubles
            And by opposing end them. To die: to sleep;
            No more; and by a sleep to say we end
            The heart-ache and the thousand natural shocks
            That flesh is heir to, 'tis a consummation
            Devoutly to be wish'd. To die, to sleep;
            To sleep: perchance to dream: ay, there's the rub;
            """.repeat(3);

    private static final String[] SPECS = {"char", "byte", "word:300", "bpe:400", "wordpiece:400", "unigram:400"};

    @Test
    void everyKindRoundTripsAndSavesAndLoads(@TempDir Path dir) throws IOException {
        String unseen = "Grüße, 2024 — the sea! ☃";
        for (String spec : SPECS) {
            Tokenizer t = Tokenizer.train(spec, TEXT);
            assertEquals(TEXT, t.decode(t.encode(TEXT)), spec);
            int[] ids = t.encode(TEXT);
            for (int id : ids) {
                assertTrue(id >= 0 && id < t.size(), spec + " id " + id);
            }
            if (!spec.equals("char")) {
                assertEquals(unseen, t.decode(t.encode(unseen)), spec);
            }
            Path file = dir.resolve(t.kind() + ".tok");
            t.save(file);
            Tokenizer back = Tokenizer.load(file);
            assertEquals(t.kind(), back.kind());
            assertEquals(t.size(), back.size(), spec);
            assertArrayEquals(ids, back.encode(TEXT), spec);
        }
    }

    @Test
    void learnedVocabulariesShortenTheText() {
        for (String spec : new String[] {"word:300", "bpe:400", "wordpiece:400", "unigram:400"}) {
            Tokenizer t = Tokenizer.train(spec, TEXT);
            // words seen once fall back to bytes; wordpiece favours pieces that stick together
            // over frequent ones, so it compresses less
            double min = spec.startsWith("bpe") || spec.startsWith("unigram") ? 2.5 : 1.5;
            assertTrue(t.charactersPerToken(TEXT) > min, spec + ": " + t.charactersPerToken(TEXT));
            assertTrue(t.size() <= 400, spec + " has " + t.size());
        }
        assertEquals(1.0, Tokenizer.train("char", TEXT).charactersPerToken(TEXT), 1e-9);
    }

    @Test
    void bytePairMergesTheMostFrequentPairFirst() {
        BytePairTokenizer t = BytePairTokenizer.train("aaab aaab aaab", 258);
        int[][] merges = t.merges();
        assertArrayEquals(new int[] {'a', 'a'}, merges[0]);
        assertEquals("aa", t.decode(256));
    }

    @Test
    void unknownCharactersGetTheirOwnToken() {
        CharacterTokenizer t = CharacterTokenizer.train("abc");
        assertArrayEquals(new int[] {0, 3, 2}, t.encode("axc"));
        assertEquals("a�c", t.decode(0, 3, 2));
    }

    @Test
    void badSpecsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> Tokenizer.train("bpe", TEXT));
        assertThrows(IllegalArgumentException.class, () -> Tokenizer.train("magic:10", TEXT));
        assertThrows(IllegalArgumentException.class, () -> Tokenizer.train("bpe:100", TEXT));
    }
}
