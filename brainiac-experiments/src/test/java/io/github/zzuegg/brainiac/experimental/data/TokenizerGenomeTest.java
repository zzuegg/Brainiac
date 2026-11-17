package io.github.zzuegg.brainiac.experimental.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.BytePairTokenizer;
import io.github.zzuegg.brainiac.data.Tokenizer;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class TokenizerGenomeTest {

    private static final String TEXT = "the cat sat on the mat. the cat ate the rat. that cat is the fattest cat. ";

    @Test
    void decodesToTheSameTokenizerItCameFrom() {
        BytePairTokenizer bpe = BytePairTokenizer.train(TEXT.repeat(5), 280);
        Tokenizer back = TokenizerGenome.of(bpe).toTokenizer();
        assertArrayEquals(bpe.encode(TEXT), back.encode(TEXT));
        assertEquals(bpe.size(), back.size());
    }

    @Test
    void mutantsAlwaysDecodeAndRoundTrip() {
        TokenizerGenome g = TokenizerGenome.of(BytePairTokenizer.train(TEXT.repeat(5), 290));
        SplittableRandom r = new SplittableRandom(1);
        for (int i = 0; i < 60; i++) {
            g = g.mutate(TEXT, r).cross(g, r);
            Tokenizer t = g.toTokenizer();
            assertEquals(TEXT, t.decode(t.encode(TEXT)));
        }
    }

    @Test
    void guidedMutationAddsTheMostFrequentPair() {
        TokenizerGenome empty = new TokenizerGenome(List.of(), 0);
        assertTrue(empty.pairCounts("aaaa").containsKey(List.of("a", "a")));
        double bytes = TokenizerScores.compressionBitsPerCharacter(empty.toTokenizer(), TEXT);
        double bpe = TokenizerScores.compressionBitsPerCharacter(BytePairTokenizer.train(TEXT.repeat(20), 270),
                TEXT.repeat(20));
        assertTrue(bpe < bytes, bpe + " vs " + bytes);
    }
}
