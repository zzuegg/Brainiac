package io.github.zzuegg.brainiac.experimental.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.llm.Generator;

import java.util.Arrays;
import java.util.Set;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class KneserNeyLmTest {

    @Test
    void higherOrdersPredictCountingAndDistributionsNormalize() {
        int[][] train = TinyLm.counting(new SplittableRandom(1), 200, 12);
        int[][] test = TinyLm.counting(new SplittableRandom(2), 50, 12);
        KneserNeyLm uni = KneserNeyLm.train(train, 1, TinyLm.VOCAB, 0.75);
        KneserNeyLm tri = KneserNeyLm.train(train, 3, TinyLm.VOCAB, 0.75);
        for (KneserNeyLm m : new KneserNeyLm[] {uni, tri}) {
            double sum = Arrays.stream(m.distribution(new int[] {3, 5})).sum();
            assertEquals(1, sum, 1e-9, m.toString());
            sum = Arrays.stream(m.distribution(new int[] {7, 7})).sum();
            assertEquals(1, sum, 1e-9, "unseen context still normalizes");
        }
        double pu = uni.perplexity(test);
        double pt = tri.perplexity(test);
        assertTrue(pu > 10, "unigram perplexity " + pu);
        assertTrue(pt < 2, "trigram perplexity " + pt);
        CharacterTokenizer tok = CharacterTokenizer.train("abcdefghijk");
        int[] out = new Generator(tri, tok).generate(new int[][] {{1, 3, 5}},
                Generator.Options.greedy(6).withStopTokens(Set.of()), null).get(0).tokens();
        assertEquals(1.0, TinyLm.countingAccuracy(new int[] {1, 3, 5}, out), 1e-9);
    }
}
