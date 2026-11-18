package io.github.zzuegg.brainiac.experimental.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class TournamentWatermarkTest {

    private static final float[] LOGITS = {1.2f, 0.4f, -0.3f, 0.9f, 0.0f};

    @Test
    void closedFormMatchesTheExplicitTournament() {
        TournamentWatermark w = new TournamentWatermark(7, 3, 2);
        int[] ctx = {3, 1, 4};
        float[] row = LOGITS.clone();
        w.process(ctx, row);
        double[] closed = LogitMath.softmax(row);
        double[] p = LogitMath.softmax(LOGITS);
        double[] freq = new double[p.length];
        SplittableRandom r = new SplittableRandom(1);
        int n = 40_000;
        for (int i = 0; i < n; i++) {
            freq[w.sampleTournament(ctx, p, r)] += 1.0 / n;
        }
        double max = 0;
        for (int t = 0; t < p.length; t++) {
            max = Math.max(max, Math.abs(freq[t] - closed[t]));
        }
        System.out.printf("Tournament closed form vs explicit 2^3 tournament: max |diff| %.4f%n", max);
        assertTrue(max < 0.01, "closed form " + Arrays.toString(closed) + " vs " + Arrays.toString(freq));
    }

    @Test
    void unbiasedOverKeysAndDetectable() {
        double[] p = LogitMath.softmax(LOGITS);
        double[] mean = new double[p.length];
        int keys = 3000;
        for (int k = 0; k < keys; k++) {
            float[] row = LOGITS.clone();
            new TournamentWatermark(k, 4, 2).process(new int[] {1, 2}, row);
            double[] q = LogitMath.softmax(row);
            for (int t = 0; t < p.length; t++) {
                mean[t] += q[t] / keys;
            }
        }
        for (int t = 0; t < p.length; t++) {
            assertEquals(p[t], mean[t], 0.01, "the key-averaged distribution is the LM's");
        }
        TournamentWatermark w = new TournamentWatermark(11, 8, 2);
        SplittableRandom r = new SplittableRandom(2);
        float[] wide = new float[50];
        int[] marked = new int[202];
        int[] plain = new int[202];
        for (int i = 2; i < marked.length; i++) {
            float[] row = wide.clone();
            w.process(Arrays.copyOf(marked, i), row);
            marked[i] = LlmTensors.sample(LogitMath.softmax(row), r);
            plain[i] = r.nextInt(wide.length);
        }
        double zMarked = w.zScore(marked, 2);
        double zPlain = w.zScore(plain, 2);
        System.out.printf("Tournament z-score on 200 tokens: watermarked %.1f, plain %.1f%n", zMarked, zPlain);
        assertTrue(zMarked > 6 && Math.abs(zPlain) < 3);
    }
}
