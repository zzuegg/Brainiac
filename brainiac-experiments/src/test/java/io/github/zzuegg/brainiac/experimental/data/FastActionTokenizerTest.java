package io.github.zzuegg.brainiac.experimental.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

class FastActionTokenizerTest {

    private static List<float[][]> smoothChunks(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[][]> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            double a = r.nextDouble(-0.8, 0.8);
            double b = r.nextDouble(-0.3, 0.3);
            float[][] c = new float[16][2];
            for (int t = 0; t < 16; t++) {
                c[t][0] = (float) (a + b * t / 16.0);
                c[t][1] = (float) (0.5 * Math.sin(a + t / 8.0));
            }
            out.add(c);
        }
        return out;
    }

    @Test
    void dctRoundTrips() {
        double[] x = {0.3, -0.1, 0.7, 0.2, -0.5};
        assertArrayEquals(x, FastActionTokenizer.idct(FastActionTokenizer.dct(x)), 1e-9);
    }

    @Test
    void fastCompressesSmoothChunksAndDecodesThem() {
        List<float[][]> train = smoothChunks(200, 1);
        FastActionTokenizer fast = FastActionTokenizer.train(train, 10f, 128);
        BinningActionTokenizer bins = new BinningActionTokenizer(16, 2, 256);
        double fastTokens = 0;
        double fastError = 0;
        double binError = 0;
        List<float[][]> test = smoothChunks(50, 2);
        for (float[][] c : test) {
            int[] t = fast.encodeChunk(c);
            fastTokens += t.length;
            fastError += error(c, fast.decodeChunk(t));
            binError += error(c, bins.decodeChunk(bins.encodeChunk(c)));
        }
        fastTokens /= test.size();
        System.out.printf("FAST: %.1f tokens per 16x2 chunk (binning 32), max abs error %.3f (binning %.3f)%n",
                fastTokens, fastError / test.size(), binError / test.size());
        assertTrue(fastTokens < 32 / 3.0, "FAST compresses: " + fastTokens);
        assertTrue(fastError / test.size() < 0.1, "and decodes close to the original");
        // the text form round-trips through the core Tokenizer interface
        String text = fast.coefficientText(test.getFirst());
        assertEquals(text, fast.decode(fast.encode(text)));
        assertArrayEquals(fast.encodeChunk(test.getFirst()), fast.encode(text));
    }

    private static double error(float[][] a, float[][] b) {
        double m = 0;
        for (int t = 0; t < a.length; t++) {
            for (int d = 0; d < a[t].length; d++) {
                m = Math.max(m, Math.abs(a[t][d] - b[t][d]));
            }
        }
        return m;
    }
}
