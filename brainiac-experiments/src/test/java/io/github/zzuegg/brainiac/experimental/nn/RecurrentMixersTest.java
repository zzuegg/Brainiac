package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** RWKV-7 and xLSTM. */
class RecurrentMixersTest {

    @Test
    void rwkv7StateMatchesNaiveGeneralizedDeltaLoops() {
        Rwkv7TimeMix mix = new Rwkv7TimeMix(8, 2, 4, new SplittableRandom(1));
        Var x = Var.of(LinearRecurrenceTest.input(2, 1, 5, 8));
        Var[] p = mix.mixInputs(x);
        float[] got = mix.wkv(p).value().toArray();
        float[] r = p[0].value().toArray();
        float[] w = p[1].value().toArray();
        float[] kt = p[2].value().toArray();
        float[] v = p[3].value().toArray();
        float[] a = p[4].value().toArray();
        float[] kh = p[5].value().toArray();
        int d = 4;
        float[] expected = new float[got.length];
        for (int h = 0; h < 2; h++) {
            double[][] s = new double[d][d];
            for (int t = 0; t < 5; t++) {
                int o = (h * 5 + t) * d;
                double[] khs = new double[d];   // kh^T S
                for (int j = 0; j < d; j++) {
                    for (int i = 0; i < d; i++) {
                        khs[j] += kh[o + i] * s[i][j];
                    }
                }
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        s[i][j] = w[o + i] * s[i][j] - a[o + i] * kh[o + i] * khs[j] + kt[o + i] * v[o + j];
                    }
                }
                for (int j = 0; j < d; j++) {
                    double y = 0;
                    for (int i = 0; i < d; i++) {
                        y += r[o + i] * s[i][j];
                    }
                    expected[o + j] = (float) y;
                }
            }
        }
        assertArrayEquals(expected, got, 1e-4f);
        for (float decay : w) {
            assertTrue(decay > 0.54 && decay < 1);
        }
    }

    @Test
    void rwkv7LearnsRecall() {
        double acc = MixerHarness.train(new Rwkv7TimeMix(16, 2, 8, new SplittableRandom(3)),
                MixerHarness.recall(6, 6, 3), 16, 300, 0.01f, 4);
        assertTrue(acc > 0.7, "accuracy " + acc);
    }

    @Test
    void mLstmParallelEqualsStabilizedRecurrence() {
        MLstmLayer m = new MLstmLayer(8, 2, new SplittableRandom(5));
        LinearRecurrenceTest.assertModesAgree(m.parameters(), m::forward, m::recurrent,
                LinearRecurrenceTest.input(6, 2, 7, 8), 1e-4f);
    }

    @Test
    void mLstmLearnsRecall() {
        double acc = MixerHarness.train(new MLstmLayer(16, 2, new SplittableRandom(7)),
                MixerHarness.recall(6, 6, 3), 16, 300, 0.01f, 8);
        assertTrue(acc > 0.7, "accuracy " + acc);
    }

    @Test
    void sLstmTracksParity() {
        double acc = MixerHarness.train(new SLstmLayer(16, 16, 4, new SplittableRandom(9)),
                MixerHarness.parity(5), 16, 300, 0.01f, 10);
        assertTrue(acc > 0.95, "accuracy " + acc);
    }
}
