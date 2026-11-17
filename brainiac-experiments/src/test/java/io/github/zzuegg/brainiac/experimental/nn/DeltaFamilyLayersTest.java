package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class DeltaFamilyLayersTest {

    private static Var input(int b, int t, int d, long seed) {
        return Var.of(Tensor.normal(Shape.of(b, t, d), 0, 1, new SplittableRandom(seed)));
    }

    @Test
    void kimiDeltaChunkwiseMatchesRecurrence() {
        KimiDeltaAttention kda = new KimiDeltaAttention(8, 2, 2, 3, new SplittableRandom(1));
        Var x = input(2, 7, 8, 2);
        assertArrayEquals(kda.recurrent(x).value().toArray(), kda.forward(x).value().toArray(), 1e-4f);
        Tensor decays = kda.decays(x);
        float min = decays.min().item();
        float max = decays.max().item();
        System.out.printf("KDA chunkwise == recurrent; per-channel decays span %.3f..%.3f%n", min, max);
        assertTrue(max - min > 0.05, "channels decay at different rates");
    }

    @Test
    void kimiDeltaGradientsReachTheLowRankGate() {
        KimiDeltaAttention kda = new KimiDeltaAttention(8, 2, 2, 4, new SplittableRandom(3));
        kda.forward(input(1, 6, 8, 4)).square().mean().backward();
        for (Var p : kda.parameters()) {
            assertTrue(p.grad() != null && p.grad().abs().sum().item() > 0, "every parameter gets a gradient");
        }
    }

    @Test
    void deltaProductChunkwiseMatchesRecurrence() {
        DeltaProductLayer dp = new DeltaProductLayer(8, 2, 2, true, 2, new SplittableRandom(5));
        Var x = input(2, 5, 8, 6);
        assertArrayEquals(dp.recurrent(x).value().toArray(), dp.forward(x).value().toArray(), 1e-4f);
        assertEquals(2, dp.householderSteps());
    }

    @Test
    void deltaProductWithTwoReflectionsComposesARotation() {
        // two Householder steps per token can rotate the state, one cannot: check the algebra
        float[] k1 = {1, 0};
        float[] k2 = {(float) Math.cos(Math.PI / 3), (float) Math.sin(Math.PI / 3)};
        float[][] h = multiply(householder(k2), householder(k1));
        // product of two reflections = rotation by 2 * 60 degrees: determinant +1, no fixed axis
        double det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
        assertEquals(1, det, 1e-6);
        assertEquals(Math.cos(2 * Math.PI / 3), h[0][0], 1e-6);
    }

    private static float[][] householder(float[] k) {
        return new float[][] {{1 - 2 * k[0] * k[0], -2 * k[0] * k[1]}, {-2 * k[1] * k[0], 1 - 2 * k[1] * k[1]}};
    }

    private static float[][] multiply(float[][] a, float[][] b) {
        float[][] c = new float[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
            }
        }
        return c;
    }

    @Test
    void logLinearParallelMatchesFenwickRecurrence() {
        LogLinearAttention ll = new LogLinearAttention(8, 2, 16, true, new SplittableRandom(7));
        Var x = input(2, 13, 8, 8);
        assertArrayEquals(ll.recurrent(x).value().toArray(), ll.forward(x).value().toArray(), 1e-4f);
        LogLinearAttention linear = new LogLinearAttention(8, 2, 16, false, new SplittableRandom(7));
        assertArrayEquals(linear.recurrent(x).value().toArray(), linear.forward(x).value().toArray(), 1e-4f);
        assertEquals(0, LogLinearAttention.level(5, 5));
        assertEquals(1, LogLinearAttention.level(5, 4));
        assertEquals(3, LogLinearAttention.level(5, 0));
        assertEquals(6, ll.levels());
    }

    @Test
    void pathScoresEqualExplicitHouseholderProducts() {
        SplittableRandom r = new SplittableRandom(9);
        int t = 5;
        int d = 3;
        Var q = Var.of(Tensor.normal(Shape.of(1, 1, t, d), 0, 1, r));
        Var k = Var.of(Tensor.normal(Shape.of(1, 1, t, d), 0, 1, r));
        Tensor wRaw = Tensor.normal(Shape.of(1, 1, t, d), 0, 1, r);
        Var w = Var.of(wRaw.div(wRaw.square().sum(-1, true).sqrt()));
        Var beta = Var.of(Tensor.uniform(Shape.of(1, 1, t), 0, 2, r));
        float[] s = PathPositionAttention.scores(q, k, w, beta).value().toArray();
        float[] qa = q.value().toArray();
        float[] ka = k.value().toArray();
        float[] wa = w.value().toArray();
        float[] ba = beta.value().toArray();
        for (int i = 0; i < t; i++) {
            for (int j = 0; j <= i; j++) {
                // apply H_{j+1} ... H_i to k_j from the left, innermost first: k <- H_{j+1} k, ...
                double[] vec = new double[d];
                for (int c = 0; c < d; c++) {
                    vec[c] = ka[j * d + c];
                }
                for (int m = j + 1; m <= i; m++) {
                    double dot = 0;
                    for (int c = 0; c < d; c++) {
                        dot += wa[m * d + c] * vec[c];
                    }
                    for (int c = 0; c < d; c++) {
                        vec[c] -= ba[m] * dot * wa[m * d + c];
                    }
                }
                double expect = 0;
                for (int c = 0; c < d; c++) {
                    expect += qa[i * d + c] * vec[c];
                }
                assertEquals(expect, s[i * t + j], 1e-4, "score " + i + "," + j);
            }
        }
    }

    @Test
    void pathAttentionIsCausal() {
        PathPositionAttention path = new PathPositionAttention(8, 2, true, new SplittableRandom(11));
        Var x = input(1, 6, 8, 12);
        float[] y = path.forward(x).value().toArray();
        float[] changed = x.value().toArray();
        for (int c = 0; c < 8; c++) {
            changed[5 * 8 + c] += 1;
        }
        float[] y2 = path.forward(Var.of(Tensor.of(Shape.of(1, 6, 8), changed))).value().toArray();
        for (int i = 0; i < 5 * 8; i++) {
            assertEquals(y[i], y2[i], 1e-5f);
        }
    }
}
