package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

import org.junit.jupiter.api.Test;

/** Linear attention, Gated DeltaNet and Mamba: two computation modes agree, and they learn. */
class LinearRecurrenceTest {

    static Tensor input(long seed, int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, new SplittableRandom(seed));
    }

    /** Both modes give the same output and the same parameter gradients. */
    static void assertModesAgree(List<Var> params, Function<Var, Var> a, Function<Var, Var> b, Tensor x, float tol) {
        Var ya = a.apply(Var.of(x));
        Var yb = b.apply(Var.of(x));
        assertArrayEquals(ya.value().toArray(), yb.value().toArray(), tol);
        Tensor w = input(99, x.shape().toArray());
        ya.mul(w).sum().backward();
        float[][] ga = params.stream().map(p -> p.grad().toArray()).toArray(float[][]::new);
        params.forEach(Var::zeroGrad);
        yb.mul(w).sum().backward();
        for (int i = 0; i < params.size(); i++) {
            assertArrayEquals(ga[i], params.get(i).grad().toArray(), tol * 10, "gradient of parameter " + i);
        }
        params.forEach(Var::zeroGrad);
    }

    @Test
    void linearAttentionParallelEqualsRecurrent() {
        LinearAttention la = new LinearAttention(8, 2, new SplittableRandom(1));
        assertModesAgree(la.parameters(), la::forward, la::recurrent, input(2, 2, 7, 8), 1e-4f);
    }

    @Test
    void gatedDeltaNetChunkwiseEqualsRecurrent() {
        GatedDeltaNet net = new GatedDeltaNet(8, 2, GatedDeltaNet.Gating.SCALAR, 3, new SplittableRandom(3));
        assertModesAgree(net.parameters(), net::chunkwise, net::recurrent, input(4, 2, 8, 8), 1e-4f);
    }

    @Test
    void gatedDeltaRecurrenceMatchesNaiveLoops() {
        // with beta = 1 and alpha = 1 the delta rule stores v exactly for a unit key
        float[] k = {0.6f, 0.8f};
        float[][] s = new float[2][2];
        float[] v1 = {1, 2};
        float[] v2 = {-3, 5};
        for (float[] v : new float[][] {v1, v2}) {
            float[] read = {k[0] * s[0][0] + k[1] * s[1][0], k[0] * s[0][1] + k[1] * s[1][1]};
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    s[i][j] += k[i] * (v[j] - read[j]);
                }
            }
        }
        float[] out = {k[0] * s[0][0] + k[1] * s[1][0], k[0] * s[0][1] + k[1] * s[1][1]};
        assertArrayEquals(v2, out, 1e-5f);
        // the unit lower-triangular inverse
        Tensor l = Tensor.of(Shape.of(4, 4), 0, 0, 0, 0, 0.5f, 0, 0, 0, -1, 2, 0, 0, 0.3f, 0.1f, -0.7f, 0);
        Var inv = GatedDeltaNet.unitLowerInverse(Var.of(l), 4);
        Tensor eye = Tensor.of(Shape.of(4, 4), 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
        assertArrayEquals(eye.toArray(), inv.value().matmul(l.add(eye)).toArray(), 1e-5f);
    }

    @Test
    void mambaParallelScanEqualsSelectiveScan() {
        MambaLayer mamba = new MambaLayer(4, 8, 4, new SplittableRandom(5));
        assertModesAgree(mamba.parameters(), mamba::forward, mamba::recurrent, input(6, 2, 6, 4), 1e-4f);
    }

    @Test
    void linearAttentionLearnsRecall() {
        double acc = MixerHarness.train(new LinearAttention(16, 2, new SplittableRandom(7)),
                MixerHarness.recall(6, 6, 3), 16, 300, 0.02f, 8);
        assertTrue(acc > 0.7, "accuracy " + acc);
    }

    @Test
    void gatedDeltaNetLearnsRecall() {
        double acc = MixerHarness.train(new GatedDeltaNet(16, 2, GatedDeltaNet.Gating.SCALAR, 4,
                new SplittableRandom(9)), MixerHarness.recall(6, 6, 3), 16, 300, 0.02f, 10);
        assertTrue(acc > 0.8, "accuracy " + acc);
    }

    @Test
    void kimiDeltaAttentionLearnsRecentRecall() {
        double acc = MixerHarness.train(new GatedDeltaNet(16, 2, GatedDeltaNet.Gating.CHANNEL, 4,
                new SplittableRandom(11)), MixerHarness.recentRecall(3, 6, 4), 16, 300, 0.02f, 12);
        assertTrue(acc > 0.8, "accuracy " + acc);
    }

    @Test
    void mambaLearnsRecall() {
        double acc = MixerHarness.train(new MambaLayer(16, 16, 8, new SplittableRandom(13)),
                MixerHarness.recall(4, 4, 2), 16, 300, 0.01f, 14);
        assertTrue(acc > 0.7, "accuracy " + acc);
    }
}
