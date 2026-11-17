package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class MixtureOfExpertsTest {

    /** A piecewise target: a different function of x in each quadrant of (x0, x1). */
    private static Tensor[] regression(SplittableRandom r, int rows) {
        float[] x = new float[rows * 4];
        float[] y = new float[rows * 4];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < 4; j++) {
                x[i * 4 + j] = (float) (r.nextDouble() * 2 - 1);
            }
            int quadrant = (x[i * 4] > 0 ? 1 : 0) + (x[i * 4 + 1] > 0 ? 2 : 0);
            for (int j = 0; j < 4; j++) {
                float v = x[i * 4 + (j + quadrant) % 4];
                y[i * 4 + j] = switch (quadrant) {
                    case 0 -> v * v;
                    case 1 -> -v;
                    case 2 -> (float) Math.sin(3 * v);
                    default -> Math.abs(v) - 0.5f;
                };
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(rows, 4), x), Tensor.of(Shape.of(rows, 4), y)};
    }

    /** Trains a [rows, 4] -> [rows, 4] layer; returns the final MSE relative to the target variance. */
    private static double fit(Layer layer, int steps, long seed, Runnable afterStep) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(0.01f);
        for (int s = 0; s < steps; s++) {
            Tensor[] b = regression(r, 64);
            Var loss = layer.forward(Var.of(b[0]), true).sub(b[1]).square().mean();
            loss.backward();
            opt.step(layer.parameters());
            if (afterStep != null) {
                afterStep.run();
            }
        }
        Tensor[] test = regression(new SplittableRandom(seed + 1), 512);
        double mse = layer.forward(Var.of(test[0])).sub(Var.of(test[1])).square().mean().value().item();
        double var = test[1].sub(test[1].mean()).square().mean().item();
        return mse / var;
    }

    @Test
    void sparseDispatchMatchesDenseComputation() {
        FineGrainedMoe moe = FineGrainedMoe.of(4, 6, 1, 8, 2, new SplittableRandom(1));
        Tensor x = Tensor.uniform(Shape.of(5, 4), -1, 1, new SplittableRandom(2));
        Var y = moe.forward(Var.of(x));
        int[][] a = moe.assignments(Var.of(x));
        float[] aff = MixerOps.softmax(Var.of(x).matmul(moe.router())).value().toArray();
        Var dense = moe.dense(Var.of(x), e -> {
            float[] w = new float[5];
            for (int n = 0; n < 5; n++) {
                for (int k : a[n]) {
                    if (k == e) {
                        w[n] = aff[n * 6 + e];
                    }
                }
            }
            return Var.of(Tensor.of(Shape.of(5, 1), w));
        });
        assertArrayEquals(dense.value().toArray(), y.value().toArray(), 1e-5f);
        assertEquals(10, Arrays.stream(moe.expertLoad()).sum());
    }

    @Test
    void lossFreeBiasBalancesExpertLoad() {
        FineGrainedMoe moe = FineGrainedMoe.of(4, 8, 0, 4, 1, new SplittableRandom(3));
        moe.router().assign(moe.router().value().mul(8f));   // a skewed router
        Tensor x = Tensor.uniform(Shape.of(256, 4), -1, 1, new SplittableRandom(4));
        moe.forward(Var.of(x));
        int before = Arrays.stream(moe.expertLoad()).max().getAsInt();
        for (int i = 0; i < 300; i++) {
            moe.updateBalanceBias(0.002f);
            moe.forward(Var.of(x));
        }
        int after = Arrays.stream(moe.expertLoad()).max().getAsInt();
        assertTrue(after < before * 0.7 && after < 64, "max load " + before + " -> " + after);
        assertNotNull(moe.balanceLoss());
    }

    @Test
    void fineGrainedMoeWithSharedExpertLearnsPiecewiseFunction() {
        FineGrainedMoe moe = FineGrainedMoe.of(4, 8, 1, 16, 2, new SplittableRandom(5));
        double rel = fit(moe, 800, 6, null);
        assertTrue(rel < 0.25, "relative mse " + rel);
    }

    @Test
    void switchingNeuronGroupsLearnPiecewiseFunction() {
        SwitchingNeuronMoe moe = new SwitchingNeuronMoe(4, 4, 8, 8, 2,
                new Activation[] {Activation.RELU, Activation.TANH, Activation.SILU}, new SplittableRandom(7));
        double rel = fit(moe, 300, 8, null);
        assertTrue(rel < 0.3, "relative mse " + rel);
    }

    @Test
    void softMoeDispatchIsNormalizedAndPermutationEquivariant() {
        SoftMoe moe = new SoftMoe(4, 3, 2, 8, new SplittableRandom(9));
        Tensor x = Tensor.uniform(Shape.of(1, 5, 4), -1, 1, new SplittableRandom(10));
        float[] d = moe.dispatchWeights(Var.of(x)).toArray();
        for (int slot = 0; slot < 6; slot++) {
            float sum = 0;
            for (int t = 0; t < 5; t++) {
                sum += d[t * 6 + slot];
            }
            assertEquals(1, sum, 1e-5f);
        }
        float[] y = moe.forward(Var.of(x)).value().toArray();
        Tensor flipped = x.flip(1);
        float[] yf = moe.forward(Var.of(flipped)).value().toArray();
        for (int t = 0; t < 5; t++) {
            for (int j = 0; j < 4; j++) {
                assertEquals(y[t * 4 + j], yf[(4 - t) * 4 + j], 1e-5f);
            }
        }
    }

    @Test
    void softMoeLearnsPiecewiseFunction() {
        SoftMoe moe = new SoftMoe(4, 4, 4, 16, new SplittableRandom(11));
        Layer perToken = new Layer() {
            @Override
            public Var forward(Var input) {
                int rows = input.shape().dim(0);
                return moe.forward(input.reshape(rows / 16, 16, 4)).reshape(rows, 4);
            }

            @Override
            public java.util.List<Var> parameters() {
                return moe.parameters();
            }
        };
        double rel = fit(perToken, 300, 12, null);
        assertTrue(rel < 0.5, "relative mse " + rel);
    }

    @Test
    void productKeysFindTheBestExpertsAndOnlyTheyGetGradients() {
        ProductKeyExperts peer = new ProductKeyExperts(4, 32, 2, 4, 8, new SplittableRandom(13));
        assertEquals(1024, peer.experts());
        Tensor x = Tensor.uniform(Shape.of(3, 4), -1, 1, new SplittableRandom(14));
        int[][] got = peer.retrieve(Var.of(x));
        // brute force over all 1024 experts
        Var q = Var.of(x).matmul(peer.parameters().get(0)).reshape(6, 8);
        float[] s1 = q.slice(1, 0, 4).matmul(peer.parameters().get(1).transpose()).value().toArray();
        float[] s2 = q.slice(1, 4, 8).matmul(peer.parameters().get(2).transpose()).value().toArray();
        for (int qi = 0; qi < 6; qi++) {
            float[] all = new float[1024];
            for (int e = 0; e < 1024; e++) {
                all[e] = s1[qi * 32 + e / 32] + s2[qi * 32 + e % 32];
            }
            int[] best = MixerOps.topK(all, 0, 1024, 4);
            assertArrayEquals(best, got[qi]);
        }
        peer.forward(Var.of(x)).sum().backward();
        float[] g = peer.parameters().get(3).grad().toArray();
        int touched = 0;
        for (int e = 0; e < 1024; e++) {
            boolean nonzero = false;
            for (int j = 0; j < 4; j++) {
                nonzero |= g[e * 4 + j] != 0;
            }
            touched += nonzero ? 1 : 0;
        }
        assertTrue(touched <= 3 * 2 * 4 && touched > 0, "touched " + touched);
    }

    @Test
    void productKeyExpertsLearnPiecewiseFunction() {
        ProductKeyExperts peer = new ProductKeyExperts(4, 16, 4, 4, 8, new SplittableRandom(15));
        double rel = fit(peer, 300, 16, null);
        assertTrue(rel < 0.5, "relative mse " + rel);
    }

    @Test
    void mixtureOfDepthsProcessesOnlyCapacityTokensAndTrainsItsPredictor() {
        GatedAttention attn = new GatedAttention(8, 2, true, 0f, new SplittableRandom(17));
        MixtureOfDepths mod = new MixtureOfDepths(attn, 8, 0.25f, new SplittableRandom(18));
        Tensor x = Tensor.uniform(Shape.of(2, 8, 8), -1, 1, new SplittableRandom(19));
        float[] y = mod.forward(Var.of(x), true).value().toArray();
        float[] in = x.toArray();
        int changed = 0;
        for (int row = 0; row < 16; row++) {
            boolean diff = false;
            for (int j = 0; j < 8; j++) {
                diff |= Math.abs(y[row * 8 + j] - in[row * 8 + j]) > 1e-7;
            }
            changed += diff ? 1 : 0;
        }
        assertEquals(4, changed);
        assertEquals(2, mod.lastSelected()[0].length);
        Optimizer opt = Optimizer.adam(0.05f);
        float first = mod.predictorLoss().value().item();
        for (int i = 0; i < 100; i++) {
            mod.forward(Var.of(x), true);
            mod.predictorLoss().backward();
            opt.step(mod.parameters());
        }
        float last = mod.predictorLoss().value().item();
        assertTrue(last < first * 0.5, "predictor loss " + first + " -> " + last);
        mod.useCausalPredictor(true);
        assertEquals(Shape.of(2, 8, 8), mod.forward(Var.of(x)).shape());
    }

    @Test
    void mixtureOfDepthsLearnsRecallAtHalfCapacity() {
        GatedAttention attn = new GatedAttention(16, 2, true, 0f, new SplittableRandom(20));
        MixtureOfDepths mod = new MixtureOfDepths(attn, 16, 0.5f, new SplittableRandom(21));
        double acc = MixerHarness.train(mod, MixerHarness.recall(6, 6, 3), 16, 300, 0.02f, 22);
        assertTrue(acc > 0.6, "accuracy " + acc);
    }
}
