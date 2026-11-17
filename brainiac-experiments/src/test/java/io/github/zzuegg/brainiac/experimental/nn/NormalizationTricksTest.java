package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.PositionalAttention.Position;
import io.github.zzuegg.brainiac.experimental.nn.TunableTransformerBlock.Norm;
import io.github.zzuegg.brainiac.experimental.nn.TunableTransformerBlock.Placement;
import io.github.zzuegg.brainiac.experimental.nn.TunableTransformerBlock.Residual;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class NormalizationTricksTest {

    private static Tensor randn(long seed, int... dims) {
        return Tensor.normal(Shape.of(dims), 0, 1, new SplittableRandom(seed));
    }

    @Test
    void dynamicTanhMatchesFormulaAndGradients() {
        DynamicTanh dyt = new DynamicTanh(3, 0.7f);
        Tensor x = Tensor.of(Shape.of(1, 3), 1f, -2f, 0.5f);
        float[] y = dyt.forward(Var.of(x)).value().toArray();
        assertEquals(Math.tanh(0.7), y[0], 1e-5);
        assertEquals(Math.tanh(-1.4), y[1], 1e-5);
        GradCheck.Result r = GradCheck.check(v -> v.get(0).mul(v.get(1)).tanh().mul(v.get(2)).square().sum(),
                List.of(randn(1, 2, 3), Tensor.full(Shape.of(1), 0.5f), randn(2, 3)));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void rezeroBlockStartsAsIdentity() {
        TunableTransformerBlock b = TunableTransformerBlock.of(2).norm(Norm.NONE).residual(Residual.REZERO)
                .build(8, new SplittableRandom(3));
        Tensor x = randn(4, 2, 5, 8);
        assertArrayEquals(x.toArray(), b.forward(Var.of(x)).value().toArray(), 1e-6f);
    }

    @Test
    void layerScaleKeepsBranchesSmallAtStart() {
        TunableTransformerBlock b = TunableTransformerBlock.of(2).residual(Residual.LAYER_SCALE)
                .build(8, new SplittableRandom(5));
        Tensor x = randn(6, 2, 5, 8);
        float[] in = x.toArray();
        float[] out = b.forward(Var.of(x)).value().toArray();
        double diff = 0;
        for (int i = 0; i < in.length; i++) {
            diff = Math.max(diff, Math.abs(in[i] - out[i]));
        }
        assertTrue(diff < 0.2, "layer scale 1e-2 keeps the block near identity, max change " + diff);
    }

    @Test
    void dropPathDropsWholeSamplesOnlyWhileTraining() {
        DropPath dp = new DropPath(0.5f, new SplittableRandom(7));
        Tensor x = Tensor.ones(Shape.of(64, 3));
        assertArrayEquals(x.toArray(), dp.forward(Var.of(x), false).value().toArray());
        float[] y = dp.forward(Var.of(x), true).value().toArray();
        int dropped = 0;
        for (int i = 0; i < 64; i++) {
            assertEquals(y[i * 3], y[i * 3 + 2]);
            assertTrue(y[i * 3] == 0f || y[i * 3] == 2f);
            dropped += y[i * 3] == 0f ? 1 : 0;
        }
        assertTrue(dropped > 16 && dropped < 48, "about half dropped: " + dropped);
        assertEquals(0.25f, DropPath.linearRate(1, 3, 0.5f), 1e-6f);
    }

    @Test
    void deepNormConstants() {
        assertEquals(Math.pow(24, 0.25), DeepNorm.alpha(12), 1e-5);
        assertEquals(Math.pow(96, -0.25), DeepNorm.beta(12), 1e-5);
    }

    @Test
    void qkNormMakesAttentionWeightsScaleInvariant() {
        PositionalAttention a = new PositionalAttention(PositionalAttention.Settings.of(8, 2, true, Position.NONE,
                true), new SplittableRandom(8));
        Tensor x = randn(9, 1, 6, 8);
        float[] y1 = a.forward(Var.of(x)).value().toArray();
        float[] y2 = a.forward(Var.of(x.mul(1000f))).value().toArray();
        for (int i = 0; i < y1.length; i++) {
            assertEquals(y1[i] * 1000f, y2[i], 1e-2f * Math.max(1, Math.abs(y2[i])));
        }
    }

    @Test
    void checkpointedBlockGivesTheSameGradients() {
        TunableTransformerBlock.Spec spec = TunableTransformerBlock.of(2).norm(Norm.RMS).position(Position.ROTARY);
        TunableTransformerBlock plain = spec.build(8, new SplittableRandom(10));
        TunableTransformerBlock ckpt = spec.checkpoint(true).build(8, new SplittableRandom(10));
        Tensor x = randn(11, 2, 4, 8);
        plain.forward(Var.of(x), true).square().sum().backward();
        ckpt.forward(Var.of(x), true).square().sum().backward();
        for (int i = 0; i < plain.parameters().size(); i++) {
            assertArrayEquals(plain.parameters().get(i).grad().toArray(), ckpt.parameters().get(i).grad().toArray(),
                    1e-4f);
        }
    }

    /** Every combination of norm, placement and residual runs, and a few train a toy task. */
    @Test
    void everyVariantTrainsAToySequenceTask() {
        Tensor x = randn(12, 8, 6, 8);
        // target: each step's features reversed in sign and shifted by the mean of the sequence
        Tensor target = x.neg().add(x.mean(1, true));
        for (Norm n : Norm.values()) {
            for (Placement p : Placement.values()) {
                for (Residual r : Residual.values()) {
                    TunableTransformerBlock.Spec spec = TunableTransformerBlock.of(2).norm(n).placement(p)
                            .residual(r).ffn(GluFeedForward.Kind.SWIGLU).ffnFactor(2).dropPath(0.1f).depth(2);
                    Model m = Model.sequential().input(6, 8).add(spec).add(spec).build();
                    Optimizer opt = Optimizer.adam(0.01f);
                    float first = 0;
                    float last = 0;
                    int steps = n == Norm.DYT && p == Placement.PERI && r == Residual.PLAIN ? 60 : 3;
                    for (int s = 0; s < steps; s++) {
                        Var loss = m.forward(Var.of(x), true).sub(target).square().mean();
                        last = loss.value().item();
                        if (s == 0) {
                            first = last;
                        }
                        assertTrue(Float.isFinite(last), spec + " diverged");
                        loss.backward();
                        opt.step(m.parameters());
                    }
                    if (steps > 3) {
                        assertTrue(last < 0.5f * first, "DyT peri-LN block learns: " + first + " -> " + last);
                    }
                }
            }
        }
    }

    @Test
    void weightStandardizationNormalizesEachFilter() {
        WeightStandardizedConv c = new WeightStandardizedConv(2, 3, new int[] {3, 3}, new int[] {1, 1},
                new SplittableRandom(13));
        float[] w = c.standardizedWeight().value().toArray();
        for (int f = 0; f < 3; f++) {
            double mean = 0;
            double sq = 0;
            for (int i = 0; i < 18; i++) {
                mean += w[f * 18 + i] / 18.0;
                sq += w[f * 18 + i] * w[f * 18 + i] / 18.0;
            }
            assertEquals(0, mean, 1e-4);
            assertEquals(1, sq, 1e-3);
        }
        Var y = c.forward(Var.of(randn(14, 2, 2, 5, 5)));
        assertEquals(Shape.of(2, 3, 5, 5), y.shape());
        GradCheck.Result r = GradCheck.check(v -> {
            Var wv = v.get(0).reshape(3, 18);
            Var cc = wv.sub(wv.mean(1, true));
            return cc.div(cc.square().mean(1, true).add(1e-5f).sqrt()).mul(Tensor.arange(18).add(1f)).sum();
        }, List.of(randn(15, 3, 2, 3, 3)));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void spectralNormFindsTheTopSingularValue() {
        SplittableRandom rnd = new SplittableRandom(16);
        Tensor w = Tensor.normal(Shape.of(6, 4), 0, 1, rnd);
        SpectralNorm sn = new SpectralNorm(6, 1, rnd);
        Var normalized = null;
        for (int i = 0; i < 50; i++) {
            normalized = sn.normalize(Var.param(w));
        }
        // the top singular value from many power iterations on W^T W
        Tensor v = Tensor.ones(Shape.of(4, 1));
        for (int i = 0; i < 500; i++) {
            v = w.transpose().matmul(w.matmul(v));
            v = v.div(v.square().sum().sqrt());
        }
        float sigma = w.matmul(v).square().sum().sqrt().item();
        assertEquals(sigma, sn.lastSigma(), 1e-3f * sigma);
        Tensor nv = normalized.value();
        Tensor u = Tensor.ones(Shape.of(4, 1));
        for (int i = 0; i < 500; i++) {
            u = nv.transpose().matmul(nv.matmul(u));
            u = u.div(u.square().sum().sqrt());
        }
        assertEquals(1f, nv.matmul(u).square().sum().sqrt().item(), 1e-3f);
        SpectralNormDense d = new SpectralNormDense(4, 3, io.github.zzuegg.brainiac.nn.Activation.RELU, rnd);
        d.forward(Var.of(randn(17, 2, 4))).sum().backward();
        assertTrue(d.parameters().get(0).grad() != null);
    }
}
