package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.PositionalAttention.Position;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class ActivationsAndPositionsTest {

    private static Tensor randn(long seed, int... dims) {
        return Tensor.normal(Shape.of(dims), 0, 1, new SplittableRandom(seed));
    }

    @Test
    void squaredReluAndSnake() {
        float[] y = ModernActivations.SQUARED_RELU.forward(Var.of(Tensor.of(-1f, 3f))).value().toArray();
        assertEquals(0f, y[0]);
        assertEquals(9f, y[1]);
        GradCheck.Result r = GradCheck.check(v -> ModernActivations.snake(v.get(0), v.get(1)).sum(),
                List.of(randn(1, 3, 4), Tensor.of(Shape.of(4), 0.5f, 1f, 1.5f, 2f)), 1e-2f, 1e-2f);
        assertTrue(r.passed(), r.toString());
        Activation snake = ModernActivations.snake().bind(4);
        assertEquals(1, snake.parameters().size());
        float[] s = snake.forward(Var.of(Tensor.of(Shape.of(1, 4), 0f, 1f, 2f, 3f))).value().toArray();
        assertEquals(1 + Math.pow(Math.sin(1), 2), s[1], 1e-5);
    }

    @Test
    void rationalStartsAsLeakyReluAndHasGradients() {
        Activation pau = ModernActivations.rational().bind(61);
        float[] xs = new float[61];
        for (int i = 0; i < xs.length; i++) {
            xs[i] = -3 + 0.1f * i;
        }
        float[] y = pau.forward(Var.of(Tensor.of(Shape.of(1, 61), xs))).value().toArray();
        for (int i = 0; i < xs.length; i++) {
            double expected = xs[i] >= 0 ? xs[i] : 0.01 * xs[i];
            assertEquals(expected, y[i], 0.15, "PAU fit at " + xs[i]);
        }
        GradCheck.Result r = GradCheck.check(v -> ModernActivations.rationalForward(v.get(0), v.get(1), v.get(2))
                .square().sum(), List.of(randn(2, 2, 3), Tensor.of(0.1f, 0.5f, 0.3f), Tensor.of(0.4f, 0.2f)));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void gluVariantsLearnARegression() {
        Tensor x = randn(3, 64, 4);
        Tensor target = x.mul(x).sum(1, true).sub(x.slice(1, 0, 1)).expand(Shape.of(64, 4));
        for (GluFeedForward.Kind k : GluFeedForward.Kind.values()) {
            Model m = Model.sequential().input(4).add(GluFeedForward.spec(k, 4)).build();
            Optimizer opt = Optimizer.adam(0.02f);
            float first = 0;
            float last = 0;
            for (int s = 0; s < 150; s++) {
                Var loss = m.forward(Var.of(x)).sub(target).square().mean();
                last = loss.value().item();
                first = s == 0 ? last : first;
                loss.backward();
                opt.step(m.parameters());
            }
            assertTrue(last < 0.3f * first, k + ": " + first + " -> " + last);
        }
    }

    /** Scores of identical tokens must depend only on the offset between them. */
    private static void assertToeplitz(Var q, Var k, int steps, double tol) {
        float[] s = q.matmul(k.transpose()).value().toArray();
        for (int i = 1; i < steps; i++) {
            for (int j = 1; j < steps; j++) {
                assertEquals(s[(i - 1) * steps + j - 1], s[i * steps + j], tol);
            }
        }
    }

    @Test
    void xposScoresDependOnRelativePositionAndDecay() {
        Tensor one = randn(4, 1, 8);
        Var x = Var.of(one.expand(Shape.of(6, 8)));
        Var q = PositionEncodings.xpos(x, 10000f, 4f, 0, true);
        Var k = PositionEncodings.xpos(x, 10000f, 4f, 0, false);
        assertToeplitz(q, k, 6, 1e-3);
        // a vector on the slowest frequency pair barely rotates, so its score shows the decay alone
        float[] slow = new float[8 * 8];
        for (int t = 0; t < 8; t++) {
            slow[t * 8 + 3] = 1;
            slow[t * 8 + 7] = 1;
        }
        Var xs = Var.of(Tensor.of(Shape.of(8, 8), slow));
        Var qq = PositionEncodings.xpos(xs, 1e9f, 2f, 0, true);
        Var kk = PositionEncodings.xpos(xs, 1e9f, 2f, 0, false);
        float[] s = qq.matmul(kk.transpose()).value().toArray();
        double zeta = (6.0 / 8 + 0.4) / 1.4;
        assertEquals(2.0, s[7 * 8 + 7], 1e-4);
        assertEquals(2 * Math.pow(zeta, 7 / 2.0), s[7 * 8], 1e-3);
    }

    @Test
    void rotary2dScoresDependOnTheTwoDimensionalOffset() {
        Tensor one = randn(5, 1, 8);
        Var x = Var.of(one.expand(Shape.of(1 + 9, 8)));
        Var r = PositionEncodings.rotary2d(x, 3, 3, 100f, 1);
        float[] s = r.matmul(r.transpose()).value().toArray();
        // patch (0,0) vs (1,1) equals patch (1,1) vs (2,2); prefix token unrotated
        assertEquals(s[1 * 10 + 5], s[5 * 10 + 9], 1e-4);
        assertEquals(s[2 * 10 + 3], s[5 * 10 + 6], 1e-4);
        assertTrue(Math.abs(s[1 * 10 + 5] - s[1 * 10 + 3]) > 1e-3, "rows and columns differ");
        float[] raw = x.value().toArray();
        float[] rot = r.value().toArray();
        for (int i = 0; i < 8; i++) {
            assertEquals(raw[i], rot[i], 0f);
        }
    }

    @Test
    void t5BucketsAreExactNearAndLogarithmicFar() {
        assertEquals(0, PositionEncodings.t5Bucket(0, false, 32, 128));
        assertEquals(3, PositionEncodings.t5Bucket(-3, false, 32, 128));
        assertEquals(0, PositionEncodings.t5Bucket(5, false, 32, 128));
        assertEquals(31, PositionEncodings.t5Bucket(-1000, false, 32, 128));
        assertTrue(PositionEncodings.t5Bucket(4, true, 32, 128) >= 16);
        int prev = -1;
        for (int d = 0; d < 200; d++) {
            int b = PositionEncodings.t5Bucket(-d, false, 32, 128);
            assertTrue(b >= prev);
            prev = b;
        }
        Var table = Var.param(randn(6, 2, 32));
        Var bias = PositionEncodings.t5Bias(table, 5, 5, false, 128);
        assertEquals(Shape.of(2, 5, 5), bias.shape());
        bias.sum().backward();
        assertTrue(table.grad().sum().item() == 50f);
    }

    @Test
    void copeCountsGatedPositions() {
        // huge q.k: every gate is 1, so the contextual position of key j for query i is i - j + 1
        int t = 4;
        Var q = Var.of(Tensor.full(Shape.of(1, 1, t, 2), 10f));
        Var k = Var.of(Tensor.full(Shape.of(1, 1, t, 2), 10f));
        float[] emb = new float[8 * 2];
        for (int p = 0; p < 8; p++) {
            emb[p * 2] = p;
        }
        float[] bias = PositionEncodings.cope(q, k, Var.of(Tensor.of(Shape.of(8, 2), emb)), true).value().toArray();
        for (int i = 0; i < t; i++) {
            for (int j = 0; j <= i; j++) {
                assertEquals(10f * (i - j + 1), bias[i * t + j], 1e-3f);
            }
        }
        GradCheck.Result r = GradCheck.check(v -> PositionEncodings.cope(v.get(0), v.get(1), v.get(2), true)
                .mul(Tensor.arange(t).add(1f)).sum(), List.of(randn(7, 1, 1, t, 2).mul(0.5f),
                        randn(8, 1, 1, t, 2).mul(0.5f), randn(9, 6, 2)), 1e-3f, 2e-2f);
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void everyPositionModeRunsForwardAndBackward() {
        for (Position p : Position.values()) {
            PositionalAttention.Settings s = PositionalAttention.Settings.of(8, 2, p != Position.ROTARY_2D, p, true);
            int steps = 5;
            if (p == Position.ROTARY_2D) {
                s = s.grid(2, 2, 1);
            }
            PositionalAttention a = new PositionalAttention(s, new SplittableRandom(10));
            Var y = a.forward(Var.of(randn(11, 2, steps, 8)));
            assertEquals(Shape.of(2, steps, 8), y.shape());
            y.square().sum().backward();
            for (Var param : a.parameters()) {
                assertTrue(param.grad() != null, p + " parameter without gradient");
            }
        }
    }
}
