package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class TransducerLossTest {

    private static final int T = 3;
    private static final int U = 2;
    private static final int V = 3;

    private static Tensor logProbs(SplittableRandom r) {
        float[] z = new float[T * (U + 1) * V];
        for (int i = 0; i < z.length; i += V) {
            double norm = 0;
            for (int k = 0; k < V; k++) {
                z[i + k] = (float) r.nextGaussian();
                norm += Math.exp(z[i + k]);
            }
            for (int k = 0; k < V; k++) {
                z[i + k] -= (float) Math.log(norm);
            }
        }
        return Tensor.of(Shape.of(1, T, U + 1, V), z);
    }

    /** Sums the probability of every alignment path on the lattice by recursion. */
    private static double brute(float[] lp, int[] y, int t, int u) {
        int base = (t * (U + 1) + u) * V;
        if (t == T - 1 && u == y.length) {
            return Math.exp(lp[base]);
        }
        double p = 0;
        if (t < T - 1) {
            p += Math.exp(lp[base]) * brute(lp, y, t + 1, u);
        }
        if (u < y.length) {
            p += Math.exp(lp[base + y[u]]) * brute(lp, y, t, u + 1);
        }
        return p;
    }

    @Test
    void lossMatchesPathEnumerationAndGradientMatchesFiniteDifferences() {
        Tensor lp = logProbs(new SplittableRandom(5));
        int[][] y = {{1, 2}};
        Var x = Var.param(lp);
        Var loss = TransducerLoss.loss(x, y);
        double exact = -Math.log(brute(lp.toArray(), y[0], 0, 0));
        assertEquals(exact, loss.value().item(), 1e-4);
        loss.backward();
        float[] g = x.grad().toArray();
        float[] base = lp.toArray();
        double worst = 0;
        for (int i = 0; i < base.length; i++) {
            float[] plus = base.clone();
            float[] minus = base.clone();
            plus[i] += 1e-3f;
            minus[i] -= 1e-3f;
            double fd = (-Math.log(brute(plus, y[0], 0, 0)) + Math.log(brute(minus, y[0], 0, 0))) / 2e-3;
            worst = Math.max(worst, Math.abs(fd - g[i]));
        }
        System.out.printf("transducer loss %.5f vs paths %.5f, worst gradient error %.2e%n",
                loss.value().item(), exact, worst);
        assertTrue(worst < 1e-3, "gradient error " + worst);
    }
}
