package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class SequenceLayersTest {

    private static Tensor random(long seed, float lo, float hi, int... dims) {
        return Tensor.uniform(Shape.of(dims), lo, hi, new SplittableRandom(seed));
    }

    @Test
    void selectiveScanMatchesTheRecurrenceByHand() {
        int batch = 2;
        int steps = 4;
        int d = 3;
        int n = 2;
        Tensor x = random(1, -1, 1, batch, steps, d);
        Tensor delta = random(2, 0.1f, 1, batch, steps, d);
        Tensor a = random(3, -1, -0.1f, d, n);
        Tensor b = random(4, -1, 1, batch, steps, n);
        Tensor c = random(5, -1, 1, batch, steps, n);
        float[] xv = x.toArray();
        float[] dv = delta.toArray();
        float[] av = a.toArray();
        float[] bv = b.toArray();
        float[] cv = c.toArray();
        float[] expected = new float[batch * steps * d];
        for (int bi = 0; bi < batch; bi++) {
            for (int f = 0; f < d; f++) {
                float[] h = new float[n];
                for (int t = 0; t < steps; t++) {
                    int at = (bi * steps + t) * d + f;
                    float y = 0;
                    for (int j = 0; j < n; j++) {
                        h[j] = (float) Math.exp(dv[at] * av[f * n + j]) * h[j]
                                + dv[at] * bv[(bi * steps + t) * n + j] * xv[at];
                        y += h[j] * cv[(bi * steps + t) * n + j];
                    }
                    expected[at] = y;
                }
            }
        }
        assertArrayEquals(expected, Functional.selectiveScan(Var.of(x), Var.of(delta), Var.of(a), Var.of(b),
                Var.of(c)).value().toArray(), 1e-5f);
        GradCheck.Result r = GradCheck.check(v -> Functional.selectiveScan(v.get(0), v.get(1), v.get(2), v.get(3),
                v.get(4)).square().sum(), List.of(x, delta, a, b, c), 1e-3f, 2e-2f);
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void linearAttentionMatchesTheKernelFormula() {
        int t = 5;
        Tensor q = random(6, -1, 1, 1, 2, t, 3);
        Tensor k = random(7, -1, 1, 1, 2, t, 3);
        Tensor v = random(8, -1, 1, 1, 2, t, 4);
        for (boolean causal : new boolean[] {false, true}) {
            float[] qv = q.toArray();
            float[] kv = k.toArray();
            float[] vv = v.toArray();
            float[] expected = new float[2 * t * 4];
            for (int h = 0; h < 2; h++) {
                for (int i = 0; i < t; i++) {
                    double den = 0;
                    double[] num = new double[4];
                    for (int j = 0; j < (causal ? i + 1 : t); j++) {
                        double w = 0;
                        for (int e = 0; e < 3; e++) {
                            w += phi(qv[(h * t + i) * 3 + e]) * phi(kv[(h * t + j) * 3 + e]);
                        }
                        den += w;
                        for (int e = 0; e < 4; e++) {
                            num[e] += w * vv[(h * t + j) * 4 + e];
                        }
                    }
                    for (int e = 0; e < 4; e++) {
                        expected[(h * t + i) * 4 + e] = (float) (num[e] / den);
                    }
                }
            }
            assertArrayEquals(expected, Functional.linearAttention(Var.of(q), Var.of(k), Var.of(v), causal).value()
                    .toArray(), 1e-5f, "causal " + causal);
            GradCheck.Result r = GradCheck.check(vs -> Functional.linearAttention(vs.get(0), vs.get(1), vs.get(2),
                    causal).square().sum(), List.of(q, k, v), 1e-3f, 2e-2f);
            assertTrue(r.passed(), r.toString());
        }
    }

    private static double phi(float x) {
        return x > 0 ? x + 1 : Math.exp(x);
    }
}
