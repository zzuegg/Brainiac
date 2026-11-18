package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class SatNetLayerTest {

    private static float loss(SatNetLayer layer, Tensor z, Tensor mask, float[] weights) {
        return layer.solve(Var.of(z), mask).mul(Tensor.of(Shape.of(2, 4), weights)).sum().value().item();
    }

    @Test
    void analyticGradientMatchesFiniteDifferences() {
        SatNetLayer layer = new SatNetLayer(4, 6, 2, new SplittableRandom(3)).sweeps(400);
        Tensor z = Tensor.of(Shape.of(2, 4), new float[] {0.2f, 0.9f, 0, 0, 0.7f, 0.1f, 0, 0});
        Tensor mask = Tensor.of(Shape.of(2, 4), new float[] {1, 1, 0, 0, 1, 1, 0, 0});
        float[] w = {0, 0, 1, -0.5f, 0, 0, 0.3f, 1};
        Var zv = Var.param(z);
        layer.solve(zv, mask).mul(Tensor.of(Shape.of(2, 4), w)).sum().backward();
        float[] gs = layer.clauses().grad().toArray();
        float[] gz = zv.grad().toArray();
        float[] s0 = layer.clauses().value().toArray();
        double eps = 1e-3;
        double worst = 0;
        double scale = 0;
        for (int idx = 0; idx < s0.length; idx += 5) {
            float[] plus = s0.clone();
            float[] minus = s0.clone();
            plus[idx] += (float) eps;
            minus[idx] -= (float) eps;
            layer.clauses().assign(Tensor.of(layer.clauses().shape(), plus));
            float lp = loss(layer, z, mask, w);
            layer.clauses().assign(Tensor.of(layer.clauses().shape(), minus));
            float lm = loss(layer, z, mask, w);
            double numeric = (lp - lm) / (2 * eps);
            worst = Math.max(worst, Math.abs(numeric - gs[idx]));
            scale = Math.max(scale, Math.abs(numeric));
        }
        layer.clauses().assign(Tensor.of(layer.clauses().shape(), s0));
        float[] zz = z.toArray();
        for (int idx : new int[] {0, 1, 4, 5}) {
            float[] plus = zz.clone();
            float[] minus = zz.clone();
            plus[idx] += (float) eps;
            minus[idx] -= (float) eps;
            double numeric = (loss(layer, Tensor.of(z.shape(), plus), mask, w)
                    - loss(layer, Tensor.of(z.shape(), minus), mask, w)) / (2 * eps);
            worst = Math.max(worst, Math.abs(numeric - gz[idx]));
            scale = Math.max(scale, Math.abs(numeric));
        }
        System.out.printf("SATNet gradient check: max abs error %.2e at gradient scale %.2e%n", worst, scale);
        assertTrue(worst < 0.05 * scale + 2e-3, "analytic and numeric gradients agree: " + worst);
    }

    @Test
    void learnsXorFromExamples() {
        SatNetLayer layer = new SatNetLayer(3, 8, 4, new SplittableRandom(1)).sweeps(20);
        Optimizer opt = Optimizer.adam(0.05f);
        float[] z = new float[4 * 3];
        float[] mask = new float[4 * 3];
        float[] y = new float[4];
        for (int i = 0; i < 4; i++) {
            z[i * 3] = i & 1;
            z[i * 3 + 1] = (i >> 1) & 1;
            mask[i * 3] = 1;
            mask[i * 3 + 1] = 1;
            y[i] = (i & 1) ^ ((i >> 1) & 1);
        }
        Tensor zt = Tensor.of(Shape.of(4, 3), z);
        Tensor mt = Tensor.of(Shape.of(4, 3), mask);
        Tensor yt = Tensor.of(Shape.of(4, 1), y);
        float last = 0;
        for (int s = 0; s < 150; s++) {
            Var p = layer.solve(Var.of(zt), mt).slice(1, 2, 3).clamp(1e-4f, 1 - 1e-4f);
            Var bce = p.log().mul(yt).add(p.neg().add(1f).log().mul(yt.neg().add(1f))).neg().mean();
            last = bce.value().item();
            bce.backward();
            opt.step(layer.parameters());
        }
        float[] out = layer.solve(Var.of(zt), mt).value().toArray();
        int right = 0;
        for (int i = 0; i < 4; i++) {
            right += (out[i * 3 + 2] > 0.5f ? 1 : 0) == (int) y[i] ? 1 : 0;
        }
        System.out.printf("SATNet XOR: %d/4 right, final BCE %.3f%n", right, last);
        assertEquals(4, right, "SATNet learns XOR's clauses");
    }
}
