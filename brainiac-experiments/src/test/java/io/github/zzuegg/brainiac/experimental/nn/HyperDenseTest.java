package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HyperDenseTest {
    @Test
    void appliesTheGeneratedMatrixPerExample() {
        HyperDense h = HyperDense.of(2, 3, 2, 8, 1);
        Tensor z = Tensor.normal(Shape.of(2, 2), 0, 1, new SplittableRandom(2));
        Tensor x = Tensor.normal(Shape.of(2, 3), 0, 1, new SplittableRandom(3));
        float[] g = h.generate(Var.of(z)).value().toArray();
        float[] xs = x.toArray();
        float[] y = h.forward(Var.of(z), Var.of(x)).value().toArray();
        for (int b = 0; b < 2; b++) {
            for (int o = 0; o < 2; o++) {
                double expect = g[b * 8 + 6 + o];
                for (int i = 0; i < 3; i++) {
                    expect += xs[b * 3 + i] * g[b * 8 + i * 2 + o];
                }
                assertEquals(expect, y[b * 2 + o], 1e-5);
            }
        }
        // packed form passes the context through
        float[] packed = h.forward(Var.concat(1, Var.of(z), Var.of(x))).value().toArray();
        assertArrayEquals(new float[] {z.toArray()[0], z.toArray()[1], y[0], y[1]}, java.util.Arrays.copyOf(packed, 4),
                1e-6f);
    }

    @Test
    void gradientsFlowIntoTheHypernetwork() {
        HyperDense h = HyperDense.of(2, 2, 2, 4, 4);
        Tensor in = Tensor.normal(Shape.of(3, 4), 0, 1, new SplittableRandom(5));
        GradCheck.Result g = GradCheck.check(v -> h.forward(v.get(0)).square().sum(), List.of(in), 1e-3f, 2e-2f);
        assertTrue(g.passed(), g.toString());
    }
}
