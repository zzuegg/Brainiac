package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class VaeTest {

    @Test
    void klIsZeroAtTheStandardNormalAndMatchesTheFormula() {
        Var zero = Var.of(Tensor.zeros(Shape.of(2, 3)));
        assertEquals(0, Functional.gaussianKl(zero, zero).value().item(), 1e-7);
        float[] mu = {0.5f, -1, 2, 0};
        float[] lv = {0, 0.5f, -1, 1};
        double expected = 0;
        for (int i = 0; i < 4; i++) {
            expected += 0.5 * (mu[i] * mu[i] + Math.exp(lv[i]) - lv[i] - 1);
        }
        expected /= 2;   // two rows of two
        float kl = Functional.gaussianKl(Var.of(Tensor.of(Shape.of(2, 2), mu)), Var.of(Tensor.of(Shape.of(2, 2), lv)))
                .value().item();
        assertEquals(expected, kl, 1e-5);
    }

    @Test
    void samplesHaveTheRightMomentsAndGradients() {
        int n = 20000;
        Var mu = Var.param(Tensor.full(Shape.of(n), 1.5f));
        Var logVar = Var.param(Tensor.full(Shape.of(n), (float) Math.log(4)));
        Var z = Functional.reparameterize(mu, logVar, new SplittableRandom(1));
        float[] v = z.value().toArray();
        double mean = 0;
        double sq = 0;
        for (float x : v) {
            mean += x / n;
        }
        for (float x : v) {
            sq += (x - mean) * (x - mean) / n;
        }
        assertEquals(1.5, mean, 0.05);
        assertEquals(4, sq, 0.15);
        z.sum().backward();
        float[] ones = new float[n];
        java.util.Arrays.fill(ones, 1);
        assertArrayEquals(ones, mu.grad().toArray(), 1e-6f);
        // d z / d logVar = 0.5 * (z - mu)
        float[] g = logVar.grad().toArray();
        for (int i = 0; i < 10; i++) {
            assertEquals(0.5f * (v[i] - 1.5f), g[i], 1e-4f);
        }
    }
}
