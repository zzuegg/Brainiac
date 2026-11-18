package io.github.zzuegg.brainiac.experiments.numerics;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.DeepEquilibriumLayer;
import io.github.zzuegg.brainiac.experimental.train.AndersonSolver;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Map;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class ImplicitDepthTest {

    @Test
    void implicitGradientMatchesFiniteDifferences() {
        DeepEquilibriumLayer layer = new DeepEquilibriumLayer(3, 6, new AndersonSolver(5, 100, 1e-7, 1.0), 0f,
                new SplittableRandom(4));
        Tensor x = Tensor.normal(Shape.of(4, 3), 0, 1, new SplittableRandom(5));
        Tensor probe = Tensor.normal(Shape.of(4, 6), 0, 1, new SplittableRandom(6));
        Var w = layer.parameters().get(2);
        layer.forward(Var.of(x)).mul(Var.of(probe)).sum().backward();
        float analytic = w.grad().toArray()[7];
        float[] base = w.value().toArray();
        double eps = 1e-2;
        double[] f = new double[2];
        for (int s = 0; s < 2; s++) {
            float[] v = base.clone();
            v[7] += (float) (s == 0 ? eps : -eps);
            w.assign(Tensor.of(w.shape(), v));
            f[s] = Var.noGrad(() -> layer.forward(Var.of(x)).mul(Var.of(probe)).sum()).value().item();
        }
        double numeric = (f[0] - f[1]) / (2 * eps);
        System.out.printf("DEQ implicit gradient %.5f, finite difference %.5f%n", analytic, numeric);
        assertEquals(numeric, analytic, 2e-3 + 0.02 * Math.abs(numeric));
    }

    @Test
    void deqMatchesUnrolledAccuracyWithConstantMemory() {
        Map<String, ImplicitDepth.Row> r = ImplicitDepth.run(16, 8, 150, 1);
        StringBuilder s = new StringBuilder("Spirals, hidden 16, 150 steps, test acc/fwd iters/bwd iters/stored per"
                + " example/ms/final solve Anderson vs plain iters:");
        r.forEach((k, v) -> s.append(String.format(" %s %.3f/%.1f/%.1f/%d/%d/%d vs %d;", k, v.testAccuracy(),
                v.forwardIterations(), v.backwardIterations(), v.storedPerExample(), v.millis(), v.andersonAtEnd(),
                v.plainAtEnd())));
        System.out.println(s);
        ImplicitDepth.Row deq = r.get("DEQ (Anderson, Jacobian reg.)");
        ImplicitDepth.Row unrolled = r.get("tied, unrolled x8");
        assertTrue(deq.testAccuracy() > unrolled.testAccuracy() - 0.2, s.toString());
        assertTrue(deq.andersonAtEnd() <= deq.plainAtEnd(), s.toString());
        assertTrue(deq.storedPerExample() < unrolled.storedPerExample(), s.toString());
    }
}
