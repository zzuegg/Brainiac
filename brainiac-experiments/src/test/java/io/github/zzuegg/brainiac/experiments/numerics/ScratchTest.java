package io.github.zzuegg.brainiac.experiments.numerics;

import io.github.zzuegg.brainiac.experimental.train.AndersonSolver;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class ScratchTest {

    @Test
    void anderson() {
        SplittableRandom r = new SplittableRandom(1);
        for (float scale : new float[] {0.5f, 1.0f, 1.5f}) {
            Tensor w = Tensor.normal(Shape.of(24, 24), 0, (float) (scale / Math.sqrt(24)), r);
            Tensor inj = Tensor.normal(Shape.of(300, 24), 0, 1, r);
            for (int m : new int[] {1, 3, 5}) {
                AndersonSolver.Result res = new AndersonSolver(m, 200, 1e-4, 1.0).solve(
                        z -> z.matmul(w).add(inj).tanh(), Tensor.zeros(inj.shape()));
                System.out.printf("SCRATCH scale %.1f m %d iterations %d residual %.2e%n", scale, m, res.iterations(),
                        res.residual());
            }
        }
    }
}
