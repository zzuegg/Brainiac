package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ContinuumMemoryBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ContinuumOptimizerTest {

    @Test
    void slowLevelsChangeOnlyOnTheirPeriodAndTheBlockStillLearns() {
        ContinuumMemoryBlock block = new ContinuumMemoryBlock(4, 16, new int[] {1, 4}, new SplittableRandom(1));
        ContinuumOptimizer opt = ContinuumOptimizer.builder(() -> Optimizer.adam(0.01f))
                .group(block.levelParameters(0), block.periods()[0])
                .group(block.levelParameters(1), block.periods()[1]).build();
        SplittableRandom r = new SplittableRandom(2);
        Tensor x = Tensor.normal(Shape.of(32, 4), 0, 1, r);
        Tensor y = x.mul(x).sub(1f);
        float[] slow0 = block.levelParameters(1).get(0).value().toArray();
        float[] fast0 = block.levelParameters(0).get(0).value().toArray();
        float first = 0;
        float last = 0;
        for (int step = 1; step <= 200; step++) {
            Var loss = block.forward(Var.of(x)).sub(y).square().mean();
            if (step == 1) {
                first = loss.value().item();
            }
            last = loss.value().item();
            loss.backward();
            opt.step(block.parameters());
            if (step == 3) {
                assertArrayEquals(slow0, block.levelParameters(1).get(0).value().toArray(), 0f);
                float[] fast = block.levelParameters(0).get(0).value().toArray();
                assertTrue(diff(fast0, fast) > 0, "the fast level moved");
            }
            if (step == 4) {
                assertTrue(diff(slow0, block.levelParameters(1).get(0).value().toArray()) > 0, "slow level moved");
            }
        }
        assertTrue(last < 0.5f * first, first + " -> " + last);
    }

    private static double diff(float[] a, float[] b) {
        double d = 0;
        for (int i = 0; i < a.length; i++) {
            d += Math.abs(a[i] - b[i]);
        }
        return d;
    }
}
