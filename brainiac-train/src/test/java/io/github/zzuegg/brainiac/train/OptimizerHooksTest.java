package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;

/** The extension points other modules build optimizers and training tricks on. */
class OptimizerHooksTest {

    @Test
    void nesterovLooksAhead() {
        Optimizer opt = Optimizer.nesterov(0.1f, 0.5f);
        Var p = Var.param(Tensor.of(0f));
        p.mul(Tensor.of(1)).sum().backward();
        opt.step(List.of(p));   // v = 1, update = 1 + 0.5 = 1.5 -> p = -0.15
        assertArrayEquals(new float[] {-0.15f}, p.value().toArray(), 1e-6f);
        p.mul(Tensor.of(1)).sum().backward();
        opt.step(List.of(p));   // v = 1.5, update = 1 + 0.75 -> p = -0.325
        assertArrayEquals(new float[] {-0.325f}, p.value().toArray(), 1e-6f);
    }

    @Test
    void resetStateForgetsChosenEntries() {
        StatefulOptimizer adam = (StatefulOptimizer) Optimizer.adam(0.01f);
        Var p = Var.param(Tensor.of(new float[] {1, 2, 3}));
        p.mul(Tensor.of(new float[] {1, -2, 3})).sum().backward();
        adam.step(List.of(p));
        Tensor[] before = adam.slots(p);
        adam.resetState(p, Tensor.of(new float[] {1, 0, 1}));
        Tensor[] after = adam.slots(p);
        for (int k = 0; k < before.length; k++) {
            float[] b = before[k].toArray();
            float[] a = after[k].toArray();
            assertEquals(b[0], a[0]);
            assertEquals(0, a[1], 0f);
            assertEquals(b[2], a[2]);
        }
    }

    /** Sharpness-aware minimization (Foret et al., 2021) in a few lines, from outside the module's internals. */
    static final class Sam extends StatefulOptimizer {
        final float rho;
        final List<Float> losses = new ArrayList<>();

        Sam(float learningRate, float rho) {
            super(learningRate);
            this.rho = rho;
        }

        @Override
        public void step(List<Var> parameters, Optimizer.Reevaluation reevaluate) {
            List<Tensor> moved = new ArrayList<>();
            for (Var p : parameters) {
                Tensor g = p.grad();
                Tensor e = g.mul(rho).div(g.square().sum().sqrt().add(1e-12f));
                moved.add(e);
                p.assign(p.value().add(e));
            }
            losses.add(reevaluate.lossAndGradients());   // the gradient at the perturbed weights
            for (int i = 0; i < parameters.size(); i++) {
                Var p = parameters.get(i);
                p.assign(p.value().sub(moved.get(i)));
            }
            step(parameters);
        }

        @Override
        protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
            p.assign(p.value().sub(g.mul(learningRate())));
            return new Tensor[0];
        }
    }

    @Test
    void optimizersCanEvaluateTheLossAgainWithinAStep() {
        Var w = Var.param(Tensor.of(new float[] {2, -1}));
        Sam sam = new Sam(0.1f, 0.05f);
        List<Float> seen = new ArrayList<>();
        TrainingLoop loop = TrainingLoop.of(List.of(w), sam).beforeStep(ps -> {
            assertNotNull(ps.getFirst().grad());
            seen.add(ps.getFirst().grad().toArray()[0]);
        });
        Supplier<Var> loss = () -> w.square().sum();
        float first = loop.microBatch(loss);
        assertEquals(5, first, 1e-6f);
        assertArrayEquals(new float[] {4}, new float[] {seen.getFirst()}, 1e-6f);   // 2 w at w = (2, -1)
        // the loss at w + rho g / |g|, then a step with that gradient from w
        float[] g = {4, -2};
        float norm = (float) Math.sqrt(20);
        float[] e = {0.05f * g[0] / norm, 0.05f * g[1] / norm};
        float perturbed = (2 + e[0]) * (2 + e[0]) + (-1 + e[1]) * (-1 + e[1]);
        assertEquals(perturbed, sam.losses.getFirst(), 1e-5f);
        assertArrayEquals(new float[] {2 - 0.1f * 2 * (2 + e[0]), -1 - 0.1f * 2 * (-1 + e[1])}, w.value().toArray(),
                1e-5f);
        assertNull(w.grad());
    }
}
