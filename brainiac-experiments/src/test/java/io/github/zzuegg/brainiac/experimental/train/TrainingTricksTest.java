package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.LearningRateSchedule;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;

class TrainingTricksTest {

    @Test
    void warmupStableDecay() {
        LearningRateSchedule s = TrainingSchedules.warmupStableDecay(10, 20, 10, 0.1f);
        assertEquals(0.1f, s.rate(1, 1f), 1e-6f);
        assertEquals(1f, s.rate(10, 1f), 1e-6f);
        assertEquals(1f, s.rate(30, 1f), 1e-6f);
        assertEquals(0.55f, s.rate(35, 1f), 1e-6f);
        assertEquals(0.1f, s.rate(40, 1f), 1e-6f);
        assertEquals(0.1f, s.rate(1000, 1f), 1e-6f);
        assertTrue(TrainingSchedules.warmupStableDecay(0, 0, 4, 0, true).rate(2, 1f) < 0.5f);
    }

    @Test
    void cosineRestartsJumpBackUp() {
        LearningRateSchedule s = TrainingSchedules.cosineWithRestarts(10, 2, 0f);
        assertEquals(1f, s.rate(1, 1f), 1e-6f);
        assertTrue(s.rate(10, 1f) < 0.05f);
        assertEquals(1f, s.rate(11, 1f), 1e-6f);    // restart
        assertEquals(0.5f, s.rate(21, 1f), 1e-6f);  // middle of the 20-step second cycle
        assertEquals(1f, s.rate(31, 1f), 1e-6f);    // third cycle starts after 10 + 20
    }

    @Test
    void oneCycleRisesThenFallsFarBelowTheStart() {
        LearningRateSchedule s = TrainingSchedules.oneCycle(100);
        assertEquals(1f / 25, s.rate(1, 1f), 1e-6f);
        assertEquals(1f, s.rate(31, 1f), 1e-6f);
        float last = s.rate(100, 1f);
        assertTrue(last < 1e-5f, "ends at base / 25 / 1e4: " + last);
        float prev = 0;
        for (int i = 1; i <= 31; i++) {
            assertTrue(s.rate(i, 1f) >= prev);
            prev = s.rate(i, 1f);
        }
    }

    @Test
    void gradientNoiseDecaysAndStillConverges() {
        Var w = Var.param(Tensor.of(3f, -2f));
        GradientNoise opt = new GradientNoise(Optimizer.sgd(0.1f), 0.1f, 0.55f, 1);
        float first = opt.currentStd();
        for (int i = 0; i < 200; i++) {
            w.square().sum().backward();
            opt.step(List.of(w));
        }
        assertTrue(opt.currentStd() < first / 4);
        assertTrue(w.value().abs().sum().item() < 0.3f, "converges to 0 despite noise: " + w.value());
    }

    @Test
    void lossScalerSkipsOverflowsAndGrows() {
        DynamicLossScaler scaler = new DynamicLossScaler(1024, 2);
        Var w = Var.param(Tensor.of(1f, 2f));
        scaler.scale(w.square().sum()).backward();
        assertTrue(scaler.unscale(List.of(w)));
        assertArrayEquals(new float[] {2f, 4f}, w.grad().toArray(), 1e-5f);
        w.setGrad(Tensor.of(Float.POSITIVE_INFINITY, 1f));
        assertFalse(scaler.unscale(List.of(w)));
        assertEquals(512f, scaler.currentScale());
        assertEquals(1, scaler.skippedSteps());
        assertTrue(w.grad() == null);
        for (int i = 0; i < 2; i++) {
            w.setGrad(Tensor.of(1f, 1f));
            scaler.unscale(List.of(w));
        }
        assertEquals(1024f, scaler.currentScale());
    }

    @Test
    void rDropAddsASymmetricKlThatVanishesForEqualPredictions() {
        Tensor logits = Tensor.normal(Shape.of(4, 3), 0, 1, new SplittableRandom(2));
        assertEquals(0f, RDrop.symmetricKl(Var.of(logits), Var.of(logits)).value().item(), 1e-6f);
        GradCheck.Result r = GradCheck.check(v -> RDrop.symmetricKl(v.get(0), v.get(1)),
                List.of(logits, Tensor.normal(Shape.of(4, 3), 0, 1, new SplittableRandom(3))));
        assertTrue(r.passed(), r.toString());
        SplittableRandom rnd = new SplittableRandom(4);
        Tensor ids = Tensor.of(0f, 1f, 2f, 0f);
        float noisy = RDrop.loss(x -> x.add(Tensor.normal(Shape.of(4, 3), 0, 1, rnd)), Var.of(logits), ids, 1f)
                .value().item();
        float plain = RDrop.loss(x -> x, Var.of(logits), ids, 1f).value().item();
        assertTrue(noisy > 0 && plain > 0);
    }

    /** On quadratics the meta-gradient is known exactly: (I - a A) c for one inner step. */
    @Test
    void secondOrderMamlMatchesTheAnalyticMetaGradient() {
        Tensor a = Tensor.of(1f, 4f, 9f);
        Tensor c = Tensor.of(1f, -2f, 0.5f);
        float alpha = 0.05f;
        for (boolean second : new boolean[] {false, true}) {
            Var w = Var.param(Tensor.of(0.3f, -0.1f, 0.7f));
            Maml.Task task = new Maml.Task() {
                @Override
                public Var supportLoss() {
                    return w.square().mul(a).sum().mul(0.5f);
                }

                @Override
                public Var queryLoss() {
                    return w.mul(c).sum();
                }
            };
            new Maml(List.of(w), Optimizer.sgd(1f), alpha, 1, second).metaStep(List.of(task));
            float[] after = w.value().toArray();
            float[] start = {0.3f, -0.1f, 0.7f};
            float[] av = a.toArray();
            float[] cv = c.toArray();
            for (int i = 0; i < 3; i++) {
                float expected = second ? cv[i] * (1 - alpha * av[i]) : cv[i];
                assertEquals(expected, start[i] - after[i], 2e-3f, (second ? "second" : "first") + " order " + i);
            }
        }
    }

    @Test
    void reptileMovesTowardsTheTaskSolutions() {
        Var w = Var.param(Tensor.of(0f, 0f));
        Tensor t1 = Tensor.of(1f, 0f);
        Tensor t2 = Tensor.of(0f, 1f);
        Reptile reptile = new Reptile(List.of(w), 0.2f, 20, 0.5f);
        List<Supplier<Var>> tasks = List.of(() -> w.sub(t1).square().sum(), () -> w.sub(t2).square().sum());
        for (int i = 0; i < 20; i++) {
            reptile.metaStep(tasks);
        }
        assertArrayEquals(new float[] {0.5f, 0.5f}, w.value().toArray(), 1e-2f);
    }
}
