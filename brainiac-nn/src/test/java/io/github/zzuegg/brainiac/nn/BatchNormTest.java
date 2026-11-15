package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class BatchNormTest {

    private static final Tensor X = Tensor.normal(Shape.of(64, 3), 5, 2, new SplittableRandom(4));

    @Test
    void trainingOutputHasZeroMeanAndUnitVariancePerFeature() {
        BatchNorm bn = new BatchNorm(3);
        Tensor y = bn.forward(Var.of(X), true).value();
        assertArrayEquals(new float[] {0, 0, 0}, y.mean(0).toArray(), 1e-4f);
        assertArrayEquals(new float[] {1, 1, 1}, y.square().mean(0).toArray(), 1e-3f);
    }

    @Test
    void runningStatisticsApproachTheDataAndAreUsedForPredictions() {
        BatchNorm bn = new BatchNorm(3, 0.5f, 1e-5f);
        for (int i = 0; i < 30; i++) {
            bn.forward(Var.of(X), true);
        }
        float[] mean = X.mean(0).toArray();
        assertArrayEquals(mean, bn.runningMean().toArray(), 1e-4f);
        float[] variance = X.sub(X.mean(0, true)).square().sum(0).div(63).toArray();
        assertArrayEquals(variance, bn.runningVariance().toArray(), 1e-3f);

        Tensor single = Tensor.of(Shape.of(1, 3), mean);
        assertArrayEquals(new float[] {0, 0, 0}, bn.forward(Var.of(single), false).value().toArray(), 1e-4f);
        assertEquals(2, bn.buffers().size());
    }

    @Test
    void inferenceDoesNotChangeRunningStatistics() {
        BatchNorm bn = new BatchNorm(3);
        bn.forward(Var.of(X), false);
        assertArrayEquals(new float[] {0, 0, 0}, bn.runningMean().toArray());
        assertArrayEquals(new float[] {1, 1, 1}, bn.runningVariance().toArray());
    }

    @Test
    void gradientsMatchFiniteDifferences() {
        Tensor x = Tensor.uniform(Shape.of(5, 3), -2, 2, new SplittableRandom(1));
        Tensor weights = Tensor.uniform(Shape.of(5, 3), -1, 1, new SplittableRandom(2));
        BatchNorm bn = new BatchNorm(3);
        GradCheck.Result r = GradCheck.check(v -> bn.forward(v.get(0), true).mul(weights).sum(), List.of(x));
        assertTrue(r.passed(), r.toString());

        Tensor gammaBeta = Tensor.uniform(Shape.of(3), 0.5f, 1.5f, new SplittableRandom(3));
        GradCheck.Result p = GradCheck.check(v -> {
            Var xv = Var.of(x);
            Var mean = xv.mean(0, false);
            Var c = xv.sub(mean);
            Var n = c.div(c.square().mean(0, false).add(1e-5f).sqrt());
            return n.mul(v.get(0)).add(v.get(1)).mul(weights).sum();
        }, List.of(gammaBeta, gammaBeta));
        assertTrue(p.passed(), p.toString());
    }

    @Test
    void builderAddsBatchNormAndValidatesShapes() {
        Model m = Model.sequential().input(3).dense(4).batchNorm().activation(Activation.RELU).dense(1).seed(1).build();
        assertEquals(3 * 4 + 4 + 4 + 4 + 4 + 1, m.parameterCount());
        assertEquals(2, m.buffers().size());
        assertThrows(IllegalArgumentException.class, () -> new BatchNorm(3).forward(Var.of(Tensor.ones(Shape.of(1, 3))), true));
        assertThrows(IllegalArgumentException.class, () -> new BatchNorm(3).forward(Var.of(Tensor.ones(Shape.of(2, 4))), false));
    }
}
