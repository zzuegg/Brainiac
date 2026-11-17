package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class AdaptersAndLossesTest {

    private static Tensor[] task(long seed, float rotate) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] x = new float[200 * 2];
        float[] y = new float[200];
        for (int i = 0; i < 200; i++) {
            float a = (float) r.nextGaussian();
            float b = (float) r.nextGaussian();
            x[2 * i] = a;
            x[2 * i + 1] = b;
            y[i] = a * Math.cos(rotate) + b * Math.sin(rotate) > 0 ? 1 : 0;
        }
        return new Tensor[] {Tensor.of(Shape.of(200, 2), x), Tensor.of(Shape.of(200), y)};
    }

    @Test
    void doraStartsAtTheBaseModelAndAdaptsOnlyItsOwnParameters() {
        Model base = Model.sequential().input(2).dense(16, Activation.TANH).dense(2).seed(1).build();
        Tensor[] a = task(1, 0);
        Optimizer opt = Optimizer.adam(0.05f);
        for (int i = 0; i < 100; i++) {
            Functional.crossEntropy(base.forward(Var.of(a[0])), a[1]).backward();
            opt.step(base.parameters());
        }
        Model adapted = DoraDense.adapt(base, 2, 3);
        assertArrayEquals(base.predict(a[0]).toArray(), adapted.predict(a[0]).toArray(), 1e-4f,
                "B starts at zero and m at the column norms");
        assertEquals(2 * 3, adapted.parameters().size(), "magnitude, A and B per layer");
        float[] frozen = base.parameters().get(0).value().toArray();
        Tensor[] b = task(2, 1.2f);   // a rotated decision boundary
        Optimizer ao = Optimizer.adam(0.05f);
        for (int i = 0; i < 150; i++) {
            Functional.crossEntropy(adapted.forward(Var.of(b[0])), b[1]).backward();
            ao.step(adapted.parameters());
        }
        double acc = accuracy(adapted.predict(b[0]), b[1]);
        assertTrue(acc > 0.9, "DoRA adapts to the new task: " + acc);
        assertArrayEquals(frozen, base.parameters().get(0).value().toArray(), "the base stays frozen");
        DoraDense layer = (DoraDense) adapted.layers().get(0);
        Dense merged = layer.merge();
        assertArrayEquals(layer.forward(Var.of(b[0])).value().toArray(), merged.forward(Var.of(b[0])).value().toArray(),
                1e-5f);
    }

    @Test
    void quantizedDoraStoresTheBaseInFewBitsAndStillAdapts() {
        Model base = Model.sequential().input(2).dense(16, Activation.TANH).dense(2).seed(1).build();
        Dense first = (Dense) base.layers().get(0);
        DoraDense q = DoraDense.quantized(first, 2, 4, 2, new SplittableRandom(3));
        assertTrue(q.frozenBytes() < first.weight().value().bytes(),
                q.frozenBytes() + " vs " + first.weight().value().bytes());
        Tensor[] a = task(1, 0);
        Tensor dequantized = first.weight().value().quantized(4, 2).unpacked();
        Dense reference = new Dense(2, 16, first.activation(), dequantized, first.bias().value());
        assertArrayEquals(reference.forward(Var.of(a[0])).value().toArray(), q.forward(Var.of(a[0])).value().toArray(),
                1e-4f, "starts equal to the quantized layer");
        Model adapted = Model.sequential().input(2).layer(q).layer(base.layers().get(1)).build();
        Tensor[] b = task(2, 1.2f);
        Optimizer ao = Optimizer.adam(0.05f);
        List<Var> params = List.of(q.parameters().get(0), q.parameters().get(1), q.parameters().get(2));
        for (int i = 0; i < 200; i++) {
            Functional.crossEntropy(adapted.forward(Var.of(b[0])), b[1]).backward();
            ao.step(params);
        }
        double acc = accuracy(adapted.predict(b[0]), b[1]);
        assertTrue(acc > 0.85, "QDoRA adapts: " + acc);
    }

    private static double accuracy(Tensor scores, Tensor ids) {
        float[] s = scores.toArray();
        float[] y = ids.toArray();
        int c = scores.shape().dim(1);
        int ok = 0;
        for (int i = 0; i < y.length; i++) {
            int best = 0;
            for (int k = 1; k < c; k++) {
                best = s[i * c + k] > s[i * c + best] ? k : best;
            }
            ok += best == Math.round(y[i]) ? 1 : 0;
        }
        return (double) ok / y.length;
    }

    @Test
    void labelSmoothingMovesTheTargetAndHasAFiniteOptimum() {
        Tensor onehot = Tensor.of(new float[][] {{0, 1, 0, 0}});
        assertArrayEquals(new float[] {0.025f, 0.925f, 0.025f, 0.025f},
                SmoothedCrossEntropy.smooth(onehot, 0.1f).toArray(), 1e-6f);
        Loss loss = new SmoothedCrossEntropy(0.1f);
        Var confident = Var.of(Tensor.of(new float[][] {{0.001f, 0.997f, 0.001f, 0.001f}}));
        Var calibrated = Var.of(Tensor.of(new float[][] {{0.025f, 0.925f, 0.025f, 0.025f}}));
        assertTrue(loss.compute(calibrated, onehot).value().item() < loss.compute(confident, onehot).value().item());
        assertEquals(loss.compute(calibrated, onehot).value().item(),
                loss.compute(calibrated, Tensor.of(Shape.of(1), 1)).value().item(), 1e-6f, "class ids work too");
    }

    @Test
    void distillationLossIsZeroForTheTeacherAndHasCorrectGradients() {
        DistillationLoss kd = new DistillationLoss(3);
        Tensor teacher = Tensor.of(new float[][] {{2, -1, 0.5f}, {0, 1, -2}});
        assertEquals(0, kd.compute(Var.of(teacher), teacher).value().item(), 1e-5f);
        assertTrue(kd.compute(Var.of(teacher.mul(-1)), teacher).value().item() > 0.01f);
        GradCheck.Result r = GradCheck.check(v -> kd.compute(v.get(0), teacher), List.of(Tensor.of(
                new float[][] {{0.3f, 0.1f, -0.4f}, {1, -0.5f, 0.2f}})));
        assertTrue(r.passed(), r.toString());
        Var mixed = kd.withLabels(Var.of(teacher), teacher, Tensor.of(Shape.of(2), 0, 1), 0.5f);
        assertTrue(mixed.value().item() > 0);
    }

    @Test
    void scheduledDropoutFollowsItsRate() {
        ScheduledDropout d = new ScheduledDropout(0, new SplittableRandom(1));
        Var x = Var.of(Tensor.full(Shape.of(100, 100), 1f));
        assertEquals(1, d.forward(x, true).value().mean().item(), 1e-6f);
        d.rate(0.5f);
        Tensor y = d.forward(x, true).value();
        float zeros = y.equalTo(Tensor.full(Shape.of(), 0f)).mean().item();
        assertEquals(0.5, zeros, 0.03);
        assertEquals(1, y.mean().item(), 0.05, "kept values are scaled up");
        assertEquals(1, d.forward(x, false).value().mean().item(), 1e-6f);
    }
}
