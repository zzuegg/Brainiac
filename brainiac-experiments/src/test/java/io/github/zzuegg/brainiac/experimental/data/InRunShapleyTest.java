package io.github.zzuegg.brainiac.experimental.data;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class InRunShapleyTest {

    @Test
    void ghostDotProductsMatchPerExampleGradients() {
        SimpleMlp net = SimpleMlp.relu(3, 5, 7, 3);
        SplittableRandom r = new SplittableRandom(4);
        int batch = 4;
        Tensor x = Tensor.normal(Shape.of(batch, 5), 0, 1, r);
        Tensor y = Tensor.of(Shape.of(batch), 0, 2, 1, 2);
        Tensor vx = Tensor.normal(Shape.of(6, 5), 0, 1, r);
        Tensor vy = Tensor.of(Shape.of(6), 1, 0, 2, 2, 1, 0);
        float lr = 0.1f;
        InRunShapley shapley = new InRunShapley(batch);
        shapley.record(net, x, y, new int[] {0, 1, 2, 3}, vx, vy, lr, false);
        List<Var> gv = Var.gradients(Functional.crossEntropy(net.forward(Var.of(vx)), vy), net.parameters(), false);
        double worst = 0;
        for (int i = 0; i < batch; i++) {
            Tensor xi = Tensor.of(Shape.of(1, 5), java.util.Arrays.copyOfRange(x.toArray(), 5 * i, 5 * i + 5));
            Tensor yi = Tensor.of(Shape.of(1), y.toArray()[i]);
            List<Var> gi = Var.gradients(Functional.crossEntropy(net.forward(Var.of(xi)), yi), net.parameters(),
                    false);
            double dot = 0;
            for (int p = 0; p < gi.size(); p++) {
                dot += gi.get(p).value().mul(gv.get(p).value()).sum().item();
            }
            worst = Math.max(worst, Math.abs(lr / batch * dot - shapley.scores()[i]));
        }
        System.out.printf("In-Run Shapley ghost dot product vs explicit per-example gradients: max error %.2e%n",
                worst);
        assertEquals(0, worst, 1e-5);
    }

    @Test
    void secondOrderTermRuns() {
        SimpleMlp net = SimpleMlp.tanh(5, 4, 6, 2);
        SplittableRandom r = new SplittableRandom(6);
        Tensor x = Tensor.normal(Shape.of(3, 4), 0, 1, r);
        Tensor y = Tensor.of(Shape.of(3), 0, 1, 1);
        InRunShapley first = new InRunShapley(3);
        InRunShapley second = new InRunShapley(3);
        first.record(net, x, y, new int[] {0, 1, 2}, x, y, 0.5f, false);
        second.record(net, x, y, new int[] {0, 1, 2}, x, y, 0.5f, true);
        double diff = 0;
        for (int i = 0; i < 3; i++) {
            diff += Math.abs(first.scores()[i] - second.scores()[i]);
        }
        System.out.printf("In-Run Shapley second-order correction: %.2e%n", diff);
        assertEquals(1, second.counts()[0]);
        org.junit.jupiter.api.Assertions.assertTrue(diff > 0 && Double.isFinite(diff));
    }
}
