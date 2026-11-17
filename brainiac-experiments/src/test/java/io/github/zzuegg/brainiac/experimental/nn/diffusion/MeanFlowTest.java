package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MeanFlowTest {

    @Test
    void doubleBackwardJvpMatchesFiniteDifferences() {
        MeanFlow.Network net = MeanFlow.mlp(3, 16, 1, 4);
        SplittableRandom r = new SplittableRandom(2);
        Tensor x = Tensor.normal(Shape.of(2, 3), 0, 1, r);
        Tensor v = Tensor.normal(Shape.of(2, 3), 0, 1, r);
        Tensor t = Tensor.of(Shape.of(2, 1), new float[] {0.7f, 0.4f});
        Tensor rr = Tensor.of(Shape.of(2, 1), new float[] {0.2f, 0.1f});
        Var z = Var.param(x);
        Var tv = Var.param(t);
        Var u = net.apply(z, Var.of(rr), tv);
        Var w = Var.param(Tensor.zeros(Shape.of(2, 3)));
        List<Var> vjp = Var.gradients(u, w, List.of(z, tv), true);
        Var dir = vjp.get(0).mul(Var.of(v)).sum().add(vjp.get(1).sum());
        float[] jvp = Var.gradients(dir, List.of(w), false).get(0).value().toArray();
        float h = 1e-3f;
        float[] plus = net.apply(Var.of(x.add(v.mul(h))), Var.of(rr), Var.of(t.add(h))).value().toArray();
        float[] minus = net.apply(Var.of(x.sub(v.mul(h))), Var.of(rr), Var.of(t.sub(h))).value().toArray();
        double maxErr = 0;
        for (int i = 0; i < jvp.length; i++) {
            double fd = (plus[i] - minus[i]) / (2 * h);
            maxErr = Math.max(maxErr, Math.abs(fd - jvp[i]) / (1 + Math.abs(jvp[i])));
            assertEquals(fd, jvp[i], 0.05 * (1 + Math.abs(jvp[i])), "component " + i);
        }
        System.out.printf("MeanFlow JVP: max relative error vs central differences %.4f%n", maxErr);
    }

    @Test
    void lossesAreFiniteAndSamplingKeepsShape() {
        MeanFlow.Network net = MeanFlow.mlp(2, 16, 1, 1);
        SplittableRandom r = new SplittableRandom(1);
        Tensor x = Tensor.normal(Shape.of(8, 2), 0, 1, r);
        float a = MeanFlow.meanFlowLoss(net, x, 0.5f, 1f, r).value().item();
        float b = MeanFlow.shortcutLoss(net, x, 0.75f, 4, r).value().item();
        assertEquals(true, Float.isFinite(a) && Float.isFinite(b));
        assertEquals(Shape.of(8, 2), MeanFlow.sample(net, x, 2).shape());
    }
}
