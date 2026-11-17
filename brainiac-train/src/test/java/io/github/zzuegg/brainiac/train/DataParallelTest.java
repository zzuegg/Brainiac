package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class DataParallelTest {

    private static Var loss(List<Var> p, Tensor x, Tensor y) {
        return Var.of(x).matmul(p.get(0)).add(p.get(1)).tanh().sub(Var.of(y)).square().mean();
    }

    @Test
    void stepsEqualFullBatchSteps() {
        SplittableRandom r = new SplittableRandom(2);
        Tensor x = Tensor.normal(Shape.of(10, 3), 0, 1, r);
        Tensor y = Tensor.normal(Shape.of(10, 2), 0, 1, r);
        Tensor w0 = Tensor.normal(Shape.of(3, 2), 0, 0.5f, r);
        Tensor b0 = Tensor.normal(Shape.of(2), 0, 0.5f, r);
        Device dev = TestDevice.get();
        Var w = Var.param(w0.to(dev));
        Var b = Var.param(b0.to(dev));
        Var wRef = Var.param(w0.to(dev));
        Var bRef = Var.param(b0.to(dev));
        Optimizer ref = Optimizer.adam(0.05f);
        try (DataParallel dp = new DataParallel(List.of(dev, Device.cpu(), dev), List.of(w, b), DataParallelTest::loss,
                Optimizer.adam(0.05f))) {
            for (int step = 0; step < 5; step++) {
                float l = dp.step(x, y);
                Var full = loss(List.of(wRef, bRef), x.to(dev), y.to(dev));
                full.backward();
                assertEquals(full.value().item(), l, 1e-5f);
                ref.step(List.of(wRef, bRef));
            }
        }
        assertArrayEquals(wRef.value().toArray(), w.value().toArray(), 1e-5f);
        assertArrayEquals(bRef.value().toArray(), b.value().toArray(), 1e-5f);
    }
}
