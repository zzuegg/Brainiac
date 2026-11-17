package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class AdversarialTest {

    @Test
    void generatorMovesToTheRealMean() {
        for (Adversarial.Loss loss : Adversarial.Loss.values()) {
            SplittableRandom r = new SplittableRandom(1);
            Var a = Var.param(Tensor.of(Shape.of(1, 1), 0.5f));
            Var b = Var.param(Tensor.of(Shape.of(1), 0f));
            Var w = Var.param(Tensor.of(Shape.of(1, 1), 0.1f));
            Var c = Var.param(Tensor.of(Shape.of(1), 0f));
            Adversarial gan = new Adversarial(z -> z.matmul(a).add(b), List.of(a, b), Optimizer.adam(0.02f),
                    x -> x.matmul(w).add(c), List.of(w, c), Optimizer.adam(0.02f), loss).withR1(1f);
            for (int step = 0; step < 400; step++) {
                Tensor real = Tensor.normal(Shape.of(64, 1), 3, 0.5f, r);
                Adversarial.Step s = gan.step(real, () -> Tensor.normal(Shape.of(64, 1), 0, 1, r));
                assertTrue(Float.isFinite(s.discriminator()) && Float.isFinite(s.generator()));
            }
            assertEquals(3, b.value().item(), 0.6, loss.toString());
        }
    }
}
