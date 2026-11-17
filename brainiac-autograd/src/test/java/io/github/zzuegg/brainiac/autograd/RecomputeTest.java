package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

import org.junit.jupiter.api.Test;

class RecomputeTest {

    static Tensor random(SplittableRandom r, int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, r);
    }

    @Test
    void givesTheSameValuesAndGradients() {
        SplittableRandom r = new SplittableRandom(1);
        Var w1 = Var.param(random(r, 5, 7));
        Var w2 = Var.param(random(r, 7, 5));
        Var w3 = Var.param(random(r, 5, 3));
        Tensor input = random(r, 4, 5);
        Function<Var, Var> block = x -> x.matmul(w1).tanh().matmul(w2).add(x);   // a residual block
        float[][] plain = run(w1, w2, w3, input, x -> block.apply(block.apply(x)));
        float[][] recomputed = run(w1, w2, w3, input, x -> {
            Var h = Recompute.apply(block, x, List.of(w1, w2));
            return Recompute.apply(block, h, List.of(w1, w2));
        });
        for (int i = 0; i < plain.length; i++) {
            assertArrayEquals(plain[i], recomputed[i], 1e-5f, "result " + i);
        }
    }

    /** Output, then the gradients of w1, w2, w3 (w3 is used outside the blocks, before and after them). */
    static float[][] run(Var w1, Var w2, Var w3, Tensor input, Function<Var, Var> body) {
        for (Var w : List.of(w1, w2, w3)) {
            w.zeroGrad();
        }
        Var x = Var.of(input).matmul(w3).matmul(w3.transpose());
        Var out = body.apply(x).matmul(w3);
        out.square().mean().backward();
        return new float[][] {out.value().toArray(), w1.grad().toArray(), w2.grad().toArray(), w3.grad().toArray()};
    }

    @Test
    void addsToGradientsAlreadyThere() {
        SplittableRandom r = new SplittableRandom(2);
        Var w = Var.param(random(r, 3, 3));
        Tensor input = random(r, 2, 3);
        Var.of(input).matmul(w).sum().backward();
        Recompute.apply(x -> x.matmul(w).tanh(), Var.of(input), List.of(w)).sum().backward();
        float[] both = w.grad().toArray();
        w.zeroGrad();
        Var.of(input).matmul(w).sum().backward();
        Var.of(input).matmul(w).tanh().sum().backward();
        assertArrayEquals(w.grad().toArray(), both, 1e-6f);
    }
}
