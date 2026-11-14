package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Activations and losses, checked against finite differences. */
class NnGradCheckTest {

    private static final Tensor X = Tensor.uniform(Shape.of(3, 4), -2, 2, new SplittableRandom(1));
    private static final Tensor ONE_HOT = Tensor.of(Shape.of(3, 4), 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1);
    private static final Tensor BINARY = Tensor.of(Shape.of(3, 4), 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1);

    private static void check(String name, java.util.function.Function<Var, Var> f, Tensor x) {
        GradCheck.Result r = GradCheck.check(v -> f.apply(v.get(0)), List.of(x));
        assertTrue(r.passed(), name + ": " + r);
    }

    @Test
    void softmax() {
        // weighted so the gradient is not trivially zero (softmax rows always sum to 1)
        check("softmax", x -> Activation.SOFTMAX.forward(x).mul(X).sum(), X);
    }

    @Test
    void activations() {
        check("tanh", x -> Activation.TANH.forward(x).sum(), X);
        check("sigmoid", x -> Activation.SIGMOID.forward(x).sum(), X);
        check("identity", x -> Activation.IDENTITY.forward(x).square().sum(), X);
    }

    @Test
    void losses() {
        check("mse", x -> Loss.MSE.compute(x, ONE_HOT), X);
        check("cross entropy", x -> Loss.CROSS_ENTROPY.compute(Activation.SOFTMAX.forward(x), ONE_HOT), X);
        check("binary cross entropy", x -> Loss.BINARY_CROSS_ENTROPY.compute(x.sigmoid(), BINARY), X);
    }

    @Test
    void denseLayerParameters() {
        Dense d = new Dense(4, 3, Activation.TANH, Init.XAVIER, new SplittableRandom(2));
        Tensor target = Tensor.uniform(Shape.of(3, 3), -1, 1, new SplittableRandom(3));
        GradCheck.Result r = GradCheck.check(v -> {
            Dense layer = new Dense(4, 3, Activation.TANH, v.get(0).value(), v.get(1).value());
            // rebuild with the check's Vars so their gradients are tracked
            Var out = Var.of(X).matmul(v.get(0)).add(v.get(1)).tanh();
            assertTrue(layer.forward(Var.of(X)).value().sub(out.value()).abs().max().item() < 1e-6);
            return Loss.MSE.compute(out, target);
        }, List.of(d.weight().value(), d.bias().value()));
        assertTrue(r.passed(), r.toString());
    }
}
