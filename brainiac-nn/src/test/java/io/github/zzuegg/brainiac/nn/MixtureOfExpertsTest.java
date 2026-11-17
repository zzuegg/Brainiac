package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

import org.junit.jupiter.api.Test;

class MixtureOfExpertsTest {

    private static Tensor random(long seed, int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, new SplittableRandom(seed));
    }

    @Test
    void matchesADenseComputationOfTheChosenExperts() {
        int n = 5;
        int d = 3;
        int e = 4;
        Tensor x = random(1, n, d);
        Tensor router = random(2, d, e);
        List<Tensor> w = new ArrayList<>();
        List<Function<Var, Var>> experts = new ArrayList<>();
        for (int i = 0; i < e; i++) {
            Tensor wi = random(10 + i, d, 2);
            w.add(wi);
            experts.add(v -> v.matmul(Var.of(wi)).tanh());
        }
        for (int k = 1; k <= 2; k++) {
            Functional.Experts r = Functional.mixtureOfExperts(Var.of(x), Var.of(router), experts, k);
            float[] probs = Activation.SOFTMAX.forward(Var.of(x).matmul(Var.of(router))).value().toArray();
            float[] expected = new float[n * 2];
            for (int t = 0; t < n; t++) {
                Integer[] order = {0, 1, 2, 3};
                int row = t;
                java.util.Arrays.sort(order, (a, b) -> Float.compare(probs[row * e + b], probs[row * e + a]));
                float total = 0;
                for (int j = 0; j < k; j++) {
                    total += probs[t * e + order[j]];
                }
                for (int j = 0; j < k; j++) {
                    float[] y = Var.of(x.slice(0, t, t + 1)).matmul(Var.of(w.get(order[j]))).tanh().value().toArray();
                    float g = probs[t * e + order[j]] / total;
                    expected[t * 2] += g * y[0];
                    expected[t * 2 + 1] += g * y[1];
                }
            }
            assertArrayEquals(expected, r.output().value().toArray(), 1e-5f, "k " + k);
            double loadSum = 0;
            for (double l : r.load()) {
                loadSum += l;
            }
            assertEquals(1, loadSum, 1e-9);
            assertTrue(r.balanceLoss().value().item() > 0);
        }
    }

    @Test
    void gradientsReachInputsRouterAndExperts() {
        Tensor x = random(3, 4, 3);
        Tensor router = random(4, 3, 3);
        Tensor w0 = random(5, 3, 2);
        Tensor w1 = random(6, 3, 2);
        Tensor w2 = random(7, 3, 2);
        Function<List<Var>, Var> f = v -> {
            List<Function<Var, Var>> experts = List.of(a -> a.matmul(v.get(2)), a -> a.matmul(v.get(3)),
                    a -> a.matmul(v.get(4)));
            Functional.Experts r = Functional.mixtureOfExperts(v.get(0), v.get(1), experts, 2);
            return r.output().square().sum().add(r.balanceLoss());
        };
        GradCheck.Result r = GradCheck.check(f, List.of(x, router, w0, w1, w2), 1e-3f, 2e-2f);
        assertTrue(r.passed(), r.toString());
    }
}
