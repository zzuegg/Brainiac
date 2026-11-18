package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** eigh and svd gradients match finite differences for sign-free losses. */
class DecompositionsTest {

    private static Tensor random(long seed, int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, new SplittableRandom(seed));
    }

    @Test
    void eighGradients() {
        Tensor w1 = random(1, 4);
        Tensor w2 = random(2, 4, 4);
        GradCheck.Result r = GradCheck.check(v -> {
            Var[] e = Decompositions.eigh(v.get(0));
            return e[0].mul(w1).sum().add(e[1].mul(e[1]).mul(w2).sum());
        }, List.of(random(3, 4, 4)), 1e-2f, 2e-2f);
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void svdGradientsForTallSquareAndWide() {
        for (int[] dims : new int[][] {{5, 3}, {3, 3}, {2, 4}}) {
            int k = Math.min(dims[0], dims[1]);
            Tensor wu = random(4, dims[0], k);
            Tensor ws = random(5, k);
            Tensor wv = random(6, k, dims[1]);
            GradCheck.Result r = GradCheck.check(v -> {
                Var[] d = Decompositions.svd(v.get(0));
                return d[0].mul(d[0]).mul(wu).sum().add(d[1].mul(ws).sum()).add(d[2].mul(d[2]).mul(wv).sum());
            }, List.of(random(7 + dims[0], dims)), 1e-2f, 2e-2f);
            assertTrue(r.passed(), java.util.Arrays.toString(dims) + " " + r);
        }
    }
}
