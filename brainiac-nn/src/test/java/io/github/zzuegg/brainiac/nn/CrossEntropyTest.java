package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.api.Test;

class CrossEntropyTest {

    static final float[] IDS = {3, 0, -1, 9, 9, 2, -1, 5, 1, 4, 7};

    @Test
    void matchesTheDefinition() {
        Tensor logits = Tensor.uniform(Shape.of(IDS.length, 10), -3, 3, new SplittableRandom(1));
        float[] l = logits.toArray();
        double sum = 0;
        int count = 0;
        for (int r = 0; r < IDS.length; r++) {
            if (IDS[r] < 0) {
                continue;
            }
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < 10; c++) {
                max = Math.max(max, l[r * 10 + c]);
            }
            double z = 0;
            for (int c = 0; c < 10; c++) {
                z += Math.exp(l[r * 10 + c] - max);
            }
            sum += max + Math.log(z) - l[r * 10 + (int) IDS[r]];
            count++;
        }
        assertEquals(sum / count, Functional.crossEntropy(Var.of(logits), Tensor.of(IDS)).value().item(), 1e-5);
    }

    /** Chunked: the same loss and gradients as the whole logits, whatever the chunk. */
    @ParameterizedTest
    @ValueSource(ints = {1, 4, 11, 100})
    void chunkedEqualsTheWholeLogits(int chunk) {
        SplittableRandom r = new SplittableRandom(2);
        Var h1 = Var.param(Tensor.uniform(Shape.of(IDS.length, 6), -1, 1, r));
        Var w1 = Var.param(Tensor.uniform(Shape.of(6, 10), -1, 1, r));
        Var h2 = Var.param(h1.value());
        Var w2 = Var.param(w1.value());
        Var whole = Functional.crossEntropy(h1.matmul(w1), Tensor.of(IDS));
        whole.backward();
        Var chunked = Functional.linearCrossEntropy(h2, w2, Tensor.of(IDS), chunk);
        chunked.backward();
        assertEquals(whole.value().item(), chunked.value().item(), 1e-5);
        assertArrayEquals(h1.grad().toArray(), h2.grad().toArray(), 1e-6f);
        assertArrayEquals(w1.grad().toArray(), w2.grad().toArray(), 1e-6f);
    }
}
