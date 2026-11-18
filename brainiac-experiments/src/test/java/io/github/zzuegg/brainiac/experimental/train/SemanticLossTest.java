package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

class SemanticLossTest {

    private static final float[] P = {0.7f, 0.2f, 0.4f, 0.9f};

    /** Brute-force probability that the number of true variables satisfies the test. */
    private static double bruteForce(java.util.function.IntPredicate count) {
        double total = 0;
        for (int mask = 0; mask < 16; mask++) {
            double w = 1;
            for (int i = 0; i < 4; i++) {
                w *= (mask >> i & 1) == 1 ? P[i] : 1 - P[i];
            }
            total += count.test(Integer.bitCount(mask)) ? w : 0;
        }
        return total;
    }

    @Test
    void weightedModelCountsAreExact() {
        Var p = Var.of(Tensor.of(Shape.of(1, 4), P));
        double one = Math.exp(-SemanticLoss.exactlyOne(p).value().item());
        double atMost2 = Math.exp(-SemanticLoss.atMostK(p, 2).value().item());
        System.out.printf("Semantic loss WMC: exactly-one %.4f (brute force %.4f), at-most-2 %.4f (%.4f)%n", one,
                bruteForce(c -> c == 1), atMost2, bruteForce(c -> c <= 2));
        assertEquals(bruteForce(c -> c == 1), one, 1e-4);
        assertEquals(bruteForce(c -> c <= 2), atMost2, 1e-4);
    }

    @Test
    void latinSquareLossIsSmallOnlyForValidSquares() {
        int[] square = {0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0};
        float[] valid = new float[64];
        float[] invalid = new float[64];
        for (int c = 0; c < 16; c++) {
            for (int v = 0; v < 4; v++) {
                valid[c * 4 + v] = square[c] == v ? 0.97f : 0.01f;
                invalid[c * 4 + v] = (c % 4) == v ? 0.97f : 0.01f;   // every row 0 1 2 3: columns repeat
            }
        }
        float good = SemanticLoss.latinSquare(Var.of(Tensor.of(Shape.of(1, 64), valid)), 4, true).value().item();
        float bad = SemanticLoss.latinSquare(Var.of(Tensor.of(Shape.of(1, 64), invalid)), 4, true).value().item();
        assertTrue(good < 5 && bad > 10 * good, "valid " + good + " invalid " + bad);
    }
}
