package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CtcLossTest {

    @Test
    void gradientMatchesFiniteDifferences() {
        Tensor logits = Tensor.normal(Shape.of(2, 6, 4), 0, 1, new SplittableRandom(1));
        // a repeated label needs a blank between its copies
        GradCheck.Result r = GradCheck.check(CtcLoss.op(new int[][] {{1, 1, 2}, {3}}), 1e-2f, 1e-2f, logits);
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void matchesBruteForceOverAllAlignments() {
        // 3 frames, classes {blank, a}; target "a": alignments a__, _a_, __a, aa_, _aa, aaa, a_?...
        Tensor logits = Tensor.normal(Shape.of(1, 3, 2), 0, 1, new SplittableRandom(2));
        float[] v = logits.toArray();
        double[][] p = new double[3][2];
        for (int t = 0; t < 3; t++) {
            double z = Math.exp(v[2 * t]) + Math.exp(v[2 * t + 1]);
            p[t][0] = Math.exp(v[2 * t]) / z;
            p[t][1] = Math.exp(v[2 * t + 1]) / z;
        }
        double total = 0;
        for (int mask = 0; mask < 8; mask++) {
            // collapse: keep paths whose a-frames form one contiguous run
            int[] path = {mask & 1, mask >> 1 & 1, mask >> 2 & 1};
            int runs = 0;
            for (int t = 0; t < 3; t++) {
                if (path[t] == 1 && (t == 0 || path[t - 1] == 0)) {
                    runs++;
                }
            }
            if (runs == 1) {
                total += p[0][path[0]] * p[1][path[1]] * p[2][path[2]];
            }
        }
        float loss = CtcLoss.loss(Var.of(logits), new int[][] {{1}}).value().item();
        assertEquals(-Math.log(total), loss, 1e-5);
        assertEquals(loss, CtcLoss.INSTANCE.compute(Var.of(logits), Tensor.of(Shape.of(1, 2), 1, -1)).value().item(),
                1e-6f);
    }

    @Test
    void greedyDecodingMergesRepeatsAndDropsBlanks() {
        // argmax path: 1 1 0 1 2 2 -> "1 1 2"
        int[] path = {1, 1, 0, 1, 2, 2};
        float[] l = new float[6 * 3];
        for (int t = 0; t < 6; t++) {
            l[t * 3 + path[t]] = 5;
        }
        assertArrayEquals(new int[] {1, 1, 2}, CtcLoss.greedyDecode(Tensor.of(Shape.of(1, 6, 3), l))[0]);
    }
}
