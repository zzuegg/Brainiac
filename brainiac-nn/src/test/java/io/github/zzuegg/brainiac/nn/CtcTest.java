package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class CtcTest {

    /** Sums the probability of every alignment by brute force. */
    private static double bruteForce(float[] p, int time, int classes, int[] target) {
        double total = 0;
        int[] path = new int[time];
        int count = (int) Math.pow(classes, time);
        for (int code = 0; code < count; code++) {
            int c = code;
            double prob = 1;
            for (int t = 0; t < time; t++) {
                path[t] = c % classes;
                c /= classes;
                prob *= p[t * classes + path[t]];
            }
            java.util.List<Integer> out = new java.util.ArrayList<>();
            int prev = -1;
            for (int t = 0; t < time; t++) {
                if (path[t] != prev && path[t] != 0) {
                    out.add(path[t]);
                }
                prev = path[t];
            }
            if (out.equals(java.util.Arrays.stream(target).boxed().toList())) {
                total += prob;
            }
        }
        return -Math.log(total);
    }

    @Test
    void lossMatchesBruteForceAndHasExactGradients() {
        SplittableRandom r = new SplittableRandom(2);
        Tensor logits = Tensor.normal(Shape.of(2, 5, 3), 0, 1, r).to(DType.FLOAT64);
        int[][] targets = {{1, 2}, {1, 1}};
        Var logp = Activation.LOG_SOFTMAX.forward(Var.of(logits));
        double loss = Ctc.loss(logp, targets, null, 0).value().toDoubleArray()[0];
        float[] p = logp.value().exp().toArray();
        double expected = 0;
        for (int b = 0; b < 2; b++) {
            expected += bruteForce(java.util.Arrays.copyOfRange(p, b * 15, b * 15 + 15), 5, 3, targets[b]);
        }
        assertEquals(expected / 2, loss, 1e-5);
        GradCheck.Result g = GradCheck.check(v -> Ctc.loss(Activation.LOG_SOFTMAX.forward(v.get(0)), targets, null, 0),
                List.of(logits));
        assertTrue(g.passed(), g.toString());
    }

    @Test
    void decodingMergesRepeatsAndDropsBlanks() {
        // steps pick: 1 1 0 1 2 2
        float[][] rows = {{0, 1, 0}, {0, 1, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 1}};
        float[] v = new float[18];
        for (int t = 0; t < 6; t++) {
            for (int c = 0; c < 3; c++) {
                v[t * 3 + c] = rows[t][c] == 1 ? (float) Math.log(0.8) : (float) Math.log(0.1);
            }
        }
        Tensor lp = Tensor.of(Shape.of(1, 6, 3), v);
        assertArrayEquals(new int[] {1, 1, 2}, Ctc.greedy(lp, 0)[0]);
        assertArrayEquals(new int[] {1, 1, 2}, Ctc.beamSearch(lp.reshape(6, 3), 8, 0));
    }

    @Test
    void beamSearchFindsTheSequenceGreedyMisses() {
        // two steps: greedy picks blank, blank (empty), but "1" has 0.3*0.3+0.3*0.4*2 = 0.33 > 0.16
        float[] p = {0.4f, 0.3f, 0.3f, 0.4f, 0.3f, 0.3f};
        float[] lp = new float[6];
        for (int i = 0; i < 6; i++) {
            lp[i] = (float) Math.log(p[i]);
        }
        Tensor t = Tensor.of(Shape.of(1, 2, 3), lp);
        assertEquals(0, Ctc.greedy(t, 0)[0].length);
        int[] beam = Ctc.beamSearch(t.reshape(2, 3), 4, 0);
        assertEquals(1, beam.length);
    }
}
