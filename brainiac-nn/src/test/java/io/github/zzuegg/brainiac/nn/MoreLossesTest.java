package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** The losses beyond MSE and cross-entropy, against hand computations. */
class MoreLossesTest {

    private static float of(Loss loss, Tensor p, Tensor t) {
        return loss.compute(Var.of(p), t).value().item();
    }

    @Test
    void regressionLosses() {
        Tensor p = Tensor.of(new float[] {0, 2, -3});
        Tensor t = Tensor.of(new float[] {0.5f, 2, 1});
        assertEquals((0.5 + 0 + 4) / 3, of(Loss.L1, p, t), 1e-6);
        // huber with delta 1: 0.5 * 0.25, 0, 1 * (4 - 0.5)
        assertEquals((0.125 + 0 + 3.5) / 3, of(Loss.huber(1), p, t), 1e-6);
    }

    @Test
    void hingeAndCosine() {
        // max(0, 1 - 0.5) = 0.5, max(0, 1 + 2) = 3
        assertEquals(1.75, of(Loss.HINGE, Tensor.of(new float[] {0.5f, 2}), Tensor.of(new float[] {1, -1})), 1e-6);
        Tensor a = Tensor.of(Shape.of(2, 2), 1, 0, 1, 1);
        Tensor b = Tensor.of(Shape.of(2, 2), 0, 1, 2, 2);
        assertEquals((1 + 0) / 2.0, of(Loss.COSINE, a, b), 1e-5);
    }

    @Test
    void logitCrossEntropyWithSmoothingAndPadding() {
        Tensor logits = Tensor.of(Shape.of(3, 3), 1, 2, 3, 0, 0, 0, 5, -1, 2);
        Tensor ids = Tensor.of(new float[] {2, -1, 0});   // the middle row is padding
        float[] l = logits.toArray();
        double plain = 0;
        double smooth = 0;
        for (int r : new int[] {0, 2}) {
            double max = Math.max(l[3 * r], Math.max(l[3 * r + 1], l[3 * r + 2]));
            double lse = max + Math.log(Math.exp(l[3 * r] - max) + Math.exp(l[3 * r + 1] - max) + Math.exp(l[3 * r + 2] - max));
            int id = r == 0 ? 2 : 0;
            plain += lse - l[3 * r + id];
            for (int c = 0; c < 3; c++) {
                smooth += ((c == id ? 0.9 : 0) + 0.1 / 3) * (lse - l[3 * r + c]);
            }
        }
        assertEquals(plain / 2, of(Loss.crossEntropyFromLogits(0), logits, ids), 1e-5);
        assertEquals(smooth / 2, of(Loss.crossEntropyFromLogits(0.1f), logits, ids), 1e-5);
    }

    @Test
    void klFocalAndDice() {
        Tensor p = Tensor.of(Shape.of(1, 2), 0.25f, 0.75f);
        Tensor t = Tensor.of(Shape.of(1, 2), 0.5f, 0.5f);
        assertEquals(0.5 * Math.log(2) + 0.5 * Math.log(2 / 3.0), of(Loss.KL_DIVERGENCE, p, t), 1e-5);
        Tensor prob = Tensor.of(new float[] {0.9f, 0.2f});
        Tensor y = Tensor.of(new float[] {1, 0});
        // gamma 0, alpha 0.5: half the binary cross-entropy
        double bce = -(Math.log(0.9) + Math.log(0.8)) / 2;
        assertEquals(bce / 2, of(Loss.focal(0, 0.5f), prob, y), 1e-5);
        double focal2 = -(0.01 * Math.log(0.9) + 0.04 * Math.log(0.8)) / 2 / 2;
        assertEquals(focal2, of(Loss.focal(2, 0.5f), prob, y), 1e-5);
        Tensor mask = Tensor.of(Shape.of(1, 4), 1, 1, 0, 0);
        Tensor pred = Tensor.of(Shape.of(1, 4), 1, 0.5f, 0.5f, 0);
        // 1 - (2 * 1.5 + 1) / (2 + 2 + 1)
        assertEquals(1 - 4.0 / 5, of(Loss.dice(1), pred, mask), 1e-6);
    }

    @Test
    void infoNcePrefersMatchingPairsAndHasGradients() {
        SplittableRandom r = new SplittableRandom(1);
        Tensor a = Tensor.normal(Shape.of(6, 4), 0, 1, r);
        float matched = Functional.infoNce(Var.of(a), Var.of(a), 0.1f).value().item();
        float shuffled = Functional.infoNce(Var.of(a), Var.of(a.flip(0)), 0.1f).value().item();
        assertTrue(matched < shuffled, matched + " vs " + shuffled);
        Tensor b = Tensor.normal(Shape.of(6, 4), 0, 1, r);
        GradCheck.Result g = GradCheck.check(x -> Functional.infoNce(x.get(0), x.get(1), 0.5f),
                java.util.List.of(a.to(DType.FLOAT64), b.to(DType.FLOAT64)));
        assertTrue(g.passed(), g.toString());
        Tensor logits = Tensor.normal(Shape.of(6, 7), 0, 1, r).to(DType.FLOAT64);
        g = GradCheck.check(x -> Loss.crossEntropyFromLogits(0.1f).compute(x.get(0),
                Tensor.of(new float[] {1, 0, 3, 2, 5, 4})), java.util.List.of(logits));
        assertTrue(g.passed(), g.toString());
    }
}
