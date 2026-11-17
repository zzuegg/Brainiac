package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

class LovaszTest {

    @Test
    void hardPredictionsGiveOneMinusMeanIou() {
        int[] labels = {0, 0, 1, 1, 2, 2, 0, 1};
        int[] predicted = {0, 1, 1, 1, 2, 0, 0, 2};
        float[] p = new float[labels.length * 3];
        float[] ids = new float[labels.length];
        for (int i = 0; i < labels.length; i++) {
            p[i * 3 + predicted[i]] = 1;
            ids[i] = labels[i];
        }
        double expected = 0;
        for (int c = 0; c < 3; c++) {
            int inter = 0;
            int union = 0;
            for (int i = 0; i < labels.length; i++) {
                boolean a = labels[i] == c;
                boolean b = predicted[i] == c;
                inter += a && b ? 1 : 0;
                union += a || b ? 1 : 0;
            }
            expected += 1 - (double) inter / union;
        }
        expected /= 3;
        float loss = Functional.lovaszSoftmax(Var.of(Tensor.of(Shape.of(labels.length, 3), p)),
                Tensor.of(Shape.of(labels.length), ids)).value().item();
        assertEquals(expected, loss, 1e-6);
    }

    @Test
    void perfectIsZeroAndGradientsFlow() {
        float[] ids = {0, 1, 1, 0};
        float[] perfect = {1, 0, 0, 1, 0, 1, 1, 0};
        assertEquals(0, Functional.lovaszSoftmax(Var.of(Tensor.of(Shape.of(4, 2), perfect)),
                Tensor.of(Shape.of(4), ids)).value().item(), 1e-7);
        Var logits = Var.param(Tensor.of(Shape.of(4, 2), 0.2f, -0.1f, 0.3f, 0.1f, -0.5f, 0.4f, 0.0f, 0.2f));
        Var loss = Functional.lovaszSoftmax(logits.exp().div(logits.exp().sum(1, true)), Tensor.of(Shape.of(4), ids));
        assertTrue(loss.value().item() > 0);
        loss.backward();
        assertNotNull(logits.grad());
    }
}
