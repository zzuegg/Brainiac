package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

class CrossEntropyPerRowTest {

    @Test
    void matchesTheMeanLossRowByRow() {
        Var logits = Var.param(Tensor.of(Shape.of(3, 4), 0.5f, -1f, 2f, 0f, 1f, 1f, 1f, 1f, -2f, 3f, 0.1f, 0.2f));
        Tensor ids = Tensor.of(2f, -1f, 1f);
        // torch.nn.functional.cross_entropy(..., reduction="none", ignore_index=-1)
        float[] rows = Functional.crossEntropyPerRow(logits, ids).value().toArray();
        assertArrayEquals(new float[] {0.34232f, 0f, 0.11563f}, rows, 1e-4f);
        assertEquals(Functional.crossEntropy(logits, ids).value().item(), (rows[0] + rows[2]) / 2, 1e-5f);
    }
}
