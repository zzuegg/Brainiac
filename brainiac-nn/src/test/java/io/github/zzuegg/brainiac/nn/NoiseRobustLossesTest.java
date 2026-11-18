package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

/** Multiclass focal, generalized and symmetric cross-entropy; values from PyTorch. */
class NoiseRobustLossesTest {

    private static final Var LOGITS = Var.of(Tensor.of(Shape.of(2, 3), 1, 2, 0.5f, 0.1f, -1, 2));
    private static final Tensor IDS = Tensor.of(1, 2);
    private static final Tensor ONE_HOT = Tensor.of(Shape.of(2, 3), 0, 1, 0, 0, 0, 1);

    @Test
    void focalFromLogits() {
        assertEquals(0.03455006331205368, Loss.focalFromLogits(2).compute(LOGITS, IDS).value().item(), 1e-6);
        assertEquals(0.03455006331205368, Loss.focalFromLogits(2).compute(LOGITS, ONE_HOT).value().item(), 1e-6);
        assertEquals(Loss.crossEntropyFromLogits(0).compute(LOGITS, IDS).value().item(),
                Loss.focalFromLogits(0).compute(LOGITS, IDS).value().item(), 1e-6);
    }

    @Test
    void generalizedAndSymmetricCrossEntropy() {
        assertEquals(0.2835715711116791, Loss.generalizedCrossEntropy(0.7f).compute(LOGITS, IDS).value().item(), 1e-6);
        assertEquals(2.508449077606201, Loss.symmetricCrossEntropy(0.1f, 1).compute(LOGITS, IDS).value().item(), 1e-5);
    }

    @Test
    void aLossCarriesItsParameters() {
        Var scale = Var.param(Tensor.of(2f));
        Loss scaled = Loss.withParameters((p, t) -> Loss.MSE.compute(p.mul(scale), t), java.util.List.of(scale));
        assertEquals(java.util.List.of(scale), scaled.parameters());
        assertEquals(java.util.List.of(), Loss.MSE.parameters());
    }
}
