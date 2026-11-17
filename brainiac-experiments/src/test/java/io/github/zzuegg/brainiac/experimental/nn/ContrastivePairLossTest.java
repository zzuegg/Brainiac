package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ContrastivePairLossTest {

    @Test
    void clipMatchesCoreInfoNceAndSigLipPrefersMatchedPairs() {
        SplittableRandom r = new SplittableRandom(1);
        Var a = Var.of(Tensor.normal(Shape.of(6, 4), 0, 1, r));
        Var b = Var.of(Tensor.normal(Shape.of(6, 4), 0, 1, r));
        float clip = ContrastivePairLoss.clip(0.1f).compute(a, b).value().item();
        assertEquals(Functional.infoNce(a, b, 0.1f).value().item(), clip, 1e-4f);
        ContrastivePairLoss sig = ContrastivePairLoss.sigLip(10, -10);
        float matched = sig.compute(a, a).value().item();
        float random = sig.compute(a, b).value().item();
        System.out.printf("SigLIP loss: matched pairs %.3f, random pairs %.3f%n", matched, random);
        assertTrue(matched < random);
        assertEquals(2, sig.parameters().size());
    }
}
