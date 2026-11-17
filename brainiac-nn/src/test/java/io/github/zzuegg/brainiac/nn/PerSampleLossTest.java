package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class PerSampleLossTest {

    @Test
    void perSampleLossesAverageToTheLoss() {
        SplittableRandom r = new SplittableRandom(3);
        Tensor logits = Tensor.uniform(Shape.of(4, 5), -2, 2, r);
        Var probs = Var.of(logits).exp().div(Var.of(logits).exp().sum(1, true));
        Var sig = Var.of(logits).sigmoid();
        Tensor ids = Tensor.of(Shape.of(4), new float[] {0, 3, 4, 1});
        Tensor bits = Tensor.uniform(Shape.of(4, 5), 0, 1, r).greater(Tensor.of(0.5f));
        Tensor values = Tensor.uniform(Shape.of(4, 5), -1, 1, r);
        Loss custom = (p, t) -> p.sub(t).abs().mean();
        Object[][] cases = {{Loss.MSE, probs, values}, {Loss.CROSS_ENTROPY, probs, ids},
            {Loss.CROSS_ENTROPY, probs, Loss.oneHot(ids, Shape.of(4, 5))}, {Loss.BINARY_CROSS_ENTROPY, sig, bits},
            {custom, probs, values}};
        for (Object[] c : cases) {
            Loss loss = (Loss) c[0];
            Var p = (Var) c[1];
            Tensor t = (Tensor) c[2];
            Var each = Loss.perSample(loss, p, t);
            assertEquals(Shape.of(4), each.shape());
            assertEquals(loss.compute(p, t).value().item(), each.mean().value().item(), 1e-5f);
        }
        // and each is that example's own loss
        float[] each = Loss.perSample(Loss.MSE, probs, values).value().toArray();
        assertArrayEquals(new float[] {Loss.MSE.compute(probs.slice(0, 2, 3), values.slice(0, 2, 3)).value().item()},
                new float[] {each[2]}, 1e-6f);
    }
}
