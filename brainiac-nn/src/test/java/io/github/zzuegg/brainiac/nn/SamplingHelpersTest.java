package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Per-row picks, straight-through and Gumbel-softmax. */
class SamplingHelpersTest {

    @Test
    void takeAlongPicksPerRowAndSendsTheGradientBack() {
        Var x = Var.param(Tensor.of(Shape.of(2, 3), 1, 2, 3, 4, 5, 6));
        Tensor idx = Tensor.of(Shape.of(2, 2), 2, 0, 1, 1);
        Var picked = x.takeAlong(idx);
        assertEquals(Shape.of(2, 2), picked.shape());
        assertArrayEquals(new float[] {3, 1, 5, 5}, picked.value().toArray());
        picked.sum().backward();
        assertArrayEquals(new float[] {1, 0, 1, 0, 2, 0}, x.grad().toArray());
        assertArrayEquals(new float[] {2, 6}, x.value().takeAlong(Tensor.of(Shape.of(2, 1), 1, 2)).toArray());
    }

    @Test
    void straightThroughHasTheForwardValueAndTheBackwardGradient() {
        Var x = Var.param(Tensor.of(new float[] {0.4f, 1.6f}));
        Var rounded = Functional.straightThrough(Var.of(Tensor.of(new float[] {0, 2})), x.mul(3));
        assertArrayEquals(new float[] {0, 2}, rounded.value().toArray(), 1e-6f);
        rounded.sum().backward();
        assertArrayEquals(new float[] {3, 3}, x.grad().toArray());
    }

    @Test
    void gumbelSoftmaxSamplesInProportion() {
        Tensor logits = Tensor.of(Shape.of(1, 3), (float) Math.log(0.6), (float) Math.log(0.3), (float) Math.log(0.1));
        Var l = Var.param(logits.expand(Shape.of(4000, 3)).add(0f));
        Var hard = Functional.gumbelSoftmax(l, 0.5f, true, new SplittableRandom(1));
        float[] counts = hard.value().sum(0).toArray();
        assertEquals(4000, counts[0] + counts[1] + counts[2], 1e-3);
        assertEquals(0.6, counts[0] / 4000, 0.03);
        assertEquals(0.3, counts[1] / 4000, 0.03);
        hard.mul(Var.of(Tensor.of(Shape.of(1, 3), 1, 2, 3))).sum().backward();
        assertTrue(l.grad().abs().sum().item() > 0);   // the soft sample's gradient
        Var soft = Functional.gumbelSoftmax(l, 1f, false, new SplittableRandom(2));
        assertArrayEquals(Tensor.full(Shape.of(4000), 1f).toArray(), soft.value().sum(1).toArray(), 1e-5f);
    }
}
