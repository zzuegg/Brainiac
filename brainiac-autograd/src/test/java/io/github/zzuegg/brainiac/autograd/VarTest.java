package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class VarTest {

    @Test
    void computesSimpleGradients() {
        Var x = Var.param(Tensor.of(1, 2, 3));
        Var y = x.square().sum();   // dy/dx = 2x
        y.backward();
        assertEquals(14f, y.value().item());
        assertArrayEquals(new float[] {2, 4, 6}, x.grad().toArray());
    }

    @Test
    void sumsBroadcastGradientsBackToTheBiasShape() {
        Var bias = Var.param(Tensor.of(0, 0));
        Var out = Var.of(Tensor.ones(Shape.of(3, 2))).add(bias).sum();
        out.backward();
        assertEquals(Shape.of(2), bias.grad().shape());
        assertArrayEquals(new float[] {3, 3}, bias.grad().toArray());
    }

    @Test
    void accumulatesUntilZeroed() {
        Var x = Var.param(Tensor.of(1, 2));
        x.mul(3).sum().backward();
        x.mul(3).sum().backward();
        assertArrayEquals(new float[] {6, 6}, x.grad().toArray());
        x.zeroGrad();
        assertNull(x.grad());
    }

    @Test
    void constantsDoNotRecordOrReceiveGradients() {
        Var c = Var.of(Tensor.of(1, 2));
        Var r = c.mul(2).sum();
        assertFalse(r.requiresGrad());
        assertThrows(IllegalStateException.class, r::backward);
        Var p = Var.param(Tensor.of(1, 1));
        c.mul(p).sum().backward();
        assertNull(c.grad());
        assertArrayEquals(new float[] {1, 2}, p.grad().toArray());
    }

    @Test
    void backwardNeedsAScalarOrASeed() {
        Var x = Var.param(Tensor.of(1, 2));
        Var y = x.mul(2);
        assertThrows(IllegalStateException.class, y::backward);
        y.backward(Tensor.of(1, 10));
        assertArrayEquals(new float[] {2, 20}, x.grad().toArray());
        assertThrows(IllegalArgumentException.class, () -> y.backward(Tensor.of(1)));
    }

    @Test
    void handlesVeryDeepGraphsWithoutStackOverflow() {
        Var x = Var.param(Tensor.scalar(1));
        Var y = x;
        for (int i = 0; i < 20_000; i++) {
            y = y.add(1);
        }
        y.backward();
        assertEquals(1f, x.grad().item());
    }

    @Test
    void canTrainALinearModelWithPlainGradientDescent() {
        // y = 2x - 1, learned from 4 points
        Tensor xs = Tensor.of(Shape.of(4, 1), -1, 0, 1, 2);
        Tensor ys = Tensor.of(Shape.of(4, 1), -3, -1, 1, 3);
        Var w = Var.param(Tensor.normal(Shape.of(1, 1), 0, 1, new SplittableRandom(1)));
        Var b = Var.param(Tensor.zeros(Shape.of(1)));
        float loss = Float.NaN;
        for (int step = 0; step < 300; step++) {
            Var l = Var.of(xs).matmul(w).add(b).sub(ys).square().mean();
            loss = l.value().item();
            l.backward();
            w = Var.param(w.value().sub(w.grad().mul(0.1f)));
            b = Var.param(b.value().sub(b.grad().mul(0.1f)));
        }
        assertTrue(loss < 1e-6, "loss " + loss);
        assertEquals(2f, w.value().item(), 1e-3);
        assertEquals(-1f, b.value().item(), 1e-3);
    }

    @Test
    void onlyParametersCanBeAssigned() {
        Var p = Var.param(Tensor.of(1, 2));
        Var before = p.mul(2);
        p.assign(Tensor.of(5, 6));
        assertArrayEquals(new float[] {5, 6}, p.value().toArray());
        assertArrayEquals(new float[] {2, 4}, before.value().toArray());
        assertThrows(IllegalArgumentException.class, () -> p.assign(Tensor.of(1)));
        assertThrows(IllegalStateException.class, () -> before.assign(Tensor.of(1, 1)));
        assertThrows(IllegalStateException.class, () -> Var.of(Tensor.of(1)).assign(Tensor.of(2)));
    }

    @Test
    void gradCheckReportsWrongGradients() {
        // detach() hides the dependency from backward(), but finite differences still see it
        GradCheck.Result r = GradCheck.check(v -> v.get(0).detach().add(v.get(0)).sum(), List.of(Tensor.of(1, 2)));
        assertFalse(r.passed(), r.toString());
        assertEquals(2.0, r.numeric(), 1e-2);
        assertEquals(1.0, r.analytic(), 1e-6);
    }
}
