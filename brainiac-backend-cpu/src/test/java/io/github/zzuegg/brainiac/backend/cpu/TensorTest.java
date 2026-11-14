package io.github.zzuegg.brainiac.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Tensor behaviour, run against the CPU backend. */
class TensorTest {

    private static final float EPS = 1e-6f;

    private static void assertValues(Tensor t, Shape shape, float... expected) {
        assertEquals(shape, t.shape());
        assertArrayEquals(expected, t.toArray(), EPS);
    }

    @Test
    void createsFromValuesRowsAndConstants() {
        assertValues(Tensor.of(Shape.of(2, 2), 1, 2, 3, 4), Shape.of(2, 2), 1, 2, 3, 4);
        assertValues(Tensor.of(new float[][] {{1, 2, 3}, {4, 5, 6}}), Shape.of(2, 3), 1, 2, 3, 4, 5, 6);
        assertValues(Tensor.of(7, 8), Shape.of(2), 7, 8);
        assertValues(Tensor.scalar(3), Shape.of(), 3);
        assertValues(Tensor.zeros(Shape.of(3)), Shape.of(3), 0, 0, 0);
        assertValues(Tensor.ones(Shape.of(2)), Shape.of(2), 1, 1);
        assertEquals(Device.cpu(), Tensor.zeros(Shape.of(1)).device());
    }

    @Test
    void rejectsWrongValueCountsAndRaggedRows() {
        assertThrows(IllegalArgumentException.class, () -> Tensor.of(Shape.of(2, 2), 1, 2, 3));
        assertThrows(IllegalArgumentException.class, () -> Tensor.of(new float[][] {{1, 2}, {3}}));
    }

    @Test
    void isImmutable() {
        float[] values = {1, 2};
        Tensor t = Tensor.of(values);
        values[0] = 9;
        t.toArray()[1] = 9;
        assertValues(t, Shape.of(2), 1, 2);
        Tensor u = t.add(1);
        assertValues(t, Shape.of(2), 1, 2);
        assertValues(u, Shape.of(2), 2, 3);
    }

    @Test
    void addsSubtractsMultipliesAndDividesElementWise() {
        Tensor a = Tensor.of(1, 2, 3);
        Tensor b = Tensor.of(4, 5, 6);
        assertValues(a.add(b), Shape.of(3), 5, 7, 9);
        assertValues(a.sub(b), Shape.of(3), -3, -3, -3);
        assertValues(a.mul(b), Shape.of(3), 4, 10, 18);
        assertValues(b.div(a), Shape.of(3), 4, 2.5f, 2);
        assertValues(a.pow(2), Shape.of(3), 1, 4, 9);
        assertValues(a.mul(2).sub(1).div(2).add(1), Shape.of(3), 1.5f, 2.5f, 3.5f);
        assertValues(a.maximum(Tensor.of(2, 2, 2)), Shape.of(3), 2, 2, 3);
        assertValues(a.minimum(Tensor.of(2, 2, 2)), Shape.of(3), 1, 2, 2);
        assertValues(a.greater(Tensor.of(2, 2, 2)), Shape.of(3), 0, 0, 1);
        assertValues(a.equalTo(Tensor.of(2, 2, 2)), Shape.of(3), 0, 1, 0);
    }

    @Test
    void broadcastsBiasRowsAndColumns() {
        Tensor x = Tensor.of(Shape.of(2, 3), 1, 2, 3, 4, 5, 6);
        assertValues(x.add(Tensor.of(10, 20, 30)), Shape.of(2, 3), 11, 22, 33, 14, 25, 36);
        assertValues(x.mul(Tensor.of(Shape.of(2, 1), 1, -1)), Shape.of(2, 3), 1, 2, 3, -4, -5, -6);
        assertValues(Tensor.of(Shape.of(2, 1), 1, 2).add(Tensor.of(10, 20, 30)),
                Shape.of(2, 3), 11, 21, 31, 12, 22, 32);
        assertValues(Tensor.scalar(1).sub(x), Shape.of(2, 3), 0, -1, -2, -3, -4, -5);
        assertThrows(IllegalArgumentException.class, () -> x.add(Tensor.of(1, 2)));
    }

    @Test
    void broadcastsInThreeDimensions() {
        Tensor a = Tensor.of(Shape.of(2, 1, 2), 1, 2, 3, 4);
        Tensor b = Tensor.of(Shape.of(3, 1), 10, 20, 30);
        assertValues(a.add(b), Shape.of(2, 3, 2),
                11, 12, 21, 22, 31, 32,
                13, 14, 23, 24, 33, 34);
    }

    @Test
    void appliesActivationsAndMathFunctions() {
        Tensor x = Tensor.of(-2, 0, 3);
        assertValues(x.neg(), Shape.of(3), 2, 0, -3);
        assertValues(x.abs(), Shape.of(3), 2, 0, 3);
        assertValues(x.relu(), Shape.of(3), 0, 0, 3);
        assertValues(x.step(), Shape.of(3), 0, 0, 1);
        assertValues(x.sign(), Shape.of(3), -1, 0, 1);
        assertValues(x.square(), Shape.of(3), 4, 0, 9);
        assertValues(x.tanh(), Shape.of(3), (float) Math.tanh(-2), 0, (float) Math.tanh(3));
        assertValues(x.exp(), Shape.of(3), (float) Math.exp(-2), 1, (float) Math.exp(3));
        assertValues(Tensor.of(1, 4).sqrt(), Shape.of(2), 1, 2);
        assertValues(Tensor.of(1, (float) Math.E).log(), Shape.of(2), 0, 1);
        assertValues(x.sigmoid(), Shape.of(3),
                (float) (1 / (1 + Math.exp(2))), 0.5f, (float) (1 / (1 + Math.exp(-3))));
    }

    @Test
    void sigmoidStaysFiniteForLargeInputs() {
        assertValues(Tensor.of(-1000, 1000).sigmoid(), Shape.of(2), 0, 1);
    }

    @Test
    void multipliesMatrices() {
        Tensor a = Tensor.of(Shape.of(2, 3), 1, 2, 3, 4, 5, 6);
        Tensor b = Tensor.of(Shape.of(3, 2), 7, 8, 9, 10, 11, 12);
        assertValues(a.matmul(b), Shape.of(2, 2), 58, 64, 139, 154);
        assertThrows(IllegalArgumentException.class, () -> a.matmul(a));
        assertThrows(IllegalArgumentException.class, () -> a.matmul(Tensor.of(1, 2, 3)));
    }

    @Test
    void transposesAndPermutes() {
        Tensor a = Tensor.of(Shape.of(2, 3), 1, 2, 3, 4, 5, 6);
        assertValues(a.transpose(), Shape.of(3, 2), 1, 4, 2, 5, 3, 6);
        assertValues(a.transpose().transpose(), Shape.of(2, 3), 1, 2, 3, 4, 5, 6);
        Tensor c = Tensor.of(Shape.of(2, 1, 3), 1, 2, 3, 4, 5, 6);
        assertValues(c.permute(2, 0, 1), Shape.of(3, 2, 1), 1, 4, 2, 5, 3, 6);
        assertThrows(IllegalArgumentException.class, () -> c.permute(0, 0, 1));
    }

    @Test
    void reshapesWithoutChangingValues() {
        Tensor a = Tensor.of(1, 2, 3, 4, 5, 6);
        assertValues(a.reshape(2, 3), Shape.of(2, 3), 1, 2, 3, 4, 5, 6);
        assertValues(a.reshape(3, 2).flatten(), Shape.of(6), 1, 2, 3, 4, 5, 6);
        assertThrows(IllegalArgumentException.class, () -> a.reshape(4, 2));
    }

    @Test
    void reducesAlongAxesAndOverEverything() {
        Tensor a = Tensor.of(Shape.of(2, 3), 1, 5, 3, 4, 2, 6);
        assertValues(a.sum(), Shape.of(), 21);
        assertValues(a.mean(), Shape.of(), 3.5f);
        assertValues(a.max(), Shape.of(), 6);
        assertValues(a.min(), Shape.of(), 1);
        assertValues(a.sum(0), Shape.of(3), 5, 7, 9);
        assertValues(a.sum(1), Shape.of(2), 9, 12);
        assertValues(a.sum(-1, true), Shape.of(2, 1), 9, 12);
        assertValues(a.mean(0), Shape.of(3), 2.5f, 3.5f, 4.5f);
        assertValues(a.max(1, false), Shape.of(2), 5, 6);
        assertValues(a.min(0, true), Shape.of(1, 3), 1, 2, 3);
        assertValues(a.argmax(1), Shape.of(2), 1, 2);
        assertValues(Tensor.of(3, 3, 1).argmax(0), Shape.of(), 0);
        assertEquals(21f, a.sum().item());
    }

    @Test
    void reducesTheMiddleAxisOfThreeDimensions() {
        Tensor a = Tensor.of(Shape.of(2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8);
        assertValues(a.sum(1), Shape.of(2, 2), 4, 6, 12, 14);
    }

    @Test
    void expandsAndSumsBackDown() {
        Tensor bias = Tensor.of(1, 2, 3);
        Tensor expanded = bias.expand(Shape.of(2, 3));
        assertValues(expanded, Shape.of(2, 3), 1, 2, 3, 1, 2, 3);
        assertValues(expanded.sumTo(Shape.of(3)), Shape.of(3), 2, 4, 6);
        assertValues(expanded.sumTo(Shape.of(1, 3)), Shape.of(1, 3), 2, 4, 6);
        assertValues(expanded.sumTo(Shape.of(2, 1)), Shape.of(2, 1), 6, 6);
        assertValues(expanded.sumTo(Shape.of()), Shape.of(), 12);
        assertThrows(IllegalArgumentException.class, () -> expanded.sumTo(Shape.of(2)));
    }

    @Test
    void itemNeedsExactlyOneValue() {
        assertEquals(4f, Tensor.of(4).item());
        assertThrows(IllegalStateException.class, () -> Tensor.of(1, 2).item());
    }

    @Test
    void moveToSameDeviceReturnsSameTensor() {
        Tensor t = Tensor.of(1, 2);
        assertSame(t, t.to(Device.cpu()));
    }

    @Test
    void randomTensorsAreReproducibleWithASeed() {
        Tensor u1 = Tensor.uniform(Shape.of(100), -1, 1, new SplittableRandom(42));
        Tensor u2 = Tensor.uniform(Shape.of(100), -1, 1, new SplittableRandom(42));
        assertArrayEquals(u1.toArray(), u2.toArray());
        for (float v : u1.toArray()) {
            assertTrue(v >= -1 && v < 1);
        }
        Tensor n = Tensor.normal(Shape.of(10_000), 2, 0.5f, new SplittableRandom(7));
        assertEquals(2f, n.mean().item(), 0.02f);
        assertEquals(0.25f, n.sub(2).square().mean().item(), 0.02f);
    }

    @Test
    void describesItself() {
        assertEquals("Tensor[2]@cpu [1.0, 2.0]", Tensor.of(1, 2).toString());
    }
}
