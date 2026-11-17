package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

import org.junit.jupiter.api.Test;

class ComplexTest {

    @Test
    void arithmetic() {
        Var a = Var.of(Tensor.of(Shape.of(2, 2), 1, 2, 3, -1));   // 1+2i, 3-i
        Var b = Var.of(Tensor.of(Shape.of(2, 2), 0, 1, 2, 2));    // i, 2+2i
        assertArrayEquals(new float[] {-2, 1, 8, 4}, Complex.mul(a, b).value().toArray(), 1e-6f);
        assertArrayEquals(new float[] {1, -2, 3, 1}, Complex.conj(a).value().toArray());
        assertArrayEquals(new float[] {(float) Math.sqrt(5), (float) Math.sqrt(10)}, Complex.abs(a).value().toArray(),
                1e-6f);
        Var p = Complex.polar(Var.of(Tensor.of(new float[] {2})), Var.of(Tensor.of(new float[] {(float) (Math.PI / 2)})));
        assertArrayEquals(new float[] {0, 2}, p.value().toArray(), 1e-6f);
        // the transform of a product is the circular convolution: check mul against fft
        Var x = Var.of(Tensor.of(Shape.of(4, 2), 1, 0, 2, 0, 0, 0, 0, 0));
        Var y = Var.of(Tensor.of(Shape.of(4, 2), 0, 0, 1, 0, 0, 0, 0, 0));
        float[] conv = Complex.mul(x.fft(false), y.fft(false)).fft(true).value().toArray();
        assertArrayEquals(new float[] {0, 0, 1, 0, 2, 0, 0, 0}, conv, 1e-5f);
    }

    @Test
    void gradients() {
        Tensor a = Tensor.uniform(Shape.of(3, 2), -1, 1, new java.util.SplittableRandom(1)).to(DType.FLOAT64);
        Tensor b = Tensor.uniform(Shape.of(3, 2), -1, 1, new java.util.SplittableRandom(2)).to(DType.FLOAT64);
        GradCheck.Result r = GradCheck.check(v -> Complex.abs(Complex.mul(v.get(0), Complex.conj(v.get(1)))).sum(),
                List.of(a, b));
        assertTrue(r.passed(), r.toString());
    }
}
