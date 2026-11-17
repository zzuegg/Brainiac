package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

class EinsumTest {

    static final Tensor A = Tensor.of(Shape.of(2, 3, 4), new float[] {0.126f, -0.132f, 0.64f, 0.105f, -0.536f, 0.362f,
        1.304f, 0.947f, -0.704f, -1.265f, -0.623f, 0.041f, -2.325f, -0.219f, -1.246f, -0.732f, -0.544f, -0.316f, 0.412f,
        1.043f, -0.129f, 1.366f, -0.665f, 0.352f});
    static final Tensor B = Tensor.of(Shape.of(2, 4, 5), new float[] {0.903f, 0.094f, -0.743f, -0.922f, -0.458f, 0.22f,
        -1.01f, -0.209f, -0.159f, 0.541f, 0.215f, 0.355f, -0.654f, -0.13f, 0.784f, 1.493f, -1.259f, 1.514f, 1.346f,
        0.781f, 0.264f, -0.314f, 1.458f, 1.96f, 1.802f, 1.315f, 0.357f, -1.208f, -0.004f, 0.656f, -1.288f, 0.395f,
        0.43f, 0.696f, -1.184f, -0.662f, -0.436f, -1.17f, 1.739f, -0.496f});
    static final Tensor C = Tensor.of(Shape.of(5), new float[] {0.329f, -0.259f, 1.583f, 1.32f, 0.633f});

    private static void check(String expr, int[] shape, float[] first, Tensor... ops) {
        Tensor r = Einsum.of(expr, ops);
        assertArrayEquals(shape, r.shape().toArray(), expr);
        assertArrayEquals(first, Arrays.copyOf(r.toArray(), first.length), 1e-4f, expr);
    }

    @Test
    void matchesNumpy() {
        // values from numpy.einsum
        check("bij,bjk->bik", new int[] {2, 3, 5}, new float[] {0.3791f, 0.24017f, -0.32562f, -0.03705f, 0.45464f,
            1.28986f}, A, B);
        check("bij,bjk->ik", new int[] {3, 5}, new float[] {1.56675f, 0.71902f, -3.13026f, -6.73334f, -2.04033f,
            -0.49042f}, A, B);
        check("bij,bjk", new int[] {3, 5}, new float[] {1.56675f, 0.71902f, -3.13026f, -6.73334f, -2.04033f,
            -0.49042f}, A, B);
        check("bij,bjk,k->bi", new int[] {2, 3}, new float[] {-0.21406f, 5.58112f, 2.20606f, -14.59145f, -2.84876f,
            -2.59453f}, A, B, C);
        check("bij->j", new int[] {4}, new float[] {-4.112f, -0.204f, -0.178f, 1.756f}, A);
        check("i,i->", new int[] {}, new float[] {4.8243f}, C, C);
        check("bij,k->bijk", new int[] {2, 3, 4, 5}, new float[] {0.04145f, -0.03263f, 0.19946f, 0.16632f, 0.07976f,
            -0.04343f}, A, C);
        assertArrayEquals(A.permute(2, 0, 1).toArray(), Einsum.of("bij->jbi", A).toArray(), 0f);
    }

    @Test
    void hasGradients() {
        GradCheck.Result r = GradCheck.check(v -> Einsum.of("bij,bjk,k->bi", v.get(0), v.get(1), v.get(2)).square().sum(),
                List.of(A, B, C));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void refusesWhatItCannotDo() {
        assertThrows(IllegalArgumentException.class, () -> Einsum.of("ii->i", Tensor.zeros(Shape.of(2, 2))));
        assertThrows(IllegalArgumentException.class, () -> Einsum.of("ij,jk->ik", A, B));
        assertThrows(IllegalArgumentException.class, () -> Einsum.of("...ij->ij", A));
    }
}
