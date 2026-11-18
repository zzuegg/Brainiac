package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

/** The 2-D transform matches numpy.fft.fft2 and inverts. */
class Fft2Test {

    private static final Tensor X = Tensor.of(Shape.of(3, 4, 2), 0, 2, 0.5f, 2.25f, 1, 2.5f, 1.5f, 2.75f, 2, 1, 2.5f,
            1.25f, 3, 1.5f, 3.5f, 1.75f, 4, 0, 4.5f, 0.25f, 5, 0.5f, 5.5f, 0.75f);

    @Test
    void matchesNumpyAndInverts() {
        float[] want = {33, 16.5f, -4.5f, 1.5f, -3, -1.5f, -1.5f, -4.5f, -8.535898f, 12.928203f, 0, 0, 0, 0, 0, 0,
            -15.464102f, -0.928203f, 0, 0, 0, 0, 0, 0};
        assertArrayEquals(want, X.fft2(false).toArray(), 1e-4f);
        assertArrayEquals(want, Var.of(X).fft2(false).value().toArray(), 1e-4f);
        assertArrayEquals(X.toArray(), X.fft2(false).fft2(true).toArray(), 1e-5f);
    }
}
