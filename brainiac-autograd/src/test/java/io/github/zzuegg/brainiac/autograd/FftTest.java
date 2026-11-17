package io.github.zzuegg.brainiac.autograd;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.SplittableRandom;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Fourier transforms against the definition, for powers of two and other lengths. */
class FftTest {

    private static float[] dft(float[] x, int rows, int n, boolean inverse) {
        float[] out = new float[x.length];
        double sign = inverse ? 1 : -1;
        for (int r = 0; r < rows; r++) {
            for (int k = 0; k < n; k++) {
                double re = 0;
                double im = 0;
                for (int j = 0; j < n; j++) {
                    double a = sign * 2 * Math.PI * ((long) j * k % n) / n;
                    double xr = x[2 * (r * n + j)];
                    double xi = x[2 * (r * n + j) + 1];
                    re += xr * Math.cos(a) - xi * Math.sin(a);
                    im += xr * Math.sin(a) + xi * Math.cos(a);
                }
                out[2 * (r * n + k)] = (float) (inverse ? re / n : re);
                out[2 * (r * n + k) + 1] = (float) (inverse ? im / n : im);
            }
        }
        return out;
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 8, 12, 64, 100, 127, 256})
    void matchesTheDefinition(int n) {
        Tensor x = Tensor.uniform(Shape.of(3, n, 2), -1, 1, new SplittableRandom(n));
        float tolerance = 1e-5f * n + 1e-5f;
        assertArrayEquals(dft(x.toArray(), 3, n, false), x.fft(false).toArray(), tolerance, "forward " + n);
        assertArrayEquals(dft(x.toArray(), 3, n, true), x.fft(true).toArray(), tolerance, "inverse " + n);
        assertArrayEquals(x.toArray(), x.fft(false).fft(true).toArray(), 1e-5f, "round trip " + n);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 7, 8, 30})
    void realTransformsRoundTrip(int n) {
        Tensor x = Tensor.uniform(Shape.of(2, n), -1, 1, new SplittableRandom(n));
        Tensor spectrum = x.rfft();
        assertEquals(Shape.of(2, n / 2 + 1, 2), spectrum.shape());
        // the kept half of the complex transform
        float[] full = Tensor.concat(2, x.reshape(2, n, 1), Tensor.zeros(Shape.of(2, n, 1))).fft(false)
                .slice(1, 0, n / 2 + 1).toArray();
        assertArrayEquals(full, spectrum.toArray(), 1e-5f);
        assertArrayEquals(x.toArray(), spectrum.irfft(n).toArray(), 1e-5f);
    }
}
