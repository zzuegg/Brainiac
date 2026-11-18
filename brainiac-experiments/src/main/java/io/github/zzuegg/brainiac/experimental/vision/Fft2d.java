package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;

/**
 * Two-dimensional discrete Fourier transforms: an in-place radix-2 row-column FFT on plain
 * arrays (any power-of-two sides), and a differentiable version on {@link Var}s that applies
 * the core 1-D {@link Var#fft(boolean)} along both spatial axes of complex values
 * {@code [..., h, w, 2]} (the layout of the core {@code Complex} helpers).
 *
 * <p>The plain transform is the workhorse of FFT convolution and spectral pooling
 * ({@link SpectralConv}); the {@code Var} transform gives exact gradients (the conjugate
 * transform) for spectral layers trained end to end. Wave 10 found no 2-D FFT anywhere in the
 * repository; this fills that gap without changing any existing class.
 */
public final class Fft2d {

    private Fft2d() {
    }

    /**
     * @param n a positive length
     * @return whether it is a power of two
     */
    public static boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    /**
     * @param n a positive length
     * @return the smallest power of two at least {@code n}
     */
    public static int nextPowerOfTwo(int n) {
        int p = 1;
        while (p < n) {
            p <<= 1;
        }
        return p;
    }

    /**
     * An in-place 1-D radix-2 FFT of a strided sequence.
     *
     * @param re real parts
     * @param im imaginary parts
     * @param offset index of the first element
     * @param stride distance between elements
     * @param n the length, a power of two
     * @param inverse whether to compute the inverse (scaled by {@code 1 / n})
     */
    public static void fft1d(double[] re, double[] im, int offset, int stride, int n, boolean inverse) {
        if (!isPowerOfTwo(n)) {
            throw new IllegalArgumentException("length must be a power of two, got " + n);
        }
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            if (i < j) {
                int a = offset + i * stride;
                int b = offset + j * stride;
                double t = re[a];
                re[a] = re[b];
                re[b] = t;
                t = im[a];
                im[a] = im[b];
                im[b] = t;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = 2 * Math.PI / len * (inverse ? 1 : -1);
            double wr = Math.cos(ang);
            double wi = Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                double cr = 1;
                double ci = 0;
                for (int k = 0; k < len / 2; k++) {
                    int a = offset + (i + k) * stride;
                    int b = offset + (i + k + len / 2) * stride;
                    double xr = re[b] * cr - im[b] * ci;
                    double xi = re[b] * ci + im[b] * cr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                    double nr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = nr;
                }
            }
        }
        if (inverse) {
            for (int i = 0; i < n; i++) {
                re[offset + i * stride] /= n;
                im[offset + i * stride] /= n;
            }
        }
    }

    /**
     * An in-place 2-D FFT of a row-major {@code h x w} complex image: every row, then every
     * column.
     *
     * @param re real parts, {@code h * w}
     * @param im imaginary parts, {@code h * w}
     * @param h rows, a power of two
     * @param w columns, a power of two
     * @param inverse whether to compute the inverse (scaled by {@code 1 / (h w)})
     */
    public static void fft2d(double[] re, double[] im, int h, int w, boolean inverse) {
        for (int y = 0; y < h; y++) {
            fft1d(re, im, y * w, 1, w, inverse);
        }
        for (int x = 0; x < w; x++) {
            fft1d(re, im, x, w, h, inverse);
        }
    }

    /**
     * The 2-D spectrum of a real image, zero-padded to {@code h x w}.
     *
     * @param image row-major pixels
     * @param ih image rows
     * @param iw image columns
     * @param h transform rows, a power of two at least {@code ih}
     * @param w transform columns, a power of two at least {@code iw}
     * @return {@code {re, im}}, each {@code h * w}
     */
    public static double[][] forward(float[] image, int ih, int iw, int h, int w) {
        double[] re = new double[h * w];
        double[] im = new double[h * w];
        for (int y = 0; y < ih; y++) {
            for (int x = 0; x < iw; x++) {
                re[y * w + x] = image[y * iw + x];
            }
        }
        fft2d(re, im, h, w, false);
        return new double[][] {re, im};
    }

    /**
     * A differentiable 2-D DFT of complex values {@code [..., h, w, 2]} with the core 1-D
     * transform along both spatial axes (any sizes the core FFT accepts).
     *
     * @param z complex values {@code [..., h, w, 2]}
     * @param inverse whether to compute the inverse transform
     * @return the transform, same shape
     */
    public static Var fft2d(Var z, boolean inverse) {
        int rank = z.shape().rank();
        if (rank < 3 || z.shape().dim(rank - 1) != 2) {
            throw new IllegalArgumentException("need complex values [..., h, w, 2], got " + z.shape());
        }
        Var rows = z.fft(inverse);
        int[] swap = new int[rank];
        for (int i = 0; i < rank; i++) {
            swap[i] = i;
        }
        swap[rank - 3] = rank - 2;
        swap[rank - 2] = rank - 3;
        return rows.permute(swap).fft(inverse).permute(swap);
    }

    /**
     * Lifts real images {@code [..., h, w]} to complex values {@code [..., h, w, 2]}.
     *
     * @param x real values
     * @return complex values with zero imaginary part
     */
    public static Var complex(Var x) {
        int[] dims = java.util.Arrays.copyOf(x.shape().toArray(), x.shape().rank() + 1);
        dims[dims.length - 1] = 1;
        Var re = x.reshape(dims);
        return Var.concat(dims.length - 1, re, re.mul(0f));
    }
}
