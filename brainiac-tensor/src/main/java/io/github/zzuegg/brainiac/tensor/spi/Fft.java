package io.github.zzuegg.brainiac.tensor.spi;

/**
 * Discrete Fourier transforms on the host, in double precision: radix 2 for powers of two,
 * Bluestein's algorithm (a power-of-two convolution) for every other length. The reference
 * for backends, and their fallback.
 */
public final class Fft {

    private Fft() {
    }

    /**
     * Transforms rows of complex values in place, stored as interleaved (real, imaginary)
     * pairs: {@code X[k] = sum_j x[j] e^(-2 pi i jk / n)}, or with {@code inverse} the
     * conjugate exponent and a factor {@code 1 / n}.
     *
     * @param data {@code rows * n} complex values, {@code 2 * rows * n} numbers
     * @param rows the number of rows
     * @param n the values per row
     * @param inverse whether to compute the inverse transform
     */
    public static void transform(float[] data, long rows, int n, boolean inverse) {
        double[] re = new double[n];
        double[] im = new double[n];
        for (long r = 0; r < rows; r++) {
            int base = Math.toIntExact(r * n * 2);
            for (int j = 0; j < n; j++) {
                re[j] = data[base + 2 * j];
                im[j] = data[base + 2 * j + 1];
            }
            transform(re, im, inverse);
            double scale = inverse ? 1.0 / n : 1.0;
            for (int j = 0; j < n; j++) {
                data[base + 2 * j] = (float) (re[j] * scale);
                data[base + 2 * j + 1] = (float) (im[j] * scale);
            }
        }
    }

    /**
     * Transforms rows of complex values in place in double precision, as the float version:
     * interleaved (real, imaginary) pairs, with the inverse scaled by {@code 1 / n}.
     *
     * @param data {@code rows * n} complex values, {@code 2 * rows * n} numbers
     * @param rows the number of rows
     * @param n the values per row
     * @param inverse whether to compute the inverse transform
     */
    public static void transform(double[] data, long rows, int n, boolean inverse) {
        double[] re = new double[n];
        double[] im = new double[n];
        for (long r = 0; r < rows; r++) {
            int base = Math.toIntExact(r * n * 2);
            for (int j = 0; j < n; j++) {
                re[j] = data[base + 2 * j];
                im[j] = data[base + 2 * j + 1];
            }
            transform(re, im, inverse);
            double scale = inverse ? 1.0 / n : 1.0;
            for (int j = 0; j < n; j++) {
                data[base + 2 * j] = re[j] * scale;
                data[base + 2 * j + 1] = im[j] * scale;
            }
        }
    }

    /**
     * One transform of any length in double precision, in place, unnormalized:
     * {@code X[k] = sum_j x[j] e^(-+2 pi i jk / n)}; divide by {@code n} after an inverse.
     *
     * @param re the real parts
     * @param im the imaginary parts, as many
     * @param inverse whether to use the conjugate exponent
     */
    public static void transform(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        if (im.length != n) {
            throw new IllegalArgumentException(n + " real and " + im.length + " imaginary parts");
        }
        if (n <= 1) {
            return;
        }
        if (Integer.bitCount(n) == 1) {
            radix2(re, im, inverse);
        } else {
            bluestein(re, im, inverse);
        }
    }

    private static void radix2(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            if (i < j) {
                double t = re[i];
                re[i] = re[j];
                re[j] = t;
                t = im[i];
                im[i] = im[j];
                im[j] = t;
            }
        }
        double sign = inverse ? 1 : -1;
        for (int len = 2; len <= n; len <<= 1) {
            int half = len >> 1;
            for (int k = 0; k < half; k++) {
                double a = sign * 2 * Math.PI * k / len;
                double wr = Math.cos(a);
                double wi = Math.sin(a);
                for (int i = k; i < n; i += len) {
                    int j = i + half;
                    double xr = re[j] * wr - im[j] * wi;
                    double xi = re[j] * wi + im[j] * wr;
                    re[j] = re[i] - xr;
                    im[j] = im[i] - xi;
                    re[i] += xr;
                    im[i] += xi;
                }
            }
        }
    }

    private static void bluestein(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        int m = Integer.highestOneBit(2 * n - 1) << 1;
        double sign = inverse ? 1 : -1;
        // chirp w[j] = e^(sign * pi i j^2 / n), with j^2 taken mod 2n to keep the angle exact
        double[] wr = new double[n];
        double[] wi = new double[n];
        for (int j = 0; j < n; j++) {
            long jj = (long) j * j % (2L * n);
            double a = sign * Math.PI * jj / n;
            wr[j] = Math.cos(a);
            wi[j] = Math.sin(a);
        }
        double[] ar = new double[m];
        double[] ai = new double[m];
        double[] br = new double[m];
        double[] bi = new double[m];
        for (int j = 0; j < n; j++) {
            ar[j] = re[j] * wr[j] - im[j] * wi[j];
            ai[j] = re[j] * wi[j] + im[j] * wr[j];
        }
        br[0] = wr[0];
        bi[0] = -wi[0];
        for (int j = 1; j < n; j++) {
            br[j] = br[m - j] = wr[j];
            bi[j] = bi[m - j] = -wi[j];
        }
        radix2(ar, ai, false);
        radix2(br, bi, false);
        for (int j = 0; j < m; j++) {
            double r = ar[j] * br[j] - ai[j] * bi[j];
            ai[j] = ar[j] * bi[j] + ai[j] * br[j];
            ar[j] = r;
        }
        radix2(ar, ai, true);
        for (int k = 0; k < n; k++) {
            double r = ar[k] / m;
            double i = ai[k] / m;
            re[k] = r * wr[k] - i * wi[k];
            im[k] = r * wi[k] + i * wr[k];
        }
    }
}
