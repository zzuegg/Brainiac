package io.github.zzuegg.brainiac.experimental.imaging;

import java.util.SplittableRandom;

/**
 * Phase retrieval: recovering an image from Fourier intensities only.
 *
 * <p>Classical oversampled Fourier magnitudes ({@link #fourierMagnitudes}): the image (real,
 * non-negative, inside a known support in the top-left corner of a twice-as-large grid) is
 * recovered from {@code |F x|} by
 * <ul>
 *   <li>{@link Method#ER}, error reduction / Gerchberg-Saxton: impose the measured magnitudes,
 *       then the object constraints;</li>
 *   <li>{@link Method#HIO}, Fienup's hybrid input-output (Appl. Opt. 1982): outside the
 *       constraints, {@code g = g - beta g'} instead of zero, a feedback that escapes ER's
 *       stagnation (finished by a few ER steps).</li>
 * </ul>
 *
 * <p>Coded diffraction patterns ({@link CodedDiffraction}): intensities {@code |F(D_l x)|^2} for
 * several random phase masks {@code D_l}, recovered by
 * <ul>
 *   <li>{@link Method#WIRTINGER_FLOW} (Candès, Li &amp; Soltanolkotabi, IEEE TIT 2015,
 *       <a href="https://arxiv.org/abs/1407.1065">arXiv:1407.1065</a>): a spectral initialisation
 *       (the top eigenvector of {@code sum y_i a_i a_i^*} by power iteration) followed by gradient
 *       descent on {@code sum (|a_i^* z|^2 - y_i)^2} with the step ramped up as
 *       {@code min(1 - exp(-k / 330), mu_max)};</li>
 *   <li>{@link Method#TRUNCATED_WF} (Chen &amp; Candès, CPAM 2017): a truncated spectral
 *       initialisation and Poisson-likelihood gradients with outlier terms dropped;</li>
 *   <li>{@link Method#PR_DEEP} (Metzler et al., ICML 2018,
 *       <a href="https://arxiv.org/abs/1803.00212">arXiv:1803.00212</a>): regularisation by
 *       denoising, {@code z -= mu (grad f(z) + lambda (z - D(z)))}, with any denoiser and the
 *       estimate kept real and non-negative (the denoiser term is applied every fourth step with
 *       four times the weight, to save denoiser calls);</li>
 *   <li>{@link Method#RANDOM_START_GD}: Wirtinger flow from a random start, the baseline the
 *       spectral initialisation is measured against.</li>
 * </ul>
 */
public final class PhaseRetrieval {

    /** The algorithms. */
    public enum Method {
        /** Error reduction (Gerchberg-Saxton). */
        ER,
        /** Hybrid input-output. */
        HIO,
        /** Wirtinger flow with spectral initialisation. */
        WIRTINGER_FLOW,
        /** Truncated Wirtinger flow. */
        TRUNCATED_WF,
        /** Wirtinger flow from a random start. */
        RANDOM_START_GD,
        /** Regularisation by denoising on the intensity loss. */
        PR_DEEP,
        /**
         * Plug-and-play half-quadratic splitting: ten gradient steps on the intensity loss, then
         * the estimate is replaced by its denoised version (the baseline prDeep improves on).
         */
        PNP_HQS
    }

    private PhaseRetrieval() {
    }

    // ---------------------------------------------------------------- Fourier magnitudes

    /**
     * Oversampled Fourier magnitudes of an image placed in the top-left corner of a
     * {@code 2 side} grid.
     *
     * @param x the image
     * @param side its side
     * @return {@code |F pad(x)|} on the {@code 2 side x 2 side} grid
     */
    public static double[] fourierMagnitudes(double[] x, int side) {
        int big = 2 * side;
        double[] re = new double[big * big];
        double[] im = new double[big * big];
        for (int r = 0; r < side; r++) {
            System.arraycopy(x, r * side, re, r * big, side);
        }
        ImageMath.fft2(re, im, big, big, false);
        double[] m = new double[big * big];
        for (int i = 0; i < m.length; i++) {
            m[i] = Math.hypot(re[i], im[i]);
        }
        return m;
    }

    /**
     * Recovers an image from oversampled Fourier magnitudes by ER or HIO.
     *
     * @param method {@link Method#ER} or {@link Method#HIO}
     * @param magnitudes the measured magnitudes
     * @param side the image side
     * @param iterations the iterations (HIO ends with 10% of them as ER)
     * @param beta HIO's feedback
     * @param r the random source of the start
     * @return the image
     */
    public static double[] fourier(Method method, double[] magnitudes, int side, int iterations, double beta,
            SplittableRandom r) {
        int big = 2 * side;
        int n = big * big;
        double[] g = new double[n];
        for (int row = 0; row < side; row++) {
            for (int c = 0; c < side; c++) {
                g[row * big + c] = r.nextDouble();
            }
        }
        int hioSteps = method == Method.HIO ? (int) (0.9 * iterations) : 0;
        for (int it = 0; it < iterations; it++) {
            double[] re = g.clone();
            double[] im = new double[n];
            ImageMath.fft2(re, im, big, big, false);
            for (int i = 0; i < n; i++) {
                double a = Math.hypot(re[i], im[i]);
                if (a > 1e-12) {
                    re[i] *= magnitudes[i] / a;
                    im[i] *= magnitudes[i] / a;
                } else {
                    re[i] = magnitudes[i];
                    im[i] = 0;
                }
            }
            ImageMath.fft2(re, im, big, big, true);
            boolean hio = it < hioSteps;
            for (int row = 0; row < big; row++) {
                for (int c = 0; c < big; c++) {
                    int i = row * big + c;
                    boolean inside = row < side && c < side && re[i] >= 0;
                    if (inside) {
                        g[i] = re[i];
                    } else {
                        g[i] = hio ? g[i] - beta * re[i] : 0;
                    }
                }
            }
        }
        double[] x = new double[side * side];
        for (int row = 0; row < side; row++) {
            for (int c = 0; c < side; c++) {
                x[row * side + c] = Math.max(0, g[row * big + c]);
            }
        }
        return x;
    }

    /**
     * The relative error up to the twin image (a 180-degree rotation inside the support).
     *
     * @param estimate the estimate
     * @param truth the truth
     * @param side the side
     * @return {@code min(|x - t|, |flip(x) - t|) / |t|}
     */
    public static double twinError(double[] estimate, double[] truth, int side) {
        double a = 0;
        double b = 0;
        double t = 0;
        for (int i = 0; i < truth.length; i++) {
            double d1 = estimate[i] - truth[i];
            double d2 = estimate[truth.length - 1 - i] - truth[i];
            a += d1 * d1;
            b += d2 * d2;
            t += truth[i] * truth[i];
        }
        return Math.sqrt(Math.min(a, b) / t);
    }

    /**
     * The estimate or its twin, whichever is closer to the truth.
     *
     * @param estimate the estimate
     * @param truth the truth
     * @return the aligned estimate
     */
    public static double[] alignTwin(double[] estimate, double[] truth) {
        double[] flip = new double[estimate.length];
        for (int i = 0; i < flip.length; i++) {
            flip[i] = estimate[flip.length - 1 - i];
        }
        return ImageMath.mse(flip, truth) < ImageMath.mse(estimate, truth) ? flip : estimate;
    }

    // ---------------------------------------------------------------- coded diffraction

    /** Coded-diffraction intensity measurements {@code y_l = |F(D_l x)|^2} of a square image. */
    public static final class CodedDiffraction {
        private final int side;
        private final double[][][] masks;

        /**
         * Creates the measurement model.
         *
         * @param side the image side
         * @param masks {@code [pattern][re or im][pixel]} phase masks
         */
        public CodedDiffraction(int side, double[][][] masks) {
            this.side = side;
            this.masks = masks;
        }

        /** @return the number of patterns */
        public int patterns() {
            return masks.length;
        }

        /** @return the number of intensity measurements */
        public int measurements() {
            return masks.length * side * side;
        }

        /**
         * The complex measurements {@code F(D_l z)}.
         *
         * @param zr real part
         * @param zi imaginary part
         * @return {@code [pattern][re or im][frequency]}
         */
        public double[][][] forward(double[] zr, double[] zi) {
            int n = side * side;
            double[][][] out = new double[masks.length][2][];
            for (int l = 0; l < masks.length; l++) {
                double[] re = new double[n];
                double[] im = new double[n];
                for (int i = 0; i < n; i++) {
                    double dr = masks[l][0][i];
                    double di = masks[l][1][i];
                    re[i] = dr * zr[i] - di * zi[i];
                    im[i] = dr * zi[i] + di * zr[i];
                }
                ImageMath.fft2(re, im, side, side, false);
                out[l][0] = re;
                out[l][1] = im;
            }
            return out;
        }

        /**
         * The adjoint {@code sum_l D_l^* F^H v_l} (with {@code F^H = n F^-1}).
         *
         * @param v {@code [pattern][re or im][frequency]}
         * @return {@code {re, im}}
         */
        public double[][] adjoint(double[][][] v) {
            int n = side * side;
            double[] or = new double[n];
            double[] oi = new double[n];
            for (int l = 0; l < masks.length; l++) {
                double[] re = v[l][0].clone();
                double[] im = v[l][1].clone();
                ImageMath.fft2(re, im, side, side, true);
                for (int i = 0; i < n; i++) {
                    double dr = masks[l][0][i];
                    double di = -masks[l][1][i];
                    or[i] += n * (dr * re[i] - di * im[i]);
                    oi[i] += n * (dr * im[i] + di * re[i]);
                }
            }
            return new double[][] {or, oi};
        }

        /**
         * Noiseless intensities of a real image.
         *
         * @param x the image
         * @return {@code [pattern * n]} intensities
         */
        public double[] intensities(double[] x) {
            double[][][] f = forward(x, new double[x.length]);
            int n = side * side;
            double[] y = new double[masks.length * n];
            for (int l = 0; l < masks.length; l++) {
                for (int i = 0; i < n; i++) {
                    y[l * n + i] = f[l][0][i] * f[l][0][i] + f[l][1][i] * f[l][1][i];
                }
            }
            return y;
        }
    }

    /** A denoiser for {@link Method#PR_DEEP} on one real image. */
    @FunctionalInterface
    public interface Denoiser {

        /**
         * Denoises.
         *
         * @param x the image
         * @param sigma the noise level
         * @return the estimate
         */
        double[] denoise(double[] x, double sigma);
    }

    private static double[][] spectralInit(CodedDiffraction a, double[] y, boolean truncated, int iterations,
            SplittableRandom r) {
        int n = a.side * a.side;
        int m = y.length;
        double mean = 0;
        for (double v : y) {
            mean += v / m;
        }
        double[] w = y.clone();
        if (truncated) {
            for (int i = 0; i < m; i++) {
                if (w[i] > 9 * mean) {
                    w[i] = 0;
                }
            }
        }
        double[] zr = new double[n];
        double[] zi = new double[n];
        for (int i = 0; i < n; i++) {
            zr[i] = r.nextGaussian();
            zi[i] = r.nextGaussian();
        }
        for (int it = 0; it < iterations; it++) {
            double nz = Math.sqrt(ImageMath.dot(zr, zr) + ImageMath.dot(zi, zi));
            for (int i = 0; i < n; i++) {
                zr[i] /= nz;
                zi[i] /= nz;
            }
            double[][][] f = a.forward(zr, zi);
            for (int l = 0; l < f.length; l++) {
                for (int i = 0; i < n; i++) {
                    f[l][0][i] *= w[l * n + i] / m;
                    f[l][1][i] *= w[l * n + i] / m;
                }
            }
            double[][] back = a.adjoint(f);
            zr = back[0];
            zi = back[1];
        }
        double nz = Math.sqrt(ImageMath.dot(zr, zr) + ImageMath.dot(zi, zi));
        // E |a_i^* x|^2 = |x|^2 for these masks, so |x| is about sqrt(mean y)
        double lambda = Math.sqrt(mean);
        for (int i = 0; i < n; i++) {
            zr[i] *= lambda / nz;
            zi[i] *= lambda / nz;
        }
        return new double[][] {zr, zi};
    }

    /**
     * Recovers a real image from coded-diffraction intensities.
     *
     * @param method {@link Method#WIRTINGER_FLOW}, {@link Method#TRUNCATED_WF},
     *     {@link Method#RANDOM_START_GD}, {@link Method#PR_DEEP} or {@link Method#PNP_HQS}
     * @param a the measurement model
     * @param y the measured intensities
     * @param iterations gradient iterations
     * @param denoiser the denoiser for prDeep and HQS ({@code null} otherwise)
     * @param lambda prDeep's regularisation weight
     * @param sigma the denoiser's noise level for prDeep
     * @param r the random source
     * @return {@code {re, im}} of the estimate (phase not aligned)
     */
    public static double[][] coded(Method method, CodedDiffraction a, double[] y, int iterations, Denoiser denoiser,
            double lambda, double sigma, SplittableRandom r) {
        int n = a.side * a.side;
        int m = y.length;
        double[][] z;
        if (method == Method.RANDOM_START_GD) {
            double mean = 0;
            for (double v : y) {
                mean += v / m;
            }
            z = new double[2][n];
            double s = Math.sqrt(mean / (2 * n));
            for (int i = 0; i < n; i++) {
                z[0][i] = s * r.nextGaussian();
                z[1][i] = s * r.nextGaussian();
            }
        } else {
            z = spectralInit(a, y, method == Method.TRUNCATED_WF, 50, r);
        }
        if (method == Method.PR_DEEP || method == Method.PNP_HQS) {
            // prDeep works on real images: rotate the global phase so the sum is real and positive
            double sumR = 0;
            double sumI = 0;
            for (int i = 0; i < n; i++) {
                sumR += z[0][i];
                sumI += z[1][i];
            }
            double ph = Math.atan2(sumI, sumR);
            for (int i = 0; i < n; i++) {
                z[0][i] = Math.max(0, z[0][i] * Math.cos(ph) + z[1][i] * Math.sin(ph));
                z[1][i] = 0;
            }
        }
        // |x|^2 is about the mean intensity, a scale that does not collapse if the start does
        double z0 = 0;
        for (double v : y) {
            z0 += v / m;
        }
        for (int k = 0; k < iterations; k++) {
            double[][][] f = a.forward(z[0], z[1]);
            double[][][] g = new double[f.length][2][n];
            if (method == Method.TRUNCATED_WF) {
                double zn = Math.sqrt(ImageMath.dot(z[0], z[0]) + ImageMath.dot(z[1], z[1]));
                double kt = 0;
                for (int l = 0; l < f.length; l++) {
                    for (int i = 0; i < n; i++) {
                        double p = f[l][0][i] * f[l][0][i] + f[l][1][i] * f[l][1][i];
                        kt += Math.abs(y[l * n + i] - p) / m;
                    }
                }
                for (int l = 0; l < f.length; l++) {
                    for (int i = 0; i < n; i++) {
                        double p = f[l][0][i] * f[l][0][i] + f[l][1][i] * f[l][1][i];
                        double amp = Math.sqrt(p);
                        double ratio = amp / zn;
                        boolean keep = ratio >= 0.3 && ratio <= 5
                                && Math.abs(y[l * n + i] - p) <= 5 * kt * ratio;
                        if (keep && p > 1e-12) {
                            double c = 2 * (p - y[l * n + i]) / p;
                            g[l][0][i] = c * f[l][0][i];
                            g[l][1][i] = c * f[l][1][i];
                        }
                    }
                }
            } else {
                for (int l = 0; l < f.length; l++) {
                    for (int i = 0; i < n; i++) {
                        double p = f[l][0][i] * f[l][0][i] + f[l][1][i] * f[l][1][i];
                        double c = (p - y[l * n + i]);
                        g[l][0][i] = c * f[l][0][i];
                        g[l][1][i] = c * f[l][1][i];
                    }
                }
            }
            double[][] grad = a.adjoint(g);
            if (method == Method.TRUNCATED_WF) {
                // the Poisson-likelihood gradient is scale-free, so the step is mu / m
                double step = 0.2 / m;
                for (int i = 0; i < n; i++) {
                    z[0][i] -= step * grad[0][i];
                    z[1][i] -= step * grad[1][i];
                }
                continue;
            }
            double mu = Math.min(1 - Math.exp(-(k + 1) / 330.0), 0.2);
            double scale = mu / z0 / m;
            for (int i = 0; i < n; i++) {
                z[0][i] -= scale * grad[0][i];
                z[1][i] -= scale * grad[1][i];
            }
            if (method == Method.PR_DEEP) {
                // the RED step, taken every fourth iteration with four times the weight
                double[] re = z[0].clone();
                double[] den = k % 4 == 3 ? denoiser.denoise(re, sigma) : re;
                for (int i = 0; i < n; i++) {
                    z[0][i] = Math.max(0, re[i] - 4 * mu * lambda * (re[i] - den[i]));
                    z[1][i] = 0;
                }
            } else if (method == Method.PNP_HQS) {
                for (int i = 0; i < n; i++) {
                    z[0][i] = Math.max(0, z[0][i]);
                    z[1][i] = 0;
                }
                if (k % 10 == 9) {
                    z[0] = denoiser.denoise(z[0], sigma);
                }
            }
        }
        return z;
    }

    /**
     * Aligns a complex estimate to a real truth by the best global phase and returns its real
     * part.
     *
     * @param z {@code {re, im}}
     * @param truth the real truth
     * @return the aligned real part
     */
    public static double[] alignPhase(double[][] z, double[] truth) {
        double cr = 0;
        double ci = 0;
        for (int i = 0; i < truth.length; i++) {
            cr += z[0][i] * truth[i];
            ci += z[1][i] * truth[i];
        }
        double ph = Math.atan2(ci, cr);
        double[] out = new double[truth.length];
        for (int i = 0; i < truth.length; i++) {
            out[i] = z[0][i] * Math.cos(ph) + z[1][i] * Math.sin(ph);
        }
        return out;
    }

    /**
     * The relative error up to a global phase, {@code min_phi |z e^{i phi} - x| / |x|}.
     *
     * @param z {@code {re, im}}
     * @param truth the real truth
     * @return the error
     */
    public static double phaseError(double[][] z, double[] truth) {
        double cr = 0;
        double ci = 0;
        double zz = 0;
        double tt = 0;
        for (int i = 0; i < truth.length; i++) {
            cr += z[0][i] * truth[i];
            ci += z[1][i] * truth[i];
            zz += z[0][i] * z[0][i] + z[1][i] * z[1][i];
            tt += truth[i] * truth[i];
        }
        double d2 = zz + tt - 2 * Math.hypot(cr, ci);
        return Math.sqrt(Math.max(0, d2) / tt);
    }
}
