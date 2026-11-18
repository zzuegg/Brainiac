package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Matrices;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Small generators for imaging inverse problems: random Shepp-Logan-style ellipse phantoms,
 * piecewise-smooth scenes (from {@link DegradedImages}) in grey and colour (with moiré-prone
 * stripes), Cartesian k-space line masks (uniform random and variable density), random phase
 * masks for coded diffraction, motion and defocus blur kernels, Bayer mosaics, inpainting masks,
 * Poisson noise, the Donoho-Johnstone 1-D test signals and Gaussian-mixture priors with
 * linear-Gaussian measurements whose posterior is a known mixture ({@link GaussianMixture}).
 * The forward operators themselves (Radon, masked FFT, single-pixel patterns) live in
 * {@code experimental.imaging}.
 */
public final class InverseProblems {

    private InverseProblems() {
    }

    // ---------------------------------------------------------------- images

    /**
     * Random ellipse phantoms in {@code [0, 1]} inside the inscribed disc: a bright skull ellipse,
     * a darker interior and 3-6 random inner ellipses of varying contrast (the Shepp-Logan
     * recipe with random parameters).
     *
     * @param n the number of images
     * @param side the side
     * @param seed the seed
     * @return the images
     */
    public static List<double[]> phantoms(int n, int side, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<double[]> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            double[] img = new double[side * side];
            double ax = 0.72 + 0.12 * r.nextDouble();
            double ay = 0.80 + 0.12 * r.nextDouble();
            ellipse(img, side, 0, 0, ax, ay, 0, 0.9);
            ellipse(img, side, 0, -0.02, ax - 0.06, ay - 0.06, 0, -0.65);
            int k = 3 + r.nextInt(4);
            for (int e = 0; e < k; e++) {
                double cx = (r.nextDouble() - 0.5) * 0.8 * ax;
                double cy = (r.nextDouble() - 0.5) * 0.8 * ay;
                double a = 0.05 + 0.25 * r.nextDouble();
                double b = 0.05 + 0.25 * r.nextDouble();
                double th = r.nextDouble() * Math.PI;
                double v = (r.nextBoolean() ? 1 : -0.5) * (0.1 + 0.35 * r.nextDouble());
                ellipse(img, side, cx, cy, a, b, th, v);
            }
            for (int p = 0; p < img.length; p++) {
                img[p] = Math.max(0, Math.min(1, img[p]));
            }
            out.add(img);
        }
        return out;
    }

    private static void ellipse(double[] img, int side, double cx, double cy, double a, double b, double th,
            double value) {
        double c = Math.cos(th);
        double s = Math.sin(th);
        for (int row = 0; row < side; row++) {
            for (int col = 0; col < side; col++) {
                double x = (2 * (col + 0.5) / side - 1) - cx;
                double y = (2 * (row + 0.5) / side - 1) - cy;
                double u = (x * c + y * s) / a;
                double v = (-x * s + y * c) / b;
                if (u * u + v * v <= 1) {
                    img[row * side + col] += value;
                }
            }
        }
    }

    /**
     * Piecewise-smooth grey scenes ({@link DegradedImages#scenes}) as doubles.
     *
     * @param n the number of images
     * @param side the side
     * @param seed the seed
     * @return the images
     */
    public static List<double[]> scenes(int n, int side, long seed) {
        List<double[]> out = new ArrayList<>(n);
        for (float[] f : DegradedImages.scenes(n, side, seed)) {
            double[] d = new double[f.length];
            for (int i = 0; i < f.length; i++) {
                d[i] = Math.max(0, Math.min(1, f[i]));
            }
            out.add(d);
        }
        return out;
    }

    /**
     * Colour scenes {@code [3][side * side]}: a grey scene's regions mapped to two random colours,
     * with a patch of fine oblique stripes (period 2-4 pixels) that makes demosaicing alias.
     *
     * @param n the number of images
     * @param side the side
     * @param stripes whether to add the stripe patch
     * @param seed the seed
     * @return the images, channels R, G, B
     */
    public static List<double[][]> colourScenes(int n, int side, boolean stripes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<double[]> grey = scenes(n, side, seed + 1);
        List<double[][]> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            double[] g = grey.get(i);
            double[] c1 = {r.nextDouble(), r.nextDouble(), r.nextDouble()};
            double[] c2 = {r.nextDouble(), r.nextDouble(), r.nextDouble()};
            double[][] img = new double[3][side * side];
            for (int p = 0; p < g.length; p++) {
                for (int ch = 0; ch < 3; ch++) {
                    img[ch][p] = 0.1 + 0.8 * (g[p] * c1[ch] + (1 - g[p]) * c2[ch]);
                }
            }
            if (stripes) {
                int x0 = r.nextInt(side / 2);
                int y0 = r.nextInt(side / 2);
                double period = 2 + 2 * r.nextDouble();
                double angle = r.nextDouble() * Math.PI;
                double[] sc = {r.nextDouble(), r.nextDouble(), r.nextDouble()};
                for (int y = y0; y < y0 + side / 2; y++) {
                    for (int x = x0; x < x0 + side / 2; x++) {
                        double t = 0.5 + 0.5 * Math.sin(2 * Math.PI * (x * Math.cos(angle) + y * Math.sin(angle))
                                / period);
                        for (int ch = 0; ch < 3; ch++) {
                            img[ch][y * side + x] = 0.1 + 0.8 * (t * sc[ch] + (1 - t) * (1 - sc[ch]));
                        }
                    }
                }
            }
            out.add(img);
        }
        return out;
    }

    // ---------------------------------------------------------------- measurement patterns

    /**
     * A Cartesian k-space mask of full phase-encoding lines (columns of the unshifted DFT grid):
     * a fully sampled centre of {@code side / 16} lines either side of DC, the rest drawn at
     * random, uniformly or with density decaying with the frequency (variable density).
     *
     * @param side the side
     * @param acceleration the undersampling factor (4 or 8)
     * @param variableDensity whether to favour low frequencies
     * @param seed the seed
     * @return {@code side x side} sampled flags
     */
    public static boolean[] kSpaceMask(int side, int acceleration, boolean variableDensity, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int lines = Math.max(1, side / acceleration);
        boolean[] chosen = new boolean[side];
        int centre = Math.max(1, side / 16);
        int count = 0;
        for (int k = -centre; k <= centre && count < lines; k++) {
            chosen[Math.floorMod(k, side)] = true;
            count++;
        }
        while (count < lines) {
            int k;
            if (variableDensity) {
                double u = r.nextDouble();
                int f = (int) Math.floor(Math.pow(u, 2) * side / 2);
                k = r.nextBoolean() ? f : -f;
            } else {
                k = r.nextInt(side);
            }
            int idx = Math.floorMod(k, side);
            if (!chosen[idx]) {
                chosen[idx] = true;
                count++;
            }
            int mirror = Math.floorMod(-k, side);
            if (!chosen[mirror]) {
                chosen[mirror] = true;
                count++;
            }
        }
        boolean[] mask = new boolean[side * side];
        for (int row = 0; row < side; row++) {
            for (int col = 0; col < side; col++) {
                mask[row * side + col] = chosen[col];
            }
        }
        return mask;
    }

    /**
     * Random coded-diffraction phase masks with entries in {@code {1, -1, i, -i}} times
     * {@code sqrt(2)/2} or {@code sqrt(3)} with probabilities 0.8 and 0.2 (Candès, Li &amp;
     * Soltanolkotabi's octanary patterns).
     *
     * @param patterns the number of masks
     * @param n the pixels per mask
     * @param seed the seed
     * @return {@code [pattern][re or im][pixel]}
     */
    public static double[][][] codedMasks(int patterns, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][][] m = new double[patterns][2][n];
        for (int l = 0; l < patterns; l++) {
            for (int i = 0; i < n; i++) {
                int ph = r.nextInt(4);
                double amp = r.nextDouble() < 0.8 ? Math.sqrt(2) / 2 : Math.sqrt(3);
                m[l][0][i] = amp * (ph == 0 ? 1 : ph == 1 ? -1 : 0);
                m[l][1][i] = amp * (ph == 2 ? 1 : ph == 3 ? -1 : 0);
            }
        }
        return m;
    }

    /**
     * A random blur kernel: linear motion (length 3 to {@code size}, any direction) or defocus
     * (a uniform disc), normalised to sum 1.
     *
     * @param size the kernel side (odd)
     * @param motion motion (true) or defocus
     * @param r the random source
     * @return {@code size x size}
     */
    public static double[] blurKernel(int size, boolean motion, SplittableRandom r) {
        double[] k = new double[size * size];
        if (motion) {
            float[] f = DegradedImages.motionKernel(size, 3 + r.nextDouble() * (size - 3), r.nextDouble() * Math.PI);
            for (int i = 0; i < k.length; i++) {
                k[i] = f[i];
            }
            return k;
        }
        double rad = 1 + r.nextDouble() * (size / 2.0 - 1);
        int c = size / 2;
        double s = 0;
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double cover = Math.max(0, Math.min(1, rad + 0.5 - Math.hypot(x - c, y - c)));
                k[y * size + x] = cover;
                s += cover;
            }
        }
        for (int i = 0; i < k.length; i++) {
            k[i] /= s;
        }
        return k;
    }

    /**
     * The RGGB Bayer pattern's channel at a pixel.
     *
     * @param row the row
     * @param col the column
     * @return 0 red, 1 green, 2 blue
     */
    public static int bayerChannel(int row, int col) {
        if (row % 2 == 0) {
            return col % 2 == 0 ? 0 : 1;
        }
        return col % 2 == 0 ? 1 : 2;
    }

    /**
     * The RGGB Bayer mosaic of a colour image, with optional Gaussian noise.
     *
     * @param rgb the image {@code [3][side * side]}
     * @param side the side
     * @param sigma the noise level
     * @param r the random source
     * @return the single-channel mosaic
     */
    public static double[] bayer(double[][] rgb, int side, double sigma, SplittableRandom r) {
        double[] m = new double[side * side];
        for (int row = 0; row < side; row++) {
            for (int col = 0; col < side; col++) {
                int i = row * side + col;
                m[i] = rgb[bayerChannel(row, col)][i] + sigma * r.nextGaussian();
            }
        }
        return m;
    }

    /**
     * A random-pixel inpainting mask.
     *
     * @param n the pixels
     * @param keep the share kept
     * @param seed the seed
     * @return true where observed
     */
    public static boolean[] pixelMask(int n, double keep, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        boolean[] m = new boolean[n];
        for (int i = 0; i < n; i++) {
            m[i] = r.nextDouble() < keep;
        }
        return m;
    }

    /**
     * A box inpainting mask: a {@code box x box} hole at a random position.
     *
     * @param side the image side
     * @param box the hole side
     * @param r the random source
     * @return true where observed
     */
    public static boolean[] boxMask(int side, int box, SplittableRandom r) {
        boolean[] m = new boolean[side * side];
        java.util.Arrays.fill(m, true);
        int y0 = r.nextInt(side - box + 1);
        int x0 = r.nextInt(side - box + 1);
        for (int y = y0; y < y0 + box; y++) {
            for (int x = x0; x < x0 + box; x++) {
                m[y * side + x] = false;
            }
        }
        return m;
    }

    /**
     * Gaussian noise.
     *
     * @param x the clean values
     * @param sigma the standard deviation
     * @param r the random source
     * @return a noisy copy
     */
    public static double[] gaussian(double[] x, double sigma, SplittableRandom r) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] + sigma * r.nextGaussian();
        }
        return y;
    }

    /**
     * Poisson counts at a photon level, rescaled back: {@code Poisson(peak x) / peak} (normal
     * approximation above 30 counts).
     *
     * @param x the non-negative intensities
     * @param peak the photons per unit intensity
     * @param r the random source
     * @return the noisy copy
     */
    public static double[] poisson(double[] x, double peak, SplittableRandom r) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double lam = Math.max(0, x[i]) * peak;
            double k;
            if (lam > 30) {
                k = Math.max(0, Math.round(lam + Math.sqrt(lam) * r.nextGaussian()));
            } else {
                double l = Math.exp(-lam);
                double p = 1;
                int c = -1;
                do {
                    c++;
                    p *= r.nextDouble();
                } while (p > l);
                k = c;
            }
            y[i] = k / peak;
        }
        return y;
    }

    // ---------------------------------------------------------------- 1-D test signals

    /**
     * The Donoho-Johnstone test signals, scaled to unit standard deviation.
     *
     * @param name {@code blocks}, {@code bumps}, {@code heavisine} or {@code doppler}
     * @param n the length
     * @return the signal
     */
    public static double[] testSignal(String name, int n) {
        double[] t = new double[n];
        double[] f = new double[n];
        for (int i = 0; i < n; i++) {
            t[i] = (i + 0.5) / n;
        }
        double[] pos = {0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81};
        switch (name) {
            case "blocks" -> {
                double[] hgt = {4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2};
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < pos.length; j++) {
                        f[i] += hgt[j] * (1 + Math.signum(t[i] - pos[j])) / 2;
                    }
                }
            }
            case "bumps" -> {
                double[] hgt = {4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2};
                double[] wth = {0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005};
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < pos.length; j++) {
                        f[i] += hgt[j] * Math.pow(1 + Math.abs((t[i] - pos[j]) / wth[j]), -4);
                    }
                }
            }
            case "heavisine" -> {
                for (int i = 0; i < n; i++) {
                    f[i] = 4 * Math.sin(4 * Math.PI * t[i]) - Math.signum(t[i] - 0.3) - Math.signum(0.72 - t[i]);
                }
            }
            case "doppler" -> {
                for (int i = 0; i < n; i++) {
                    f[i] = Math.sqrt(t[i] * (1 - t[i])) * Math.sin(2 * Math.PI * 1.05 / (t[i] + 0.05));
                }
            }
            default -> throw new IllegalArgumentException("unknown signal " + name);
        }
        double mean = 0;
        for (double v : f) {
            mean += v / n;
        }
        double var = 0;
        for (double v : f) {
            var += (v - mean) * (v - mean) / n;
        }
        double sd = Math.sqrt(var);
        for (int i = 0; i < n; i++) {
            f[i] = (f[i] - mean) / sd;
        }
        return f;
    }

    // ---------------------------------------------------------------- Gaussian mixtures

    /**
     * A Gaussian-mixture prior: {@code k} isotropic components with means spread in
     * {@code [-scale, scale]^d} and standard deviations {@code 0.15-0.4 scale / 2}.
     *
     * @param d the dimension
     * @param k the components
     * @param scale the spread of the means
     * @param seed the seed
     * @return the mixture
     */
    public static GaussianMixture mixturePrior(int d, int k, double scale, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] w = new double[k];
        double[][] mu = new double[k][d];
        double[][][] cov = new double[k][d][d];
        double sum = 0;
        for (int c = 0; c < k; c++) {
            w[c] = 0.5 + r.nextDouble();
            sum += w[c];
            for (int j = 0; j < d; j++) {
                mu[c][j] = scale * (2 * r.nextDouble() - 1);
            }
            double s = scale * (0.15 + 0.25 * r.nextDouble()) / 2;
            for (int j = 0; j < d; j++) {
                cov[c][j][j] = s * s;
            }
        }
        for (int c = 0; c < k; c++) {
            w[c] /= sum;
        }
        return new GaussianMixture(w, mu, cov);
    }

    /**
     * A Gaussian mixture with full covariances: sampling, log density with gradient, the
     * denoiser {@code E[x0 | x0 + sigma e]} (Tweedie), and the exact posterior under
     * {@code y = A x + n}, {@code n ~ N(0, sigma_y^2 I)}, which is again a Gaussian mixture.
     */
    public static final class GaussianMixture {
        private final double[] weights;
        private final double[][] means;
        private final double[][][] covariances;
        private final double[][][] precisions;
        private final double[][][] choleskies;
        private final double[] logNorms;
        private final double[] isotropic;

        /**
         * Creates the mixture.
         *
         * @param weights the component weights, summing to 1
         * @param means the means {@code [k][d]}
         * @param covariances the covariances {@code [k][d][d]}
         */
        public GaussianMixture(double[] weights, double[][] means, double[][][] covariances) {
            int k = weights.length;
            int d = means[0].length;
            this.weights = weights.clone();
            this.means = means;
            this.covariances = covariances;
            this.precisions = new double[k][][];
            this.choleskies = new double[k][][];
            this.logNorms = new double[k];
            double[] iso = new double[k];
            boolean allIso = true;
            for (int c = 0; c < k; c++) {
                precisions[c] = Matrices.inverse(covariances[c]);
                choleskies[c] = Matrices.cholesky(covariances[c]);
                double logDet = 0;
                for (int j = 0; j < d; j++) {
                    logDet += 2 * Math.log(choleskies[c][j][j]);
                }
                logNorms[c] = Math.log(weights[c]) - 0.5 * logDet - 0.5 * d * Math.log(2 * Math.PI);
                iso[c] = covariances[c][0][0];
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        double expect = i == j ? iso[c] : 0;
                        if (Math.abs(covariances[c][i][j] - expect) > 1e-15) {
                            allIso = false;
                        }
                    }
                }
            }
            this.isotropic = allIso ? iso : null;
        }

        /** @return the dimension */
        public int dimension() {
            return means[0].length;
        }

        /** @return the number of components */
        public int components() {
            return weights.length;
        }

        /** @return the weights (a copy) */
        public double[] weights() {
            return weights.clone();
        }

        /**
         * One component's mean.
         *
         * @param c the component
         * @return its mean (a copy)
         */
        public double[] mean(int c) {
            return means[c].clone();
        }

        /**
         * Draws samples.
         *
         * @param n how many
         * @param r the random source
         * @return {@code [n][d]}
         */
        public double[][] sample(int n, SplittableRandom r) {
            int d = dimension();
            double[][] out = new double[n][d];
            for (int i = 0; i < n; i++) {
                double u = r.nextDouble();
                int c = 0;
                while (c < weights.length - 1 && u > weights[c]) {
                    u -= weights[c];
                    c++;
                }
                double[] z = new double[d];
                for (int j = 0; j < d; j++) {
                    z[j] = r.nextGaussian();
                }
                for (int j = 0; j < d; j++) {
                    double s = means[c][j];
                    for (int q = 0; q <= j; q++) {
                        s += choleskies[c][j][q] * z[q];
                    }
                    out[i][j] = s;
                }
            }
            return out;
        }

        /**
         * The log density and its gradient.
         *
         * @param x the point
         * @param gradient receives the gradient, or {@code null}
         * @return {@code log p(x)}
         */
        public double logDensity(double[] x, double[] gradient) {
            int k = weights.length;
            int d = dimension();
            double[] lp = new double[k];
            double[][] g = new double[k][d];
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < k; c++) {
                double[] diff = new double[d];
                for (int j = 0; j < d; j++) {
                    diff[j] = x[j] - means[c][j];
                }
                double q = 0;
                for (int i = 0; i < d; i++) {
                    double s = 0;
                    for (int j = 0; j < d; j++) {
                        s += precisions[c][i][j] * diff[j];
                    }
                    g[c][i] = -s;
                    q += diff[i] * s;
                }
                lp[c] = logNorms[c] - 0.5 * q;
                max = Math.max(max, lp[c]);
            }
            double sum = 0;
            for (int c = 0; c < k; c++) {
                sum += Math.exp(lp[c] - max);
            }
            if (gradient != null) {
                java.util.Arrays.fill(gradient, 0);
                for (int c = 0; c < k; c++) {
                    double rc = Math.exp(lp[c] - max) / sum;
                    for (int j = 0; j < d; j++) {
                        gradient[j] += rc * g[c][j];
                    }
                }
            }
            return max + Math.log(sum);
        }

        /**
         * The component responsibilities at a point.
         *
         * @param x the point
         * @return the posterior component probabilities
         */
        public double[] responsibilities(double[] x) {
            int k = weights.length;
            int d = dimension();
            double[] lp = new double[k];
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < k; c++) {
                double q = 0;
                for (int i = 0; i < d; i++) {
                    double s = 0;
                    for (int j = 0; j < d; j++) {
                        s += precisions[c][i][j] * (x[j] - means[c][j]);
                    }
                    q += (x[i] - means[c][i]) * s;
                }
                lp[c] = logNorms[c] - 0.5 * q;
                max = Math.max(max, lp[c]);
            }
            double sum = 0;
            for (int c = 0; c < k; c++) {
                lp[c] = Math.exp(lp[c] - max);
                sum += lp[c];
            }
            for (int c = 0; c < k; c++) {
                lp[c] /= sum;
            }
            return lp;
        }

        /**
         * The minimum-mean-square-error denoiser {@code E[x0 | x0 + sigma e = x]} (exact for
         * isotropic components; general covariances are inverted per call).
         *
         * @param x the noisy point
         * @param sigma the noise level
         * @return the posterior mean
         */
        public double[] denoise(double[] x, double sigma) {
            int k = weights.length;
            int d = dimension();
            double s2 = sigma * sigma;
            double[] lp = new double[k];
            double[][] m = new double[k][d];
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < k; c++) {
                if (isotropic != null) {
                    double v = isotropic[c] + s2;
                    double q = 0;
                    for (int j = 0; j < d; j++) {
                        double diff = x[j] - means[c][j];
                        q += diff * diff;
                        m[c][j] = means[c][j] + isotropic[c] / v * diff;
                    }
                    lp[c] = Math.log(weights[c]) - 0.5 * d * Math.log(v) - 0.5 * q / v;
                } else {
                    double[][] cov = new double[d][d];
                    for (int i = 0; i < d; i++) {
                        cov[i] = covariances[c][i].clone();
                        cov[i][i] += s2;
                    }
                    double[][] inv = Matrices.inverse(cov);
                    double[] diff = new double[d];
                    for (int j = 0; j < d; j++) {
                        diff[j] = x[j] - means[c][j];
                    }
                    double[] t = new double[d];
                    double q = 0;
                    for (int i = 0; i < d; i++) {
                        for (int j = 0; j < d; j++) {
                            t[i] += inv[i][j] * diff[j];
                        }
                        q += diff[i] * t[i];
                    }
                    for (int i = 0; i < d; i++) {
                        double s = means[c][i];
                        for (int j = 0; j < d; j++) {
                            s += covariances[c][i][j] * t[j];
                        }
                        m[c][i] = s;
                    }
                    lp[c] = Math.log(weights[c]) - 0.5 * Math.log(Matrices.determinant(cov)) - 0.5 * q;
                }
                max = Math.max(max, lp[c]);
            }
            double sum = 0;
            for (int c = 0; c < k; c++) {
                lp[c] = Math.exp(lp[c] - max);
                sum += lp[c];
            }
            double[] out = new double[d];
            for (int c = 0; c < k; c++) {
                for (int j = 0; j < d; j++) {
                    out[j] += lp[c] / sum * m[c][j];
                }
            }
            return out;
        }

        /**
         * The exact posterior under a linear-Gaussian measurement.
         *
         * @param a the measurement matrix {@code [m][d]}
         * @param sigmaY the measurement noise
         * @param y the measurement
         * @return the posterior mixture
         */
        public GaussianMixture posterior(double[][] a, double sigmaY, double[] y) {
            int k = weights.length;
            int d = dimension();
            int m = a.length;
            double[] lw = new double[k];
            double[][] mu = new double[k][];
            double[][][] cov = new double[k][][];
            double max = Double.NEGATIVE_INFINITY;
            double iv = 1 / (sigmaY * sigmaY);
            for (int c = 0; c < k; c++) {
                double[][] prec = new double[d][d];
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        double s = precisions[c][i][j];
                        for (int q = 0; q < m; q++) {
                            s += a[q][i] * a[q][j] * iv;
                        }
                        prec[i][j] = s;
                    }
                }
                cov[c] = Matrices.inverse(prec);
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < i; j++) {
                        double avg = 0.5 * (cov[c][i][j] + cov[c][j][i]);
                        cov[c][i][j] = avg;
                        cov[c][j][i] = avg;
                    }
                }
                double[] rhs = new double[d];
                for (int i = 0; i < d; i++) {
                    double s = 0;
                    for (int j = 0; j < d; j++) {
                        s += precisions[c][i][j] * means[c][j];
                    }
                    for (int q = 0; q < m; q++) {
                        s += a[q][i] * y[q] * iv;
                    }
                    rhs[i] = s;
                }
                mu[c] = new double[d];
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        mu[c][i] += cov[c][i][j] * rhs[j];
                    }
                }
                // evidence N(y; A mu_c, A S_c A^T + sigma_y^2 I)
                double[][] sy = new double[m][m];
                double[] pred = new double[m];
                for (int p = 0; p < m; p++) {
                    for (int j = 0; j < d; j++) {
                        pred[p] += a[p][j] * means[c][j];
                    }
                    for (int q = 0; q < m; q++) {
                        double s = 0;
                        for (int i = 0; i < d; i++) {
                            for (int j = 0; j < d; j++) {
                                s += a[p][i] * covariances[c][i][j] * a[q][j];
                            }
                        }
                        sy[p][q] = s + (p == q ? sigmaY * sigmaY : 0);
                    }
                }
                double[][] syi = Matrices.inverse(sy);
                double q = 0;
                for (int p = 0; p < m; p++) {
                    for (int r = 0; r < m; r++) {
                        q += (y[p] - pred[p]) * syi[p][r] * (y[r] - pred[r]);
                    }
                }
                lw[c] = Math.log(weights[c]) - 0.5 * Math.log(Matrices.determinant(sy)) - 0.5 * q;
                max = Math.max(max, lw[c]);
            }
            double sum = 0;
            for (int c = 0; c < k; c++) {
                lw[c] = Math.exp(lw[c] - max);
                sum += lw[c];
            }
            for (int c = 0; c < k; c++) {
                lw[c] = Math.max(lw[c] / sum, 1e-300);
            }
            return new GaussianMixture(lw, mu, cov);
        }
    }
}
