package io.github.zzuegg.brainiac.experimental.imaging;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Discrete wavelet transforms by lifting and wavelet-shrinkage denoising.
 *
 * <p>The transforms ({@link #forward1d}, {@link #forward2d} and their inverses) use periodic
 * boundaries and the lifting steps of Daubechies &amp; Sweldens (1998) for {@link Wavelet#HAAR},
 * {@link Wavelet#DB4} (Daubechies' four-tap orthogonal wavelet) and {@link Wavelet#CDF97} (the
 * biorthogonal 9/7 of JPEG 2000, nearly orthogonal with this scaling). The 2-D transform is
 * separable, in the Mallat layout (the coarse band in the top-left corner).
 *
 * <p>Denoising ({@link #denoise2d}, {@link #denoise1d}) thresholds the detail coefficients with
 * <ul>
 *   <li>{@link Rule#VISU_SHRINK}: the universal threshold {@code sigma sqrt(2 ln n)} (Donoho
 *       &amp; Johnstone, Biometrika 1994), which oversmooths;</li>
 *   <li>{@link Rule#SURE_SHRINK}: per subband, the threshold minimising Stein's unbiased risk
 *       estimate, falling back to the universal one in sparse subbands (the hybrid rule of
 *       Donoho &amp; Johnstone, JASA 1995);</li>
 *   <li>{@link Rule#BAYES_SHRINK}: per subband, {@code sigma^2 / sigma_x} from a generalised
 *       Gaussian prior (Chang, Yu &amp; Vetterli, IEEE TIP 2000);</li>
 * </ul>
 * soft or hard, with the noise level estimated by the median absolute deviation of the finest
 * diagonal subband over 0.6745, and optional cycle spinning ({@link #cycleSpin}, Coifman &amp;
 * Donoho 1995): averaging the estimate over circular shifts to remove the transform's
 * shift-variance artefacts.
 */
public final class Wavelets {

    /** The wavelets. */
    public enum Wavelet {
        /** Haar. */
        HAAR,
        /** Daubechies' orthogonal 4-tap wavelet (two vanishing moments). */
        DB4,
        /** The Cohen-Daubechies-Feauveau 9/7 biorthogonal wavelet. */
        CDF97
    }

    /** Threshold rules. */
    public enum Rule {
        /** The universal threshold. */
        VISU_SHRINK,
        /** Per-subband SURE (hybrid). */
        SURE_SHRINK,
        /** Per-subband BayesShrink. */
        BAYES_SHRINK
    }

    private static final double S3 = Math.sqrt(3);
    private static final double A97 = -1.586134342059924;
    private static final double B97 = -0.052980118572961;
    private static final double G97 = 0.882911075530934;
    private static final double D97 = 0.443506852043971;
    private static final double Z97 = 1.149604398860241;

    private Wavelets() {
    }

    // ---------------------------------------------------------------- 1-D lifting

    private static void liftForward(double[] x, int off, int stride, int n, Wavelet w) {
        int h = n / 2;
        double[] s = new double[h];
        double[] d = new double[h];
        for (int i = 0; i < h; i++) {
            s[i] = x[off + 2 * i * stride];
            d[i] = x[off + (2 * i + 1) * stride];
        }
        switch (w) {
            case HAAR -> {
                for (int i = 0; i < h; i++) {
                    d[i] -= s[i];
                    s[i] += d[i] / 2;
                    s[i] *= Math.sqrt(2);
                    d[i] /= Math.sqrt(2);
                }
            }
            case DB4 -> {
                for (int i = 0; i < h; i++) {
                    s[i] += S3 * d[i];
                }
                double[] d1 = new double[h];
                for (int i = 0; i < h; i++) {
                    d1[i] = d[i] - S3 / 4 * s[i] - (S3 - 2) / 4 * s[Math.floorMod(i - 1, h)];
                }
                for (int i = 0; i < h; i++) {
                    s[i] -= d1[(i + 1) % h];
                }
                for (int i = 0; i < h; i++) {
                    s[i] *= (S3 - 1) / Math.sqrt(2);
                    d[i] = d1[i] * (S3 + 1) / Math.sqrt(2);
                }
            }
            case CDF97 -> {
                for (int i = 0; i < h; i++) {
                    d[i] += A97 * (s[i] + s[(i + 1) % h]);
                }
                for (int i = 0; i < h; i++) {
                    s[i] += B97 * (d[i] + d[Math.floorMod(i - 1, h)]);
                }
                for (int i = 0; i < h; i++) {
                    d[i] += G97 * (s[i] + s[(i + 1) % h]);
                }
                for (int i = 0; i < h; i++) {
                    s[i] += D97 * (d[i] + d[Math.floorMod(i - 1, h)]);
                }
                for (int i = 0; i < h; i++) {
                    s[i] *= Z97;
                    d[i] /= Z97;
                }
            }
            default -> throw new IllegalStateException();
        }
        for (int i = 0; i < h; i++) {
            x[off + i * stride] = s[i];
            x[off + (h + i) * stride] = d[i];
        }
    }

    private static void liftInverse(double[] x, int off, int stride, int n, Wavelet w) {
        int h = n / 2;
        double[] s = new double[h];
        double[] d = new double[h];
        for (int i = 0; i < h; i++) {
            s[i] = x[off + i * stride];
            d[i] = x[off + (h + i) * stride];
        }
        switch (w) {
            case HAAR -> {
                for (int i = 0; i < h; i++) {
                    s[i] /= Math.sqrt(2);
                    d[i] *= Math.sqrt(2);
                    s[i] -= d[i] / 2;
                    d[i] += s[i];
                }
            }
            case DB4 -> {
                double[] d1 = new double[h];
                for (int i = 0; i < h; i++) {
                    s[i] /= (S3 - 1) / Math.sqrt(2);
                    d1[i] = d[i] / ((S3 + 1) / Math.sqrt(2));
                }
                for (int i = 0; i < h; i++) {
                    s[i] += d1[(i + 1) % h];
                }
                for (int i = 0; i < h; i++) {
                    d[i] = d1[i] + S3 / 4 * s[i] + (S3 - 2) / 4 * s[Math.floorMod(i - 1, h)];
                }
                for (int i = 0; i < h; i++) {
                    s[i] -= S3 * d[i];
                }
            }
            case CDF97 -> {
                for (int i = 0; i < h; i++) {
                    s[i] /= Z97;
                    d[i] *= Z97;
                }
                for (int i = 0; i < h; i++) {
                    s[i] -= D97 * (d[i] + d[Math.floorMod(i - 1, h)]);
                }
                for (int i = 0; i < h; i++) {
                    d[i] -= G97 * (s[i] + s[(i + 1) % h]);
                }
                for (int i = 0; i < h; i++) {
                    s[i] -= B97 * (d[i] + d[Math.floorMod(i - 1, h)]);
                }
                for (int i = 0; i < h; i++) {
                    d[i] -= A97 * (s[i] + s[(i + 1) % h]);
                }
            }
            default -> throw new IllegalStateException();
        }
        for (int i = 0; i < h; i++) {
            x[off + 2 * i * stride] = s[i];
            x[off + (2 * i + 1) * stride] = d[i];
        }
    }

    /**
     * The 1-D transform: coarse coefficients first, then details from coarse to fine.
     *
     * @param x the signal, length divisible by {@code 2^levels}
     * @param w the wavelet
     * @param levels the levels
     * @return the coefficients
     */
    public static double[] forward1d(double[] x, Wavelet w, int levels) {
        double[] c = x.clone();
        int n = c.length;
        for (int l = 0; l < levels; l++) {
            liftForward(c, 0, 1, n, w);
            n /= 2;
        }
        return c;
    }

    /**
     * Inverts {@link #forward1d}.
     *
     * @param c the coefficients
     * @param w the wavelet
     * @param levels the levels
     * @return the signal
     */
    public static double[] inverse1d(double[] c, Wavelet w, int levels) {
        double[] x = c.clone();
        int n = x.length >> (levels - 1);
        for (int l = 0; l < levels; l++) {
            liftInverse(x, 0, 1, n, w);
            n *= 2;
        }
        return x;
    }

    /**
     * The separable 2-D transform in the Mallat layout.
     *
     * @param x the image, side divisible by {@code 2^levels}
     * @param side its side
     * @param w the wavelet
     * @param levels the levels
     * @return the coefficients
     */
    public static double[] forward2d(double[] x, int side, Wavelet w, int levels) {
        double[] c = x.clone();
        int n = side;
        for (int l = 0; l < levels; l++) {
            for (int r = 0; r < n; r++) {
                liftForward(c, r * side, 1, n, w);
            }
            for (int col = 0; col < n; col++) {
                liftForward(c, col, side, n, w);
            }
            n /= 2;
        }
        return c;
    }

    /**
     * Inverts {@link #forward2d}.
     *
     * @param c the coefficients
     * @param side the side
     * @param w the wavelet
     * @param levels the levels
     * @return the image
     */
    public static double[] inverse2d(double[] c, int side, Wavelet w, int levels) {
        double[] x = c.clone();
        int n = side >> (levels - 1);
        for (int l = 0; l < levels; l++) {
            for (int col = 0; col < n; col++) {
                liftInverse(x, col, side, n, w);
            }
            for (int r = 0; r < n; r++) {
                liftInverse(x, r * side, 1, n, w);
            }
            n *= 2;
        }
        return x;
    }

    /**
     * The detail subbands of a 2-D transform, finest first: per level the HL, LH and HH bands.
     *
     * @param side the side
     * @param levels the levels
     * @return the flat indices of each subband
     */
    public static List<int[]> subbands2d(int side, int levels) {
        List<int[]> out = new ArrayList<>();
        int n = side;
        for (int l = 0; l < levels; l++) {
            int h = n / 2;
            int[][] origins = {{0, h}, {h, 0}, {h, h}};
            for (int[] o : origins) {
                int[] idx = new int[h * h];
                int k = 0;
                for (int r = 0; r < h; r++) {
                    for (int col = 0; col < h; col++) {
                        idx[k++] = (o[0] + r) * side + o[1] + col;
                    }
                }
                out.add(idx);
            }
            n = h;
        }
        return out;
    }

    /**
     * The detail subbands of a 1-D transform, finest first.
     *
     * @param n the length
     * @param levels the levels
     * @return the indices of each subband
     */
    public static List<int[]> subbands1d(int n, int levels) {
        List<int[]> out = new ArrayList<>();
        int len = n;
        for (int l = 0; l < levels; l++) {
            int h = len / 2;
            int[] idx = new int[h];
            for (int i = 0; i < h; i++) {
                idx[i] = h + i;
            }
            out.add(idx);
            len = h;
        }
        return out;
    }

    /**
     * The robust noise estimate {@code median(|d|) / 0.6745} of the finest subband.
     *
     * @param coefficients the coefficients
     * @param finest the finest diagonal (or 1-D detail) subband
     * @return the estimate
     */
    public static double noiseSigma(double[] coefficients, int[] finest) {
        double[] a = new double[finest.length];
        for (int i = 0; i < a.length; i++) {
            a[i] = Math.abs(coefficients[finest[i]]);
        }
        Arrays.sort(a);
        return a[a.length / 2] / 0.6745;
    }

    /**
     * SURE for soft thresholding unit-variance data at {@code t}.
     *
     * @param x the noisy coefficients divided by sigma
     * @param t the threshold
     * @return the risk estimate
     */
    public static double sure(double[] x, double t) {
        double s = x.length;
        for (double v : x) {
            double a = Math.abs(v);
            if (a <= t) {
                s -= 2;
                s += v * v;
            } else {
                s += t * t;
            }
        }
        return s;
    }

    /**
     * The threshold of a rule on one subband.
     *
     * @param rule the rule
     * @param band the subband's coefficients
     * @param sigma the noise level
     * @param total the signal length (for the universal threshold)
     * @return the threshold
     */
    public static double threshold(Rule rule, double[] band, double sigma, int total) {
        double universal = sigma * Math.sqrt(2 * Math.log(total));
        switch (rule) {
            case VISU_SHRINK:
                return universal;
            case BAYES_SHRINK: {
                double var = 0;
                for (double v : band) {
                    var += v * v;
                }
                var /= band.length;
                double sx = Math.sqrt(Math.max(var - sigma * sigma, 0));
                return sx == 0 ? maxAbs(band) : Math.min(sigma * sigma / sx, maxAbs(band));
            }
            case SURE_SHRINK: {
                int d = band.length;
                double[] x = new double[d];
                double energy = 0;
                for (int i = 0; i < d; i++) {
                    x[i] = band[i] / sigma;
                    energy += x[i] * x[i] - 1;
                }
                double sparsity = Math.pow(Math.log(d) / Math.log(2), 1.5) / Math.sqrt(d);
                double tu = Math.sqrt(2 * Math.log(d));
                if (energy / d <= sparsity) {
                    return sigma * tu;
                }
                double[] a = new double[d];
                for (int i = 0; i < d; i++) {
                    a[i] = Math.abs(x[i]);
                }
                Arrays.sort(a);
                double best = Double.MAX_VALUE;
                double bestT = 0;
                double cumSq = 0;
                for (int k = 0; k <= d; k++) {
                    double t = k == 0 ? 0 : a[k - 1];
                    if (t > tu) {
                        break;
                    }
                    if (k > 0) {
                        cumSq += a[k - 1] * a[k - 1];
                    }
                    double risk = d - 2.0 * k + cumSq + (d - k) * t * t;
                    if (risk < best) {
                        best = risk;
                        bestT = t;
                    }
                }
                return sigma * bestT;
            }
            default:
                throw new IllegalStateException();
        }
    }

    private static double maxAbs(double[] v) {
        double m = 0;
        for (double x : v) {
            m = Math.max(m, Math.abs(x));
        }
        return m;
    }

    private static double shrink(double v, double t, boolean soft) {
        if (Math.abs(v) <= t) {
            return 0;
        }
        return soft ? v - Math.signum(v) * t : v;
    }

    private static void thresholdBands(double[] c, List<int[]> bands, Rule rule, boolean soft, double sigma,
            double scale, int total) {
        for (int[] band : bands) {
            double[] b = new double[band.length];
            for (int i = 0; i < band.length; i++) {
                b[i] = c[band[i]];
            }
            double t = scale * threshold(rule, b, sigma, total);
            for (int idx : band) {
                c[idx] = shrink(c[idx], t, soft);
            }
        }
    }

    /**
     * Denoises an image by wavelet shrinkage.
     *
     * @param x the noisy image
     * @param side its side
     * @param w the wavelet
     * @param levels the levels
     * @param rule the threshold rule
     * @param soft soft (true) or hard thresholding
     * @param sigma the noise level, or {@code NaN} to estimate it
     * @param scale a multiplier on the thresholds (1 for the rule as published)
     * @return the estimate
     */
    public static double[] denoise2d(double[] x, int side, Wavelet w, int levels, Rule rule, boolean soft,
            double sigma, double scale) {
        double[] c = forward2d(x, side, w, levels);
        List<int[]> bands = subbands2d(side, levels);
        double s = Double.isNaN(sigma) ? noiseSigma(c, bands.get(2)) : sigma;
        thresholdBands(c, bands, rule, soft, s, scale, x.length);
        return inverse2d(c, side, w, levels);
    }

    /**
     * Denoises a 1-D signal by wavelet shrinkage.
     *
     * @param x the noisy signal
     * @param w the wavelet
     * @param levels the levels
     * @param rule the threshold rule
     * @param soft soft or hard
     * @param sigma the noise level, or {@code NaN} to estimate it
     * @param scale a multiplier on the thresholds
     * @return the estimate
     */
    public static double[] denoise1d(double[] x, Wavelet w, int levels, Rule rule, boolean soft, double sigma,
            double scale) {
        double[] c = forward1d(x, w, levels);
        List<int[]> bands = subbands1d(x.length, levels);
        double s = Double.isNaN(sigma) ? noiseSigma(c, bands.get(0)) : sigma;
        thresholdBands(c, bands, rule, soft, s, scale, x.length);
        return inverse1d(c, w, levels);
    }

    /**
     * The best single threshold per subband chosen with the clean signal (the oracle risk SURE is
     * compared to).
     *
     * @param x the noisy signal
     * @param clean the clean signal
     * @param w the wavelet
     * @param levels the levels
     * @return the oracle estimate
     */
    public static double[] oracle1d(double[] x, double[] clean, Wavelet w, int levels) {
        double[] c = forward1d(x, w, levels);
        double[] t = forward1d(clean, w, levels);
        for (int[] band : subbands1d(x.length, levels)) {
            double best = Double.MAX_VALUE;
            double bestT = 0;
            double hi = 0;
            for (int i : band) {
                hi = Math.max(hi, Math.abs(c[i]));
            }
            for (int k = 0; k <= 200; k++) {
                double th = hi * k / 200;
                double err = 0;
                for (int i : band) {
                    double e = shrink(c[i], th, true) - t[i];
                    err += e * e;
                }
                if (err < best) {
                    best = err;
                    bestT = th;
                }
            }
            for (int i : band) {
                c[i] = shrink(c[i], bestT, true);
            }
        }
        return inverse1d(c, w, levels);
    }

    /**
     * Cycle spinning: averages a shift-variant denoiser over circular shifts
     * {@code (dy, dx) in [0, shifts)^2}.
     *
     * @param x the noisy image
     * @param side its side
     * @param shifts the shifts per axis
     * @param denoiser the denoiser
     * @return the averaged estimate
     */
    public static double[] cycleSpin(double[] x, int side, int shifts, java.util.function.UnaryOperator<double[]> denoiser) {
        double[] acc = new double[x.length];
        for (int dy = 0; dy < shifts; dy++) {
            for (int dx = 0; dx < shifts; dx++) {
                double[] s = new double[x.length];
                for (int r = 0; r < side; r++) {
                    for (int c = 0; c < side; c++) {
                        s[r * side + c] = x[((r + dy) % side) * side + (c + dx) % side];
                    }
                }
                double[] d = denoiser.apply(s);
                for (int r = 0; r < side; r++) {
                    for (int c = 0; c < side; c++) {
                        acc[((r + dy) % side) * side + (c + dx) % side] += d[r * side + c];
                    }
                }
            }
        }
        for (int i = 0; i < acc.length; i++) {
            acc[i] /= shifts * shifts;
        }
        return acc;
    }
}
