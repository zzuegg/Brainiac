package io.github.zzuegg.brainiac.experimental.dpmech;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Correlated noise for private prefix sums, the noise model of DP-FTRL and DP matrix
 * factorisation. The workload is the lower-triangular all-ones matrix {@code A} (prefix
 * sums of {@code n} gradients); a factorisation {@code A = B C} releases {@code C g + z} and
 * reconstructs {@code A g + B z}, so the model sees the per-step noise {@code C⁻¹ z}. The
 * sensitivity is the largest norm of a sum of {@code C}'s columns over the steps a user can
 * touch; the error is the {@code B z} term.
 *
 * <ul>
 *   <li>{@link Kind#INDEPENDENT}: {@code C = I}, plain DP-SGD noise without amplification.</li>
 *   <li>{@link Kind#TREE}: the binary-tree mechanism, every step in about {@code log₂ n}
 *       dyadic nodes, each prefix the sum of its decomposition's nodes (Kairouz et al.,
 *       ICML 2021, <a href="https://arxiv.org/abs/2103.00039">arXiv:2103.00039</a>).</li>
 *   <li>{@link Kind#BANDED_MF}: a lower-triangular banded Toeplitz {@code C} with {@code b}
 *       coefficients, optimised for {@code sens² · ‖A C⁻¹‖²_F / n} by Nelder-Mead from the
 *       truncated square-root factorisation (Choquette-Choo et al., NeurIPS 2023,
 *       <a href="https://arxiv.org/abs/2306.08153">arXiv:2306.08153</a>; this restricts their
 *       general banded matrices to Toeplitz ones). Noise needs {@code b - 1} buffers.</li>
 *   <li>{@link Kind#BLT}: the buffered linear Toeplitz mechanism with {@code c_0 = 1},
 *       {@code c_k = Σ_j ω_j θ_j^{k-1}}, fitted the same way; {@code C⁻¹ z} streams from
 *       {@code d} buffers {@code S_j ← θ_j S_j + w}, memory independent of {@code n}
 *       (McMahan et al. 2024, <a href="https://arxiv.org/abs/2408.08868">arXiv:2408.08868</a>).</li>
 * </ul>
 *
 * <p>The sensitivity for {@code k} participations at separation {@code s} uses the columns
 * {@code 0, s, 2s, ...}, the worst case for the non-negative decreasing coefficients
 * produced here.
 */
public final class CorrelatedNoise {

    /** The mechanism. */
    public enum Kind {
        /** independent noise */
        INDEPENDENT,
        /** binary tree aggregation */
        TREE,
        /** banded Toeplitz matrix factorisation */
        BANDED_MF,
        /** buffered linear Toeplitz */
        BLT
    }

    private final Kind kind;
    private final int steps;
    private final double[] coefficients;
    private final double[] theta;
    private final double[] omega;

    private CorrelatedNoise(Kind kind, int steps, double[] coefficients, double[] theta, double[] omega) {
        this.kind = kind;
        this.steps = steps;
        this.coefficients = coefficients;
        this.theta = theta;
        this.omega = omega;
    }

    /**
     * Independent noise.
     *
     * @param steps {@code n}
     * @return the mechanism
     */
    public static CorrelatedNoise independent(int steps) {
        return new CorrelatedNoise(Kind.INDEPENDENT, steps, new double[] {1}, null, null);
    }

    /**
     * The binary tree.
     *
     * @param steps {@code n}
     * @return the mechanism
     */
    public static CorrelatedNoise tree(int steps) {
        return new CorrelatedNoise(Kind.TREE, steps, null, null, null);
    }

    /**
     * An optimised banded Toeplitz factorisation.
     *
     * @param steps {@code n}
     * @param bands {@code b}
     * @param participations {@code k} for the objective's sensitivity
     * @param separation the steps between a user's participations
     * @param iterations Nelder-Mead iterations
     * @return the mechanism
     */
    public static CorrelatedNoise banded(int steps, int bands, int participations, int separation, int iterations) {
        double[] start = new double[bands - 1];
        double r = 1;
        for (int k = 1; k < bands; k++) {
            r *= (2.0 * k - 1) / (2.0 * k);
            start[k - 1] = r;
        }
        ToDoubleFunction<double[]> loss = x -> {
            double[] c = new double[bands];
            c[0] = 1;
            for (int k = 1; k < bands; k++) {
                c[k] = Math.max(0, x[k - 1]);
            }
            CorrelatedNoise m = new CorrelatedNoise(Kind.BANDED_MF, steps, c, null, null);
            return m.squaredError(participations, separation);
        };
        double[] best = nelderMead(loss, start, 0.1, iterations);
        double[] c = new double[bands];
        c[0] = 1;
        for (int k = 1; k < bands; k++) {
            c[k] = Math.max(0, best[k - 1]);
        }
        return new CorrelatedNoise(Kind.BANDED_MF, steps, c, null, null);
    }

    /**
     * A fitted buffered linear Toeplitz mechanism.
     *
     * @param steps {@code n}
     * @param buffers {@code d}
     * @param participations {@code k}
     * @param separation the steps between participations
     * @param iterations Nelder-Mead iterations
     * @return the mechanism
     */
    public static CorrelatedNoise blt(int steps, int buffers, int participations, int separation, int iterations) {
        double[] start = new double[2 * buffers];
        for (int j = 0; j < buffers; j++) {
            double t = 1 - Math.pow(0.1, j + 1);
            start[j] = Math.log(t / (1 - t));
            start[buffers + j] = Math.log(0.5 / buffers);
        }
        ToDoubleFunction<double[]> loss = x -> fromParameters(steps, buffers, x).squaredError(participations,
                separation);
        double[] best = nelderMead(loss, start, 0.5, iterations);
        return fromParameters(steps, buffers, best);
    }

    private static CorrelatedNoise fromParameters(int steps, int buffers, double[] x) {
        double[] th = new double[buffers];
        double[] om = new double[buffers];
        for (int j = 0; j < buffers; j++) {
            th[j] = 1 / (1 + Math.exp(-x[j]));
            om[j] = Math.exp(Math.min(x[buffers + j], 5));
        }
        return new CorrelatedNoise(Kind.BLT, steps, null, th, om);
    }

    /** @return the mechanism */
    public Kind kind() {
        return kind;
    }

    /** @return the number of steps {@code n} */
    public int steps() {
        return steps;
    }

    /** @return noise vectors kept between steps (the per-coordinate state) */
    public int state() {
        return switch (kind) {
            case INDEPENDENT -> 0;
            case TREE -> 32 - Integer.numberOfLeadingZeros(steps);
            case BANDED_MF -> coefficients.length - 1;
            case BLT -> theta.length;
        };
    }

    /**
     * The coefficients {@code c_0 ... c_{n-1}} of a Toeplitz {@code C}.
     *
     * @return the coefficients (not for the tree)
     */
    public double[] toeplitz() {
        double[] c = new double[steps];
        if (kind == Kind.BLT) {
            c[0] = 1;
            double[] pow = new double[theta.length];
            java.util.Arrays.fill(pow, 1);
            for (int k = 1; k < steps; k++) {
                for (int j = 0; j < theta.length; j++) {
                    c[k] += omega[j] * pow[j];
                    pow[j] *= theta[j];
                }
            }
        } else if (kind != Kind.TREE) {
            System.arraycopy(coefficients, 0, c, 0, Math.min(steps, coefficients.length));
        }
        return c;
    }

    /**
     * The sensitivity of {@code C g} for a user in {@code k} steps {@code s} apart, with
     * unit-norm gradients.
     *
     * @param participations {@code k}
     * @param separation {@code s}
     * @return {@code max ‖Σ C[:, t_i]‖}
     */
    public double sensitivity(int participations, int separation) {
        int k = Math.max(1, Math.min(participations, (steps - 1) / Math.max(1, separation) + 1));
        if (kind == Kind.TREE) {
            Map<Long, Integer> nodes = new HashMap<>();
            for (int i = 0; i < k; i++) {
                int s = i * separation;
                for (int l = 0; (1L << l) <= steps; l++) {
                    long block = s >> l;
                    if ((block + 1) << l <= steps) {
                        nodes.merge(((long) l << 40) | block, 1, Integer::sum);
                    }
                }
            }
            double sum = 0;
            for (int c : nodes.values()) {
                sum += c * (double) c;
            }
            return Math.sqrt(sum);
        }
        double[] c = toeplitz();
        double sum = 0;
        for (int t = 0; t < steps; t++) {
            double v = 0;
            for (int i = 0; i < k; i++) {
                int lag = t - i * separation;
                if (lag >= 0) {
                    v += c[lag];
                }
            }
            sum += v * v;
        }
        return Math.sqrt(sum);
    }

    /**
     * The mean variance of the prefix-sum error per unit noise, {@code ‖B‖²_F / n}.
     *
     * @return the mean prefix error variance
     */
    public double meanPrefixVariance() {
        if (kind == Kind.TREE) {
            double s = 0;
            for (int t = 1; t <= steps; t++) {
                s += Integer.bitCount(t);
            }
            return s / steps;
        }
        double[] d = inverse();
        double acc = 0;
        double s = 0;
        for (int m = 0; m < steps; m++) {
            acc += d[m];
            s += (steps - m) * acc * acc;
        }
        return s / steps;
    }

    /**
     * The root-mean-square prefix-sum error when the noise is calibrated to the
     * sensitivity, per unit noise multiplier.
     *
     * @param participations {@code k}
     * @param separation the steps between participations
     * @return {@code sens · sqrt(‖B‖²_F / n)}
     */
    public double prefixRmse(int participations, int separation) {
        return Math.sqrt(squaredError(participations, separation));
    }

    private double squaredError(int participations, int separation) {
        double sens = sensitivity(participations, separation);
        return sens * sens * meanPrefixVariance();
    }

    /** The coefficients of {@code C⁻¹} (first column), by forward substitution or buffers. */
    private double[] inverse() {
        double[] d = new double[steps];
        if (kind == Kind.BLT) {
            double[] s = new double[theta.length];
            for (int t = 0; t < steps; t++) {
                double past = 0;
                for (int j = 0; j < theta.length; j++) {
                    past += omega[j] * s[j];
                }
                d[t] = (t == 0 ? 1 : 0) - past;
                for (int j = 0; j < theta.length; j++) {
                    s[j] = theta[j] * s[j] + d[t];
                }
            }
            return d;
        }
        double[] c = coefficients;
        for (int m = 0; m < steps; m++) {
            double v = m == 0 ? 1 : 0;
            for (int j = 1; j < Math.min(c.length, m + 1); j++) {
                v -= c[j] * d[m - j];
            }
            d[m] = v / c[0];
        }
        return d;
    }

    /**
     * A stream of per-step noise {@code (C⁻¹ z)_t} for a vector of {@code dim} coordinates,
     * with standard normal {@code z}.
     *
     * @param dim the coordinates
     * @param random the noise source
     * @return the stream
     */
    public Stream stream(int dim, SplittableRandom random) {
        return new Stream(this, dim, random);
    }

    /** Generates the per-step noise of a mechanism with its minimal state. */
    public static final class Stream {

        private final CorrelatedNoise m;
        private final int dim;
        private final SplittableRandom random;
        private final double[][] buffers;
        private int t;

        private Stream(CorrelatedNoise m, int dim, SplittableRandom random) {
            this.m = Objects.requireNonNull(m);
            this.dim = dim;
            this.random = random;
            int b = switch (m.kind) {
                case INDEPENDENT -> 0;
                case TREE -> 40;
                case BANDED_MF -> m.coefficients.length - 1;
                case BLT -> m.theta.length;
            };
            this.buffers = new double[b][dim];
        }

        /** @return the next step's noise */
        public double[] next() {
            double[] out = new double[dim];
            switch (m.kind) {
                case INDEPENDENT -> {
                    for (int i = 0; i < dim; i++) {
                        out[i] = random.nextGaussian();
                    }
                }
                case TREE -> {
                    int time = t + 1;
                    int j = Integer.numberOfTrailingZeros(time);
                    for (int i = 0; i < dim; i++) {
                        double z = random.nextGaussian();
                        double cleared = 0;
                        for (int l = 0; l < j; l++) {
                            cleared += buffers[l][i];
                            buffers[l][i] = 0;
                        }
                        buffers[j][i] = z;
                        out[i] = z - cleared;
                    }
                }
                case BANDED_MF -> {
                    double[] c = m.coefficients;
                    for (int i = 0; i < dim; i++) {
                        double v = random.nextGaussian();
                        for (int k = 1; k < c.length; k++) {
                            v -= c[k] * buffers[k - 1][i];
                        }
                        v /= c[0];
                        for (int k = c.length - 2; k > 0; k--) {
                            buffers[k][i] = buffers[k - 1][i];
                        }
                        if (c.length > 1) {
                            buffers[0][i] = v;
                        }
                        out[i] = v;
                    }
                }
                case BLT -> {
                    for (int i = 0; i < dim; i++) {
                        double past = 0;
                        for (int j = 0; j < m.theta.length; j++) {
                            past += m.omega[j] * buffers[j][i];
                        }
                        double v = random.nextGaussian() - past;
                        for (int j = 0; j < m.theta.length; j++) {
                            buffers[j][i] = m.theta[j] * buffers[j][i] + v;
                        }
                        out[i] = v;
                    }
                }
            }
            t++;
            return out;
        }
    }

    /**
     * A small Nelder-Mead minimiser.
     *
     * @param f the objective
     * @param start the starting point
     * @param step the initial simplex size
     * @param iterations iterations
     * @return the best point found
     */
    static double[] nelderMead(ToDoubleFunction<double[]> f, double[] start, double step, int iterations) {
        int d = start.length;
        double[][] p = new double[d + 1][];
        double[] v = new double[d + 1];
        for (int i = 0; i <= d; i++) {
            p[i] = start.clone();
            if (i > 0) {
                p[i][i - 1] += step;
            }
            v[i] = f.applyAsDouble(p[i]);
        }
        for (int it = 0; it < iterations; it++) {
            Integer[] order = new Integer[d + 1];
            for (int i = 0; i <= d; i++) {
                order[i] = i;
            }
            java.util.Arrays.sort(order, (a, b) -> Double.compare(v[a], v[b]));
            double[][] sp = new double[d + 1][];
            double[] sv = new double[d + 1];
            for (int i = 0; i <= d; i++) {
                sp[i] = p[order[i]];
                sv[i] = v[order[i]];
            }
            System.arraycopy(sp, 0, p, 0, d + 1);
            System.arraycopy(sv, 0, v, 0, d + 1);
            double[] centroid = new double[d];
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    centroid[j] += p[i][j] / d;
                }
            }
            double[] reflected = blend(centroid, p[d], -1);
            double fr = f.applyAsDouble(reflected);
            if (fr < v[0]) {
                double[] expanded = blend(centroid, p[d], -2);
                double fe = f.applyAsDouble(expanded);
                p[d] = fe < fr ? expanded : reflected;
                v[d] = Math.min(fe, fr);
            } else if (fr < v[d - 1]) {
                p[d] = reflected;
                v[d] = fr;
            } else {
                double[] contracted = blend(centroid, p[d], 0.5);
                double fc = f.applyAsDouble(contracted);
                if (fc < v[d]) {
                    p[d] = contracted;
                    v[d] = fc;
                } else {
                    for (int i = 1; i <= d; i++) {
                        p[i] = blend(p[0], p[i], 0.5);
                        v[i] = f.applyAsDouble(p[i]);
                    }
                }
            }
        }
        int best = 0;
        for (int i = 1; i <= d; i++) {
            best = v[i] < v[best] ? i : best;
        }
        return p[best];
    }

    /** {@code c + t (x - c)}. */
    private static double[] blend(double[] c, double[] x, double t) {
        double[] out = new double[c.length];
        for (int j = 0; j < c.length; j++) {
            out[j] = c[j] + t * (x[j] - c[j]);
        }
        return out;
    }
}
