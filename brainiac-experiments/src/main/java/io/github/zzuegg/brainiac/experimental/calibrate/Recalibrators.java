package io.github.zzuegg.brainiac.experimental.calibrate;

import io.github.zzuegg.brainiac.experimental.train.Calibration;

import java.util.Arrays;

/**
 * The classic post-hoc recalibration maps, all fitted on validation logits with one
 * {@link #fit} call and applied as a {@link Recalibrator}:
 *
 * <ul>
 *   <li>{@link Method#TEMPERATURE}: {@code softmax(z / T)} via {@link Calibration#fitTemperature}
 *       (Guo et al. 2017, <a href="https://arxiv.org/abs/1706.04599">arXiv:1706.04599</a>), the
 *       baseline every other map is measured against;</li>
 *   <li>{@link Method#HISTOGRAM}: one-vs-rest equal-mass histogram binning of {@code p_k},
 *       each bin's score replaced by its empirical frequency (Zadrozny &amp; Elkan 2001);</li>
 *   <li>{@link Method#ISOTONIC}: one-vs-rest isotonic regression by pool-adjacent-violators,
 *       linearly interpolated, then renormalised (Zadrozny &amp; Elkan 2002);</li>
 *   <li>{@link Method#PLATT}: one-vs-rest {@code sigmoid(a z_k + c)} (Platt 1999);</li>
 *   <li>{@link Method#VECTOR}: a per-class temperature and bias on the logits;</li>
 *   <li>{@link Method#MATRIX}: a full affine map {@code W z + b} on the logits (lightly ridged);</li>
 *   <li>{@link Method#BETA}: one-vs-rest beta calibration {@code sigmoid(a ln p - b ln(1-p) + c)},
 *       a three-parameter family containing the identity (Kull et al., AISTATS 2017);</li>
 *   <li>{@link Method#DIRICHLET}: matrix scaling on log-probabilities with ODIR regularisation,
 *       {@code λ/(k(k-1)) Σ_{i≠j} W_ij² + μ/k Σ b²} (Kull et al., NeurIPS 2019,
 *       <a href="https://arxiv.org/abs/1910.12656">arXiv:1910.12656</a>), the multiclass
 *       generalisation of beta calibration.</li>
 * </ul>
 *
 * One-vs-rest maps renormalise their per-class scores (with a floor of {@code 1e-6}, so
 * binning and isotonic maps that put zero mass on the true class show up as a large but finite
 * NLL).
 */
public final class Recalibrators {

    /** A recalibration map. */
    public enum Method {
        /** the uncalibrated softmax */
        NONE,
        /** temperature scaling */
        TEMPERATURE,
        /** histogram binning */
        HISTOGRAM,
        /** isotonic regression */
        ISOTONIC,
        /** Platt scaling */
        PLATT,
        /** vector scaling */
        VECTOR,
        /** matrix scaling */
        MATRIX,
        /** beta calibration */
        BETA,
        /** Dirichlet calibration with ODIR */
        DIRICHLET
    }

    /** Bins of {@link Method#HISTOGRAM}. */
    public static final int HISTOGRAM_BINS = 15;
    /** Floor of renormalised one-vs-rest scores. */
    public static final double FLOOR = 1e-6;

    private Recalibrators() {
    }

    /**
     * Fits a map on validation logits.
     *
     * @param method the map
     * @param logits validation logits {@code [n][k]}
     * @param labels validation labels
     * @return the fitted map
     */
    public static Recalibrator fit(Method method, double[][] logits, int[] labels) {
        int k = logits[0].length;
        int n = logits.length;
        return switch (method) {
            case NONE -> AffineLogitFit::softmaxRows;
            case TEMPERATURE -> {
                double t = Calibration.fitTemperature(logits, labels);
                yield z -> Calibration.withTemperature(z, t);
            }
            case VECTOR -> {
                AffineLogitFit a = AffineLogitFit.fit(logits, labels, AffineLogitFit.Structure.DIAGONAL, 0, 0, 0,
                        0, 500);
                yield a::probabilities;
            }
            case MATRIX -> {
                AffineLogitFit a = AffineLogitFit.fit(logits, labels, AffineLogitFit.Structure.FULL, 0, 0,
                        1e-3, 1e-3, 500);
                yield a::probabilities;
            }
            case DIRICHLET -> {
                double lambda = 1e-2;
                AffineLogitFit a = AffineLogitFit.fit(AffineLogitFit.logProbabilities(logits, 1, -30), labels,
                        AffineLogitFit.Structure.FULL, 0, 0, lambda / (k * (k - 1.0)), lambda / k, 500);
                yield z -> a.probabilities(AffineLogitFit.logProbabilities(z, 1, -30));
            }
            case PLATT, BETA -> {
                double[][] probs = AffineLogitFit.softmaxRows(logits);
                double[][] coef = new double[k][];
                for (int c = 0; c < k; c++) {
                    double[][] f = new double[n][];
                    double[] y = new double[n];
                    for (int i = 0; i < n; i++) {
                        f[i] = features(method, logits[i][c], probs[i][c]);
                        y[i] = labels[i] == c ? 1 : 0;
                    }
                    coef[c] = logistic(f, y, 1e-3);
                }
                yield z -> {
                    double[][] p = AffineLogitFit.softmaxRows(z);
                    double[][] s = new double[z.length][k];
                    for (int i = 0; i < z.length; i++) {
                        for (int c = 0; c < k; c++) {
                            s[i][c] = sigmoid(dot(coef[c], features(method, z[i][c], p[i][c])));
                        }
                    }
                    return AffineLogitFit.renormalise(s, FLOOR);
                };
            }
            case HISTOGRAM, ISOTONIC -> {
                double[][] probs = AffineLogitFit.softmaxRows(logits);
                double[][][] maps = new double[k][][];
                for (int c = 0; c < k; c++) {
                    double[] score = new double[n];
                    double[] y = new double[n];
                    for (int i = 0; i < n; i++) {
                        score[i] = probs[i][c];
                        y[i] = labels[i] == c ? 1 : 0;
                    }
                    maps[c] = method == Method.HISTOGRAM ? histogram(score, y, HISTOGRAM_BINS) : isotonic(score, y);
                }
                yield z -> {
                    double[][] p = AffineLogitFit.softmaxRows(z);
                    double[][] s = new double[z.length][k];
                    for (int i = 0; i < z.length; i++) {
                        for (int c = 0; c < k; c++) {
                            s[i][c] = method == Method.HISTOGRAM ? lookupBin(maps[c], p[i][c])
                                    : interpolate(maps[c], p[i][c]);
                        }
                    }
                    return AffineLogitFit.renormalise(s, FLOOR);
                };
            }
        };
    }

    private static double[] features(Method method, double logit, double p) {
        if (method == Method.PLATT) {
            return new double[] {logit, 1};
        }
        double q = Math.min(1 - 1e-12, Math.max(1e-12, p));
        return new double[] {Math.log(q), -Math.log(1 - q), 1};
    }

    // ------------------------------------------------------------------ binary helpers

    /**
     * Binary logistic regression by Newton's method (IRLS) with a small ridge.
     *
     * @param f feature rows (include a constant column for an intercept)
     * @param y targets in {@code [0, 1]}
     * @param ridge the ridge on every coefficient
     * @return the coefficients
     */
    public static double[] logistic(double[][] f, double[] y, double ridge) {
        int d = f[0].length;
        double[] beta = new double[d];
        if (d == 3) {
            beta[0] = 1;
            beta[1] = 1; // beta calibration starts at the identity map
        } else {
            beta[0] = 1;
        }
        for (int it = 0; it < 50; it++) {
            double[] g = new double[d];
            double[][] h = new double[d][d];
            for (int i = 0; i < f.length; i++) {
                double p = sigmoid(dot(beta, f[i]));
                double wgt = Math.max(1e-9, p * (1 - p));
                for (int a = 0; a < d; a++) {
                    g[a] += (p - y[i]) * f[i][a];
                    for (int b = 0; b < d; b++) {
                        h[a][b] += wgt * f[i][a] * f[i][b];
                    }
                }
            }
            for (int a = 0; a < d; a++) {
                g[a] += ridge * beta[a];
                h[a][a] += ridge + 1e-9;
            }
            double[] delta = solve(h, g);
            double before = logLoss(f, y, beta, ridge);
            double step = 1;
            double[] next = new double[d];
            for (int half = 0; half < 30; half++) {
                for (int a = 0; a < d; a++) {
                    next[a] = beta[a] - step * delta[a];
                }
                if (logLoss(f, y, next, ridge) <= before) {
                    break;
                }
                step /= 2;
            }
            double norm = 0;
            for (int a = 0; a < d; a++) {
                norm += Math.abs(beta[a] - next[a]);
                beta[a] = next[a];
            }
            if (norm < 1e-9) {
                break;
            }
        }
        return beta;
    }

    private static double logLoss(double[][] f, double[] y, double[] beta, double ridge) {
        double s = 0;
        for (int i = 0; i < f.length; i++) {
            double z = dot(beta, f[i]);
            // log(1 + e^z) - y z, stably
            s += Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z))) - y[i] * z;
        }
        return s + 0.5 * ridge * dot(beta, beta);
    }

    /**
     * Solves a small dense linear system by Gaussian elimination with partial pivoting.
     *
     * @param a the matrix (not modified)
     * @param b the right-hand side (not modified)
     * @return {@code x} with {@code a x = b}
     */
    public static double[] solve(double[][] a, double[] b) {
        int n = b.length;
        double[][] m = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, m[i], 0, n);
            m[i][n] = b[i];
        }
        for (int c = 0; c < n; c++) {
            int piv = c;
            for (int r = c + 1; r < n; r++) {
                if (Math.abs(m[r][c]) > Math.abs(m[piv][c])) {
                    piv = r;
                }
            }
            double[] t = m[c];
            m[c] = m[piv];
            m[piv] = t;
            double d = Math.abs(m[c][c]) < 1e-300 ? 1e-300 : m[c][c];
            for (int r = c + 1; r < n; r++) {
                double fct = m[r][c] / d;
                for (int q = c; q <= n; q++) {
                    m[r][q] -= fct * m[c][q];
                }
            }
        }
        double[] x = new double[n];
        for (int r = n - 1; r >= 0; r--) {
            double s = m[r][n];
            for (int q = r + 1; q < n; q++) {
                s -= m[r][q] * x[q];
            }
            x[r] = s / (Math.abs(m[r][r]) < 1e-300 ? 1e-300 : m[r][r]);
        }
        return x;
    }

    static double sigmoid(double z) {
        return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
    }

    static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /**
     * Equal-mass histogram binning.
     *
     * @param score scores
     * @param y binary targets
     * @param bins the number of bins
     * @return {@code {upper edges, bin frequencies}}
     */
    public static double[][] histogram(double[] score, double[] y, int bins) {
        Integer[] order = sortedOrder(score);
        int n = score.length;
        double[] edge = new double[bins];
        double[] freq = new double[bins];
        for (int b = 0; b < bins; b++) {
            int lo = b * n / bins;
            int hi = Math.max(lo + 1, (b + 1) * n / bins);
            double s = 0;
            for (int i = lo; i < hi && i < n; i++) {
                s += y[order[i]];
            }
            freq[b] = s / (Math.min(hi, n) - lo);
            edge[b] = b == bins - 1 ? Double.POSITIVE_INFINITY
                    : 0.5 * (score[order[Math.min(hi, n) - 1]] + score[order[Math.min(hi, n - 1)]]);
        }
        return new double[][] {edge, freq};
    }

    private static double lookupBin(double[][] map, double s) {
        int b = 0;
        while (b < map[0].length - 1 && s > map[0][b]) {
            b++;
        }
        return map[1][b];
    }

    /**
     * Isotonic regression by pool-adjacent-violators.
     *
     * @param score scores
     * @param y targets
     * @return {@code {block score means, block fitted values}}, increasing in score
     */
    public static double[][] isotonic(double[] score, double[] y) {
        Integer[] order = sortedOrder(score);
        int n = score.length;
        double[] xs = new double[n];
        double[] ys = new double[n];
        double[] ws = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = score[order[i]];
            ys[i] = y[order[i]];
            ws[i] = 1;
        }
        return pav(xs, ys, ws, n);
    }

    /**
     * Weighted pool-adjacent-violators on points already sorted by {@code x}.
     *
     * @param x sorted inputs
     * @param y targets
     * @param w weights
     * @param n the number of points used
     * @return {@code {block x means, block values}}
     */
    static double[][] pav(double[] x, double[] y, double[] w, int n) {
        double[] bx = new double[n];
        double[] by = new double[n];
        double[] bw = new double[n];
        int m = 0;
        for (int i = 0; i < n; i++) {
            bx[m] = x[i] * w[i];
            by[m] = y[i] * w[i];
            bw[m] = w[i];
            m++;
            while (m > 1 && by[m - 2] / bw[m - 2] >= by[m - 1] / bw[m - 1]) {
                bx[m - 2] += bx[m - 1];
                by[m - 2] += by[m - 1];
                bw[m - 2] += bw[m - 1];
                m--;
            }
        }
        double[][] out = new double[2][m];
        for (int i = 0; i < m; i++) {
            out[0][i] = bx[i] / bw[i];
            out[1][i] = by[i] / bw[i];
        }
        return out;
    }

    static double interpolate(double[][] map, double s) {
        double[] xs = map[0];
        double[] vs = map[1];
        if (s <= xs[0]) {
            return vs[0];
        }
        if (s >= xs[xs.length - 1]) {
            return vs[vs.length - 1];
        }
        int i = Arrays.binarySearch(xs, s);
        if (i >= 0) {
            return vs[i];
        }
        int hi = -i - 1;
        int lo = hi - 1;
        double t = (s - xs[lo]) / (xs[hi] - xs[lo]);
        return vs[lo] + t * (vs[hi] - vs[lo]);
    }

    private static Integer[] sortedOrder(double[] score) {
        Integer[] order = new Integer[score.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(score[a], score[b]));
        return order;
    }
}
