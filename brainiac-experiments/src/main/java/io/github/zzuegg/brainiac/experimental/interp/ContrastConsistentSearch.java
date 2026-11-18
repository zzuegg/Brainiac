package io.github.zzuegg.brainiac.experimental.interp;

import java.util.SplittableRandom;

/**
 * Contrast-consistent search (CCS, Burns et al., ICLR 2023): find a "truth" direction in a
 * model's hidden states without labels. For each yes/no statement both completions are run
 * ({@code x+}: "... Yes", {@code x-}: "... No") and each set of hidden states is normalised
 * (its own mean removed, divided by its own standard deviation), so the probe cannot simply
 * read which answer token was appended. A logistic probe {@code p(x) = sigmoid(w.x + b)} is
 * then trained to be
 *
 * <ul>
 *   <li>consistent: {@code p(x+) = 1 - p(x-)}, loss {@code (p(x+) - (1 - p(x-)))^2};</li>
 *   <li>confident: not ½ for both, loss {@code min(p(x+), p(x-))^2};</li>
 * </ul>
 *
 * <p>with several random restarts, keeping the lowest loss. The truth label is
 * {@code (p(x+) + 1 - p(x-)) / 2 > ½}, up to a global sign that unsupervised methods cannot
 * know, so accuracy is reported as {@code max(acc, 1 - acc)}, as in the paper. Follow-up work
 * (Farquhar et al., 2023) showed that CCS latches onto any salient binary feature that is
 * consistent across the pair, so experiments should include a distractor.
 *
 * <p>{@link #contrastivePca} is the paper's unsupervised baseline: the top principal
 * component of the normalised differences {@code x+ - x-}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2212.03827">arXiv:2212.03827</a>.
 */
public final class ContrastConsistentSearch {

    private final double[] w;
    private final double bias;
    private final double[][] norm;
    private final double loss;

    private ContrastConsistentSearch(double[] w, double bias, double[][] norm, double loss) {
        this.w = w;
        this.bias = bias;
        this.norm = norm;
        this.loss = loss;
    }

    /** The per-set normalisation: {mean+, std+, mean-, std-}. */
    private static double[][] statistics(float[][] pos, float[][] neg) {
        int d = pos[0].length;
        double[][] s = new double[4][d];
        for (int k = 0; k < 2; k++) {
            float[][] x = k == 0 ? pos : neg;
            for (float[] row : x) {
                for (int j = 0; j < d; j++) {
                    s[2 * k][j] += row[j] / (double) x.length;
                }
            }
            for (float[] row : x) {
                for (int j = 0; j < d; j++) {
                    double c = row[j] - s[2 * k][j];
                    s[2 * k + 1][j] += c * c / x.length;
                }
            }
            for (int j = 0; j < d; j++) {
                s[2 * k + 1][j] = Math.sqrt(s[2 * k + 1][j]) + 1e-6;
            }
        }
        return s;
    }

    private static double[] normalise(float[] x, double[][] s, boolean positive) {
        int k = positive ? 0 : 2;
        double[] out = new double[x.length];
        for (int j = 0; j < x.length; j++) {
            out[j] = (x[j] - s[k][j]) / s[k + 1][j];
        }
        return out;
    }

    /**
     * Fits CCS.
     *
     * @param pos hidden states of the "yes" completions {@code [n][d]}
     * @param neg hidden states of the "no" completions, same order
     * @param restarts random restarts
     * @param epochs full-batch Adam steps per restart
     * @param seed the seed
     * @return the probe with the lowest CCS loss
     */
    public static ContrastConsistentSearch fit(float[][] pos, float[][] neg, int restarts, int epochs, long seed) {
        double[][] s = statistics(pos, neg);
        int n = pos.length;
        int d = pos[0].length;
        double[][] xp = new double[n][];
        double[][] xn = new double[n][];
        for (int i = 0; i < n; i++) {
            xp[i] = normalise(pos[i], s, true);
            xn[i] = normalise(neg[i], s, false);
        }
        SplittableRandom r = new SplittableRandom(seed);
        ContrastConsistentSearch best = null;
        for (int restart = 0; restart < restarts; restart++) {
            double[] w = new double[d + 1];
            double scale = 1 / Math.sqrt(d);
            for (int j = 0; j < d; j++) {
                w[j] = r.nextGaussian() * scale;
            }
            double[] m = new double[d + 1];
            double[] v = new double[d + 1];
            double lr = 1e-2;
            for (int e = 1; e <= epochs; e++) {
                double[] g = new double[d + 1];
                for (int i = 0; i < n; i++) {
                    double pp = InterpMath.sigmoid(dot(w, xp[i]) + w[d]);
                    double pn = InterpMath.sigmoid(dot(w, xn[i]) + w[d]);
                    double cons = pp - (1 - pn);
                    // d/dpp and d/dpn of cons^2 + min(pp, pn)^2
                    double gp = 2 * cons + (pp <= pn ? 2 * pp : 0);
                    double gn = 2 * cons + (pn < pp ? 2 * pn : 0);
                    double sp = gp * pp * (1 - pp) / n;
                    double sn = gn * pn * (1 - pn) / n;
                    for (int j = 0; j < d; j++) {
                        g[j] += sp * xp[i][j] + sn * xn[i][j];
                    }
                    g[d] += sp + sn;
                }
                for (int j = 0; j <= d; j++) {
                    m[j] = 0.9 * m[j] + 0.1 * g[j];
                    v[j] = 0.999 * v[j] + 0.001 * g[j] * g[j];
                    w[j] -= lr * (m[j] / (1 - Math.pow(0.9, e))) / (Math.sqrt(v[j] / (1 - Math.pow(0.999, e))) + 1e-8);
                }
            }
            double l = loss(w, xp, xn);
            if (best == null || l < best.loss) {
                double[] wd = java.util.Arrays.copyOf(w, d);
                best = new ContrastConsistentSearch(wd, w[d], s, l);
            }
        }
        return best;
    }

    private static double dot(double[] w, double[] x) {
        double s = 0;
        for (int j = 0; j < x.length; j++) {
            s += w[j] * x[j];
        }
        return s;
    }

    private static double loss(double[] w, double[][] xp, double[][] xn) {
        int d = xp[0].length;
        double l = 0;
        for (int i = 0; i < xp.length; i++) {
            double pp = InterpMath.sigmoid(dot(w, xp[i]) + w[d]);
            double pn = InterpMath.sigmoid(dot(w, xn[i]) + w[d]);
            double cons = pp - (1 - pn);
            double conf = Math.min(pp, pn);
            l += (cons * cons + conf * conf) / xp.length;
        }
        return l;
    }

    /** @return the CCS loss of the chosen restart */
    public double loss() {
        return loss;
    }

    /**
     * The probe's credence that the statement is true.
     *
     * @param pos the "yes" hidden state
     * @param neg the "no" hidden state
     * @return {@code (p(x+) + 1 - p(x-)) / 2}
     */
    public double credence(float[] pos, float[] neg) {
        double pp = InterpMath.sigmoid(dot(w, normalise(pos, norm, true)) + bias);
        double pn = InterpMath.sigmoid(dot(w, normalise(neg, norm, false)) + bias);
        return (pp + 1 - pn) / 2;
    }

    /**
     * Sign-free accuracy against labels.
     *
     * @param pos "yes" states
     * @param neg "no" states
     * @param truth the labels
     * @return {@code max(acc, 1 - acc)}
     */
    public double accuracy(float[][] pos, float[][] neg, boolean[] truth) {
        int hit = 0;
        for (int i = 0; i < pos.length; i++) {
            hit += (credence(pos[i], neg[i]) > 0.5) == truth[i] ? 1 : 0;
        }
        double a = (double) hit / pos.length;
        return Math.max(a, 1 - a);
    }

    /**
     * The contrastive-PCA baseline: project the normalised difference {@code x+ - x-} on its
     * top principal component (fitted on the given pairs) and read the sign.
     *
     * @param pos "yes" states
     * @param neg "no" states
     * @param truth the labels, used only to score
     * @return the sign-free accuracy
     */
    public static double contrastivePca(float[][] pos, float[][] neg, boolean[] truth) {
        double[][] s = statistics(pos, neg);
        int n = pos.length;
        double[][] diff = new double[n][];
        for (int i = 0; i < n; i++) {
            double[] a = normalise(pos[i], s, true);
            double[] b = normalise(neg[i], s, false);
            for (int j = 0; j < a.length; j++) {
                a[j] -= b[j];
            }
            diff[i] = a;
        }
        double[][] c = InterpMath.centre(diff);
        double[][] u = io.github.zzuegg.brainiac.experimental.tensor.ThinSvd.of(c).v();
        int hit = 0;
        for (int i = 0; i < n; i++) {
            double proj = 0;
            for (int j = 0; j < c[i].length; j++) {
                proj += c[i][j] * u[j][0];
            }
            hit += (proj > 0) == truth[i] ? 1 : 0;
        }
        double a = (double) hit / n;
        return Math.max(a, 1 - a);
    }

    /**
     * A random direction, the floor: sign-free accuracy of a random probe on the
     * normalised difference.
     *
     * @param pos "yes" states
     * @param neg "no" states
     * @param truth labels
     * @param seed the seed
     * @return the sign-free accuracy
     */
    public static double randomDirection(float[][] pos, float[][] neg, boolean[] truth, long seed) {
        double[][] s = statistics(pos, neg);
        SplittableRandom r = new SplittableRandom(seed);
        double[] w = new double[pos[0].length];
        for (int j = 0; j < w.length; j++) {
            w[j] = r.nextGaussian();
        }
        int hit = 0;
        for (int i = 0; i < pos.length; i++) {
            double[] a = normalise(pos[i], s, true);
            double[] b = normalise(neg[i], s, false);
            hit += (dot(w, a) - dot(w, b) > 0) == truth[i] ? 1 : 0;
        }
        double a = (double) hit / pos.length;
        return Math.max(a, 1 - a);
    }
}
