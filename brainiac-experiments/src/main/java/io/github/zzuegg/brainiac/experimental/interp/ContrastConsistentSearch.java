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
 * <p>Wave 10 follow-up (item 10.3.17): with several prompt templates pooled, the
 * {@linkplain #fit(float[][], float[][], int[], int, int, long) per-template fit} normalises each
 * template's states by that template's own statistics; {@link #ensembleAccuracy} averages a
 * statement's credences over templates; and {@link #crc} is contrastive representation
 * clustering, the paper's simpler alternative, fitted on one set and applied to another. The
 * original methods and their defaults are unchanged.
 *
 * <p>Papers: <a href="https://arxiv.org/abs/2212.03827">arXiv:2212.03827</a>; Farquhar et al.,
 * <a href="https://arxiv.org/abs/2312.10029">arXiv:2312.10029</a>.
 */
public final class ContrastConsistentSearch {

    private final double[] w;
    private final double bias;
    private final double[][] norm;
    private final double loss;
    private final double[][][] templateNorm;

    private ContrastConsistentSearch(double[] w, double bias, double[][] norm, double loss, double[][][] templateNorm) {
        this.w = w;
        this.bias = bias;
        this.norm = norm;
        this.loss = loss;
        this.templateNorm = templateNorm;
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
        return search(xp, xn, d, restarts, epochs, seed, s, null);
    }

    /** Adam restarts on normalised pairs; keeps the lowest CCS loss. */
    private static ContrastConsistentSearch search(double[][] xp, double[][] xn, int d, int restarts, int epochs,
            long seed, double[][] s, double[][][] templateNorm) {
        int n = xp.length;
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
                best = new ContrastConsistentSearch(wd, w[d], s, l, templateNorm);
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

    // ------------------------------------------------------------------ wave 10 follow-up (10.3.17)

    /** Per-template normalisation statistics, {@code [templates][4][d]}. */
    private static double[][][] templateStatistics(float[][] pos, float[][] neg, int[] template) {
        int t = 0;
        for (int k : template) {
            t = Math.max(t, k + 1);
        }
        double[][][] out = new double[t][][];
        for (int k = 0; k < t; k++) {
            int count = 0;
            for (int v : template) {
                count += v == k ? 1 : 0;
            }
            if (count == 0) {
                continue;
            }
            float[][] p = new float[count][];
            float[][] q = new float[count][];
            int c = 0;
            for (int i = 0; i < template.length; i++) {
                if (template[i] == k) {
                    p[c] = pos[i];
                    q[c++] = neg[i];
                }
            }
            out[k] = statistics(p, q);
        }
        return out;
    }

    private double[][] statsFor(int template) {
        return templateNorm != null && template < templateNorm.length && templateNorm[template] != null
                ? templateNorm[template] : norm;
    }

    /**
     * CCS with per-template normalisation (Burns et al.'s recipe when several prompt templates
     * are pooled): each template's "yes" and "no" hidden states are normalised by that
     * template's own mean and standard deviation, which removes template-specific offsets as
     * well as the answer-token direction.
     *
     * @param pos hidden states of the "yes" completions {@code [n][d]}
     * @param neg hidden states of the "no" completions, same order
     * @param template the template of every row, {@code 0..T-1}
     * @param restarts random restarts
     * @param epochs full-batch Adam steps per restart
     * @param seed the seed
     * @return the probe with the lowest CCS loss
     */
    public static ContrastConsistentSearch fit(float[][] pos, float[][] neg, int[] template, int restarts, int epochs,
            long seed) {
        double[][][] ts = templateStatistics(pos, neg, template);
        int n = pos.length;
        double[][] xp = new double[n][];
        double[][] xn = new double[n][];
        for (int i = 0; i < n; i++) {
            xp[i] = normalise(pos[i], ts[template[i]], true);
            xn[i] = normalise(neg[i], ts[template[i]], false);
        }
        return search(xp, xn, pos[0].length, restarts, epochs, seed, statistics(pos, neg), ts);
    }

    /**
     * The credence with a template's normalisation (the pooled one if the template is unknown
     * or the probe was fitted without templates).
     *
     * @param pos the "yes" hidden state
     * @param neg the "no" hidden state
     * @param template the template
     * @return {@code (p(x+) + 1 - p(x-)) / 2}
     */
    public double credence(float[] pos, float[] neg, int template) {
        double[][] s = statsFor(template);
        double pp = InterpMath.sigmoid(dot(w, normalise(pos, s, true)) + bias);
        double pn = InterpMath.sigmoid(dot(w, normalise(neg, s, false)) + bias);
        return (pp + 1 - pn) / 2;
    }

    /**
     * Credences of many pairs.
     *
     * @param pos "yes" states
     * @param neg "no" states
     * @param template each row's template
     * @return one credence per row
     */
    public double[] credences(float[][] pos, float[][] neg, int[] template) {
        double[] c = new double[pos.length];
        for (int i = 0; i < pos.length; i++) {
            c[i] = credence(pos[i], neg[i], template[i]);
        }
        return c;
    }

    /**
     * Sign-free accuracy of credences against labels.
     *
     * @param credence one credence per statement
     * @param truth the labels
     * @return {@code max(acc, 1 - acc)} of {@code credence > 1/2}
     */
    public static double signFreeAccuracy(double[] credence, boolean[] truth) {
        int hit = 0;
        for (int i = 0; i < truth.length; i++) {
            hit += (credence[i] > 0.5) == truth[i] ? 1 : 0;
        }
        double a = (double) hit / truth.length;
        return Math.max(a, 1 - a);
    }

    /**
     * The prompt-ensemble accuracy: each statement's credences under several templates are
     * averaged before thresholding.
     *
     * @param credences {@code [templates][statements]}
     * @param truth the labels
     * @return the sign-free accuracy of the averaged credence
     */
    public static double ensembleAccuracy(double[][] credences, boolean[] truth) {
        double[] mean = new double[truth.length];
        for (double[] c : credences) {
            for (int i = 0; i < truth.length; i++) {
                mean[i] += c[i] / credences.length;
            }
        }
        return signFreeAccuracy(mean, truth);
    }

    /**
     * Contrastive representation clustering (CRC, Burns et al.): the top principal component
     * of the per-template normalised differences {@code x+ - x-}, fitted on one set and applied
     * to others; the credence is {@code sigmoid} of the projection.
     *
     * @param direction the unit principal direction {@code [d]}
     * @param center the mean normalised difference
     * @param templateNorm per-template statistics
     */
    public record Crc(double[] direction, double[] center, double[][][] templateNorm) {

        /**
         * @param pos the "yes" state
         * @param neg the "no" state
         * @param template its template
         * @return {@code sigmoid(u . (diff - center))}
         */
        public double credence(float[] pos, float[] neg, int template) {
            double[][] s = templateNorm[Math.min(template, templateNorm.length - 1)];
            double[] a = normalise(pos, s, true);
            double[] b = normalise(neg, s, false);
            double proj = 0;
            for (int j = 0; j < a.length; j++) {
                proj += (a[j] - b[j] - center[j]) * direction[j];
            }
            return InterpMath.sigmoid(proj);
        }

        /**
         * @param pos "yes" states
         * @param neg "no" states
         * @param template each row's template
         * @return one credence per row
         */
        public double[] credences(float[][] pos, float[][] neg, int[] template) {
            double[] c = new double[pos.length];
            for (int i = 0; i < pos.length; i++) {
                c[i] = credence(pos[i], neg[i], template[i]);
            }
            return c;
        }
    }

    /**
     * Fits CRC with per-template normalisation.
     *
     * @param pos "yes" states
     * @param neg "no" states
     * @param template each row's template
     * @return the fitted direction
     */
    public static Crc crc(float[][] pos, float[][] neg, int[] template) {
        double[][][] ts = templateStatistics(pos, neg, template);
        int n = pos.length;
        double[][] diff = new double[n][];
        for (int i = 0; i < n; i++) {
            double[] a = normalise(pos[i], ts[template[i]], true);
            double[] b = normalise(neg[i], ts[template[i]], false);
            for (int j = 0; j < a.length; j++) {
                a[j] -= b[j];
            }
            diff[i] = a;
        }
        double[] center = InterpMath.mean(diff);
        double[][] c = InterpMath.centre(diff);
        double[][] v = io.github.zzuegg.brainiac.experimental.tensor.ThinSvd.of(c).v();
        double[] u = new double[center.length];
        for (int j = 0; j < u.length; j++) {
            u[j] = v[j][0];
        }
        return new Crc(u, center, ts);
    }
}
