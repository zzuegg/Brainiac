package io.github.zzuegg.brainiac.experimental.privacy;

import io.github.zzuegg.brainiac.experimental.ml.NormalTails;

import java.util.SplittableRandom;

/**
 * Private aggregation of teacher ensembles, PATE (Papernot et al., ICLR 2017,
 * <a href="https://arxiv.org/abs/1610.05755">arXiv:1610.05755</a>; ICLR 2018,
 * <a href="https://arxiv.org/abs/1802.08908">arXiv:1802.08908</a>). Teachers trained on
 * disjoint parts of the private data vote on each query of a student; the student only sees a
 * noisy argmax of the vote counts, so privacy is spent per answered query, and very little when
 * the teachers agree.
 *
 * <ul>
 *   <li>{@link #laplaceMax}: LNMax, Laplace noise of scale {@code 1/γ} on every count, a
 *       {@code 2γ}-DP answer (accounted as {@code α (2γ)² / 2} RDP).</li>
 *   <li>{@link #gnMax}: Gaussian noise {@code N(0, σ²)}; data-independent RDP {@code α/σ²},
 *       tightened per query by the data-dependent bound of Papernot et al. 2018 (Theorem 6 and
 *       Proposition 10) from {@code q}, a union-bound estimate of the chance that the noisy
 *       argmax differs from the true one.</li>
 *   <li>{@link #confidentGnMax}: answers only when the noisy largest count clears a threshold
 *       {@code T} (noise {@code σ1}); the check costs {@code α/(2σ1²)} RDP (sensitivity 1 of the
 *       largest count, accounted data-independently here, a conservative choice), answered
 *       queries then cost the GNMax bound with {@code σ2}.</li>
 * </ul>
 *
 * <p>The accumulated RDP curve is converted to {@code (ε, δ)} with the same conversion as
 * {@code RdpAccountant} (Balle et al. 2020), minimised over the orders.
 */
public final class TeacherEnsembleAggregation {

    private static final double[] ORDERS;

    static {
        double[] o = new double[80];
        int k = 0;
        for (double a = 1.25; a < 10; a += 0.25) {
            o[k++] = a;
        }
        for (double a = 10; a <= 100; a += 5) {
            o[k++] = a;
        }
        o[k++] = 128;
        o[k++] = 256;
        ORDERS = java.util.Arrays.copyOf(o, k);
    }

    private final int classes;
    private final double[] rdp = new double[ORDERS.length];
    private final SplittableRandom random;
    private int answered;
    private int asked;

    private TeacherEnsembleAggregation(int classes, long seed) {
        this.classes = classes;
        this.random = new SplittableRandom(seed);
    }

    /**
     * An aggregator with no queries answered yet.
     *
     * @param classes the number of classes
     * @param seed the seed of the noise
     * @return the aggregator
     */
    public static TeacherEnsembleAggregation of(int classes, long seed) {
        return new TeacherEnsembleAggregation(classes, seed);
    }

    /**
     * Vote counts from teacher predictions.
     *
     * @param predictions one predicted class per teacher
     * @param classes the number of classes
     * @return the histogram
     */
    public static double[] votes(int[] predictions, int classes) {
        double[] v = new double[classes];
        for (int p : predictions) {
            v[p]++;
        }
        return v;
    }

    /**
     * LNMax.
     *
     * @param counts the vote counts
     * @param gamma the inverse Laplace scale
     * @return the noisy argmax
     */
    public int laplaceMax(double[] counts, double gamma) {
        asked++;
        answered++;
        double eps = 2 * gamma;
        for (int i = 0; i < ORDERS.length; i++) {
            rdp[i] += Math.min(eps, ORDERS[i] * eps * eps / 2);
        }
        int best = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < counts.length; c++) {
            double u = random.nextDouble() - 0.5;
            double noisy = counts[c] - Math.signum(u) * Math.log(1 - 2 * Math.abs(u)) / gamma;
            if (noisy > bestValue) {
                bestValue = noisy;
                best = c;
            }
        }
        return best;
    }

    /**
     * GNMax with data-dependent accounting.
     *
     * @param counts the vote counts
     * @param sigma the noise standard deviation
     * @return the noisy argmax
     */
    public int gnMax(double[] counts, double sigma) {
        asked++;
        answered++;
        double logQ = logQ(counts, sigma);
        for (int i = 0; i < ORDERS.length; i++) {
            rdp[i] += rdpGaussian(logQ, sigma, ORDERS[i]);
        }
        return noisyArgmax(counts, sigma);
    }

    /**
     * Confident-GNMax.
     *
     * @param counts the vote counts
     * @param threshold the threshold {@code T} on the noisy largest count
     * @param sigma1 the noise of the threshold check
     * @param sigma2 the noise of the answer
     * @return the noisy argmax, or -1 when the teachers do not agree enough
     */
    public int confidentGnMax(double[] counts, double threshold, double sigma1, double sigma2) {
        asked++;
        for (int i = 0; i < ORDERS.length; i++) {
            rdp[i] += ORDERS[i] / (2 * sigma1 * sigma1);
        }
        double max = 0;
        for (double c : counts) {
            max = Math.max(max, c);
        }
        if (max + random.nextGaussian() * sigma1 < threshold) {
            return -1;
        }
        answered++;
        double logQ = logQ(counts, sigma2);
        for (int i = 0; i < ORDERS.length; i++) {
            rdp[i] += rdpGaussian(logQ, sigma2, ORDERS[i]);
        }
        return noisyArgmax(counts, sigma2);
    }

    private int noisyArgmax(double[] counts, double sigma) {
        int best = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < counts.length; c++) {
            double noisy = counts[c] + random.nextGaussian() * sigma;
            if (noisy > bestValue) {
                bestValue = noisy;
                best = c;
            }
        }
        return best;
    }

    /** @return queries asked */
    public int asked() {
        return asked;
    }

    /** @return queries answered */
    public int answered() {
        return answered;
    }

    /**
     * @param delta the failure probability
     * @return the {@code ε} spent so far
     */
    public double epsilon(double delta) {
        if (asked == 0) {
            return 0;
        }
        double best = Double.POSITIVE_INFINITY;
        for (int i = 0; i < ORDERS.length; i++) {
            double a = ORDERS[i];
            double eps = rdp[i] + Math.log((a - 1) / a) - (Math.log(delta) + Math.log(a)) / (a - 1);
            best = Math.min(best, eps);
        }
        return Math.max(best, 0);
    }

    /**
     * The log of a union-bound estimate of {@code q = P(noisy argmax != argmax)} under
     * {@code N(0, σ²)} noise: {@code Σ_(i != i*) P(N(0, 2σ²) > n_(i*) - n_i)}, capped at
     * {@code 1 - 1/k}.
     *
     * @param counts the vote counts
     * @param sigma the noise
     * @return {@code log q}
     */
    public static double logQ(double[] counts, double sigma) {
        int top = 0;
        for (int c = 1; c < counts.length; c++) {
            top = counts[c] > counts[top] ? c : top;
        }
        double max = Double.NEGATIVE_INFINITY;
        double[] terms = new double[counts.length];
        for (int c = 0; c < counts.length; c++) {
            terms[c] = c == top ? Double.NEGATIVE_INFINITY
                    : NormalTails.logSf((counts[top] - counts[c]) / (Math.sqrt(2) * sigma));
            max = Math.max(max, terms[c]);
        }
        double sum = 0;
        for (double t : terms) {
            sum += Math.exp(t - max);
        }
        return Math.min(max + Math.log(sum), Math.log(1 - 1.0 / counts.length));
    }

    /**
     * The data-dependent RDP of one GNMax answer at an order (Papernot et al. 2018,
     * Proposition 10), falling back to the data-independent {@code α/σ²}.
     *
     * @param logQ the log of the probability bound {@code q}
     * @param sigma the noise
     * @param order the order {@code α > 1}
     * @return the RDP
     */
    public static double rdpGaussian(double logQ, double sigma, double order) {
        double variance = sigma * sigma;
        double independent = order / variance;
        if (Double.isInfinite(logQ)) {
            return 0;
        }
        double muHi2 = Math.sqrt(variance * -logQ);
        double muHi1 = muHi2 + 1;
        if (!(muHi1 > order && muHi2 > 1)) {
            return independent;
        }
        double rdpHi1 = muHi1 / variance;
        double rdpHi2 = muHi2 / variance;
        double logA2 = (muHi2 - 1) * rdpHi2;
        boolean applies = logQ <= logA2 - muHi2 * (Math.log(1 + 1 / (muHi1 - 1)) + Math.log(1 + 1 / (muHi2 - 1)))
                && -logQ > rdpHi2;
        if (!applies) {
            return independent;
        }
        double log1q = log1mexp(logQ);
        double logA = (order - 1) * (log1q - log1mexp((logQ + rdpHi2) * (1 - 1 / muHi2)));
        double logB = (order - 1) * (rdpHi1 - logQ / (muHi1 - 1));
        double x = log1q + logA;
        double y = logQ + logB;
        double logS = Math.max(x, y) + Math.log1p(Math.exp(-Math.abs(x - y)));
        return Math.min(independent, logS / (order - 1));
    }

    /** {@code log(1 - exp(x))} for {@code x < 0}. */
    private static double log1mexp(double x) {
        return x < -0.693 ? Math.log1p(-Math.exp(x)) : Math.log(-Math.expm1(x));
    }

    /** @return the number of classes */
    public int classes() {
        return classes;
    }
}
