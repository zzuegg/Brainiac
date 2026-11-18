package io.github.zzuegg.brainiac.experimental.dpmech;

import java.util.SplittableRandom;

/**
 * MWEM, private query release by multiplicative weights and the exponential mechanism
 * (Hardt, Ligett and McSherry, NeurIPS 2012, <a href="https://arxiv.org/abs/1012.4763">arXiv:1012.4763</a>).
 * The data are rows of a discrete table with column cardinalities {@code cards}; the
 * workload is a set of 2-way marginals. MWEM keeps a distribution {@code A} over the whole
 * domain, starting uniform. In each of {@code T} rounds it picks the marginal that {@code A}
 * answers worst with the exponential mechanism ({@code ε/2T}, score: the {@code ℓ1} error in
 * counts, sensitivity 1), measures it with Laplace noise ({@code ε/2T}, the marginal's
 * {@code ℓ1} sensitivity is 1 under add/remove), and updates {@code A} by multiplicative
 * weights {@code A(x) ← A(x) exp(q(x)(m - n A(q)) / 2n)} over all measurements so far. The
 * row count {@code n} is treated as public. The released {@code A} answers any query and
 * {@link #sample} draws a synthetic table from it.
 *
 * <p>{@link #noisyMarginals} is the independent-noise baseline: every workload marginal
 * answered directly with Laplace noise, the budget split evenly.
 */
public final class Mwem {

    private Mwem() {
    }

    /**
     * Runs MWEM.
     *
     * @param rows the private table, values in {@code [0, cards[j])}
     * @param cards the column cardinalities (their product must be small, about 10⁴)
     * @param workload column pairs {@code {a, b}}
     * @param epsilon the total budget
     * @param rounds {@code T}
     * @param seed the seed
     * @return the released distribution over the domain (index by {@link #index})
     */
    public static double[] run(int[][] rows, int[] cards, int[][] workload, double epsilon, int rounds, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int size = domain(cards);
        int n = rows.length;
        double[] a = new double[size];
        java.util.Arrays.fill(a, 1.0 / size);
        double[][] real = new double[workload.length][];
        for (int q = 0; q < workload.length; q++) {
            real[q] = marginal(rows, cards, workload[q][0], workload[q][1]);
        }
        double eps = epsilon / (2 * rounds);
        int[] measuredQuery = new int[rounds];
        double[][] measured = new double[rounds][];
        for (int t = 0; t < rounds; t++) {
            double[] score = new double[workload.length];
            for (int q = 0; q < workload.length; q++) {
                double[] m = marginal(a, cards, workload[q][0], workload[q][1]);
                for (int c = 0; c < m.length; c++) {
                    score[q] += Math.abs(real[q][c] - n * m[c]);
                }
            }
            int q = PrivateSelection.select(PrivateSelection.Mechanism.EXPONENTIAL, score, eps, 1, r);
            double[] noisy = real[q].clone();
            for (int c = 0; c < noisy.length; c++) {
                noisy[c] += LocalDp.laplace(1 / eps, r);
            }
            measuredQuery[t] = q;
            measured[t] = noisy;
            for (int pass = 0; pass < 10; pass++) {
                for (int s = 0; s <= t; s++) {
                    update(a, cards, workload[measuredQuery[s]], measured[s], n);
                }
            }
        }
        return a;
    }

    private static void update(double[] a, int[] cards, int[] pair, double[] noisy, int n) {
        double[] m = marginal(a, cards, pair[0], pair[1]);
        int[] tuple = new int[cards.length];
        double z = 0;
        for (int x = 0; x < a.length; x++) {
            decode(x, cards, tuple);
            int c = tuple[pair[0]] * cards[pair[1]] + tuple[pair[1]];
            a[x] *= Math.exp((noisy[c] - n * m[c]) / (2.0 * n));
            z += a[x];
        }
        for (int x = 0; x < a.length; x++) {
            a[x] /= z;
        }
    }

    /**
     * Every 2-way marginal answered with Laplace noise of scale {@code |W|/ε}.
     *
     * @param rows the private table
     * @param cards the cardinalities
     * @param workload the column pairs
     * @param epsilon the total budget
     * @param seed the seed
     * @return the noisy counts of each marginal
     */
    public static double[][] noisyMarginals(int[][] rows, int[] cards, int[][] workload, double epsilon, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] out = new double[workload.length][];
        for (int q = 0; q < workload.length; q++) {
            out[q] = marginal(rows, cards, workload[q][0], workload[q][1]);
            for (int c = 0; c < out[q].length; c++) {
                out[q][c] += LocalDp.laplace(workload.length / epsilon, r);
            }
        }
        return out;
    }

    /**
     * The counts of a 2-way marginal of a table.
     *
     * @param rows the table
     * @param cards the cardinalities
     * @param a one column
     * @param b another column
     * @return counts, cell {@code v_a cards[b] + v_b}
     */
    public static double[] marginal(int[][] rows, int[] cards, int a, int b) {
        double[] m = new double[cards[a] * cards[b]];
        for (int[] row : rows) {
            m[row[a] * cards[b] + row[b]]++;
        }
        return m;
    }

    /**
     * The probabilities of a 2-way marginal of a distribution over the domain.
     *
     * @param dist the distribution
     * @param cards the cardinalities
     * @param a one column
     * @param b another column
     * @return probabilities, cell {@code v_a cards[b] + v_b}
     */
    public static double[] marginal(double[] dist, int[] cards, int a, int b) {
        double[] m = new double[cards[a] * cards[b]];
        int[] tuple = new int[cards.length];
        for (int x = 0; x < dist.length; x++) {
            decode(x, cards, tuple);
            m[tuple[a] * cards[b] + tuple[b]] += dist[x];
        }
        return m;
    }

    /**
     * The mean workload error: the average over marginals of the {@code ℓ1} distance between
     * the real and the released marginal, as a share of the rows (twice the total variation).
     *
     * @param rows the real table
     * @param cards the cardinalities
     * @param workload the column pairs
     * @param released the released counts of each marginal
     * @return the mean error
     */
    public static double workloadError(int[][] rows, int[] cards, int[][] workload, double[][] released) {
        double s = 0;
        for (int q = 0; q < workload.length; q++) {
            double[] real = marginal(rows, cards, workload[q][0], workload[q][1]);
            for (int c = 0; c < real.length; c++) {
                s += Math.abs(real[c] - released[q][c]) / rows.length / workload.length;
            }
        }
        return s;
    }

    /**
     * The marginals of a synthetic table, as released counts.
     *
     * @param synthetic the table
     * @param cards the cardinalities
     * @param workload the column pairs
     * @param n the real row count the counts are scaled to
     * @return the counts of each marginal
     */
    public static double[][] marginalsOf(int[][] synthetic, int[] cards, int[][] workload, int n) {
        double[][] out = new double[workload.length][];
        for (int q = 0; q < workload.length; q++) {
            out[q] = marginal(synthetic, cards, workload[q][0], workload[q][1]);
            for (int c = 0; c < out[q].length; c++) {
                out[q][c] *= n / (double) synthetic.length;
            }
        }
        return out;
    }

    /**
     * Draws rows from a distribution over the domain.
     *
     * @param dist the distribution
     * @param cards the cardinalities
     * @param n rows
     * @param seed the seed
     * @return the synthetic table
     */
    public static int[][] sample(double[] dist, int[] cards, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] cdf = new double[dist.length];
        double acc = 0;
        for (int x = 0; x < dist.length; x++) {
            acc += dist[x];
            cdf[x] = acc;
        }
        int[][] out = new int[n][cards.length];
        for (int i = 0; i < n; i++) {
            double u = r.nextDouble() * acc;
            int lo = 0;
            int hi = dist.length - 1;
            while (lo < hi) {
                int mid = (lo + hi) >>> 1;
                if (cdf[mid] < u) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            decode(lo, cards, out[i]);
        }
        return out;
    }

    /**
     * The domain size.
     *
     * @param cards the cardinalities
     * @return their product
     */
    public static int domain(int[] cards) {
        int s = 1;
        for (int c : cards) {
            s = Math.multiplyExact(s, c);
        }
        return s;
    }

    /**
     * The domain index of a tuple (mixed radix, last column fastest).
     *
     * @param tuple the values
     * @param cards the cardinalities
     * @return the index
     */
    public static int index(int[] tuple, int[] cards) {
        int x = 0;
        for (int j = 0; j < cards.length; j++) {
            x = x * cards[j] + tuple[j];
        }
        return x;
    }

    static void decode(int x, int[] cards, int[] tuple) {
        for (int j = cards.length - 1; j >= 0; j--) {
            tuple[j] = x % cards[j];
            x /= cards[j];
        }
    }
}
