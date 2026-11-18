package io.github.zzuegg.brainiac.experimental.testing;

import io.github.zzuegg.brainiac.experimental.ml.IsolationForest;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Novelty detection with false-discovery-rate control over a batch of test points, around any
 * anomaly score.
 *
 * <ul>
 *   <li>{@link Method#SPLIT_BH}/{@link Method#SPLIT_STOREY} (Bates, Candes, Lei, Romano and
 *       Sesia, Annals 2023, <a href="https://arxiv.org/abs/2104.08279">arXiv:2104.08279</a>):
 *       train the score on half the reference inliers, turn test scores into conformal p-values
 *       {@code (1 + #{calibration >= s}) / (n_cal + 1)} with the other half, and run BH (or
 *       Storey-BH); the p-values are PRDS, so the FDR is controlled.</li>
 *   <li>{@link Method#FULL_E_BH}/{@link Method#K_BLOCK_E_BH} (Lee, Popov and Ren, 2025,
 *       <a href="https://arxiv.org/abs/2501.02703">arXiv:2501.02703</a>): train the score on all
 *       reference data plus the test block (unlabelled, order-invariant), find the threshold
 *       {@code T = inf{t : m/(n+1) (1 + #{ref >= t}) / max(1, #{test >= t}) <= alpha}}, form the
 *       e-values {@code e_j = (n+1) 1{V_j >= T} / (1 + #{ref >= T})}, and run e-BH. No reference
 *       point is spent on calibration only. K blocks train one score per block of test points
 *       ({@code K = 1} is the canonical full-conformal version).</li>
 * </ul>
 */
public final class ConformalNovelty {

    /** The procedure. */
    public enum Method {
        /** Split conformal p-values with BH. */
        SPLIT_BH,
        /** Split conformal p-values with Storey's adaptive BH. */
        SPLIT_STOREY,
        /** Full-conformal e-values with e-BH (one block). */
        FULL_E_BH,
        /** Full-conformal e-values with e-BH, one score per block of the test set. */
        K_BLOCK_E_BH
    }

    /** Trains an anomaly score (larger = more anomalous) on unlabelled rows. */
    @FunctionalInterface
    public interface ScoreFunction {

        /**
         * Trains the score.
         *
         * @param training the rows to fit on (order must not matter)
         * @return the score
         */
        ToDoubleFunction<double[]> fit(double[][] training);
    }

    private ConformalNovelty() {
    }

    /**
     * An isolation-forest score.
     *
     * @param trees trees
     * @param sampleSize the subsample per tree
     * @param seed the seed
     * @return the score function
     */
    public static ScoreFunction isolationForest(int trees, int sampleSize, long seed) {
        return training -> {
            IsolationForest f = IsolationForest.fit(training, trees, Math.min(sampleSize, training.length), seed);
            return f::score;
        };
    }

    /**
     * The mean distance to the {@code k} nearest training rows (a point in the training set
     * also counts itself, which full conformal requires for exchangeability).
     *
     * @param k neighbours
     * @return the score function
     */
    public static ScoreFunction knnDistance(int k) {
        return training -> x -> {
            double[] best = new double[k];
            java.util.Arrays.fill(best, Double.POSITIVE_INFINITY);
            for (double[] t : training) {
                double d = 0;
                for (int j = 0; j < x.length; j++) {
                    d += (x[j] - t[j]) * (x[j] - t[j]);
                }
                if (d < best[k - 1]) {
                    int i = k - 1;
                    while (i > 0 && best[i - 1] > d) {
                        best[i] = best[i - 1];
                        i--;
                    }
                    best[i] = d;
                }
            }
            double s = 0;
            for (double b : best) {
                s += Math.sqrt(b);
            }
            return s / k;
        };
    }

    /**
     * Flags novelties in a test batch.
     *
     * @param method the procedure
     * @param score the score
     * @param reference inlier rows
     * @param test the batch
     * @param alpha the target FDR
     * @param blocks blocks for {@link Method#K_BLOCK_E_BH}
     * @param seed the seed (split and blocks)
     * @return the flags
     */
    public static boolean[] detect(Method method, ScoreFunction score, double[][] reference, double[][] test,
            double alpha, int blocks, long seed) {
        return switch (method) {
            case SPLIT_BH, SPLIT_STOREY -> {
                double[] p = splitPValues(score, reference, test, seed);
                yield MultipleTesting.reject(method == Method.SPLIT_BH ? MultipleTesting.Method.BH
                        : MultipleTesting.Method.STOREY_BH, p, alpha);
            }
            case FULL_E_BH -> MultipleTesting.reject(MultipleTesting.Method.E_BH,
                    fullEValues(score, reference, test, alpha, 1, seed), alpha);
            case K_BLOCK_E_BH -> MultipleTesting.reject(MultipleTesting.Method.E_BH,
                    fullEValues(score, reference, test, alpha, blocks, seed), alpha);
        };
    }

    /**
     * Split-conformal p-values.
     *
     * @param score the score
     * @param reference inliers (half train, half calibrate)
     * @param test the batch
     * @param seed the split seed
     * @return a p-value per test row
     */
    public static double[] splitPValues(ScoreFunction score, double[][] reference, double[][] test, long seed) {
        int n = reference.length;
        int[] perm = permutation(n, new SplittableRandom(seed));
        int half = n / 2;
        double[][] tr = new double[half][];
        double[][] cal = new double[n - half][];
        for (int i = 0; i < n; i++) {
            if (i < half) {
                tr[i] = reference[perm[i]];
            } else {
                cal[i - half] = reference[perm[i]];
            }
        }
        ToDoubleFunction<double[]> f = score.fit(tr);
        double[] cs = new double[cal.length];
        for (int i = 0; i < cal.length; i++) {
            cs[i] = f.applyAsDouble(cal[i]);
        }
        double[] p = new double[test.length];
        for (int j = 0; j < test.length; j++) {
            double s = f.applyAsDouble(test[j]);
            int ge = 0;
            for (double c : cs) {
                if (c >= s) {
                    ge++;
                }
            }
            p[j] = (1.0 + ge) / (cal.length + 1);
        }
        return p;
    }

    /**
     * Full-conformal e-values with {@code K} blocks.
     *
     * @param score the score
     * @param reference inliers
     * @param test the batch
     * @param alphaTilde the level that sets the threshold
     * @param blocks the number of blocks
     * @param seed the block assignment seed
     * @return an e-value per test row
     */
    public static double[] fullEValues(ScoreFunction score, double[][] reference, double[][] test, double alphaTilde,
            int blocks, long seed) {
        int n = reference.length;
        int m = test.length;
        int[] perm = permutation(m, new SplittableRandom(seed));
        double[] e = new double[m];
        int k = Math.max(1, Math.min(blocks, m));
        for (int b = 0; b < k; b++) {
            List<Integer> block = new ArrayList<>();
            for (int a = 0; a < m; a++) {
                if (a % k == b) {
                    block.add(perm[a]);
                }
            }
            double[][] train = new double[n + block.size()][];
            System.arraycopy(reference, 0, train, 0, n);
            for (int a = 0; a < block.size(); a++) {
                train[n + a] = test[block.get(a)];
            }
            ToDoubleFunction<double[]> f = score.fit(train);
            double[] vr = new double[n];
            double[] vt = new double[m];
            for (int i = 0; i < n; i++) {
                vr[i] = f.applyAsDouble(reference[i]);
            }
            for (int j = 0; j < m; j++) {
                vt[j] = f.applyAsDouble(test[j]);
            }
            double t = threshold(vr, vt, alphaTilde);
            int refAbove = 0;
            for (double v : vr) {
                if (v >= t) {
                    refAbove++;
                }
            }
            for (int j : block) {
                e[j] = vt[j] >= t ? (n + 1.0) / (1 + refAbove) : 0;
            }
        }
        return e;
    }

    private static double threshold(double[] vr, double[] vt, double alpha) {
        int n = vr.length;
        int m = vt.length;
        double[] all = new double[n + m];
        System.arraycopy(vr, 0, all, 0, n);
        System.arraycopy(vt, 0, all, n, m);
        java.util.Arrays.sort(all);
        for (double t : all) {
            int ra = 0;
            int ta = 0;
            for (double v : vr) {
                if (v >= t) {
                    ra++;
                }
            }
            for (double v : vt) {
                if (v >= t) {
                    ta++;
                }
            }
            if ((double) m / (n + 1) * (1 + ra) / Math.max(1, ta) <= alpha) {
                return t;
            }
        }
        return Double.POSITIVE_INFINITY;
    }

    private static int[] permutation(int n, SplittableRandom r) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        return a;
    }
}
