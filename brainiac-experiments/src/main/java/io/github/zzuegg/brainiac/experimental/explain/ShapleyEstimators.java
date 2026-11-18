package io.github.zzuegg.brainiac.experimental.explain;

import io.github.zzuegg.brainiac.tensor.Matrices;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Model-agnostic Shapley value estimators under a common budget of coalition evaluations.
 *
 * <ul>
 *   <li>{@link Estimator#PERMUTATION} (Štrumbelj and Kononenko, KAIS 2014): the mean marginal
 *       contribution of each feature along random orders; each order costs {@code n - 1}
 *       evaluations once {@code v(empty)} and {@code v(N)} are known.</li>
 *   <li>{@link Estimator#ANTITHETIC_PERMUTATION}: every order is paired with its reverse, whose
 *       contributions are negatively correlated, cutting the variance.</li>
 *   <li>{@link Estimator#KERNEL_SHAP_PAIRED} (Covert and Lee, AISTATS 2021,
 *       <a href="https://arxiv.org/abs/2012.01536">arXiv:2012.01536</a>): KernelSHAP with
 *       coalitions sampled by the Shapley kernel and each one's complement added, solved as the
 *       efficiency-constrained least-squares problem.</li>
 *   <li>{@link Estimator#LEVERAGE_SHAP} (Musco and Witter, ICLR 2025,
 *       <a href="https://arxiv.org/abs/2410.01917">arXiv:2410.01917</a>): the same regression
 *       with the constraint projected out ({@code A = Z P}, {@code P = I - 11^T / n}); the rows
 *       of {@code A} have leverage scores {@code 1 / C(n, |z|)}, so coalitions are sampled by
 *       leverage: every size gets equal mass, pairs {@code (z, complement)} without replacement,
 *       each row reweighted by its kernel weight over its inclusion probability. The counts per
 *       size are the stratified deterministic ones of the authors' code.</li>
 * </ul>
 */
public final class ShapleyEstimators {

    private ShapleyEstimators() {
    }

    /** The estimators. */
    public enum Estimator {
        /** Plain permutation sampling. */
        PERMUTATION,
        /** Permutations paired with their reverses. */
        ANTITHETIC_PERMUTATION,
        /** KernelSHAP with paired kernel sampling. */
        KERNEL_SHAP_PAIRED,
        /** Leverage-score sampled KernelSHAP. */
        LEVERAGE_SHAP
    }

    /**
     * Estimates Shapley values.
     *
     * @param estimator the estimator
     * @param game the game
     * @param n the number of players
     * @param budget coalition evaluations, including {@code v(empty)} and {@code v(N)}
     * @param random the random source
     * @return one value per player
     */
    public static double[] estimate(Estimator estimator, CoalitionGame game, int n, int budget,
            SplittableRandom random) {
        return switch (estimator) {
            case PERMUTATION -> permutation(game, n, budget, false, random);
            case ANTITHETIC_PERMUTATION -> permutation(game, n, budget, true, random);
            case KERNEL_SHAP_PAIRED -> kernelPaired(game, n, budget, random);
            case LEVERAGE_SHAP -> leverage(game, n, budget, random);
        };
    }

    private static double[] permutation(CoalitionGame game, int n, int budget, boolean antithetic,
            SplittableRandom random) {
        int orders = Math.max(antithetic ? 2 : 1, (budget - 2) / Math.max(1, n - 1));
        if (antithetic) {
            orders -= orders % 2;
        }
        int[][] perms = new int[orders][];
        for (int k = 0; k < orders; k++) {
            if (antithetic && k % 2 == 1) {
                int[] prev = perms[k - 1];
                int[] rev = new int[n];
                for (int i = 0; i < n; i++) {
                    rev[i] = prev[n - 1 - i];
                }
                perms[k] = rev;
            } else {
                perms[k] = shuffled(n, random);
            }
        }
        boolean[][] batch = new boolean[2 + orders * (n - 1)][];
        batch[0] = new boolean[n];
        batch[1] = new boolean[n];
        java.util.Arrays.fill(batch[1], true);
        int b = 2;
        for (int[] perm : perms) {
            boolean[] s = new boolean[n];
            for (int i = 0; i < n - 1; i++) {
                s[perm[i]] = true;
                batch[b++] = s.clone();
            }
        }
        double[] v = game.values(batch);
        double[] phi = new double[n];
        b = 2;
        for (int[] perm : perms) {
            double prev = v[0];
            for (int i = 0; i < n; i++) {
                double cur = i == n - 1 ? v[1] : v[b++];
                phi[perm[i]] += (cur - prev) / orders;
                prev = cur;
            }
        }
        return phi;
    }

    private static int[] shuffled(int n, SplittableRandom random) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int k = random.nextInt(i + 1);
            int t = p[i];
            p[i] = p[k];
            p[k] = t;
        }
        return p;
    }

    private static double binomial(int n, int k) {
        double r = 1;
        for (int i = 1; i <= k; i++) {
            r = r * (n - k + i) / i;
        }
        return r;
    }

    /** The KernelSHAP weight of one coalition of size {@code s}. */
    private static double kernelWeight(int n, int s) {
        return 1.0 / (binomial(n, s) * s * (n - s));
    }

    private static double[] kernelPaired(CoalitionGame game, int n, int budget, SplittableRandom random) {
        double[] q = new double[n];
        double total = 0;
        for (int s = 1; s < n; s++) {
            q[s] = 1.0 / (s * (n - s));
            total += q[s];
        }
        List<boolean[]> rows = new ArrayList<>();
        while (rows.size() + 2 <= budget - 2) {
            double u = random.nextDouble() * total;
            int size = 1;
            while (size < n - 1 && u > q[size]) {
                u -= q[size];
                size++;
            }
            boolean[] z = ShapleyInteractions.randomOfSize(n, size, random);
            boolean[] c = new boolean[n];
            for (int i = 0; i < n; i++) {
                c[i] = !z[i];
            }
            rows.add(z);
            rows.add(c);
        }
        double[] w = new double[rows.size()];
        java.util.Arrays.fill(w, 1.0);
        return solve(game, n, rows, w);
    }

    private static double[] leverage(CoalitionGame game, int n, int budget, SplittableRandom random) {
        int m = budget - 2;
        // stratified counts: size s (paired with n - s) gets min(C(n,s), 2c) coalitions
        double lo = 0;
        double hi = Math.max(1, m);
        for (int it = 0; it < 100; it++) {
            double c = (lo + hi) / 2;
            double e = 0;
            for (int s = 1; s < n; s++) {
                e += Math.min(binomial(n, s), 2 * c);
            }
            if (e > m) {
                hi = c;
            } else {
                lo = c;
            }
        }
        double c = lo;
        List<boolean[]> rows = new ArrayList<>();
        List<Double> weights = new ArrayList<>();
        for (int s = 1; s <= n / 2; s++) {
            double count = binomial(n, s);
            double take = Math.min(count, 2 * c);
            int k = (int) Math.floor(take);
            if (random.nextDouble() < take - k) {
                k++;
            }
            if (k == 0) {
                continue;
            }
            boolean middle = 2 * s == n;
            // pairs (z, complement): for the middle size a pair covers two coalitions of that size
            int pairs = middle ? Math.max(1, k / 2) : k;
            double inclusion = middle ? Math.min(1, 2.0 * pairs / count) : Math.min(1, pairs / count);
            List<boolean[]> chosen = distinctOfSize(n, s, pairs, middle, random);
            double weight = kernelWeight(n, s) / inclusion;
            for (boolean[] z : chosen) {
                boolean[] comp = new boolean[n];
                for (int i = 0; i < n; i++) {
                    comp[i] = !z[i];
                }
                rows.add(z);
                weights.add(weight);
                rows.add(comp);
                weights.add(weight);
            }
        }
        double[] w = new double[weights.size()];
        for (int i = 0; i < w.length; i++) {
            w[i] = weights.get(i);
        }
        return solve(game, n, rows, w);
    }

    /** Distinct coalitions of one size, none the complement of another when {@code middle}. */
    private static List<boolean[]> distinctOfSize(int n, int s, int k, boolean middle, SplittableRandom random) {
        double count = binomial(n, s);
        Set<String> seen = new HashSet<>();
        List<boolean[]> out = new ArrayList<>();
        int limit = (int) Math.min(k, middle ? count / 2 : count);
        int attempts = 0;
        while (out.size() < limit && attempts < 100 * limit + 100) {
            attempts++;
            boolean[] z = ShapleyInteractions.randomOfSize(n, s, random);
            if (middle && !z[0]) {
                for (int i = 0; i < n; i++) {
                    z[i] = !z[i];
                }
            }
            String key = java.util.Arrays.toString(z);
            if (seen.add(key)) {
                out.add(z);
            }
        }
        return out;
    }

    /**
     * The projected weighted regression: {@code phi = phi_perp + (v(N) - v(0)) / n}, where
     * {@code phi_perp} minimises {@code ||W^1/2 (Z P x - b)||} with
     * {@code b = v(z) - v(0) - |z| (v(N) - v(0)) / n}.
     */
    private static double[] solve(CoalitionGame game, int n, List<boolean[]> rows, double[] w) {
        boolean[][] batch = new boolean[rows.size() + 2][];
        batch[0] = new boolean[n];
        batch[1] = new boolean[n];
        java.util.Arrays.fill(batch[1], true);
        for (int r = 0; r < rows.size(); r++) {
            batch[r + 2] = rows.get(r);
        }
        double[] v = game.values(batch);
        double empty = v[0];
        double gap = v[1] - empty;
        double[][] ata = new double[n][n];
        double[] atb = new double[n];
        double[] a = new double[n];
        for (int r = 0; r < rows.size(); r++) {
            boolean[] z = rows.get(r);
            int size = 0;
            for (boolean b : z) {
                size += b ? 1 : 0;
            }
            double frac = (double) size / n;
            for (int i = 0; i < n; i++) {
                a[i] = (z[i] ? 1 : 0) - frac;   // the row of Z P
            }
            double b = v[r + 2] - empty - size * gap / n;
            for (int i = 0; i < n; i++) {
                atb[i] += w[r] * a[i] * b;
                for (int j = 0; j < n; j++) {
                    ata[i][j] += w[r] * a[i] * a[j];
                }
            }
        }
        double scale = 0;
        for (int i = 0; i < n; i++) {
            scale = Math.max(scale, ata[i][i]);
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                ata[i][j] += scale / n;   // pins the component along 1 to zero
            }
            ata[i][i] += 1e-12 * scale;
        }
        double[] x = Matrices.solve(ata, atb);
        double mean = java.util.Arrays.stream(x).average().orElse(0);
        double[] phi = new double[n];
        for (int i = 0; i < n; i++) {
            phi[i] = x[i] - mean + gap / n;
        }
        return phi;
    }
}
