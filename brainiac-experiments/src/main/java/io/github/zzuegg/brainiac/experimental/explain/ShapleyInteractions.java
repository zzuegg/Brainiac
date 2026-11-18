package io.github.zzuegg.brainiac.experimental.explain;

import io.github.zzuegg.brainiac.tensor.Matrices;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Pairwise Shapley interaction indices, the order-2 explanations that first-order Shapley
 * values hide.
 *
 * <ul>
 *   <li>{@link Index#SII}, the Shapley interaction index (Grabisch and Roubens, IJGT 1999):
 *       {@code I(ij) = sum_{T in N\ij} (n-t-2)! t! / (n-1)! Delta_ij(T)} with the discrete
 *       derivative {@code Delta_ij(T) = v(T+ij) - v(T+i) - v(T+j) + v(T)}; the first-order
 *       terms are the Shapley values.</li>
 *   <li>{@link Index#K_SII}, the k-Shapley values for k = 2 (Bord et al. 2023): the SII pairs
 *       with singletons {@code phi_i - 1/2 sum_j I(ij)} (the Bernoulli-number aggregation), an
 *       efficient explanation summing to {@code v(N) - v(empty)}.</li>
 *   <li>{@link Index#STII}, the Shapley-Taylor index of order 2 (Sundararajan et al. 2020):
 *       singletons {@code v(i) - v(empty)}, pairs {@code (2/n) sum_T Delta_ij(T) / C(n-1, t)}.</li>
 *   <li>{@link Index#FAITH_SHAP} (Tsai, Yeh and Ravikumar, JMLR 2023,
 *       <a href="https://arxiv.org/abs/2203.00870">arXiv:2203.00870</a>): the order-2 polynomial
 *       {@code v(T) ~ v(empty) + sum_{S in T, |S| <= 2} beta_S} fitted by least squares under
 *       the Shapley kernel, constrained to be exact at the full coalition.</li>
 * </ul>
 *
 * <p>Methods: {@link Method#EXACT} enumerates all {@code 2^n} coalitions;
 * {@link Method#SHAP_IQ} (Fumagalli et al., NeurIPS 2023,
 * <a href="https://arxiv.org/abs/2303.01179">arXiv:2303.01179</a>) writes every cardinal
 * interaction index as {@code sum_T v(T) gamma_S(t, |T and S|)} and estimates all of them from
 * one shared sample of coalitions (sizes drawn by the Shapley kernel, the border sizes that the
 * budget can cover enumerated exactly, values centred at {@code v(empty)}); {@link Method#PERMUTATION}
 * is the per-pair permutation baseline (four evaluations per sample per pair); {@link Method#TREE}
 * is exact for a {@link TreeView}: for every tree and background row the interventional game
 * depends only on the features where the row and the reference split differently, so the indices
 * are computed on that small sub-game and summed (SII and the Shapley value are invariant to the
 * null players left out), the role TreeSHAP-IQ (Muschalik et al., AAAI 2024) plays.
 */
public final class ShapleyInteractions {

    private ShapleyInteractions() {
    }

    /** The interaction index. */
    public enum Index {
        /** Shapley interaction index. */
        SII,
        /** k-Shapley values, k = 2. */
        K_SII,
        /** Shapley-Taylor interaction index, order 2. */
        STII,
        /** Faithful Shapley interaction index, order 2. */
        FAITH_SHAP
    }

    /** How the index is computed. */
    public enum Method {
        /** All coalitions. */
        EXACT,
        /** One shared coalition sample for all pairs. */
        SHAP_IQ,
        /** Per-pair permutation sampling (SII pairs only; singletons are zero). */
        PERMUTATION,
        /** Exact on tree ensembles (SII and K_SII). */
        TREE
    }

    /**
     * An order-2 explanation.
     *
     * @param first the first-order terms
     * @param pairs the symmetric pair terms, zero diagonal
     * @param evaluations the coalitions valued
     */
    public record Interactions(double[] first, double[][] pairs, long evaluations) {

        /**
         * The pairs ranked by absolute interaction, strongest first.
         *
         * @return {@code [rank][2]} feature pairs
         */
        public int[][] ranking() {
            int n = first.length;
            List<int[]> all = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    all.add(new int[] {i, j});
                }
            }
            all.sort((a, b) -> Double.compare(Math.abs(pairs[b[0]][b[1]]), Math.abs(pairs[a[0]][a[1]])));
            return all.toArray(new int[0][]);
        }

        /** @return the sum of all terms, {@code v(N) - v(empty)} for efficient indices */
        public double total() {
            double s = 0;
            for (int i = 0; i < first.length; i++) {
                s += first[i];
                for (int j = i + 1; j < first.length; j++) {
                    s += pairs[i][j];
                }
            }
            return s;
        }
    }

    private static double[] factorials(int n) {
        double[] f = new double[n + 2];
        f[0] = 1;
        for (int i = 1; i < f.length; i++) {
            f[i] = f[i - 1] * i;
        }
        return f;
    }

    private static double binomial(int n, int k) {
        if (k < 0 || k > n) {
            return 0;
        }
        double r = 1;
        for (int i = 1; i <= k; i++) {
            r = r * (n - k + i) / i;
        }
        return r;
    }

    // ------------------------------------------------------------------ exact

    /**
     * Exact indices by enumerating every coalition.
     *
     * @param index the index
     * @param game the game
     * @param n the number of players, at most about 14
     * @return the explanation
     */
    public static Interactions exact(Index index, CoalitionGame game, int n) {
        double[] v = CoalitionGame.table(game, n);
        Interactions r = fromTable(index, v, n);
        return new Interactions(r.first(), r.pairs(), 1L << n);
    }

    /** Indices of a game given as a full table indexed by bit mask. */
    static Interactions fromTable(Index index, double[] v, int n) {
        if (index == Index.FAITH_SHAP) {
            int full = 1 << n;
            List<boolean[]> rows = new ArrayList<>();
            List<Double> values = new ArrayList<>();
            List<Double> weights = new ArrayList<>();
            for (int s = 1; s < full - 1; s++) {
                rows.add(CoalitionGame.members(s, n));
                values.add(v[s]);
                weights.add(kernel(n, Integer.bitCount(s)));
            }
            return faith(rows, values, weights, v[0], v[full - 1], n, full);
        }
        double[] f = factorials(n);
        double[] first = new double[n];
        double[][] pairs = new double[n][n];
        int full = 1 << n;
        for (int t = 0; t < full; t++) {
            int size = Integer.bitCount(t);
            for (int i = 0; i < n; i++) {
                int bi = 1 << i;
                if ((t & bi) != 0) {
                    continue;
                }
                if (index != Index.STII) {
                    first[i] += f[size] * f[n - size - 1] / f[n] * (v[t | bi] - v[t]);
                }
                for (int j = i + 1; j < n; j++) {
                    int bj = 1 << j;
                    if ((t & bj) != 0) {
                        continue;
                    }
                    double delta = v[t | bi | bj] - v[t | bi] - v[t | bj] + v[t];
                    double w = index == Index.STII ? 2.0 / n / binomial(n - 1, size)
                            : f[size] * f[n - size - 2] / f[n - 1];
                    pairs[i][j] += w * delta;
                }
            }
        }
        symmetrise(pairs);
        if (index == Index.STII) {
            for (int i = 0; i < n; i++) {
                first[i] = v[1 << i] - v[0];
            }
        } else if (index == Index.K_SII) {
            toKsii(first, pairs);
        }
        return new Interactions(first, pairs, full);
    }

    private static void symmetrise(double[][] pairs) {
        for (int i = 0; i < pairs.length; i++) {
            for (int j = i + 1; j < pairs.length; j++) {
                pairs[j][i] = pairs[i][j];
            }
        }
    }

    private static void toKsii(double[] first, double[][] pairs) {
        for (int i = 0; i < first.length; i++) {
            double s = 0;
            for (int j = 0; j < first.length; j++) {
                if (j != i) {
                    s += pairs[i][j];
                }
            }
            first[i] -= s / 2;
        }
    }

    /** The Shapley kernel weight of a coalition of size {@code t}. */
    private static double kernel(int n, int t) {
        return (n - 1) / (binomial(n, t) * t * (n - t));
    }

    /** The constrained weighted least squares of Faith-Shap over the given rows. */
    private static Interactions faith(List<boolean[]> rows, List<Double> values, List<Double> weights, double empty,
            double fullValue, int n, long evaluations) {
        int pairCount = n * (n - 1) / 2;
        int m = n + pairCount;
        double[][] ata = new double[m + 1][m + 1];
        double[] atb = new double[m + 1];
        double[] feat = new double[m];
        for (int r = 0; r < rows.size(); r++) {
            boolean[] s = rows.get(r);
            int k = 0;
            for (int i = 0; i < n; i++) {
                feat[k++] = s[i] ? 1 : 0;
            }
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    feat[k++] = s[i] && s[j] ? 1 : 0;
                }
            }
            double w = weights.get(r);
            double y = values.get(r) - empty;
            for (int a = 0; a < m; a++) {
                if (feat[a] == 0) {
                    continue;
                }
                atb[a] += w * y;
                for (int b = 0; b < m; b++) {
                    ata[a][b] += w * feat[b];
                }
            }
        }
        for (int a = 0; a < m; a++) {
            ata[a][a] += 1e-10;
            ata[a][m] = 1;
            ata[m][a] = 1;
        }
        atb[m] = fullValue - empty;
        double[] beta = Matrices.solve(ata, atb);
        double[] first = new double[n];
        double[][] pairs = new double[n][n];
        int k = 0;
        for (int i = 0; i < n; i++) {
            first[i] = beta[k++];
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                pairs[i][j] = beta[k++];
            }
        }
        symmetrise(pairs);
        return new Interactions(first, pairs, evaluations);
    }

    // ------------------------------------------------------------------ SHAP-IQ

    /**
     * SHAP-IQ: every index estimated from one shared sample of about {@code budget} coalitions.
     *
     * @param index the index
     * @param game the game
     * @param n the number of players
     * @param budget the number of coalitions to value
     * @param random the random source
     * @return the estimate
     */
    public static Interactions shapIq(Index index, CoalitionGame game, int n, int budget, SplittableRandom random) {
        // sizes 1..n-1 drawn by the Shapley kernel; enumerate border sizes the budget covers
        double[] q = new double[n + 1];
        for (int t = 1; t < n; t++) {
            q[t] = 1.0 / (t * (n - t));
        }
        boolean[] exactSize = new boolean[n + 1];
        exactSize[0] = true;
        exactSize[n] = true;
        int remaining = budget - 2 - (index == Index.STII ? n : 0);
        boolean changed = true;
        while (changed) {
            changed = false;
            double total = 0;
            for (int t = 1; t < n; t++) {
                if (!exactSize[t]) {
                    total += q[t];
                }
            }
            for (int lo = 1, hi = n - 1; lo <= hi; lo++, hi--) {
                if (exactSize[lo]) {
                    continue;
                }
                double count = binomial(n, lo) + (lo == hi ? 0 : binomial(n, hi));
                double expected = remaining * (q[lo] + (lo == hi ? 0 : q[hi])) / total;
                if (count <= expected && count <= remaining) {
                    exactSize[lo] = true;
                    exactSize[hi] = true;
                    remaining -= (int) count;
                    changed = true;
                    break;
                }
            }
        }
        List<boolean[]> coalitions = new ArrayList<>();
        List<Double> weight = new ArrayList<>();   // 1 for enumerated, 1/(K p(T)) for sampled
        for (int t = 0; t <= n; t++) {
            if (exactSize[t]) {
                for (long mask : masksOfSize(n, t)) {
                    coalitions.add(CoalitionGame.members(mask, n));
                    weight.add(1.0);
                }
            }
        }
        double total = 0;
        for (int t = 1; t < n; t++) {
            if (!exactSize[t]) {
                total += q[t];
            }
        }
        int samples = Math.max(0, remaining);
        if (total > 0) {
            for (int k = 0; k < samples; k++) {
                double u = random.nextDouble() * total;
                int t = -1;
                for (int s = 1; s < n; s++) {
                    if (!exactSize[s]) {
                        t = s;
                        u -= q[s];
                        if (u <= 0) {
                            break;
                        }
                    }
                }
                coalitions.add(randomOfSize(n, t, random));
                double p = q[t] / total / binomial(n, t);
                weight.add(1.0 / (samples * p));
            }
        }
        int extra = index == Index.STII ? n : 0;
        boolean[][] batch = new boolean[coalitions.size() + extra][];
        for (int c = 0; c < coalitions.size(); c++) {
            batch[c] = coalitions.get(c);
        }
        for (int i = 0; i < extra; i++) {
            batch[coalitions.size() + i] = CoalitionGame.members(1L << i, n);
        }
        double[] v = game.values(batch);
        double empty = 0;
        double fullValue = 0;
        for (int c = 0; c < coalitions.size(); c++) {
            int size = count(coalitions.get(c));
            if (size == 0) {
                empty = v[c];
            } else if (size == n) {
                fullValue = v[c];
            }
        }
        long evaluations = batch.length;
        if (index == Index.FAITH_SHAP) {
            List<boolean[]> rows = new ArrayList<>();
            List<Double> values = new ArrayList<>();
            List<Double> weights = new ArrayList<>();
            for (int c = 0; c < coalitions.size(); c++) {
                int t = count(coalitions.get(c));
                if (t == 0 || t == n) {
                    continue;
                }
                rows.add(coalitions.get(c));
                values.add(v[c]);
                weights.add(kernel(n, t) * weight.get(c));
            }
            return faith(rows, values, weights, empty, fullValue, n, evaluations);
        }
        double[] f = factorials(n);
        double[] first = new double[n];
        double[][] pairs = new double[n][n];
        for (int c = 0; c < coalitions.size(); c++) {
            boolean[] s = coalitions.get(c);
            int t = count(s);
            double val = (v[c] - empty) * weight.get(c);
            if (index != Index.STII) {
                for (int i = 0; i < n; i++) {
                    int l = s[i] ? 1 : 0;
                    double g = (l == 1 ? 1 : -1) * f[t - l] * f[n - (t - l) - 1] / f[n];
                    if (l == 1 || t < n) {
                        first[i] += val * g;
                    }
                }
            }
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    int l = (s[i] ? 1 : 0) + (s[j] ? 1 : 0);
                    int base = t - l;
                    if (base > n - 2) {
                        continue;
                    }
                    double m = index == Index.STII ? 2.0 / n / binomial(n - 1, base)
                            : f[base] * f[n - base - 2] / f[n - 1];
                    pairs[i][j] += val * (l == 1 ? -m : m);
                }
            }
        }
        symmetrise(pairs);
        if (index == Index.STII) {
            for (int i = 0; i < n; i++) {
                first[i] = v[coalitions.size() + i] - empty;
            }
        } else if (index == Index.K_SII) {
            toKsii(first, pairs);
        }
        return new Interactions(first, pairs, evaluations);
    }

    private static int count(boolean[] s) {
        int c = 0;
        for (boolean b : s) {
            if (b) {
                c++;
            }
        }
        return c;
    }

    private static List<Long> masksOfSize(int n, int t) {
        List<Long> out = new ArrayList<>();
        collect(n, t, 0, 0L, out);
        return out;
    }

    private static void collect(int n, int t, int from, long mask, List<Long> out) {
        if (t == 0) {
            out.add(mask);
            return;
        }
        for (int i = from; i <= n - t; i++) {
            collect(n, t - 1, i + 1, mask | (1L << i), out);
        }
    }

    static boolean[] randomOfSize(int n, int t, SplittableRandom random) {
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) {
            perm[i] = i;
        }
        boolean[] s = new boolean[n];
        for (int i = 0; i < t; i++) {
            int k = i + random.nextInt(n - i);
            int tmp = perm[i];
            perm[i] = perm[k];
            perm[k] = tmp;
            s[perm[i]] = true;
        }
        return s;
    }

    // ------------------------------------------------------------------ permutation baseline

    /**
     * Per-pair permutation sampling of the SII: each sample draws the coalition preceding the
     * merged pair in a random order and values its four discrete-derivative coalitions.
     *
     * @param game the game
     * @param n the number of players
     * @param budget the total number of coalitions to value, shared evenly over the pairs
     * @param random the random source
     * @return SII pair estimates (singletons zero)
     */
    public static Interactions permutation(CoalitionGame game, int n, int budget, SplittableRandom random) {
        int pairCount = n * (n - 1) / 2;
        int perPair = Math.max(1, budget / (4 * pairCount));
        double[][] pairs = new double[n][n];
        List<boolean[]> batch = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                for (int s = 0; s < perPair; s++) {
                    int t = random.nextInt(n - 1);
                    boolean[] base = randomOfSizeExcluding(n, t, i, j, random);
                    boolean[] bi = base.clone();
                    bi[i] = true;
                    boolean[] bj = base.clone();
                    bj[j] = true;
                    boolean[] bij = bi.clone();
                    bij[j] = true;
                    batch.add(base);
                    batch.add(bi);
                    batch.add(bj);
                    batch.add(bij);
                }
            }
        }
        double[] v = game.values(batch.toArray(new boolean[0][]));
        int k = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double s = 0;
                for (int r = 0; r < perPair; r++) {
                    s += v[k + 3] - v[k + 1] - v[k + 2] + v[k];
                    k += 4;
                }
                pairs[i][j] = s / perPair;
            }
        }
        symmetrise(pairs);
        return new Interactions(new double[n], pairs, v.length);
    }

    private static boolean[] randomOfSizeExcluding(int n, int t, int a, int b, SplittableRandom random) {
        int[] others = new int[n - 2];
        for (int i = 0, k = 0; i < n; i++) {
            if (i != a && i != b) {
                others[k++] = i;
            }
        }
        boolean[] s = new boolean[n];
        for (int i = 0; i < t; i++) {
            int k = i + random.nextInt(others.length - i);
            int tmp = others[i];
            others[i] = others[k];
            others[k] = tmp;
            s[others[i]] = true;
        }
        return s;
    }

    // ------------------------------------------------------------------ trees

    /**
     * Exact interventional SII or k-SII of a tree ensemble.
     *
     * @param index {@link Index#SII} or {@link Index#K_SII}
     * @param model the ensemble
     * @param x the row
     * @param background the reference rows
     * @return the explanation ({@code evaluations} counts sub-game coalitions)
     */
    public static Interactions tree(Index index, TreeView model, double[] x, double[][] background) {
        if (index != Index.SII && index != Index.K_SII) {
            throw new IllegalArgumentException("TREE supports SII and K_SII, got " + index);
        }
        int p = model.features();
        double[] first = new double[p];
        double[][] pairs = new double[p][p];
        long evaluations = 0;
        for (TreeView.Tree t : model.trees()) {
            for (double[] z : background) {
                boolean[] differs = new boolean[p];
                mark(t, x, z, 0, differs);
                int[] players = java.util.stream.IntStream.range(0, p).filter(j -> differs[j]).toArray();
                int d = players.length;
                if (d == 0) {
                    continue;
                }
                double[] v = new double[1 << d];
                double[] row = new double[p];
                for (int s = 0; s < v.length; s++) {
                    System.arraycopy(z, 0, row, 0, p);
                    for (int k = 0; k < d; k++) {
                        if ((s & (1 << k)) != 0) {
                            row[players[k]] = x[players[k]];
                        }
                    }
                    v[s] = t.predict(row);
                }
                evaluations += v.length;
                Interactions sub = fromTable(Index.SII, v, d);
                for (int a = 0; a < d; a++) {
                    first[players[a]] += sub.first()[a];
                    for (int b = 0; b < d; b++) {
                        pairs[players[a]][players[b]] += sub.pairs()[a][b];
                    }
                }
            }
        }
        double w = model.scale() / background.length;
        for (int i = 0; i < p; i++) {
            first[i] *= w;
            for (int j = 0; j < p; j++) {
                pairs[i][j] *= w;
            }
        }
        if (index == Index.K_SII) {
            toKsii(first, pairs);
        }
        return new Interactions(first, pairs, evaluations);
    }

    private static void mark(TreeView.Tree t, double[] x, double[] z, int node, boolean[] differs) {
        int f = t.feature()[node];
        if (f < 0) {
            return;
        }
        int xc = x[f] <= t.threshold()[node] ? t.left()[node] : t.right()[node];
        int zc = z[f] <= t.threshold()[node] ? t.left()[node] : t.right()[node];
        if (xc == zc) {
            mark(t, x, z, xc, differs);
        } else {
            differs[f] = true;
            mark(t, x, z, xc, differs);
            mark(t, x, z, zc, differs);
        }
    }
}
