package io.github.zzuegg.brainiac.experimental.boost;

import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * CatBoost's fixes for <em>prediction shift</em> (Prokhorenkova et al., "CatBoost: unbiased
 * boosting with categorical features", NeurIPS 2018,
 * <a href="https://arxiv.org/abs/1706.09516">arXiv:1706.09516</a>). Replacing a category by the
 * mean target of its rows leaks each row's own label, and so do the residuals of ordinary
 * boosting, whose trees were fitted on the very row they are evaluated on. CatBoost draws random
 * permutations of the training rows and only ever looks at rows <em>earlier</em> in a
 * permutation:
 * <ul>
 *   <li><em>ordered target statistics</em>: a row's category value becomes
 *       {@code (sum of earlier targets in the category + a * prior) / (earlier count + a)};</li>
 *   <li><em>ordered boosting</em>: a row's gradient comes from a supporting model trained only on
 *       the rows before it. As in the paper's implementation, supporting models exist for prefixes
 *       of length {@code 2^k}, and a row at position {@code p} uses the one for
 *       {@code 2^floor(log2 p)}; each tree's structure is chosen on one permutation and then
 *       every permutation's supporting models get their own leaf values;</li>
 *   <li><em>oblivious trees</em>: every node at one depth uses the same split, so a tree is a
 *       table indexed by {@code depth} bits;</li>
 *   <li>optional <em>combinations</em> of categorical columns (all pairs) as extra categories.</li>
 * </ul>
 * {@link Mode#PLAIN} keeps the ordered statistics but boosts on the final model's own residuals
 * (the paper's "Plain" mode). Static helpers give the leaky and out-of-fold target encodings that
 * baselines feed to other boosters.
 */
public final class OrderedBoosting {

    /** How gradients are computed. */
    public enum Mode {
        /** Gradients from supporting models that never saw the row. */
        ORDERED,
        /** Gradients from the final model, as ordinary boosting. */
        PLAIN
    }

    /**
     * Settings.
     *
     * @param trees boosting rounds
     * @param depth oblivious tree depth
     * @param learningRate the shrinkage
     * @param lambda L2 regularisation of leaf values
     * @param permutations random permutations for ordered statistics and boosting
     * @param priorWeight the weight {@code a} of the prior in target statistics
     * @param combinations whether pairs of categorical columns are added as categories
     * @param bins thresholds per numeric column
     * @param mode ordered or plain boosting
     * @param seed the permutations' seed
     */
    public record Settings(int trees, int depth, double learningRate, double lambda, int permutations,
            double priorWeight, boolean combinations, int bins, Mode mode, long seed) {

        /** @return 200 depth-4 trees, learning rate 0.1, 4 permutations, prior weight 1 */
        public static Settings defaults() {
            return new Settings(200, 4, 0.1, 3, 4, 1, false, 32, Mode.ORDERED, 1);
        }

        /**
         * @param m the mode
         * @return a copy with that mode
         */
        public Settings withMode(Mode m) {
            return new Settings(trees, depth, learningRate, lambda, permutations, priorWeight, combinations, bins, m,
                    seed);
        }

        /**
         * @param n rounds
         * @return a copy with that many trees
         */
        public Settings withTrees(int n) {
            return new Settings(n, depth, learningRate, lambda, permutations, priorWeight, combinations, bins, mode,
                    seed);
        }

        /**
         * @param on whether pairs of categorical columns are added
         * @return a copy with that option
         */
        public Settings withCombinations(boolean on) {
            return new Settings(trees, depth, learningRate, lambda, permutations, priorWeight, on, bins, mode, seed);
        }
    }

    /** An oblivious tree: one (feature, threshold) per level and {@code 2^depth} leaf values. */
    private record Oblivious(int[] feature, double[] threshold, double[] leaf) {
        int index(double[] row) {
            int k = 0;
            for (int d = 0; d < feature.length; d++) {
                k = 2 * k + (row[feature[d]] >= threshold[d] ? 1 : 0);
            }
            return k;
        }
    }

    private final GradientBoostedTrees.Loss loss;
    private final double base;
    private final List<Oblivious> trees;
    private final double learningRate;
    private final int numeric;
    private final List<Map<Long, double[]>> stats;   // per categorical column: key -> {sum, count}
    private final int[][] pairs;
    private final double prior;
    private final double priorWeight;

    private OrderedBoosting(GradientBoostedTrees.Loss loss, double base, List<Oblivious> trees, double learningRate,
            int numeric, List<Map<Long, double[]>> stats, int[][] pairs, double prior, double priorWeight) {
        this.loss = loss;
        this.base = base;
        this.trees = trees;
        this.learningRate = learningRate;
        this.numeric = numeric;
        this.stats = stats;
        this.pairs = pairs;
        this.prior = prior;
        this.priorWeight = priorWeight;
    }

    /**
     * Trains.
     *
     * @param numeric numeric features {@code [rows][numeric]} (may have zero columns)
     * @param categorical category ids {@code [rows][columns]}
     * @param y targets (0/1 for a binary loss)
     * @param loss the loss's derivatives
     * @param s the settings
     * @return the model
     */
    public static OrderedBoosting fit(double[][] numeric, int[][] categorical, double[] y,
            GradientBoostedTrees.Loss loss, Settings s) {
        int n = y.length;
        int[][] pairs = s.combinations() ? pairs(categorical[0].length) : new int[0][];
        long[][] keys = keys(categorical, pairs);
        int cats = keys[0].length;
        int num = numeric[0].length;
        int features = num + cats;
        double prior = Arrays.stream(y).average().orElse(0);
        SplittableRandom random = new SplittableRandom(s.seed());
        int perms = Math.max(1, s.permutations());
        // permutation 0 is used by the final model, 1.. by the supporting models
        int[][] position = new int[perms + 1][];
        double[][][] rows = new double[perms + 1][][];
        for (int p = 0; p <= perms; p++) {
            int[] order = permutation(n, random);
            position[p] = new int[n];
            for (int i = 0; i < n; i++) {
                position[p][order[i]] = i;
            }
            rows[p] = new double[n][features];
            for (int i = 0; i < n; i++) {
                System.arraycopy(numeric[i], 0, rows[p][i], 0, num);
            }
            for (int c = 0; c < cats; c++) {
                Map<Long, double[]> running = new HashMap<>();
                for (int i : order) {
                    double[] acc = running.computeIfAbsent(keys[i][c], k -> new double[2]);
                    rows[p][i][num + c] = (acc[0] + s.priorWeight() * prior) / (acc[1] + s.priorWeight());
                    acc[0] += y[i];
                    acc[1] += 1;
                }
            }
        }
        double[][][] thresholds = new double[perms + 1][features][];
        for (int p = 0; p <= perms; p++) {
            for (int f = 0; f < features; f++) {
                thresholds[p][f] = LightBoost.thresholds(rows[p], f, s.bins());
            }
        }
        double base = loss.base(y);
        double[] fin = new double[n];
        Arrays.fill(fin, base);
        int levels = 1;
        while ((1 << levels) < n) {
            levels++;
        }
        // support[p][k][i]: prediction of the model on the first 2^k rows of permutation p, for rows
        // at positions below 2^(k+1)
        double[][][] support = new double[perms + 1][levels + 1][n];
        for (double[][] a : support) {
            for (double[] b : a) {
                Arrays.fill(b, base);
            }
        }
        double[] g = new double[n];
        double[] h = new double[n];
        double[] one = new double[1];
        double[] oneY = new double[1];
        double[] oneG = new double[1];
        double[] oneH = new double[1];
        List<Oblivious> trees = new ArrayList<>();
        for (int t = 0; t < s.trees(); t++) {
            int p = s.mode() == Mode.ORDERED ? 1 + t % perms : 0;
            if (s.mode() == Mode.ORDERED) {
                for (int i = 0; i < n; i++) {
                    int pos = position[p][i];
                    one[0] = pos == 0 ? base : support[p][log2(pos)][i];
                    oneY[0] = y[i];
                    loss.derivatives(one, oneY, oneG, oneH);
                    g[i] = oneG[0];
                    h[i] = oneH[0];
                }
            } else {
                loss.derivatives(fin, y, g, h);
            }
            Oblivious structure = structure(rows[p], thresholds[p], g, h, s);
            // final model: leaf values from all rows on permutation 0's statistics
            loss.derivatives(fin, y, g, h);
            double[] leaf = leafValues(structure, rows[0], g, h, n, s.lambda());
            Oblivious tree = new Oblivious(structure.feature(), structure.threshold(), leaf);
            trees.add(tree);
            for (int i = 0; i < n; i++) {
                fin[i] += s.learningRate() * leaf[tree.index(rows[0][i])];
            }
            if (s.mode() == Mode.ORDERED) {
                for (int q = 1; q <= perms; q++) {
                    int[] idx = new int[n];
                    for (int i = 0; i < n; i++) {
                        idx[i] = structure.index(rows[q][i]);
                    }
                    for (int k = 0; (1 << k) < n && k <= levels; k++) {
                        int prefix = 1 << k;
                        double[] pred = support[q][k];
                        double[] gk = new double[n];
                        double[] hk = new double[n];
                        loss.derivatives(pred, y, gk, hk);
                        double[] sg = new double[structure.leaf().length];
                        double[] sh = new double[sg.length];
                        for (int i = 0; i < n; i++) {
                            if (position[q][i] < prefix) {
                                sg[idx[i]] += gk[i];
                                sh[idx[i]] += hk[i];
                            }
                        }
                        for (int i = 0; i < n; i++) {
                            if (position[q][i] < 2 * prefix) {
                                pred[i] -= s.learningRate() * sg[idx[i]] / (sh[idx[i]] + s.lambda());
                            }
                        }
                    }
                }
            }
        }
        List<Map<Long, double[]>> stats = new ArrayList<>();
        for (int c = 0; c < cats; c++) {
            Map<Long, double[]> m = new HashMap<>();
            for (int i = 0; i < n; i++) {
                double[] acc = m.computeIfAbsent(keys[i][c], k -> new double[2]);
                acc[0] += y[i];
                acc[1] += 1;
            }
            stats.add(m);
        }
        return new OrderedBoosting(loss, base, List.copyOf(trees), s.learningRate(), num, stats, pairs, prior,
                s.priorWeight());
    }

    private static int log2(int v) {
        return 31 - Integer.numberOfLeadingZeros(v);
    }

    private static double[] leafValues(Oblivious tree, double[][] rows, double[] g, double[] h, int n,
            double lambda) {
        double[] sg = new double[tree.leaf().length];
        double[] sh = new double[sg.length];
        for (int i = 0; i < n; i++) {
            int k = tree.index(rows[i]);
            sg[k] += g[i];
            sh[k] += h[i];
        }
        double[] v = new double[sg.length];
        for (int k = 0; k < v.length; k++) {
            v[k] = -sg[k] / (sh[k] + lambda);
        }
        return v;
    }

    /** Greedy oblivious tree: per level, the split that maximises the summed gain over all leaves. */
    private static Oblivious structure(double[][] rows, double[][] thresholds, double[] g, double[] h, Settings s) {
        int n = rows.length;
        int depth = s.depth();
        int[] leaf = new int[n];
        int[] feature = new int[depth];
        double[] threshold = new double[depth];
        int used = 0;
        for (int d = 0; d < depth; d++) {
            int leaves = 1 << d;
            double bestGain = -1;
            int bestF = -1;
            double bestT = 0;
            for (int f = 0; f < thresholds.length; f++) {
                double[] th = thresholds[f];
                if (th.length == 0) {
                    continue;
                }
                int bins = th.length + 1;
                double[] hg = new double[leaves * bins];
                double[] hh = new double[leaves * bins];
                for (int i = 0; i < n; i++) {
                    int b = Arrays.binarySearch(th, rows[i][f]);
                    b = b >= 0 ? b + 1 : -b - 1;
                    hg[leaf[i] * bins + b] += g[i];
                    hh[leaf[i] * bins + b] += h[i];
                }
                double[] gains = new double[th.length];
                for (int l = 0; l < leaves; l++) {
                    double gs = 0;
                    double hs = 0;
                    for (int b = 0; b < bins; b++) {
                        gs += hg[l * bins + b];
                        hs += hh[l * bins + b];
                    }
                    double parent = gs * gs / (hs + s.lambda());
                    double gl = 0;
                    double hl = 0;
                    for (int k = 0; k < th.length; k++) {
                        gl += hg[l * bins + k];
                        hl += hh[l * bins + k];
                        double gr = gs - gl;
                        double hr = hs - hl;
                        gains[k] += gl * gl / (hl + s.lambda()) + gr * gr / (hr + s.lambda()) - parent;
                    }
                }
                for (int k = 0; k < th.length; k++) {
                    if (gains[k] > bestGain) {
                        bestGain = gains[k];
                        bestF = f;
                        bestT = th[k];
                    }
                }
            }
            if (bestF < 0) {
                break;
            }
            feature[d] = bestF;
            threshold[d] = bestT;
            used++;
            for (int i = 0; i < n; i++) {
                leaf[i] = 2 * leaf[i] + (rows[i][bestF] >= bestT ? 1 : 0);
            }
        }
        return new Oblivious(Arrays.copyOf(feature, used), Arrays.copyOf(threshold, used), new double[1 << used]);
    }

    private static int[][] pairs(int cats) {
        List<int[]> p = new ArrayList<>();
        for (int a = 0; a < cats; a++) {
            for (int b = a + 1; b < cats; b++) {
                p.add(new int[] {a, b});
            }
        }
        return p.toArray(new int[0][]);
    }

    private static long[][] keys(int[][] categorical, int[][] pairs) {
        long[][] out = new long[categorical.length][];
        for (int i = 0; i < categorical.length; i++) {
            out[i] = keyRow(categorical[i], pairs);
        }
        return out;
    }

    private static long[] keyRow(int[] cats, int[][] pairs) {
        long[] k = new long[cats.length + pairs.length];
        for (int c = 0; c < cats.length; c++) {
            k[c] = cats[c];
        }
        for (int q = 0; q < pairs.length; q++) {
            k[cats.length + q] = ((long) cats[pairs[q][0]] << 32) | (cats[pairs[q][1]] & 0xffffffffL);
        }
        return k;
    }

    private static int[] permutation(int n, SplittableRandom random) {
        int[] p = new int[n];
        Arrays.setAll(p, i -> i);
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    /**
     * The model's feature row for new data: numeric columns, then each category's target
     * statistic over all training rows.
     *
     * @param numericRow numeric features
     * @param categoryRow category ids
     * @return the row the trees read
     */
    public double[] encode(double[] numericRow, int[] categoryRow) {
        long[] keys = keyRow(categoryRow, pairs);
        double[] row = new double[numeric + keys.length];
        System.arraycopy(numericRow, 0, row, 0, numeric);
        for (int c = 0; c < keys.length; c++) {
            double[] acc = stats.get(c).get(keys[c]);
            double sum = acc == null ? 0 : acc[0];
            double count = acc == null ? 0 : acc[1];
            row[numeric + c] = (sum + priorWeight * prior) / (count + priorWeight);
        }
        return row;
    }

    /**
     * @param numericRow numeric features
     * @param categoryRow category ids
     * @return the raw score
     */
    public double raw(double[] numericRow, int[] categoryRow) {
        double[] row = encode(numericRow, categoryRow);
        double r = base;
        for (Oblivious t : trees) {
            r += learningRate * t.leaf()[t.index(row)];
        }
        return r;
    }

    /**
     * @param numericRow numeric features
     * @param categoryRow category ids
     * @return the loss's link of the raw score
     */
    public double predict(double[] numericRow, int[] categoryRow) {
        return loss.link(raw(numericRow, categoryRow));
    }

    /**
     * Leaky target encoding: each category replaced by the smoothed mean target of all its rows,
     * the row itself included (the encoding CatBoost was built to replace).
     *
     * @param train training categories {@code [rows][columns]}
     * @param y training targets
     * @param apply rows to encode (pass {@code train} for the training rows themselves)
     * @param priorWeight the prior's weight {@code a}
     * @return encoded columns {@code [rows][columns]}
     */
    public static double[][] targetEncode(int[][] train, double[] y, int[][] apply, double priorWeight) {
        int cols = train[0].length;
        double prior = Arrays.stream(y).average().orElse(0);
        double[][] out = new double[apply.length][cols];
        for (int c = 0; c < cols; c++) {
            Map<Integer, double[]> m = new HashMap<>();
            for (int i = 0; i < train.length; i++) {
                double[] acc = m.computeIfAbsent(train[i][c], k -> new double[2]);
                acc[0] += y[i];
                acc[1]++;
            }
            for (int i = 0; i < apply.length; i++) {
                double[] acc = m.get(apply[i][c]);
                out[i][c] = acc == null ? prior : (acc[0] + priorWeight * prior) / (acc[1] + priorWeight);
            }
        }
        return out;
    }

    /**
     * Out-of-fold target encoding of the training rows: each row is encoded from the other
     * folds only.
     *
     * @param train training categories
     * @param y training targets
     * @param folds the number of folds
     * @param priorWeight the prior's weight
     * @param seed the fold assignment seed
     * @return encoded training columns
     */
    public static double[][] outOfFoldEncode(int[][] train, double[] y, int folds, double priorWeight, long seed) {
        int n = train.length;
        int[] fold = new int[n];
        int[] perm = permutation(n, new SplittableRandom(seed));
        for (int i = 0; i < n; i++) {
            fold[perm[i]] = i % folds;
        }
        double[][] out = new double[n][];
        for (int f = 0; f < folds; f++) {
            List<Integer> in = new ArrayList<>();
            List<Integer> outRows = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                (fold[i] == f ? outRows : in).add(i);
            }
            int[][] fitX = in.stream().map(i -> train[i]).toArray(int[][]::new);
            double[] fitY = in.stream().mapToDouble(i -> y[i]).toArray();
            int[][] applyX = outRows.stream().map(i -> train[i]).toArray(int[][]::new);
            double[][] enc = targetEncode(fitX, fitY, applyX, priorWeight);
            for (int k = 0; k < outRows.size(); k++) {
                out[outRows.get(k)] = enc[k];
            }
        }
        return out;
    }

    /** @return the number of trees */
    public int size() {
        return trees.size();
    }
}
