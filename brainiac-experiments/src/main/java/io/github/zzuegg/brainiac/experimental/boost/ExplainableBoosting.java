package io.github.zzuegg.brainiac.experimental.boost;

import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * An explainable boosting machine: a GA²M, {@code f(x) = b + sum_i f_i(x_i) + sum_(i,j) f_ij(x_i, x_j)},
 * learned by cyclic gradient boosting (Lou, Caruana, Gehrke and Hooker, "Accurate intelligible
 * models with pairwise interactions", KDD 2013,
 * <a href="https://doi.org/10.1145/2487575.2487579">doi:10.1145/2487575.2487579</a>; Nori et al.,
 * "InterpretML", 2019, <a href="https://arxiv.org/abs/1909.09223">arXiv:1909.09223</a>).
 * <ul>
 *   <li>Main effects: round-robin over the features, each step fits a tiny tree (a few leaves)
 *       on <em>one</em> feature's quantile bins to the current gradients, with a very low
 *       learning rate so the order of the features does not matter;</li>
 *   <li>FAST pair ranking: on the residuals of the main-effect model, every pair is scored by
 *       the best gain of a single 2-D cut into four quadrants (on a coarse grid with prefix
 *       sums), and the top pairs are kept;</li>
 *   <li>pair terms: cyclic boosting of the kept pairs, each step a quadrant split of a coarse
 *       2-D table.</li>
 * </ul>
 * Every term is a lookup table that can be plotted ({@link #shape}, {@link #pairValue}), so the
 * model is exactly interpretable; {@link #importance} gives each term's mean absolute
 * contribution. Losses are core {@link GradientBoostedTrees.Loss} derivatives.
 */
public final class ExplainableBoosting {

    /**
     * Settings.
     *
     * @param rounds cyclic passes over the main effects
     * @param learningRate the shrinkage (small, so feature order does not matter)
     * @param bins quantile thresholds per feature for main effects
     * @param leaves leaves per one-feature tree
     * @param interactions how many FAST-ranked pairs to add
     * @param pairRounds cyclic passes over the pairs
     * @param pairBins coarse bins per feature for FAST and the pair tables
     * @param lambda L2 regularisation of leaf values
     * @param minLeaf the least rows per leaf
     */
    public record Settings(int rounds, double learningRate, int bins, int leaves, int interactions, int pairRounds,
            int pairBins, double lambda, int minLeaf) {

        /** @return 300 rounds at 0.05, 32 bins, 3-leaf trees, 2 pairs over 100 rounds on an 8x8 grid */
        public static Settings defaults() {
            return new Settings(300, 0.05, 32, 3, 2, 100, 8, 1, 5);
        }

        /**
         * @param k how many pairs
         * @return a copy with that many interactions
         */
        public Settings withInteractions(int k) {
            return new Settings(rounds, learningRate, bins, leaves, k, pairRounds, pairBins, lambda, minLeaf);
        }

        /**
         * @param n main-effect rounds
         * @return a copy with that many rounds
         */
        public Settings withRounds(int n) {
            return new Settings(n, learningRate, bins, leaves, interactions, pairRounds, pairBins, lambda, minLeaf);
        }
    }

    private final GradientBoostedTrees.Loss loss;
    private final double intercept;
    private final double[][] thresholds;
    private final double[][] shapes;
    private final double[][] coarse;   // per feature: coarse thresholds
    private final int[][] pairs;
    private final double[][][] tables;
    private final double[] importance;
    private final int[][] ranking;
    private final double[] rankingGain;

    private ExplainableBoosting(GradientBoostedTrees.Loss loss, double intercept, double[][] thresholds,
            double[][] shapes, double[][] coarse, int[][] pairs, double[][][] tables, double[] importance,
            int[][] ranking, double[] rankingGain) {
        this.loss = loss;
        this.intercept = intercept;
        this.thresholds = thresholds;
        this.shapes = shapes;
        this.coarse = coarse;
        this.pairs = pairs;
        this.tables = tables;
        this.importance = importance;
        this.ranking = ranking;
        this.rankingGain = rankingGain;
    }

    private static int bin(double[] th, double v) {
        int b = Arrays.binarySearch(th, v);
        return b >= 0 ? b + 1 : -b - 1;
    }

    /**
     * Trains.
     *
     * @param x rows {@code [n][d]}
     * @param y targets
     * @param loss the loss's derivatives
     * @param s the settings
     * @return the model
     */
    public static ExplainableBoosting fit(double[][] x, double[] y, GradientBoostedTrees.Loss loss, Settings s) {
        int n = x.length;
        int d = x[0].length;
        double[][] thresholds = new double[d][];
        double[][] coarse = new double[d][];
        int[][] fine = new int[d][n];
        int[][] rough = new int[d][n];
        double[][] shapes = new double[d][];
        for (int f = 0; f < d; f++) {
            thresholds[f] = LightBoost.thresholds(x, f, s.bins());
            coarse[f] = LightBoost.thresholds(x, f, s.pairBins() - 1);
            shapes[f] = new double[thresholds[f].length + 1];
            for (int i = 0; i < n; i++) {
                fine[f][i] = bin(thresholds[f], x[i][f]);
                rough[f][i] = bin(coarse[f], x[i][f]);
            }
        }
        double base = loss.base(y);
        double[] raw = new double[n];
        Arrays.fill(raw, base);
        double[] g = new double[n];
        double[] h = new double[n];
        for (int round = 0; round < s.rounds(); round++) {
            for (int f = 0; f < d; f++) {
                int bins = shapes[f].length;
                if (bins < 2) {
                    continue;
                }
                loss.derivatives(raw, y, g, h);
                double[] hg = new double[bins];
                double[] hh = new double[bins];
                int[] hc = new int[bins];
                for (int i = 0; i < n; i++) {
                    hg[fine[f][i]] += g[i];
                    hh[fine[f][i]] += h[i];
                    hc[fine[f][i]]++;
                }
                double[] update = oneFeatureTree(hg, hh, hc, s);
                for (int b = 0; b < bins; b++) {
                    shapes[f][b] += s.learningRate() * update[b];
                }
                for (int i = 0; i < n; i++) {
                    raw[i] += s.learningRate() * update[fine[f][i]];
                }
            }
        }
        // FAST: rank all pairs on the main-effect residuals
        loss.derivatives(raw, y, g, h);
        List<int[]> all = new ArrayList<>();
        List<Double> gains = new ArrayList<>();
        for (int a = 0; a < d; a++) {
            for (int b = a + 1; b < d; b++) {
                all.add(new int[] {a, b});
                gains.add(fastGain(rough[a], rough[b], coarse[a].length + 1, coarse[b].length + 1, g, h, s.lambda()));
            }
        }
        Integer[] order = new Integer[all.size()];
        Arrays.setAll(order, i -> i);
        Arrays.sort(order, (p, q) -> Double.compare(gains.get(q), gains.get(p)));
        int[][] ranking = new int[order.length][];
        double[] rankingGain = new double[order.length];
        for (int k = 0; k < order.length; k++) {
            ranking[k] = all.get(order[k]);
            rankingGain[k] = gains.get(order[k]);
        }
        int keep = Math.min(s.interactions(), ranking.length);
        int[][] pairs = Arrays.copyOf(ranking, keep);
        double[][][] tables = new double[keep][][];
        for (int p = 0; p < keep; p++) {
            tables[p] = new double[coarse[pairs[p][0]].length + 1][coarse[pairs[p][1]].length + 1];
        }
        for (int round = 0; round < s.pairRounds(); round++) {
            for (int p = 0; p < keep; p++) {
                int a = pairs[p][0];
                int b = pairs[p][1];
                int ba = tables[p].length;
                int bb = tables[p][0].length;
                loss.derivatives(raw, y, g, h);
                double[][] hg = new double[ba][bb];
                double[][] hh = new double[ba][bb];
                for (int i = 0; i < n; i++) {
                    hg[rough[a][i]][rough[b][i]] += g[i];
                    hh[rough[a][i]][rough[b][i]] += h[i];
                }
                Cut cut = bestQuadrants(hg, hh, s.lambda());
                double[] q = new double[4];
                double[] qh = new double[4];
                for (int i = 0; i < ba; i++) {
                    for (int j = 0; j < bb; j++) {
                        int k = (i <= cut.a() ? 0 : 2) + (j <= cut.b() ? 0 : 1);
                        q[k] += hg[i][j];
                        qh[k] += hh[i][j];
                    }
                }
                for (int i = 0; i < ba; i++) {
                    for (int j = 0; j < bb; j++) {
                        int k = (i <= cut.a() ? 0 : 2) + (j <= cut.b() ? 0 : 1);
                        tables[p][i][j] -= s.learningRate() * q[k] / (qh[k] + s.lambda());
                    }
                }
                for (int i = 0; i < n; i++) {
                    int k = (rough[a][i] <= cut.a() ? 0 : 2) + (rough[b][i] <= cut.b() ? 0 : 1);
                    raw[i] -= s.learningRate() * q[k] / (qh[k] + s.lambda());
                }
            }
        }
        // centre every term on the training rows and move the means into the intercept
        double intercept = base;
        double[] importance = new double[d + keep];
        for (int f = 0; f < d; f++) {
            double mean = 0;
            for (int i = 0; i < n; i++) {
                mean += shapes[f][fine[f][i]] / n;
            }
            for (int b = 0; b < shapes[f].length; b++) {
                shapes[f][b] -= mean;
            }
            intercept += mean;
            for (int i = 0; i < n; i++) {
                importance[f] += Math.abs(shapes[f][fine[f][i]]) / n;
            }
        }
        for (int p = 0; p < keep; p++) {
            double mean = 0;
            for (int i = 0; i < n; i++) {
                mean += tables[p][rough[pairs[p][0]][i]][rough[pairs[p][1]][i]] / n;
            }
            for (double[] row : tables[p]) {
                for (int j = 0; j < row.length; j++) {
                    row[j] -= mean;
                }
            }
            intercept += mean;
            for (int i = 0; i < n; i++) {
                importance[d + p] += Math.abs(tables[p][rough[pairs[p][0]][i]][rough[pairs[p][1]][i]]) / n;
            }
        }
        return new ExplainableBoosting(loss, intercept, thresholds, shapes, coarse, pairs, tables, importance,
                ranking, rankingGain);
    }

    /** A best-first tree over ordered bins with up to {@code leaves} leaves; returns per-bin leaf values. */
    private static double[] oneFeatureTree(double[] hg, double[] hh, int[] hc, Settings s) {
        int bins = hg.length;
        List<int[]> segments = new ArrayList<>();   // [from, to) bin ranges
        segments.add(new int[] {0, bins});
        while (segments.size() < s.leaves()) {
            double best = 1e-12;
            int bestSeg = -1;
            int bestCut = -1;
            for (int k = 0; k < segments.size(); k++) {
                int[] seg = segments.get(k);
                double gs = 0;
                double hs = 0;
                int cs = 0;
                for (int b = seg[0]; b < seg[1]; b++) {
                    gs += hg[b];
                    hs += hh[b];
                    cs += hc[b];
                }
                double gl = 0;
                double hl = 0;
                int cl = 0;
                for (int b = seg[0]; b + 1 < seg[1]; b++) {
                    gl += hg[b];
                    hl += hh[b];
                    cl += hc[b];
                    if (cl < s.minLeaf() || cs - cl < s.minLeaf()) {
                        continue;
                    }
                    double gain = gl * gl / (hl + s.lambda()) + (gs - gl) * (gs - gl) / (hs - hl + s.lambda())
                            - gs * gs / (hs + s.lambda());
                    if (gain > best) {
                        best = gain;
                        bestSeg = k;
                        bestCut = b + 1;
                    }
                }
            }
            if (bestSeg < 0) {
                break;
            }
            int[] seg = segments.remove(bestSeg);
            segments.add(new int[] {seg[0], bestCut});
            segments.add(new int[] {bestCut, seg[1]});
        }
        double[] out = new double[bins];
        for (int[] seg : segments) {
            double gs = 0;
            double hs = 0;
            for (int b = seg[0]; b < seg[1]; b++) {
                gs += hg[b];
                hs += hh[b];
            }
            double v = -gs / (hs + s.lambda());
            for (int b = seg[0]; b < seg[1]; b++) {
                out[b] = v;
            }
        }
        return out;
    }

    private static double fastGain(int[] a, int[] b, int ba, int bb, double[] g, double[] h, double lambda) {
        double[][] hg = new double[ba][bb];
        double[][] hh = new double[ba][bb];
        for (int i = 0; i < a.length; i++) {
            hg[a[i]][b[i]] += g[i];
            hh[a[i]][b[i]] += h[i];
        }
        return bestQuadrants(hg, hh, lambda).gain();
    }

    /** A quadrant cut: rows with coarse bin {@code <= a} and {@code <= b} on each side. */
    private record Cut(int a, int b, double gain) {
    }

    /** The best single 2-D cut into four quadrants by prefix sums. */
    private static Cut bestQuadrants(double[][] hg, double[][] hh, double lambda) {
        int ba = hg.length;
        int bb = hg[0].length;
        double[][] pg = new double[ba + 1][bb + 1];
        double[][] ph = new double[ba + 1][bb + 1];
        for (int i = 0; i < ba; i++) {
            for (int j = 0; j < bb; j++) {
                pg[i + 1][j + 1] = hg[i][j] + pg[i][j + 1] + pg[i + 1][j] - pg[i][j];
                ph[i + 1][j + 1] = hh[i][j] + ph[i][j + 1] + ph[i + 1][j] - ph[i][j];
            }
        }
        double gt = pg[ba][bb];
        double ht = ph[ba][bb];
        double parent = gt * gt / (ht + lambda);
        double best = 0;
        int ca = 0;
        int cb = 0;
        for (int i = 1; i < ba; i++) {
            for (int j = 1; j < bb; j++) {
                double g00 = pg[i][j];
                double h00 = ph[i][j];
                double g01 = pg[i][bb] - g00;
                double h01 = ph[i][bb] - h00;
                double g10 = pg[ba][j] - g00;
                double h10 = ph[ba][j] - h00;
                double g11 = gt - g00 - g01 - g10;
                double h11 = ht - h00 - h01 - h10;
                double gain = g00 * g00 / (h00 + lambda) + g01 * g01 / (h01 + lambda) + g10 * g10 / (h10 + lambda)
                        + g11 * g11 / (h11 + lambda) - parent;
                if (gain > best) {
                    best = gain;
                    ca = i - 1;
                    cb = j - 1;
                }
            }
        }
        return new Cut(ca, cb, best);
    }

    /**
     * @param x a row
     * @return the raw additive score
     */
    public double raw(double[] x) {
        double r = intercept;
        for (int f = 0; f < shapes.length; f++) {
            r += shapes[f][bin(thresholds[f], x[f])];
        }
        for (int p = 0; p < pairs.length; p++) {
            r += pairValue(p, x[pairs[p][0]], x[pairs[p][1]]);
        }
        return r;
    }

    /**
     * @param x a row
     * @return the loss's link of the raw score
     */
    public double predict(double[] x) {
        return loss.link(raw(x));
    }

    /**
     * The learned shape function of one feature (centred on the training rows).
     *
     * @param feature the feature
     * @param value its value
     * @return {@code f_feature(value)}
     */
    public double shape(int feature, double value) {
        return shapes[feature][bin(thresholds[feature], value)];
    }

    /**
     * A pair term's value.
     *
     * @param pair the index into {@link #pairs()}
     * @param a the first feature's value
     * @param b the second feature's value
     * @return {@code f_ij(a, b)}
     */
    public double pairValue(int pair, double a, double b) {
        int i = pairs[pair][0];
        int j = pairs[pair][1];
        return tables[pair][bin(coarse[i], a)][bin(coarse[j], b)];
    }

    /** @return the kept pairs, each {@code {i, j}} */
    public int[][] pairs() {
        return Arrays.stream(pairs).map(int[]::clone).toArray(int[][]::new);
    }

    /** @return every pair ranked by FAST gain, best first */
    public int[][] pairRanking() {
        return Arrays.stream(ranking).map(int[]::clone).toArray(int[][]::new);
    }

    /** @return the FAST gains of {@link #pairRanking()} */
    public double[] pairRankingGains() {
        return rankingGain.clone();
    }

    /** @return each term's mean absolute contribution: main effects, then pairs */
    public double[] importance() {
        return importance.clone();
    }
}
