package io.github.zzuegg.brainiac.experimental.explain;

import java.util.Arrays;
import java.util.function.Function;

/**
 * Global effect plots of a black-box model and Friedman's interaction statistic.
 *
 * <ul>
 *   <li>{@link Effect#PDP}, partial dependence (Friedman, Annals of Statistics 2001): the mean
 *       prediction over the data while one feature is set to each grid value.</li>
 *   <li>{@link Effect#ICE} and {@link Effect#C_ICE}, individual conditional expectation
 *       (Goldstein et al., JCGS 2015, <a href="https://arxiv.org/abs/1309.6392">arXiv:1309.6392</a>):
 *       one such curve per row, optionally centred at the first grid value; non-parallel curves
 *       reveal interactions.</li>
 *   <li>{@link Effect#ALE}, accumulated local effects (Apley and Zhu, JRSS-B 2020,
 *       <a href="https://arxiv.org/abs/1612.08468">arXiv:1612.08468</a>): the prediction's
 *       finite differences across each quantile bin, averaged over only the rows in that bin and
 *       accumulated, so the model is never evaluated far from the data; unbiased for additive
 *       effects under correlated inputs where partial dependence extrapolates.</li>
 *   <li>{@link #ale2}, the second-order ALE surface of a pair with both main effects removed.</li>
 *   <li>{@link #hStatistic} and {@link #hTotal}, Friedman and Popescu's H-squared (AoAS 2008): the
 *       share of the variance of a pair's (or one feature's) partial dependence that the
 *       one-way effects do not explain.</li>
 * </ul>
 *
 * <p>Models are batch functions {@code double[][] -> double[]}, as in
 * {@code ModelAgnosticExplainer}. Curves are centred to mean zero over the data, so they compare
 * with an additive component directly.
 */
public final class EffectPlots {

    /** The kinds of one-feature curves. */
    public enum Effect {
        /** Partial dependence, centred. */
        PDP,
        /** Individual conditional expectation (one curve per row; {@link Curve#values} is their mean). */
        ICE,
        /** Centred ICE: each curve minus its value at the first grid point. */
        C_ICE,
        /** Accumulated local effects. */
        ALE
    }

    /**
     * An effect curve.
     *
     * @param grid the feature values
     * @param values the effect at each grid value, centred
     * @param individual for ICE and C_ICE the per-row curves {@code [rows][grid]}, else null
     */
    public record Curve(double[] grid, double[] values, double[][] individual) {

        /**
         * Linear interpolation of the curve, flat outside the grid.
         *
         * @param v a feature value
         * @return the effect at {@code v}
         */
        public double at(double v) {
            if (v <= grid[0]) {
                return values[0];
            }
            int last = grid.length - 1;
            if (v >= grid[last]) {
                return values[last];
            }
            int k = Arrays.binarySearch(grid, v);
            if (k >= 0) {
                return values[k];
            }
            int hi = -k - 1;
            int lo = hi - 1;
            double t = (v - grid[lo]) / (grid[hi] - grid[lo]);
            return values[lo] + t * (values[hi] - values[lo]);
        }
    }

    private final Function<double[][], double[]> model;

    /**
     * Creates the explainer.
     *
     * @param model the model, a batch of rows to one score per row
     */
    public EffectPlots(Function<double[][], double[]> model) {
        this.model = model;
    }

    /**
     * Quantile grid points of a feature over the data.
     *
     * @param x the data
     * @param feature the feature
     * @param points how many points, at least 2
     * @return the distinct quantiles {@code 0, 1/(points-1), ..., 1}, ascending
     */
    public static double[] quantileGrid(double[][] x, int feature, int points) {
        double[] v = column(x, feature);
        Arrays.sort(v);
        double[] g = new double[points];
        for (int k = 0; k < points; k++) {
            g[k] = v[(int) Math.round((double) k * (v.length - 1) / (points - 1))];
        }
        return Arrays.stream(g).distinct().toArray();
    }

    private static double[] column(double[][] x, int feature) {
        double[] v = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            v[i] = x[i][feature];
        }
        return v;
    }

    /** {@code [rows][grid]}: the prediction of every row with the feature set to every grid value. */
    private double[][] grid(double[][] x, int feature, double[] grid) {
        int n = x.length;
        double[][] batch = new double[n * grid.length][];
        for (int i = 0; i < n; i++) {
            for (int g = 0; g < grid.length; g++) {
                double[] row = x[i].clone();
                row[feature] = grid[g];
                batch[i * grid.length + g] = row;
            }
        }
        double[] f = model.apply(batch);
        double[][] out = new double[n][grid.length];
        for (int i = 0; i < n; i++) {
            System.arraycopy(f, i * grid.length, out[i], 0, grid.length);
        }
        return out;
    }

    /**
     * One effect curve.
     *
     * @param effect the kind
     * @param x the data the effect is averaged over
     * @param feature the feature
     * @param points grid points (for ALE, bins plus one)
     * @return the curve
     */
    public Curve curve(Effect effect, double[][] x, int feature, int points) {
        if (effect == Effect.ALE) {
            return ale(x, feature, points - 1);
        }
        double[] g = quantileGrid(x, feature, points);
        double[][] ice = grid(x, feature, g);
        if (effect == Effect.C_ICE) {
            for (double[] row : ice) {
                double base = row[0];
                for (int k = 0; k < row.length; k++) {
                    row[k] -= base;
                }
            }
        }
        double[] mean = new double[g.length];
        for (double[] row : ice) {
            for (int k = 0; k < g.length; k++) {
                mean[k] += row[k] / x.length;
            }
        }
        Curve c = new Curve(g, mean, effect == Effect.PDP ? null : ice);
        return centred(c, x, feature);
    }

    /** Shifts a curve so its mean over the data's feature values is zero. */
    private static Curve centred(Curve c, double[][] x, int feature) {
        double m = 0;
        for (double[] row : x) {
            m += c.at(row[feature]) / x.length;
        }
        double[] v = c.values().clone();
        for (int k = 0; k < v.length; k++) {
            v[k] -= m;
        }
        return new Curve(c.grid(), v, c.individual());
    }

    /**
     * First-order accumulated local effects over quantile bins.
     *
     * @param x the data
     * @param feature the feature
     * @param bins the number of bins
     * @return the centred ALE curve at the bin edges
     */
    public Curve ale(double[][] x, int feature, int bins) {
        double[] edges = quantileGrid(x, feature, bins + 1);
        int b = edges.length - 1;
        double[] sum = new double[b];
        int[] count = new int[b];
        int n = x.length;
        double[][] batch = new double[2 * n][];
        int[] bin = new int[n];
        for (int i = 0; i < n; i++) {
            bin[i] = binOf(edges, x[i][feature]);
            double[] lo = x[i].clone();
            double[] hi = x[i].clone();
            lo[feature] = edges[bin[i]];
            hi[feature] = edges[bin[i] + 1];
            batch[2 * i] = lo;
            batch[2 * i + 1] = hi;
        }
        double[] f = model.apply(batch);
        for (int i = 0; i < n; i++) {
            sum[bin[i]] += f[2 * i + 1] - f[2 * i];
            count[bin[i]]++;
        }
        double[] acc = new double[b + 1];
        for (int k = 0; k < b; k++) {
            acc[k + 1] = acc[k] + (count[k] > 0 ? sum[k] / count[k] : 0);
        }
        return centred(new Curve(edges, acc, null), x, feature);
    }

    private static int binOf(double[] edges, double v) {
        int b = edges.length - 1;
        for (int k = 0; k < b - 1; k++) {
            if (v <= edges[k + 1]) {
                return k;
            }
        }
        return b - 1;
    }

    /**
     * Second-order ALE: the accumulated second differences of a pair over a grid of quantile
     * cells, with both first-order effects of the accumulated surface removed and centred.
     *
     * @param x the data
     * @param j the first feature
     * @param k the second feature
     * @param bins bins per feature
     * @return the surface at the cell corners {@code [bins_j + 1][bins_k + 1]}
     */
    public double[][] ale2(double[][] x, int j, int k, int bins) {
        double[] ej = quantileGrid(x, j, bins + 1);
        double[] ek = quantileGrid(x, k, bins + 1);
        int bj = ej.length - 1;
        int bk = ek.length - 1;
        int n = x.length;
        double[][] batch = new double[4 * n][];
        int[] cj = new int[n];
        int[] ck = new int[n];
        for (int i = 0; i < n; i++) {
            cj[i] = binOf(ej, x[i][j]);
            ck[i] = binOf(ek, x[i][k]);
            for (int c = 0; c < 4; c++) {
                double[] row = x[i].clone();
                row[j] = ej[cj[i] + (c & 1)];
                row[k] = ek[ck[i] + (c >> 1)];
                batch[4 * i + c] = row;
            }
        }
        double[] f = model.apply(batch);
        double[][] sum = new double[bj][bk];
        int[][] count = new int[bj][bk];
        for (int i = 0; i < n; i++) {
            double d = f[4 * i + 3] - f[4 * i + 2] - f[4 * i + 1] + f[4 * i];
            sum[cj[i]][ck[i]] += d;
            count[cj[i]][ck[i]]++;
        }
        double[][] acc = new double[bj + 1][bk + 1];
        for (int a = 1; a <= bj; a++) {
            for (int c = 1; c <= bk; c++) {
                double mean = count[a - 1][c - 1] > 0 ? sum[a - 1][c - 1] / count[a - 1][c - 1] : 0;
                acc[a][c] = acc[a - 1][c] + acc[a][c - 1] - acc[a - 1][c - 1] + mean;
            }
        }
        // remove the main effects: subtract row and column means, add back the grand mean
        double grand = 0;
        double[] rowMean = new double[bj + 1];
        double[] colMean = new double[bk + 1];
        for (int a = 0; a <= bj; a++) {
            for (int c = 0; c <= bk; c++) {
                rowMean[a] += acc[a][c] / (bk + 1);
                colMean[c] += acc[a][c] / (bj + 1);
                grand += acc[a][c] / ((bj + 1) * (bk + 1));
            }
        }
        for (int a = 0; a <= bj; a++) {
            for (int c = 0; c <= bk; c++) {
                acc[a][c] = acc[a][c] - rowMean[a] - colMean[c] + grand;
            }
        }
        return acc;
    }

    /** Centred partial dependence on a set of features, evaluated at each row's own values. */
    private double[] partialAtRows(double[][] x, int[] features) {
        int n = x.length;
        double[][] batch = new double[n * n][];
        for (int i = 0; i < n; i++) {
            for (int m = 0; m < n; m++) {
                double[] row = x[m].clone();
                for (int f : features) {
                    row[f] = x[i][f];
                }
                batch[i * n + m] = row;
            }
        }
        double[] f = model.apply(batch);
        double[] pd = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int m = 0; m < n; m++) {
                s += f[i * n + m];
            }
            pd[i] = s / n;
        }
        return centre(pd);
    }

    private static double[] centre(double[] v) {
        double m = Arrays.stream(v).average().orElse(0);
        return Arrays.stream(v).map(a -> a - m).toArray();
    }

    /**
     * Pairwise H-squared of every pair of features, each one-way partial dependence computed once.
     *
     * @param x the data, a small sample (cost is {@code n^2} model calls per feature and pair)
     * @return {@code [p][p]} symmetric, zero on the diagonal
     */
    public double[][] hStatistics(double[][] x) {
        int p = x[0].length;
        double[][] one = new double[p][];
        for (int j = 0; j < p; j++) {
            one[j] = partialAtRows(x, new int[] {j});
        }
        double[][] h = new double[p][p];
        for (int j = 0; j < p; j++) {
            for (int k = j + 1; k < p; k++) {
                h[j][k] = h[k][j] = h2(partialAtRows(x, new int[] {j, k}), one[j], one[k]);
            }
        }
        return h;
    }

    private static double h2(double[] pair, double[] a, double[] b) {
        double num = 0;
        double den = 0;
        for (int i = 0; i < pair.length; i++) {
            double r = pair[i] - a[i] - b[i];
            num += r * r;
            den += pair[i] * pair[i];
        }
        return den > 0 ? num / den : 0;
    }

    /**
     * Friedman's pairwise H-squared: the share of the variance of the pair's centred partial
     * dependence not explained by the two one-way partial dependences.
     *
     * @param x the data
     * @param j the first feature
     * @param k the second feature
     * @return {@code H^2_jk} in {@code [0, 1]} (can exceed 1 with tiny joint effects)
     */
    public double hStatistic(double[][] x, int j, int k) {
        return h2(partialAtRows(x, new int[] {j, k}), partialAtRows(x, new int[] {j}),
                partialAtRows(x, new int[] {k}));
    }

    /**
     * Friedman's total H-squared of one feature: the share of the prediction's variance not
     * explained by the feature's partial dependence plus the partial dependence on all others.
     *
     * @param x the data
     * @param j the feature
     * @return {@code H^2_j}
     */
    public double hTotal(double[][] x, int j) {
        int p = x[0].length;
        int[] rest = new int[p - 1];
        for (int a = 0, b = 0; a < p; a++) {
            if (a != j) {
                rest[b++] = a;
            }
        }
        double[] f = centre(model.apply(x));
        return h2(f, partialAtRows(x, new int[] {j}), partialAtRows(x, rest));
    }

    /**
     * Permutation importance: the increase of the mean squared error when one feature's column
     * is shuffled.
     *
     * @param x the data
     * @param y the targets
     * @param seed the shuffle seed
     * @return one importance per feature
     */
    public double[] permutationImportance(double[][] x, double[] y, long seed) {
        java.util.SplittableRandom r = new java.util.SplittableRandom(seed);
        double base = mse(model.apply(x), y);
        int p = x[0].length;
        double[] imp = new double[p];
        for (int j = 0; j < p; j++) {
            double[][] shuffled = new double[x.length][];
            int[] perm = new int[x.length];
            for (int i = 0; i < perm.length; i++) {
                perm[i] = i;
            }
            for (int i = perm.length - 1; i > 0; i--) {
                int t = r.nextInt(i + 1);
                int s = perm[i];
                perm[i] = perm[t];
                perm[t] = s;
            }
            for (int i = 0; i < x.length; i++) {
                shuffled[i] = x[i].clone();
                shuffled[i][j] = x[perm[i]][j];
            }
            imp[j] = mse(model.apply(shuffled), y) - base;
        }
        return imp;
    }

    private static double mse(double[] f, double[] y) {
        double s = 0;
        for (int i = 0; i < f.length; i++) {
            s += (f[i] - y[i]) * (f[i] - y[i]);
        }
        return s / f.length;
    }
}
