package io.github.zzuegg.brainiac.experimental.causal;

import io.github.zzuegg.brainiac.experimental.ml.RegressionForest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Causal forests as generalized random forests (Wager and Athey, JASA 2018,
 * <a href="https://arxiv.org/abs/1510.04342">arXiv:1510.04342</a>; Athey, Tibshirani and Wager,
 * Annals of Statistics 2019, <a href="https://arxiv.org/abs/1610.01271">arXiv:1610.01271</a>).
 *
 * <p>How it works:
 * <ol>
 *   <li><em>Local centering</em>: the outcome and treatment are residualised with cross-fitted
 *       nuisance models, {@code y~ = y - m(x)}, {@code t~ = t - e(x)} (as in the R-learner).</li>
 *   <li><em>Effect splitting</em>: at every node the forest computes the node's effect
 *       {@code tau_P} from the residuals and relabels each unit with its influence on it,
 *       {@code rho_i = (t~_i - mean t~)((y~_i - mean y~) - tau_P (t~_i - mean t~)) / var(t~)}, then
 *       picks the CART split of {@code rho} (GRF's gradient tree). This maximises the
 *       heterogeneity of the effect between the children rather than outcome fit; the grower is
 *       the shared {@link RegressionForest#grow}.</li>
 *   <li><em>Honesty</em>: each tree splits on half of its subsample and keeps the other half as
 *       its leaf members.</li>
 *   <li><em>Forest weights</em>: a query's weight on unit {@code i} is how often {@code i} shares
 *       a leaf with it (normalised per tree); the effect solves the locally weighted moment
 *       equation {@code sum alpha_i (t~_i - tbar)((y~_i - ybar) - tau (t~_i - tbar)) = 0}.</li>
 *   <li><em>Bootstrap of little bags</em>: trees come in groups that share one half-sample; the
 *       spread of the group means of the linearised score, minus the within-group spread, gives
 *       the variance, hence confidence intervals for {@code tau(x)}.</li>
 * </ol>
 */
public final class CausalForest {

    /**
     * Settings.
     *
     * @param trees trees, a multiple of {@code groupSize}
     * @param groupSize trees per little bag, at least 2
     * @param minLeaf the fewest estimation units per leaf (and growing units per child)
     * @param maxDepth the depth limit
     * @param featuresPerSplit candidate features per split, 0 for {@code d / 3}
     * @param centering the cross-fitted nuisance model for {@code m(x)} and {@code e(x)}
     * @param seed the seed
     */
    public record Settings(int trees, int groupSize, int minLeaf, int maxDepth, int featuresPerSplit,
            Regressor centering, long seed) {

        /**
         * Defaults: 200 trees in bags of 2, leaves of 5, depth 10, regression-forest centering.
         *
         * @param seed the seed
         * @return the settings
         */
        public static Settings defaults(long seed) {
            return new Settings(200, 2, 5, 10, 0,
                    Regressor.forest(RegressionForest.Settings.defaults().withTrees(50).withSeed(seed)), seed);
        }

        /**
         * @param n trees
         * @return a copy with that many trees
         */
        public Settings withTrees(int n) {
            return new Settings(n, groupSize, minLeaf, maxDepth, featuresPerSplit, centering, seed);
        }
    }

    private record HonestTree(RegressionForest.Tree tree, int[][] members) {
    }

    private final double[][] x;
    private final double[] ry;
    private final double[] rt;
    private final List<HonestTree> trees;
    private final int groupSize;

    private CausalForest(double[][] x, double[] ry, double[] rt, List<HonestTree> trees, int groupSize) {
        this.x = x;
        this.ry = ry;
        this.rt = rt;
        this.trees = trees;
        this.groupSize = groupSize;
    }

    /**
     * Grows a causal forest.
     *
     * @param x covariates {@code [n][d]}
     * @param t treatments, 0 or 1
     * @param y outcomes
     * @param s the settings
     * @return the forest
     */
    public static CausalForest fit(double[][] x, int[] t, double[] y, Settings s) {
        int n = x.length;
        int[] fold = CrossFitting.folds(n, 5, s.seed());
        double[] m = CrossFitting.predict(x, y, null, s.centering(), fold);
        double[] e = CrossFitting.predict(x, CrossFitting.asDouble(t), null, s.centering(), fold);
        double[] ry = new double[n];
        double[] rt = new double[n];
        for (int i = 0; i < n; i++) {
            ry[i] = y[i] - m[i];
            rt[i] = t[i] - e[i];
        }
        SplittableRandom random = new SplittableRandom(s.seed());
        List<HonestTree> trees = new ArrayList<>();
        int groups = Math.max(1, s.trees() / s.groupSize());
        EffectScore score = new EffectScore(ry, rt);
        for (int g = 0; g < groups; g++) {
            int[] half = Arrays.copyOf(permutation(n, random), n / 2);
            for (int b = 0; b < s.groupSize(); b++) {
                int[] perm = permutation(half.length, random);
                int size = half.length / 2;
                int[] grow = new int[size / 2];
                int[] estimate = new int[size - size / 2];
                for (int i = 0; i < size; i++) {
                    if (i < grow.length) {
                        grow[i] = half[perm[i]];
                    } else {
                        estimate[i - grow.length] = half[perm[i]];
                    }
                }
                RegressionForest.Tree tree = RegressionForest.grow(x, grow, s.maxDepth(), s.minLeaf(),
                        s.featuresPerSplit(), score, random, rows -> 0);
                trees.add(new HonestTree(tree, members(tree, x, estimate)));
            }
        }
        return new CausalForest(x, ry, rt, trees, s.groupSize());
    }

    private static int[][] members(RegressionForest.Tree tree, double[][] x, int[] rows) {
        int nodes = tree.feature().length;
        List<List<Integer>> lists = new ArrayList<>();
        for (int k = 0; k < nodes; k++) {
            lists.add(new ArrayList<>());
        }
        for (int r : rows) {
            lists.get(tree.leaf(x[r])).add(r);
        }
        int[][] out = new int[nodes][];
        for (int k = 0; k < nodes; k++) {
            out[k] = lists.get(k).stream().mapToInt(Integer::intValue).toArray();
        }
        return out;
    }

    /**
     * An effect estimate with its variance.
     *
     * @param tau the estimated CATE
     * @param variance the bootstrap-of-little-bags variance
     */
    public record Prediction(double tau, double variance) {

        /**
         * @param z the normal quantile, such as 1.645 for a 90 % interval
         * @return {@code {lower, upper}}
         */
        public double[] interval(double z) {
            double sd = Math.sqrt(Math.max(variance, 0));
            return new double[] {tau - z * sd, tau + z * sd};
        }
    }

    /**
     * Estimates the effect and its variance at a point.
     *
     * @param query a covariate row
     * @return the prediction
     */
    public Prediction predict(double[] query) {
        int n = x.length;
        double[] alpha = new double[n];
        double[][] perTree = new double[trees.size()][];
        int used = 0;
        for (int b = 0; b < trees.size(); b++) {
            HonestTree h = trees.get(b);
            int[] leaf = h.members()[h.tree().leaf(query)];
            if (leaf.length == 0) {
                continue;
            }
            used++;
            double w = 1.0 / leaf.length;
            for (int i : leaf) {
                alpha[i] += w;
            }
        }
        if (used == 0) {
            return new Prediction(0, Double.POSITIVE_INFINITY);
        }
        double sw = 0;
        double tbar = 0;
        double ybar = 0;
        for (int i = 0; i < n; i++) {
            if (alpha[i] > 0) {
                sw += alpha[i];
                tbar += alpha[i] * rt[i];
                ybar += alpha[i] * ry[i];
            }
        }
        tbar /= sw;
        ybar /= sw;
        double num = 0;
        double den = 0;
        for (int i = 0; i < n; i++) {
            if (alpha[i] > 0) {
                double dt = rt[i] - tbar;
                num += alpha[i] * dt * (ry[i] - ybar);
                den += alpha[i] * dt * dt;
            }
        }
        double tau = num / Math.max(den, 1e-12);
        double hess = den / sw;
        // linearised score per tree: psi_b = mean over the tree's leaf of (t - tbar)((y - ybar) - tau (t - tbar))
        for (int b = 0; b < trees.size(); b++) {
            HonestTree h = trees.get(b);
            int[] leaf = h.members()[h.tree().leaf(query)];
            if (leaf.length == 0) {
                perTree[b] = null;
                continue;
            }
            double psi = 0;
            for (int i : leaf) {
                double dt = rt[i] - tbar;
                psi += dt * ((ry[i] - ybar) - tau * dt) / leaf.length;
            }
            perTree[b] = new double[] {psi};
        }
        int groups = trees.size() / groupSize;
        double grand = 0;
        int counted = 0;
        double[] groupMean = new double[groups];
        boolean[] valid = new boolean[groups];
        for (int g = 0; g < groups; g++) {
            double s = 0;
            int c = 0;
            for (int b = g * groupSize; b < (g + 1) * groupSize; b++) {
                if (perTree[b] != null) {
                    s += perTree[b][0];
                    c++;
                }
            }
            if (c == groupSize) {
                valid[g] = true;
                groupMean[g] = s / c;
                grand += groupMean[g];
                counted++;
            }
        }
        if (counted < 2) {
            return new Prediction(tau, Double.POSITIVE_INFINITY);
        }
        grand /= counted;
        double between = 0;
        double within = 0;
        for (int g = 0; g < groups; g++) {
            if (!valid[g]) {
                continue;
            }
            between += (groupMean[g] - grand) * (groupMean[g] - grand) / counted;
            double w = 0;
            for (int b = g * groupSize; b < (g + 1) * groupSize; b++) {
                w += (perTree[b][0] - groupMean[g]) * (perTree[b][0] - groupMean[g]);
            }
            within += w / (groupSize * (groupSize - 1.0)) / counted;
        }
        double v = Math.max(between - within, 0.1 * between);
        return new Prediction(tau, v / (hess * hess));
    }

    /**
     * @param rows covariate rows
     * @return the estimated CATE per row
     */
    public double[] predict(double[][] rows) {
        double[] out = new double[rows.length];
        for (int i = 0; i < rows.length; i++) {
            out[i] = predict(rows[i]).tau();
        }
        return out;
    }

    /** GRF's gradient-based effect split: CART on the per-node influence pseudo-outcome. */
    private static final class EffectScore implements RegressionForest.SplitScore {
        private final double[] ry;
        private final double[] rt;
        private final double[] rho;
        private int[] cached;
        private double[] prefix = new double[0];
        private double total;

        EffectScore(double[] ry, double[] rt) {
            this.ry = ry;
            this.rt = rt;
            this.rho = new double[ry.length];
        }

        @Override
        public void node(int[] rows) {
            double tm = 0;
            double ym = 0;
            for (int r : rows) {
                tm += rt[r];
                ym += ry[r];
            }
            tm /= rows.length;
            ym /= rows.length;
            double num = 0;
            double den = 0;
            for (int r : rows) {
                double dt = rt[r] - tm;
                num += dt * (ry[r] - ym);
                den += dt * dt;
            }
            double tau = num / Math.max(den, 1e-12);
            double var = Math.max(den / rows.length, 1e-12);
            total = 0;
            for (int r : rows) {
                double dt = rt[r] - tm;
                rho[r] = dt * ((ry[r] - ym) - tau * dt) / var;
                total += rho[r];
            }
            cached = null;
        }

        @Override
        public double gain(int[] rows, int cut) {
            if (rows != cached) {
                prefix = new double[rows.length + 1];
                for (int i = 0; i < rows.length; i++) {
                    prefix[i + 1] = prefix[i] + rho[rows[i]];
                }
                cached = rows;
            }
            double l = prefix[cut];
            double r = total - l;
            return l * l / cut + r * r / (rows.length - cut);
        }
    }

    private static int[] permutation(int n, SplittableRandom random) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }
}
