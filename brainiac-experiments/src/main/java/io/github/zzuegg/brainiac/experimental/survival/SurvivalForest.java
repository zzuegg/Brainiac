package io.github.zzuegg.brainiac.experimental.survival;

import io.github.zzuegg.brainiac.experimental.ml.RegressionForest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Random survival forests (Ishwaran, Kogalur, Blackstone and Lauer, Annals of Applied Statistics
 * 2008, <a href="https://arxiv.org/abs/0811.1645">arXiv:0811.1645</a>). Every tree grows on a
 * bootstrap sample and splits each node by the largest two-sample <em>log-rank</em> statistic
 * among a few random cut points of a few random features, so children differ in their survival
 * curves rather than in a mean; each leaf stores the Nelson-Aalen cumulative hazard of its units on
 * a time grid, and the forest averages the trees' cumulative hazards. Because leaves hold whole
 * curves, the forest captures crossing hazards with no proportionality assumption. The CART
 * bookkeeping is the shared {@link RegressionForest#grow}.
 */
public final class SurvivalForest {

    private final double[] grid;
    private final List<RegressionForest.Tree> trees;
    private final List<double[]> leafHazards;

    private SurvivalForest(double[] grid, List<RegressionForest.Tree> trees, List<double[]> leafHazards) {
        this.grid = grid;
        this.trees = trees;
        this.leafHazards = leafHazards;
    }

    /**
     * Grows a forest.
     *
     * @param x covariates {@code [n][d]}
     * @param time observed times
     * @param event 0 if censored (any non-zero value is the event)
     * @param grid the times at which leaf cumulative hazards are stored
     * @param trees the number of trees
     * @param minLeaf the fewest units per leaf (Ishwaran's default is 3 deaths; 10 units here)
     * @param cutsPerFeature random candidate cut points per feature (the {@code nsplit} option)
     * @param seed the seed
     * @return the forest
     */
    public static SurvivalForest fit(double[][] x, double[] time, int[] event, double[] grid, int trees,
            int minLeaf, int cutsPerFeature, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<RegressionForest.Tree> grown = new ArrayList<>();
        List<double[]> hazards = new ArrayList<>();
        LogRank score = new LogRank(time, event, cutsPerFeature, random);
        for (int b = 0; b < trees; b++) {
            int[] rows = RegressionForest.subsample(x.length, 1, false, random)[0];
            grown.add(RegressionForest.grow(x, rows, 30, minLeaf, 0, score, random, leaf -> {
                hazards.add(nelsonAalen(leaf, time, event, grid));
                return hazards.size() - 1;
            }));
        }
        return new SurvivalForest(grid.clone(), grown, hazards);
    }

    private static double[] nelsonAalen(int[] rows, double[] time, int[] event, double[] grid) {
        int[] o = Arrays.stream(rows).boxed().sorted((a, b) -> Double.compare(time[a], time[b]))
                .mapToInt(Integer::intValue).toArray();
        double[] h = new double[grid.length];
        double acc = 0;
        int atRisk = o.length;
        int k = 0;
        int g = 0;
        while (k < o.length) {
            double t = time[o[k]];
            int deaths = 0;
            int leaving = 0;
            while (k < o.length && time[o[k]] == t) {
                deaths += event[o[k]] != 0 ? 1 : 0;
                leaving++;
                k++;
            }
            while (g < grid.length && grid[g] < t) {
                h[g++] = acc;
            }
            acc += deaths / (double) atRisk;
            atRisk -= leaving;
        }
        while (g < grid.length) {
            h[g++] = acc;
        }
        return h;
    }

    /**
     * @param x covariates
     * @return the forest's cumulative hazard on its grid
     */
    public double[] cumulativeHazard(double[] x) {
        double[] h = new double[grid.length];
        for (RegressionForest.Tree t : trees) {
            double[] leaf = leafHazards.get((int) t.predict(x));
            for (int g = 0; g < h.length; g++) {
                h[g] += leaf[g] / trees.size();
            }
        }
        return h;
    }

    /**
     * @param x covariate rows
     * @return {@code [n][grid]} survival {@code exp(-H(t | x))}
     */
    public double[][] survival(double[][] x) {
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            double[] h = cumulativeHazard(x[i]);
            out[i] = new double[h.length];
            for (int g = 0; g < h.length; g++) {
                out[i][g] = Math.exp(-h[g]);
            }
        }
        return out;
    }

    /**
     * The mortality risk score of Ishwaran et al.: the sum of the cumulative hazard over the grid.
     *
     * @param x covariates
     * @return the risk (higher means earlier events)
     */
    public double mortality(double[] x) {
        double s = 0;
        for (double v : cumulativeHazard(x)) {
            s += v;
        }
        return s;
    }

    /** @return the grid */
    public double[] grid() {
        return grid.clone();
    }

    /** The two-sample log-rank statistic at a few random cut points per feature. */
    private static final class LogRank implements RegressionForest.SplitScore {
        private final double[] time;
        private final int[] event;
        private final int cuts;
        private final SplittableRandom random;
        private int[] byTime = new int[0];
        private final boolean[] left;
        private int[] cachedOrder;
        private boolean[] candidate = new boolean[0];

        LogRank(double[] time, int[] event, int cuts, SplittableRandom random) {
            this.time = time;
            this.event = event;
            this.cuts = cuts;
            this.random = random;
            this.left = new boolean[time.length];
        }

        @Override
        public void node(int[] rows) {
            byTime = Arrays.stream(rows).boxed().sorted((a, b) -> Double.compare(time[a], time[b]))
                    .mapToInt(Integer::intValue).toArray();
            cachedOrder = null;
        }

        @Override
        public double gain(int[] rows, int cut) {
            if (rows != cachedOrder) {
                cachedOrder = rows;
                candidate = new boolean[rows.length + 1];
                for (int k = 0; k < cuts; k++) {
                    candidate[1 + random.nextInt(rows.length - 1)] = true;
                }
            }
            if (!candidate[cut]) {
                return Double.NaN;
            }
            for (int i = 0; i < rows.length; i++) {
                left[rows[i]] = i < cut;
            }
            double atRisk = rows.length;
            double atRiskLeft = cut;
            double num = 0;
            double var = 0;
            int k = 0;
            while (k < byTime.length) {
                double t = time[byTime[k]];
                double d = 0;
                double dl = 0;
                double leaving = 0;
                double leavingLeft = 0;
                while (k < byTime.length && time[byTime[k]] == t) {
                    int r = byTime[k];
                    boolean ev = event[r] != 0;
                    d += ev ? 1 : 0;
                    dl += ev && left[r] ? 1 : 0;
                    leaving++;
                    leavingLeft += left[r] ? 1 : 0;
                    k++;
                }
                if (d > 0 && atRisk > 1) {
                    double share = atRiskLeft / atRisk;
                    num += dl - d * share;
                    var += d * share * (1 - share) * (atRisk - d) / (atRisk - 1);
                }
                atRisk -= leaving;
                atRiskLeft -= leavingLeft;
            }
            return var > 0 ? Math.abs(num) / Math.sqrt(var) : Double.NaN;
        }
    }
}
