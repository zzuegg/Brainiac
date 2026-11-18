package io.github.zzuegg.brainiac.experiments.explainability;

import io.github.zzuegg.brainiac.experimental.data.PlantedFunctions;
import io.github.zzuegg.brainiac.experimental.explain.CoalitionGame;
import io.github.zzuegg.brainiac.experimental.explain.EffectPlots;
import io.github.zzuegg.brainiac.experimental.explain.ShapleyEstimators;
import io.github.zzuegg.brainiac.experimental.explain.ShapleyInteractions;
import io.github.zzuegg.brainiac.experimental.explain.TreeShap;
import io.github.zzuegg.brainiac.experimental.explain.TreeView;
import io.github.zzuegg.brainiac.experimental.train.ModelAgnosticExplainer;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.Arrays;
import java.util.EnumMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

/**
 * Wave 13 items 13.1.2-13.1.4: exact tree Shapley values, Shapley interactions and fast
 * model-agnostic estimators, each against brute-force enumeration.
 *
 * <ul>
 *   <li>{@link #trees}: {@link TreeShap} on a boosted {@link TreeView} (8 features) against
 *       brute force over all 256 coalitions, against {@link ModelAgnosticExplainer#kernelShap}
 *       in error and time per row; the credit path-dependent values give to a correlated
 *       irrelevant feature; and gain importance against mean |TreeSHAP| for a continuous noise
 *       feature among binary relevant ones.</li>
 *   <li>{@link #interactions}: precision of the five planted interacting pairs among the top
 *       five for exact k-SII, SHAP-IQ and per-pair permutation sampling at equal budget, the
 *       H-statistic and the path-dependent TreeSHAP interaction matrix, plus each estimator's
 *       error to the exact SII.</li>
 *   <li>{@link #estimators}: mean squared error to exact interventional TreeSHAP of the
 *       {@link ShapleyEstimators} and the existing KernelSHAP at budgets of 2p, 8p and 32p.</li>
 * </ul>
 */
public final class ShapleyMethods {

    private ShapleyMethods() {
    }

    /**
     * The tree table.
     *
     * @param pathError max error of path-dependent TreeSHAP to brute force on the cover game
     * @param interventionalError max error of interventional TreeSHAP to brute force
     * @param treeShapMicros microseconds per row of interventional TreeSHAP
     * @param kernelShapMicros microseconds per row of KernelSHAP with 2000 samples
     * @param kernelShapError max error of that KernelSHAP to the exact values
     * @param speedup KernelSHAP time over TreeSHAP time
     * @param pathIrrelevantShare share of mean |phi| that path-dependent values put on irrelevant
     *     features correlated with relevant ones ({@code rho = 0.9})
     * @param interventionalIrrelevantShare the same for interventional values
     * @param gainNoiseRank the rank (0 = top) of a continuous noise feature by core GBT gain
     * @param shapNoiseRank its rank by mean |interventional TreeSHAP|
     */
    public record TreeTable(double pathError, double interventionalError, double treeShapMicros,
            double kernelShapMicros, double kernelShapError, double speedup, double pathIrrelevantShare,
            double interventionalIrrelevantShare, int gainNoiseRank, int shapNoiseRank) {
    }

    /**
     * Runs the tree table.
     *
     * @param seed the seed
     * @return the table
     */
    public static TreeTable trees(long seed) {
        PlantedFunctions.Table data = PlantedFunctions.interactions(600, 8, seed);
        TreeView model = TreeView.boost(data.x(), data.y(), 100, 4, 0.1, seed);
        TreeShap shap = new TreeShap(model);
        double[][] background = Arrays.copyOfRange(data.x(), 500, 520);
        double pathErr = 0;
        double intErr = 0;
        double kernelErr = 0;
        long treeNanos = 0;
        long kernelNanos = 0;
        ModelAgnosticExplainer kernel = new ModelAgnosticExplainer(model::predict);
        SplittableRandom r = new SplittableRandom(seed);
        int rows = 4;
        for (int i = 0; i < rows; i++) {
            double[] x = data.x()[i];
            long t0 = System.nanoTime();
            double[] exact = shap.values(TreeShap.Mode.INTERVENTIONAL, x, background);
            treeNanos += System.nanoTime() - t0;
            double[] brute = TreeShap.bruteForce(8, m -> TreeShap.interventionalGame(model::predict, x, background, m));
            double[] path = shap.values(TreeShap.Mode.PATH_DEPENDENT, x, null);
            double[] pathBrute = TreeShap.bruteForce(8, m -> shap.conditionalExpectation(x, CoalitionGame.members(m, 8)));
            t0 = System.nanoTime();
            double[] k = kernel.kernelShap(x, background, 2000, r);
            kernelNanos += System.nanoTime() - t0;
            for (int j = 0; j < 8; j++) {
                intErr = Math.max(intErr, Math.abs(exact[j] - brute[j]));
                pathErr = Math.max(pathErr, Math.abs(path[j] - pathBrute[j]));
                kernelErr = Math.max(kernelErr, Math.abs(k[j] - exact[j]));
            }
        }
        double treeMicros = treeNanos / 1e3 / rows;
        double kernelMicros = kernelNanos / 1e3 / rows;

        // correlated irrelevant features: y depends on x0..x3 only, x4..x7 follow the AR(1) chain
        SplittableRandom cr = new SplittableRandom(seed + 1);
        double[][] cx = PlantedFunctions.gaussian(600, 8, 0.9, cr);
        double[] cy = new double[600];
        for (int i = 0; i < 600; i++) {
            cy[i] = cx[i][0] + cx[i][1] * cx[i][1] + Math.sin(2 * cx[i][2]) + cx[i][3] + 0.1 * cr.nextGaussian();
        }
        TreeView cm = TreeView.boost(cx, cy, 100, 4, 0.1, seed);
        TreeShap cs = new TreeShap(cm);
        double[][] cb = Arrays.copyOfRange(cx, 500, 520);
        double[] pathMass = new double[8];
        double[] intMass = new double[8];
        for (int i = 0; i < 30; i++) {
            double[] p = cs.values(TreeShap.Mode.PATH_DEPENDENT, cx[i], null);
            double[] q = cs.values(TreeShap.Mode.INTERVENTIONAL, cx[i], cb);
            for (int j = 0; j < 8; j++) {
                pathMass[j] += Math.abs(p[j]);
                intMass[j] += Math.abs(q[j]);
            }
        }

        // a continuous noise feature among binary relevant features
        SplittableRandom nr = new SplittableRandom(seed + 2);
        double[][] nx = new double[600][5];
        double[] ny = new double[600];
        for (int i = 0; i < 600; i++) {
            for (int j = 0; j < 5; j++) {
                nx[i][j] = j == 3 ? nr.nextDouble() : nr.nextInt(2);
            }
            ny[i] = nx[i][0] + nx[i][1] + 0.3 * nx[i][0] * nx[i][2] + 0.5 * nr.nextGaussian();
        }
        GradientBoostedTrees gbt = GradientBoostedTrees.fit(nx, ny, GradientBoostedTrees.Objective.REGRESSION,
                GradientBoostedTrees.Settings.defaults());
        TreeView nm = TreeView.boost(nx, ny, 100, 4, 0.1, seed);
        TreeShap ns = new TreeShap(nm);
        double[] meanAbs = new double[5];
        for (int i = 0; i < 100; i++) {
            double[] v = ns.values(TreeShap.Mode.INTERVENTIONAL, nx[i], Arrays.copyOfRange(nx, 500, 520));
            for (int j = 0; j < 5; j++) {
                meanAbs[j] += Math.abs(v[j]);
            }
        }
        return new TreeTable(pathErr, intErr, treeMicros, kernelMicros, kernelErr, kernelMicros / treeMicros,
                share(pathMass, 4), share(intMass, 4), rankOf(gbt.featureImportance(), 3), rankOf(meanAbs, 3));
    }

    private static double share(double[] mass, int from) {
        double all = Arrays.stream(mass).sum();
        double tail = 0;
        for (int j = from; j < mass.length; j++) {
            tail += mass[j];
        }
        return tail / all;
    }

    private static int rankOf(double[] importance, int feature) {
        int rank = 0;
        for (double v : importance) {
            if (v > importance[feature]) {
                rank++;
            }
        }
        return rank;
    }

    /** Interaction detectors compared in {@link #interactions}. */
    public enum Detector {
        /** Exact k-SII (k = 2) by enumeration. */
        K_SII_EXACT,
        /** SHAP-IQ estimate of SII at the budget. */
        SHAP_IQ,
        /** Per-pair permutation estimate of SII at the same budget. */
        PERMUTATION,
        /** Exact Faith-Shap by enumeration. */
        FAITH_SHAP,
        /** Friedman's H-statistic on 40 rows. */
        H_STATISTIC,
        /** The path-dependent TreeSHAP interaction matrix (trees only). */
        TREESHAP_INTERACTION
    }

    /**
     * One detector's scores.
     *
     * @param precision share of the 5 planted pairs among its top 5 (global mean |pair| over rows)
     * @param errorToExact mean absolute error of the pair terms to exact SII (NaN where not an SII estimate)
     * @param evaluations coalitions valued per row
     */
    public record DetectorScores(double precision, double errorToExact, double evaluations) {
    }

    /**
     * The interaction table on {@link PlantedFunctions#interactions} with 8 features.
     *
     * @param trees whether to explain the boosted trees (else an MLP)
     * @param budget the coalition budget of the sampling estimators per row
     * @param seed the seed
     * @return the scores per detector
     */
    public static Map<Detector, DetectorScores> interactions(boolean trees, int budget, long seed) {
        PlantedFunctions.Table data = PlantedFunctions.interactions(800, 8, seed);
        int p = 8;
        TreeView tv = trees ? TreeView.boost(data.x(), data.y(), 100, 4, 0.1, seed) : null;
        Function<double[][], double[]> f = trees ? tv::predict
                : Models.output(Models.mlpRegressor(data.x(), data.y(), 32, 1500, seed));
        double[][] background = Arrays.copyOfRange(data.x(), 700, 710);
        int rows = 8;
        Map<Detector, double[][]> global = new EnumMap<>(Detector.class);
        Map<Detector, Double> error = new EnumMap<>(Detector.class);
        Map<Detector, Double> calls = new EnumMap<>(Detector.class);
        for (Detector d : Detector.values()) {
            global.put(d, new double[p][p]);
            error.put(d, 0.0);
            calls.put(d, 0.0);
        }
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < rows; i++) {
            double[] x = data.x()[i];
            CoalitionGame game = CoalitionGame.interventional(f, x, background);
            ShapleyInteractions.Interactions sii = ShapleyInteractions.exact(ShapleyInteractions.Index.SII, game, p);
            Map<Detector, ShapleyInteractions.Interactions> res = new EnumMap<>(Detector.class);
            res.put(Detector.K_SII_EXACT, ShapleyInteractions.exact(ShapleyInteractions.Index.K_SII, game, p));
            res.put(Detector.FAITH_SHAP, ShapleyInteractions.exact(ShapleyInteractions.Index.FAITH_SHAP, game, p));
            res.put(Detector.SHAP_IQ, ShapleyInteractions.shapIq(ShapleyInteractions.Index.SII, game, p, budget, r));
            res.put(Detector.PERMUTATION, ShapleyInteractions.permutation(game, p, budget, r));
            if (trees) {
                double[][] m = new TreeShap(tv).interactionValues(x);
                res.put(Detector.TREESHAP_INTERACTION, new ShapleyInteractions.Interactions(new double[p], m, 0));
            }
            for (var e : res.entrySet()) {
                double[][] g = global.get(e.getKey());
                double err = 0;
                for (int a = 0; a < p; a++) {
                    for (int b = 0; b < p; b++) {
                        if (a != b) {
                            g[a][b] += Math.abs(e.getValue().pairs()[a][b]);
                            err += Math.abs(e.getValue().pairs()[a][b] - sii.pairs()[a][b]) / (p * (p - 1));
                        }
                    }
                }
                error.merge(e.getKey(), err / rows, Double::sum);
                calls.merge(e.getKey(), e.getValue().evaluations() / (double) rows, Double::sum);
            }
        }
        double[][] h = new EffectPlots(f).hStatistics(Arrays.copyOf(data.x(), 40));
        global.put(Detector.H_STATISTIC, h);
        Map<Detector, DetectorScores> out = new EnumMap<>(Detector.class);
        for (Detector d : Detector.values()) {
            if (d == Detector.TREESHAP_INTERACTION && !trees) {
                continue;
            }
            boolean siiEstimate = d == Detector.SHAP_IQ || d == Detector.PERMUTATION || d == Detector.K_SII_EXACT;
            out.put(d, new DetectorScores(precision(global.get(d)), siiEstimate ? error.get(d) : Double.NaN,
                    d == Detector.H_STATISTIC ? 40 * 40 * 37 : calls.get(d)));
        }
        return out;
    }

    private static double precision(double[][] m) {
        ShapleyInteractions.Interactions ranked = new ShapleyInteractions.Interactions(new double[m.length], m, 0);
        int[][] top = ranked.ranking();
        int hits = 0;
        for (int k = 0; k < PlantedFunctions.PLANTED_PAIRS.length; k++) {
            for (int[] planted : PlantedFunctions.PLANTED_PAIRS) {
                if (top[k][0] == planted[0] && top[k][1] == planted[1]) {
                    hits++;
                }
            }
        }
        return hits / (double) PlantedFunctions.PLANTED_PAIRS.length;
    }

    /** The estimators compared in {@link #estimators}, the four blocks plus the existing KernelSHAP. */
    public enum EstimatorRow {
        /** {@link ShapleyEstimators.Estimator#PERMUTATION}. */
        PERMUTATION,
        /** {@link ShapleyEstimators.Estimator#ANTITHETIC_PERMUTATION}. */
        ANTITHETIC_PERMUTATION,
        /** {@link ShapleyEstimators.Estimator#KERNEL_SHAP_PAIRED}. */
        KERNEL_SHAP_PAIRED,
        /** {@link ShapleyEstimators.Estimator#LEVERAGE_SHAP}. */
        LEVERAGE_SHAP,
        /** {@link ModelAgnosticExplainer#kernelShap} unchanged. */
        KERNEL_SHAP_EXISTING
    }

    /**
     * Mean squared error to exact interventional TreeSHAP, averaged over rows and features.
     *
     * @param p the number of features (8, 16 or 32)
     * @param multiples budgets as multiples of {@code p}
     * @param rows rows explained
     * @param seed the seed
     * @return {@code [estimator][budget]} MSE
     */
    public static Map<EstimatorRow, double[]> estimators(int p, int[] multiples, int rows, long seed) {
        PlantedFunctions.Table data = PlantedFunctions.interactions(600, p, seed);
        TreeView model = TreeView.boost(data.x(), data.y(), 60, 4, 0.15, seed);
        double[][] background = Arrays.copyOfRange(data.x(), 500, 510);
        TreeShap shap = new TreeShap(model);
        Map<EstimatorRow, double[]> out = new EnumMap<>(EstimatorRow.class);
        for (EstimatorRow e : EstimatorRow.values()) {
            out.put(e, new double[multiples.length]);
        }
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < rows; i++) {
            double[] x = data.x()[i];
            double[] exact = shap.values(TreeShap.Mode.INTERVENTIONAL, x, background);
            CoalitionGame game = CoalitionGame.interventional(model::predict, x, background);
            for (int b = 0; b < multiples.length; b++) {
                int budget = multiples[b] * p;
                for (EstimatorRow e : EstimatorRow.values()) {
                    double[] est;
                    if (e == EstimatorRow.KERNEL_SHAP_EXISTING) {
                        est = new ModelAgnosticExplainer(model::predict).kernelShap(x, background, budget, r);
                    } else {
                        AtomicLong used = new AtomicLong();
                        est = ShapleyEstimators.estimate(ShapleyEstimators.Estimator.valueOf(e.name()),
                                CoalitionGame.counting(game, used), p, budget, r);
                    }
                    double se = 0;
                    for (int j = 0; j < p; j++) {
                        se += (est[j] - exact[j]) * (est[j] - exact[j]);
                    }
                    out.get(e)[b] += se / p / rows;
                }
            }
        }
        return out;
    }
}
