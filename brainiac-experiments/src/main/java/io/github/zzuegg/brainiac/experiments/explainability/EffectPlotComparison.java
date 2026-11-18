package io.github.zzuegg.brainiac.experiments.explainability;

import io.github.zzuegg.brainiac.experimental.boost.ExplainableBoosting;
import io.github.zzuegg.brainiac.experimental.data.PlantedFunctions;
import io.github.zzuegg.brainiac.experimental.explain.EffectPlots;
import io.github.zzuegg.brainiac.experimental.ml.RandomForest;
import io.github.zzuegg.brainiac.experimental.train.ModelAgnosticExplainer;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 13 item 13.1.1: partial dependence against accumulated local effects, and Friedman's
 * H-statistic, on {@link PlantedFunctions#additive} (five planted additive shapes and an
 * {@code x5 x6} interaction) at input correlation {@code rho = 0} and {@code 0.9}.
 *
 * <p>For each model (the true function, core {@link GradientBoostedTrees}, a regression
 * {@link RandomForest} and an MLP) the centred PDP and ALE curves of features 0-4 are compared
 * with the centred planted shapes by the root-mean-square error over the data; baselines are the
 * {@link ExplainableBoosting} shape functions and the KernelSHAP dependence
 * ({@link ModelAgnosticExplainer#kernelShap} value of feature {@code j} against {@code x_j}) of
 * the GBT. The H-statistic's AUROC for picking the planted pair out of all 21 pairs is compared
 * with EBM's FAST pair ranking.
 */
public final class EffectPlotComparison {

    private EffectPlotComparison() {
    }

    /** The explained models. */
    public enum Fitted {
        /** The noiseless generating function. */
        TRUTH,
        /** Core gradient-boosted trees. */
        GBT,
        /** A regression random forest. */
        FOREST,
        /** A two-layer tanh MLP. */
        MLP
    }

    /**
     * One row of the effect table.
     *
     * @param model the model
     * @param rho the input correlation
     * @param pdpError RMS error of the PDP curves against the planted shapes, averaged over features 0-4
     * @param aleError the same for ALE
     * @param hAuroc AUROC of the H-statistic for the planted pair among all pairs
     */
    public record Row(Fitted model, double rho, double pdpError, double aleError, double hAuroc) {
    }

    /**
     * The baselines at one correlation.
     *
     * @param rho the input correlation
     * @param ebmError RMS error of EBM's shape functions
     * @param shapError RMS error of the GBT's KernelSHAP dependence
     * @param ebmPairAuroc AUROC of EBM's FAST ranking for the planted pair
     */
    public record Baselines(double rho, double ebmError, double shapError, double ebmPairAuroc) {
    }

    /**
     * Fits one model.
     *
     * @param model which model
     * @param data the training table
     * @param seed the seed
     * @return its batch prediction function
     */
    static Function<double[][], double[]> fit(Fitted model, PlantedFunctions.Table data, long seed) {
        return switch (model) {
            case TRUTH -> rows -> Arrays.stream(rows).mapToDouble(PlantedFunctions::additiveTruth).toArray();
            case GBT -> {
                GradientBoostedTrees g = GradientBoostedTrees.fit(data.x(), data.y(),
                        GradientBoostedTrees.Objective.REGRESSION,
                        GradientBoostedTrees.Settings.defaults().withTrees(150).withDepth(3));
                yield g::predict;
            }
            case FOREST -> {
                RandomForest f = RandomForest.fitRegression(data.x(), data.y(),
                        new RandomForest.Settings(RandomForest.Mode.RANDOM_FOREST, 40, 10, 5, 0, seed));
                yield rows -> Arrays.stream(rows).mapToDouble(f::predictValue).toArray();
            }
            case MLP -> Models.output(Models.mlpRegressor(data.x(), data.y(), 32, 1000, seed));
        };
    }

    private static double curveError(Function<Double, Double> curve, double[][] x, int j) {
        double meanTruth = 0;
        double meanCurve = 0;
        for (double[] row : x) {
            meanTruth += PlantedFunctions.shape(j, row[j]) / x.length;
            meanCurve += curve.apply(row[j]) / x.length;
        }
        double se = 0;
        for (double[] row : x) {
            double d = (curve.apply(row[j]) - meanCurve) - (PlantedFunctions.shape(j, row[j]) - meanTruth);
            se += d * d;
        }
        return Math.sqrt(se / x.length);
    }

    private static double pairAuroc(double[][] h, int a, int b) {
        int p = h.length;
        double target = h[a][b];
        double below = 0;
        int others = 0;
        for (int i = 0; i < p; i++) {
            for (int j = i + 1; j < p; j++) {
                if (i == a && j == b) {
                    continue;
                }
                others++;
                below += h[i][j] < target ? 1 : h[i][j] == target ? 0.5 : 0;
            }
        }
        return below / others;
    }

    /**
     * Measures one model at one correlation.
     *
     * @param model the model
     * @param rho the correlation
     * @param seed the seed
     * @return the row
     */
    public static Row run(Fitted model, double rho, long seed) {
        PlantedFunctions.Table data = PlantedFunctions.additive(800, rho, seed);
        Function<double[][], double[]> f = fit(model, data, seed);
        EffectPlots plots = new EffectPlots(f);
        double[][] eval = Arrays.copyOf(data.x(), 300);
        double pdp = 0;
        double ale = 0;
        for (int j = 0; j < 5; j++) {
            EffectPlots.Curve c = plots.curve(EffectPlots.Effect.PDP, eval, j, 21);
            pdp += curveError(c::at, eval, j) / 5;
            EffectPlots.Curve a = plots.curve(EffectPlots.Effect.ALE, eval, j, 21);
            ale += curveError(a::at, eval, j) / 5;
        }
        double[][] h = plots.hStatistics(Arrays.copyOf(data.x(), 40));
        return new Row(model, rho, pdp, ale, pairAuroc(h, 5, 6));
    }

    /**
     * Measures the baselines at one correlation.
     *
     * @param rho the correlation
     * @param seed the seed
     * @return the baselines
     */
    public static Baselines baselines(double rho, long seed) {
        PlantedFunctions.Table data = PlantedFunctions.additive(800, rho, seed);
        ExplainableBoosting ebm = ExplainableBoosting.fit(data.x(), data.y(),
                GradientBoostedTrees.Objective.REGRESSION.loss(), ExplainableBoosting.Settings.defaults());
        double[][] eval = Arrays.copyOf(data.x(), 300);
        double ebmError = 0;
        for (int j = 0; j < 5; j++) {
            int jj = j;
            ebmError += curveError(v -> ebm.shape(jj, v), eval, j) / 5;
        }
        GradientBoostedTrees g = GradientBoostedTrees.fit(data.x(), data.y(), GradientBoostedTrees.Objective.REGRESSION,
                GradientBoostedTrees.Settings.defaults().withTrees(150).withDepth(3));
        ModelAgnosticExplainer shap = new ModelAgnosticExplainer(g::predict);
        double[][] background = Arrays.copyOfRange(data.x(), 300, 320);
        int rows = 40;
        double[][] phi = new double[rows][];
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < rows; i++) {
            phi[i] = shap.kernelShap(data.x()[i], background, 200, r);
        }
        double shapError = 0;
        for (int j = 0; j < 5; j++) {
            double mt = 0;
            double mp = 0;
            for (int i = 0; i < rows; i++) {
                mt += PlantedFunctions.shape(j, data.x()[i][j]) / rows;
                mp += phi[i][j] / rows;
            }
            double se = 0;
            for (int i = 0; i < rows; i++) {
                double d = (phi[i][j] - mp) - (PlantedFunctions.shape(j, data.x()[i][j]) - mt);
                se += d * d;
            }
            shapError += Math.sqrt(se / rows) / 5;
        }
        int[][] ranking = ebm.pairRanking();
        double[] gains = ebm.pairRankingGains();
        double[][] h = new double[7][7];
        for (int k = 0; k < ranking.length; k++) {
            h[ranking[k][0]][ranking[k][1]] = gains[k];
            h[ranking[k][1]][ranking[k][0]] = gains[k];
        }
        return new Baselines(rho, ebmError, shapError, pairAuroc(h, 5, 6));
    }
}
