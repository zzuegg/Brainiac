package io.github.zzuegg.brainiac.experiments.statistics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.SparseDesigns;

import java.util.Map;

import org.junit.jupiter.api.Test;

class RegressionModelsTest {

    private static String fmt(Map<String, ?> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> {
            sb.append(k).append(' ');
            if (v instanceof double[] a) {
                for (double x : a) {
                    sb.append(String.format("%.3f ", x));
                }
            } else {
                sb.append(String.format("%.3f", (Double) v));
            }
            sb.append("; ");
        });
        return sb.toString();
    }

    @Test
    void highBreakdownEstimatorsSurviveContamination() {
        Map<String, Double> v = RegressionModels.robust(200, 5, 0.1, false, 1);
        Map<String, Double> l = RegressionModels.robust(200, 5, 0.3, true, 2);
        System.out.printf("robust ||b-beta||: 10%% vertical: %s| 30%% leverage: %s%n", fmt(v), fmt(l));
        assertTrue(v.get("HUBER") < 0.5 * v.get("OLS"), fmt(v));
        assertTrue(l.get("LTS") < 0.5 * l.get("OLS") && l.get("MM") < 0.5 * l.get("OLS"), fmt(l));
        assertTrue(l.get("HUBER") > l.get("MM"), fmt(l));
        double[] auc = RegressionModels.outlierDetection(300, 5, 0.2, 3);
        double[] mom = RegressionModels.medianOfMeans(200, 400, 4);
        System.out.printf("outlier AUROC classical %.3f MCD %.3f; t2 location |err| mean %.3f MoM %.3f, p99 %.3f vs %.3f%n",
                auc[0], auc[1], mom[0], mom[1], mom[2], mom[3]);
        assertTrue(auc[1] > 0.95 && auc[1] > auc[0]);
        assertTrue(mom[3] < mom[2]);
    }

    @Test
    void linearQuantileRegressionAndRearrangement() {
        Map<String, double[]> q = RegressionModels.quantiles(200, 5);
        System.out.printf("quantiles n=200 {pinball, cov80, cov90}: %s%n", fmt(q));
        assertTrue(q.get("linear QR LP")[0] < q.get("quantile forest")[0], fmt(q));
        assertTrue(Math.abs(q.get("linear QR LP")[2] - 0.9) < 0.06, fmt(q));
        double[] c = RegressionModels.crossing(25, 6);
        System.out.printf("crossing n=25 9 levels: rate %.3f -> %.3f, pinball %.4f -> %.4f%n", c[0], c[1], c[2], c[3]);
        assertTrue(c[1] == 0 && c[3] <= c[2] + 1e-12);
    }

    @Test
    void additiveModelsAndMars() {
        Map<String, double[]> f = RegressionModels.smooth(RegressionModels.SmoothTask.FRIEDMAN1, 300, 7);
        Map<String, double[]> a = RegressionModels.smooth(RegressionModels.SmoothTask.ADDITIVE, 300, 8);
        System.out.printf("Friedman1 {RMSE, ms}: %s| additive: %s%n", fmt(f), fmt(a));
        assertTrue(f.get("MARS")[0] < f.get("GAM (P-splines)")[0], fmt(f));
        assertTrue(a.get("GAM (P-splines)")[0] < a.get("poly-2 ridge")[0], fmt(a));
        assertTrue(a.get("GAM (P-splines)")[0] < 1.2 * a.get("EBM")[0], fmt(a));
    }

    @Test
    void matchedCountGlmsBeatLogOls() {
        Map<String, double[]> nb = RegressionModels.counts(SparseDesigns.CountFamily.NEGATIVE_BINOMIAL, 3, 0, 500, 9);
        Map<String, double[]> zip = RegressionModels.counts(SparseDesigns.CountFamily.ZERO_INFLATED_POISSON, 1, 0.5,
                500, 10);
        Map<String, double[]> tw = RegressionModels.counts(SparseDesigns.CountFamily.TWEEDIE, 2, 0, 500, 11);
        System.out.printf("counts {deviance, |zero share err|, cov90}: NB: %s| ZIP: %s| Tweedie: %s%n", fmt(nb),
                fmt(zip), fmt(tw));
        assertTrue(zip.get("ZERO_INFLATED_POISSON GLM")[0] < zip.get("OLS on log(1+y)")[0], fmt(zip));
        assertTrue(zip.get("ZERO_INFLATED_POISSON GLM")[1] < 0.03, fmt(zip));
        assertTrue(zip.get("POISSON GLM")[1] > zip.get("ZERO_INFLATED_POISSON GLM")[1], fmt(zip));
        assertTrue(nb.get("NEGATIVE_BINOMIAL GLM")[2] > nb.get("POISSON GLM")[2], fmt(nb));
        assertTrue(tw.get("TWEEDIE GLM")[0] < tw.get("OLS on log(1+y)")[0], fmt(tw));
    }
}
