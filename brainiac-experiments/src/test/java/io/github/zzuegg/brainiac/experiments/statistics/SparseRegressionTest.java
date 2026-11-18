package io.github.zzuegg.brainiac.experiments.statistics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.SparseDesigns;

import java.util.Map;

import org.junit.jupiter.api.Test;

class SparseRegressionTest {

    private static String fmt(Map<String, double[]> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> {
            sb.append(k).append(' ');
            for (double x : v) {
                sb.append(String.format("%.3f ", x));
            }
            sb.append("; ");
        });
        return sb.toString();
    }

    @Test
    void lassoBeatsRidgeWhenSparseAndOneSeImprovesSupport() {
        SparseDesigns.Regression d = SparseDesigns.linear(100, 200, 5, 0.5, 3, SparseDesigns.Noise.GAUSSIAN,
                SparseDesigns.Coefficients.UNIFORM, 1);
        Map<String, double[]> r = SparseRegression.linearMethods(d, 1);
        System.out.printf("sparse n=100 p=200 s=5 rho=0.5 {testMSE, supportF1}: %s%n", fmt(r));
        assertTrue(r.get("lasso CV min")[0] < r.get("ridge (holdout)")[0], fmt(r));
        assertTrue(r.get("lasso CV 1-SE")[1] >= r.get("lasso CV min")[1], fmt(r));
    }

    @Test
    void correlatedDesignFavoursElasticNetOrTies() {
        SparseDesigns.Regression d = SparseDesigns.linear(100, 50, 10, 0.9, 2, SparseDesigns.Noise.GAUSSIAN,
                SparseDesigns.Coefficients.UNIFORM, 2);
        Map<String, double[]> r = SparseRegression.linearMethods(d, 2);
        System.out.printf("sparse n=100 p=50 s=10 rho=0.9 {testMSE, supportF1}: %s%n", fmt(r));
        assertTrue(r.get("elastic net CV min")[0] < 1.3 * r.get("lasso CV min")[0], fmt(r));
    }

    @Test
    void strongRulesCutUpdatesAndLarsAgrees() {
        SparseDesigns.Regression d = SparseDesigns.linear(200, 1000, 10, 0.3, 3, SparseDesigns.Noise.GAUSSIAN,
                SparseDesigns.Coefficients.UNIFORM, 3);
        double[] c = SparseRegression.pathCost(d, 30);
        System.out.printf("lasso path p=1000: updates screened %.0f vs plain %.0f (%.1fx), max diff %.2e, "
                + "LARS vs CD %.2e, ms %.0f vs %.0f%n", c[0], c[1], c[1] / c[0], c[2], c[3], c[4], c[5]);
        assertTrue(c[1] > 3 * c[0]);
        assertTrue(c[2] < 1e-5);
        assertTrue(c[3] < 1e-5);
    }

    @Test
    void foldedConcavePenaltiesReduceEstimationError() {
        SparseDesigns.Regression d = SparseDesigns.linear(150, 100, 10, 0.3, 4, SparseDesigns.Noise.GAUSSIAN,
                SparseDesigns.Coefficients.MIXED, 4);
        Map<String, double[]> r = SparseRegression.nonConvex(d, 4);
        System.out.printf("mixed coefficients {||b-beta||, F1, FP, large-coef error}: %s%n", fmt(r));
        assertTrue(r.get("MCP")[0] < r.get("LASSO")[0], fmt(r));
        assertTrue(r.get("SCAD")[3] < r.get("LASSO")[3], fmt(r));
    }

    @Test
    void groupLassoRecoversWholeGroups() {
        SparseDesigns.Regression d = SparseDesigns.grouped(100, 20, 5, 3, 0.5, 2, 5);
        Map<String, double[]> r = SparseRegression.grouped(d, 5, 5);
        System.out.printf("grouped 20x5, 3 active {group F1, feature F1, test MSE, nnz}: %s%n", fmt(r));
        assertTrue(r.get("group lasso")[1] > r.get("lasso")[1], fmt(r));
        assertTrue(r.get("group lasso")[2] < r.get("lasso")[2], fmt(r));
    }

    @Test
    void graphicalModelsBeatThresholdedInverseWhenNSmall() {
        for (SparseDesigns.GraphShape shape : new SparseDesigns.GraphShape[] {SparseDesigns.GraphShape.CHAIN,
            SparseDesigns.GraphShape.HUB}) {
            SparseDesigns.GaussianGraph g = SparseDesigns.gaussianGraph(shape, 30, 50, 6);
            Map<String, double[]> r = SparseRegression.graphs(g);
            System.out.printf("GGM %s p=30 n=50 {edge F1, Frobenius}: %s%n", shape, fmt(r));
            assertTrue(Math.max(r.get("GLASSO")[0], r.get("NEIGHBOURHOOD_AND")[0])
                    > r.get("thresholded inverse")[0], fmt(r));
        }
    }
}
