package io.github.zzuegg.brainiac.experiments.estimation;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class RandomisedLinearAlgebraTest {

    private static String line(Map<String, double[]> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> {
            sb.append(k).append(' ');
            for (double x : v) {
                sb.append(String.format("%.3g ", x));
            }
            sb.append("; ");
        });
        return sb.toString();
    }

    @Test
    void randomisedSvdIsNearOptimalWithPowerIterations() {
        Map<String, double[]> r = RandomisedLinearAlgebra.lowRank(800, 200, 20, 1);
        System.out.printf("rank-20 error / optimal, speed-up: %s%n", line(r));
        assertTrue(r.get("polynomial / q=2")[0] < 1.05, line(r));
        assertTrue(r.get("exponential / q=2")[0] < 1.05, line(r));
        assertTrue(r.get("polynomial / q=0")[0] > r.get("polynomial / q=2")[0], line(r));
        assertTrue(r.get("polynomial / q=2")[1] > 1, line(r));
    }

    @Test
    void sketchAndPreconditionSolvesIllConditionedLeastSquares() {
        Map<String, double[]> r = RandomisedLinearAlgebra.leastSquares(6000, 50, 1e8, 2);
        System.out.printf("kappa 1e8 least squares rel. error, iterations: %s%n", line(r));
        for (String s : new String[] {"GAUSSIAN", "COUNT_SKETCH", "SRHT"}) {
            assertTrue(r.get("sketch+LSQR " + s)[0] < 1e-4, line(r));
            assertTrue(r.get("sketch+LSQR " + s)[1] < 60, line(r));
        }
        assertTrue(r.get("plain LSQR (100 it)")[0] > 1e-2, line(r));
        assertTrue(r.get("normal equations")[0] > r.get("sketch+LSQR GAUSSIAN")[0] * 100, line(r));
    }

    @Test
    void lanczosQuadratureEstimatesSpectralSums() {
        double[] ld = RandomisedLinearAlgebra.logDeterminant(300, 10, 30, 3);
        Map<String, double[]> h = RandomisedLinearAlgebra.hessianSpectrum(4);
        System.out.printf("GP logdet exact %.2f SLQ %.2f (rel %.4f); MLP Hessian: %s%n", ld[0], ld[1], ld[2],
                line(h));
        assertTrue(ld[2] < 0.05, String.valueOf(ld[2]));
        double[] top = h.get("top eigenvalue");
        assertTrue(Math.abs(top[1] - top[0]) / top[0] < 0.01, line(h));
    }
}
