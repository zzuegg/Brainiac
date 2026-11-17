package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.classic.GaussianProcessUncertainty.Scores;

import java.util.Map;

import org.junit.jupiter.api.Test;

class GaussianProcessUncertaintyTest {

    @Test
    void exactGpIsBestCalibratedInsideTheGap() {
        Map<String, Scores> r = GaussianProcessUncertainty.gapRegression(1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s nll %.2f/%.2f crps %.3f/%.3f cov %.2f/%.2f; ", k, v.nllIn(),
                v.nllGap(), v.crpsIn(), v.crpsGap(), v.coverageIn(), v.coverageGap())));
        System.out.printf("gap regression (in/gap): %s%n", sb);
        Scores gp = r.get("exact GP");
        assertTrue(gp.coverageIn() > 0.85 && gp.coverageGap() > 0.85, gp.toString());
        assertTrue(gp.nllIn() < 0, gp.toString());
        for (String k : new String[] {"deep ensemble", "MC dropout", "last-layer Laplace"}) {
            assertTrue(gp.nllGap() < r.get(k).nllGap() + 0.05, k + " " + r.get(k) + " vs GP " + gp);
        }
    }

    @Test
    void sparseVariationalGpApproachesTheExactGpWithMoreInducingPoints() {
        Map<String, Double> r = GaussianProcessUncertainty.sparseScaling(new int[] {4, 8, 16, 32}, 2);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s %.3f, ", k, v)));
        System.out.printf("5k-point test NLL: %s%n", sb);
        assertTrue(r.get("SVGP M=32") < r.get("SVGP M=4") - 0.2, r.toString());
        assertTrue(r.get("SVGP M=32") < r.get("exact GP (n=1000)") + 0.1, r.toString());
    }

    @Test
    void deepKernelLearningOnSeasonalForecasts() {
        Map<String, double[]> r = GaussianProcessUncertainty.deepKernel(3);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s nll %.3f rmse %.3f cov %.2f; ", k, v[0], v[1], v[2])));
        System.out.printf("seasonal one-step: %s%n", sb);
        double[] dkl = r.get("DKL");
        assertTrue(dkl[2] > 0.8, "DKL coverage " + dkl[2]);
        assertTrue(dkl[1] < r.get("MLP alone")[1] * 1.5, "DKL rmse " + dkl[1]);
        assertTrue(dkl[0] < r.get("MLP alone")[0] + 0.3, "DKL nll " + dkl[0]);
    }
}
