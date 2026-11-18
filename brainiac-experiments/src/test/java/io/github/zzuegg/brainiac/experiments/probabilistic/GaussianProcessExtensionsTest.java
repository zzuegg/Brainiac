package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class GaussianProcessExtensionsTest {

    private static String line(Map<String, double[]> r) {
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

    private static String flat(Map<String, Double> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s %.3f; ", k, v)));
        return sb.toString();
    }

    @Test
    void coregionalisationFillsTheGap() {
        Map<String, double[]> r = GaussianProcessExtensions.multiOutput(40, 120, 1);
        System.out.printf("multi-output gap, RMSE / NLL: %s%n", line(r));
        assertTrue(r.get("ICM (Q=1, R=2)")[0] < r.get("independent GP")[0], line(r));
        assertTrue(r.get("LMC (Q=2, R=1)")[0] < r.get("independent GP")[0], line(r));
    }

    @Test
    void heteroscedasticNoiseImprovesLikelihood() {
        Map<String, double[]> r = GaussianProcessExtensions.heteroscedastic(150, 100, 2);
        System.out.printf("heteroscedastic, RMSE / NLL: %s%n", line(r));
        assertTrue(r.get("heteroscedastic GP (ML-EM)")[1] < r.get("homoscedastic GP")[1], line(r));
    }

    @Test
    void gplvmLatentSpaces() {
        Map<String, Double> r = GaussianProcessExtensions.latentVariables(40, 150, 3);
        System.out.printf("latent 5-NN accuracy: %s%n", flat(r));
        assertTrue(r.get("manifold 20-D / GPLVM") > r.get("manifold 20-D / PCA"), flat(r));
        assertTrue(r.get("digits 64-D / GPLVM") > r.get("digits 64-D / PCA"), flat(r));
    }
}
