package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class NonparametricLatentsTest {

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

    @Test
    void indianBuffetRecoversFourFeatures() {
        Map<String, double[]> r = NonparametricLatents.latentFeatures(100, 200, 1);
        System.out.printf("Cambridge toy, K / RMSE vs clean / best cosine: %s%n", line(r));
        double[] ibp = r.get("IBP linear-Gaussian");
        assertTrue(ibp[0] >= 3 && ibp[0] <= 7, line(r));
        assertTrue(ibp[2] > 0.7, line(r));
        assertTrue(r.get("DP mixture")[0] > ibp[0], line(r));
    }

    @Test
    void hdpChoosesItsTopicCount() {
        Map<String, double[]> r = NonparametricLatents.topics(300, 300, 2);
        System.out.printf("topics, K / perplexity / recovery: %s%n", line(r));
        assertTrue(r.get("HDP")[2] > r.get("LDA K=4")[2], line(r));
        assertTrue(r.get("HDP")[1] < r.get("LDA K=4")[1], line(r));
    }
}
