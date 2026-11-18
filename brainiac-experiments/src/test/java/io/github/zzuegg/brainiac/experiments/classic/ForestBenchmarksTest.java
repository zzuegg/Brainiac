package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class ForestBenchmarksTest {

    private static String line(Map<String, Double> m) {
        StringBuilder sb = new StringBuilder();
        m.forEach((k, v) -> sb.append(String.format("%s %.3f, ", k, v)));
        return sb.toString();
    }

    @Test
    void forestsAreCompetitiveTabularBaselines() {
        Map<String, Double> r = ForestBenchmarks.tabular(60, 600, 1);
        System.out.printf("tabular test accuracy: %s%n", line(r));
        assertTrue(r.get("random forest") > 0.8, r.toString());
        assertTrue(r.get("extra trees") > 0.75, r.toString());
        assertTrue(Math.abs(r.get("RF OOB error") - (1 - r.get("random forest"))) < 0.05, r.toString());
    }

    @Test
    void isolationForestFindsPlantedOutliers() {
        Map<String, Double> r = ForestBenchmarks.anomalies(2);
        System.out.printf("anomaly AUROC: %s%n", line(r));
        assertTrue(r.get("isolation forest") > 0.9, r.toString());
        assertTrue(r.get("isolation forest") > r.get("autoencoder") - 0.05, r.toString());
    }
}
