package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class SeriesClassifiersElasticTest {

    @Test
    void elasticDistancesOnWarpedTemplates() {
        Map<String, double[]> r = SeriesClassifiers.elastic(SeriesClassifiers.split("warped", 45, 60, 64, 6), 20, 6);
        r.forEach((k, v) -> System.out.printf("elastic warped %s: acc %.3f (%.0f ms)%n", k, v[0], v[1]));
        assertTrue(r.get("DTW 1-NN (full)")[0] > r.get("Euclidean 1-NN")[0], "DTW beats lock-step on warps");
        assertTrue(r.get("Elastic Ensemble")[0] >= r.get("Euclidean 1-NN")[0], "EE");
        assertTrue(r.get("Proximity Forest")[0] > 0.6, "PF " + r.get("Proximity Forest")[0]);
        assertTrue(r.get("LB_Keogh pruned share")[0] > 0.1, "LB_Keogh prunes some candidates");
    }
}
