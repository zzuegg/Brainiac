package io.github.zzuegg.brainiac.experiments.statistics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class ComparisonsTest {

    private static String fmt(Map<String, ?> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> {
            sb.append(k).append(' ');
            if (v instanceof double[] a) {
                for (double x : a) {
                    sb.append(String.format("%.4f ", x));
                }
            } else {
                sb.append(String.format("%.3f", (Double) v));
            }
            sb.append("; ");
        });
        return sb.toString();
    }

    @Test
    void bradleyTerryBeatsWinRateOnSparseGraphs() {
        Map<String, Double> sparse = Comparisons.pairwise(100, 4, 3, 1);
        Map<String, Double> dense = Comparisons.pairwise(100, 20, 3, 2);
        System.out.printf("pairwise Kendall tau: 4/item %s| 20/item %s%n", fmt(sparse), fmt(dense));
        assertTrue(sparse.get("BRADLEY_TERRY_MM") > sparse.get("win rate"), fmt(sparse));
        Map<String, Double> agg = Comparisons.aggregation(30, 15, 3);
        System.out.printf("aggregation of 15 PL rankings: %s%n", fmt(agg));
        assertTrue(agg.get("KEMENY_LOCAL_SEARCH") >= agg.get("BORDA") - 0.02, fmt(agg));
        assertTrue(agg.get("PLACKETT_LUCE") > agg.get("single voter"), fmt(agg));
    }

    @Test
    void bayesianRatingsBeatEloOnNewcomers() {
        Map<String, double[]> r = Comparisons.ratings(100, 20000, 500, 10, 0.01, 4);
        System.out.printf("ratings {log loss all, newcomers' first 10}: %s%n", fmt(r));
        assertTrue(r.get("TrueSkill")[1] < r.get("Elo K=32")[1], fmt(r));
        assertTrue(r.get("Glicko-2")[1] < r.get("Elo K=32")[1], fmt(r));
    }
}
