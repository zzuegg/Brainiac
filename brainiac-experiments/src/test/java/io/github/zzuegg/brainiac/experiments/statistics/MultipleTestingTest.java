package io.github.zzuegg.brainiac.experiments.statistics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.SparseDesigns;

import java.util.Map;

import org.junit.jupiter.api.Test;

class MultipleTestingTest {

    private static String fmt(Map<String, double[]> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s fdr %.3f pow %.3f; ", k, v[0], v[1])));
        return sb.toString();
    }

    @Test
    void bhControlsFdrWithMorePowerThanBonferroni() {
        Map<String, double[]> r = MultipleTesting.batch(1000, 0.8, 2.5, 0, 100, 0.1, 1);
        System.out.printf("batch m=1000 pi0=0.8 indep: %s%n", fmt(r));
        assertTrue(r.get("BH")[0] <= 0.12, fmt(r));
        assertTrue(r.get("UNCORRECTED")[0] > 0.2, fmt(r));
        assertTrue(r.get("BH")[1] > 2 * r.get("BONFERRONI")[1], fmt(r));
        assertTrue(r.get("STOREY_BH")[1] >= r.get("BH")[1], fmt(r));
        Map<String, double[]> c = MultipleTesting.batch(1000, 0.95, 3, 0.7, 100, 0.1, 2);
        System.out.printf("batch m=1000 pi0=0.95 rho=0.7: %s%n", fmt(c));
        assertTrue(c.get("BY")[0] <= 0.1 && c.get("E_BH")[0] <= 0.1, fmt(c));
    }

    @Test
    void onlineProceduresControlFdr() {
        Map<String, double[]> r = MultipleTesting.online(1000, 0.8, 3, 0, SparseDesigns.Placement.UNIFORM, 100,
                0.1, 3);
        System.out.printf("online pi0=0.8 uniform: %s%n", fmt(r));
        for (String k : new String[] {"LOND", "LORD_PLUS_PLUS", "SAFFRON", "E_LORD", "E_SAFFRON"}) {
            assertTrue(r.get(k)[0] <= 0.13, fmt(r));
        }
        assertTrue(r.get("SAFFRON")[1] > r.get("LORD_PLUS_PLUS")[1], fmt(r));
        Map<String, double[]> d = MultipleTesting.online(1000, 0.95, 3, 0.7, SparseDesigns.Placement.EARLY, 100,
                0.1, 4);
        System.out.printf("online pi0=0.95 early rho=0.7: %s%n", fmt(d));
        assertTrue(d.get("E_LORD")[0] <= 0.1, fmt(d));
    }

    @Test
    void knockoffsControlFdrWhereLassoCvDoesNot() {
        Map<String, double[]> r = MultipleTesting.selection(300, 60, 15, 0.3, 4, 0.2, 5);
        System.out.printf("selection n=300 p=60 s=15 rho=0.3 q=0.2: %s%n", fmt(r));
        assertTrue(r.get("knockoff+ (equi)")[0] <= 0.3, fmt(r));
        assertTrue(r.get("lasso CV support")[0] > r.get("knockoff+ (equi)")[0], fmt(r));
        assertTrue(r.get("knockoff+ (SDP)")[1] > 0.3, fmt(r));
    }

    @Test
    void conformalNoveltyControlsFdrWhereFixedThresholdsDoNot() {
        Map<String, double[]> r = MultipleTesting.novelty(1000, 100, 0.1, 2.5, 5, 0.1, 6);
        System.out.printf("novelty ref=1000 m=100 outliers 10%%: %s%n", fmt(r));
        assertTrue(r.get("IF ref 95% quantile")[0] > 0.2, fmt(r));
        assertTrue(r.get("SPLIT_BH (kNN)")[0] <= 0.2, fmt(r));
        assertTrue(r.get("FULL_E_BH (kNN)")[0] <= 0.2, fmt(r));
        assertTrue(r.get("SPLIT_BH (kNN)")[1] > 0.3, fmt(r));
    }
}
