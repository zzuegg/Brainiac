package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class HstuTransductionTest {

    @Test
    void hstuRetrievesAndRanksWithOneModel() {
        SequentialRecommendation task = new SequentialRecommendation(500, 120, 16, 8, 2);
        Map<String, double[]> r = task.transduction(150, 16, 24, 32);
        StringBuilder line = new StringBuilder("HSTU vs SASRec HR@10/NDCG@10/click AUC (params):");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f/%.3f (%.0f)", k, v[0], v[1], v[2], v[3])));
        System.out.println(line);
        double[] hstu = r.get("HSTU d=32");
        assertTrue(hstu[0] > 0.5, line.toString());
        assertTrue(hstu[0] > r.get("SASRec d=32")[0] - 0.08, line.toString());
        assertTrue(hstu[2] > 0.55, line.toString());
    }
}
