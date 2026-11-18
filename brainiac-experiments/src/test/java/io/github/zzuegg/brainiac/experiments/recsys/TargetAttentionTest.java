package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class TargetAttentionTest {

    @Test
    void targetAttentionPredictsClicks() {
        SequentialRecommendation task = new SequentialRecommendation(250, 120, 22, 8, 1);
        Map<String, Double> r = task.clickAuc(300);
        System.out.printf("click AUC: FM %.3f, mean pooling %.3f, DIN %.3f%n", r.get("FM"), r.get("mean pooling"),
                r.get("DIN"));
        assertTrue(r.get("DIN") > 0.6, r.toString());
    }
}
