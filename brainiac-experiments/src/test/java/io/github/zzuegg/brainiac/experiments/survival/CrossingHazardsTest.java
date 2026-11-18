package io.github.zzuegg.brainiac.experiments.survival;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class CrossingHazardsTest {

    @Test
    void flexibleHazardModelsBeatCoxWhenHazardsCross() {
        Map<String, SurvivalShootout.Score> r = SurvivalShootout.crossing(0.3, 1000, 400, 30, 1);
        String line = SurvivalShootoutTest.line("crossing hazards, td-C/IBS:", r);
        System.out.println(line);
        double flexible = Math.max(Math.max(r.get("discrete-time hazard net").cIndex(), r.get("DeepHit").cIndex()),
                r.get("random survival forest").cIndex());
        assertTrue(flexible > r.get("Cox").cIndex(), line);
        assertTrue(r.get("random survival forest").brier() < r.get("Cox").brier(), line);
    }

    @Test
    void deepHitLearnsCauseSpecificIncidence() {
        Map<String, double[]> r = SurvivalShootout.competing(1000, 400, 1);
        StringBuilder line = new StringBuilder("competing risks CIF error c1/c2, td-C c1/c2:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f %.3f/%.3f |", k, v[0], v[1], v[2], v[3])));
        System.out.println(line);
        double[] hit = r.get("DeepHit (2 cause heads)");
        assertTrue(hit[2] > 0.6 && hit[3] > 0.55, line.toString());
        assertTrue(hit[0] < 0.15 && hit[1] < 0.15, line.toString());
    }
}
