package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ChangePointsTest {

    @Test
    void peltAndBocpdBeatTheBaselines() {
        Map<String, ChangePoints.Score> r = ChangePoints.run(800, 3, 5);
        r.forEach((k, v) -> System.out.printf("change points %s: %s%n", k, v));
        assertTrue(r.get("PELT").f1() > 0.6, r.toString());
        assertTrue(r.get("PELT").f1() > r.get("Kalman innovations").f1(), r.toString());
        assertTrue(r.get("BOCPD").recallVariance() > r.get("Kalman innovations").recallVariance(), r.toString());
        assertTrue(r.get("single change model").f1() < r.get("PELT").f1(), r.toString());
    }
}
