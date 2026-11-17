package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.classic.FactorGraphInference.Score;

import java.util.Map;

import org.junit.jupiter.api.Test;

class FactorGraphInferenceTest {

    private static String line(Map<String, Score> m) {
        StringBuilder sb = new StringBuilder();
        m.forEach((k, v) -> sb.append(String.format("%s err %.4f (%.1f ms); ", k, v.error(), v.millis())));
        return sb.toString();
    }

    @Test
    void sprinklerAndAlarmMarginals() {
        double[] rain = FactorGraphInference.sprinkler().exactMarginal(2);
        assertEquals(0.708, rain[1], 0.002, "P(rain | wet grass)");
        double[] burglary = FactorGraphInference.alarm().exactMarginal(0);
        Map<String, Score> s = FactorGraphInference.compare(FactorGraphInference.sprinkler(), 5000, 1);
        Map<String, Score> a = FactorGraphInference.compare(FactorGraphInference.alarm(), 5000, 2);
        System.out.printf("P(rain|wet) %.3f, P(burglary|calls) %.3f; sprinkler: %s alarm: %s%n", rain[1], burglary[1],
                line(s), line(a));
        assertTrue(a.get("loopy BP").error() < 1e-6, "BP is exact on the alarm tree: " + line(a));
        assertTrue(s.get("loopy BP").error() < 0.05, line(s));
        assertTrue(s.get("Gibbs").error() < 0.03, line(s));
    }

    @Test
    void spinGlassApproximations() {
        Map<String, Score> r = FactorGraphInference.isingGrid(4, 3000, 3);
        System.out.printf("4x4 spin glass marginals: %s%n", line(r));
        assertTrue(r.get("loopy BP").error() < 0.05, line(r));
        assertTrue(r.get("Gibbs").error() < 0.03, line(r));
        assertTrue(r.get("p-bit machine").error() < 0.03, line(r));
    }
}
