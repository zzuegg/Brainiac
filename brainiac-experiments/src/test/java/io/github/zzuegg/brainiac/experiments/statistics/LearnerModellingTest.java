package io.github.zzuegg.brainiac.experiments.statistics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class LearnerModellingTest {

    private static String fmt(Map<String, double[]> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s %.3f/%.3f/%.1fs; ", k, v[0], v[1], v[2])));
        return sb.toString();
    }

    @Test
    void knowledgeTracingBeatsRunningAccuracy() {
        Map<String, double[]> plain = LearnerModelling.run(200, 100, 10, 60, 30, 0.0, false, 60, 1);
        System.out.printf("KT {AUC/mastery corr/fit s} no forgetting: %s%n", fmt(plain));
        Map<String, double[]> drift = LearnerModelling.run(300, 100, 10, 60, 50, 0.05, true, 0, 2);
        System.out.printf("KT {AUC/mastery corr/fit s} forgetting+transfer: %s%n", fmt(drift));
        for (Map<String, double[]> r : java.util.List.of(plain, drift)) {
            assertTrue(r.get("BKT")[0] > r.get("running accuracy")[0], fmt(r));
            assertTrue(r.get("BKT")[1] > r.get("running accuracy")[1], fmt(r));
        }
        // the neural models are small and briefly trained here: only check they learn something
        for (String m : new String[] {"DKT", "SIMPLE_KT", "DKT2_LITE"}) {
            assertTrue(plain.get(m)[0] > 0.55, fmt(plain));
        }
    }
}
