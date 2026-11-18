package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class MultiTaskRankingTest {

    @Test
    void gatedExpertsHandleConflictingTasks() {
        Map<String, double[]> r = MultiTaskRanking.correlationSweep(-0.5, 500, 1);
        StringBuilder line = new StringBuilder("multi-task AUC t1/t2 at rho -0.5:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f |", k, v[0], v[1])));
        System.out.println(line);
        for (Map.Entry<String, double[]> e : r.entrySet()) {
            assertTrue(e.getValue()[0] > 0.6 && e.getValue()[1] > 0.55, line.toString());
        }
        double mmoe = r.get("MMOE")[0] + r.get("MMOE")[1];
        double ple = r.get("PLE")[0] + r.get("PLE")[1];
        double shared = r.get("SHARED_BOTTOM")[0] + r.get("SHARED_BOTTOM")[1];
        assertTrue(Math.max(mmoe, ple) > shared - 0.02, line.toString());
    }

    @Test
    void esmmLearnsConversionOverTheEntireSpace() {
        Map<String, double[]> r = MultiTaskRanking.entireSpace(600, 1);
        StringBuilder line = new StringBuilder("CVR AUC (all impressions)/calibration/CTR AUC:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f/%.3f |", k, v[0], v[1], v[2])));
        System.out.println(line);
        assertTrue(r.get("ESMM (shared bottom)")[0] > 0.6, line.toString());
        assertTrue(r.get("ESMM (shared bottom)")[0] > r.get("click-space CVR (shared bottom)")[0] - 0.05,
                line.toString());
    }
}
