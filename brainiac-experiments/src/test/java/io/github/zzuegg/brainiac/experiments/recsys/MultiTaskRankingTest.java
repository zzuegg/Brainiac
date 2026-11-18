package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class MultiTaskRankingTest {

    @Test
    void gatedExpertsHandleConflictingTasks() {
        Map<String, double[]> r = MultiTaskRanking.correlationSweep(-0.5, 250, 1);
        StringBuilder line = new StringBuilder("multi-task AUC t1/t2 at rho -0.5:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f |", k, v[0], v[1])));
        System.out.printf("%s%n", line);
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
        Map<String, double[]> r = MultiTaskRanking.entireSpace(400, 1);
        StringBuilder line = new StringBuilder("CVR AUC (all impressions)/calibration/CTR AUC:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f/%.3f |", k, v[0], v[1], v[2])));
        System.out.printf("%s%n", line);
        // Genuine negative: against click-space CVR towers that already share a body with the CTR
        // task, ESMM loses about 0.08 CVR AUC even at 7% CTR (its CVR target carries the same
        // information; cf. the ESCM2 critique, Wang et al. SIGIR 2022). Sanity checks only.
        assertTrue(r.get("ESMM (shared bottom)")[0] > 0.6, line.toString());
        assertTrue(r.get("ESMM (shared bottom)")[2] > 0.75, line.toString());
        assertTrue(r.get("click-space CVR (shared bottom)")[0] > 0.65, line.toString());
    }
}
