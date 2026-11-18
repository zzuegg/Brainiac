package io.github.zzuegg.brainiac.experiments.multiagent;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class CooperativeMarlFactorisationTest {

    @Test
    void climbing() {
        Map<String, CooperativeMarl.Score> r = CooperativeMarl.factorisation(CooperativeMarl.Task.CLIMBING, 600, 2);
        System.out.printf("11.6.13 climbing greedy return (optimum 11): %s%n", CooperativeMarl.line(r));
        assertTrue(r.get("QTRAN_BASE").mean() >= r.get("VDN").mean());
    }

    @Test
    void twoStep() {
        Map<String, CooperativeMarl.Score> r = CooperativeMarl.factorisation(CooperativeMarl.Task.TWO_STEP, 600, 2);
        System.out.printf("11.6.13 two-step greedy return (optimum 8): %s%n", CooperativeMarl.line(r));
        assertTrue(r.get("QFIX_SUM").mean() >= r.get("VDN").mean());
    }
}
