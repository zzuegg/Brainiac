package io.github.zzuegg.brainiac.experiments.multiagent;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class CooperativeMarlHeterogeneousTest {

    @Test
    void scoutCarrier() {
        Map<String, CooperativeMarl.Score> r = CooperativeMarl.heterogeneous(CooperativeMarl.HeteroTask.SCOUT_CARRIER,
                3000, 2);
        System.out.printf("11.6.16 scout-carrier return (optimum ~1): %s%n", CooperativeMarl.line(r));
        assertTrue(r.get("HAPPO").mean() > 0.3);
    }
}
