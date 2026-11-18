package io.github.zzuegg.brainiac.experiments.symbolic;

import io.github.zzuegg.brainiac.experimental.data.PlanningDomains;
import io.github.zzuegg.brainiac.experimental.symbolic.StripsTask;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SatSolvingTest {

    private static SatSolving.Row find(List<SatSolving.Row> rows, String task, String method) {
        return rows.stream().filter(r -> r.task().startsWith(task) && r.method().startsWith(method)).findFirst()
            .orElseThrow();
    }

    private static String brief(List<SatSolving.Row> rows) {
        return rows.stream().map(SatSolving.Row::toString).collect(Collectors.joining("; "));
    }

    @Test
    void cdclAgainstDpllAndWalkSat() {
        List<SatSolving.Row> rows = SatSolving.cdcl(new int[] {50, 100, 200}, 50, 3, new int[] {5, 7}, 1);
        System.out.printf("13.2.9 CDCL: %s%n", brief(rows));
        SatSolving.Row big = find(rows, "3-SAT n=200", "CDCL");
        assertEquals(3, big.solved(), "CDCL decides 200-variable 3-SAT at the threshold");
        assertTrue(find(rows, "PHP-7", "CDCL").conflicts() > 10 * find(rows, "PHP-5", "CDCL").conflicts(),
            "pigeonhole stays exponential");
        List<SatSolving.Row> ab = SatSolving.ablation(120, 3, 5, false, 1);
        System.out.printf("13.2.9 ablation (work = conflicts): %s%n", brief(ab));
        System.out.printf("13.2.9 NeuroSAT SR(20) accuracy after SR(10) training: %.2f (CDCL decides 1.00)%n",
            SatSolving.neuroSat(10, 20, 150, 1));
    }

    @Test
    void rlafGuidance() {
        Map<String, double[]> r = SatSolving.rlaf(15, 30, 12, 12, 5, 6, 1);
        System.out.printf("13.2.10 RLAF {mean decisions / plain VSIDS, median ratio}: %s%n", r.entrySet().stream()
            .map(e -> String.format("%s=%.2f/%.2f", e.getKey(), e.getValue()[0], e.getValue()[1]))
            .collect(Collectors.joining("; ")));
        assertTrue(r.get("n=15 RLAF")[0] < r.get("n=15 random weights")[0] + 0.5, "RLAF is no worse than random");
    }

    @Test
    void planningAsSat() {
        Map<String, StripsTask> tasks = new LinkedHashMap<>();
        tasks.put("gripper-3", PlanningDomains.gripper(3));
        tasks.put("logistics-2c-3", PlanningDomains.logistics(2, 1, 3, false, 1));
        tasks.put("blocksworld-5", PlanningDomains.blocksworld(5, 1));
        List<SatSolving.Row> rows = SatSolving.planning(tasks, 1);
        System.out.printf("13.2.16 SATPlan (work = makespan or plan length): %s%n", brief(rows));
        for (String t : tasks.keySet()) {
            double fa = find(rows, t, "SATPlan FORALL_STEP linear").conflicts();
            double ex = find(rows, t, "SATPlan EXISTS_STEP linear").conflicts();
            double seq = find(rows, t, "SATPlan SEQUENTIAL").conflicts();
            assertEquals(seq, find(rows, t, "A* LM-cut").conflicts(), 1e-9, "sequential SATPlan is length-optimal");
            assertTrue(ex <= fa && fa <= seq, "exists-step <= forall-step <= sequential horizons");
        }
    }
}
