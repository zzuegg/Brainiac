package io.github.zzuegg.brainiac.experiments.symbolic;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ClassicalPlanningTest {

    static ClassicalPlanning.Row find(List<ClassicalPlanning.Row> rows, String domain, String method) {
        return rows.stream().filter(r -> r.domain().startsWith(domain) && r.method().startsWith(method)).findFirst()
            .orElseThrow();
    }

    static String brief(List<ClassicalPlanning.Row> rows) {
        return rows.stream().map(r -> String.format("%s/%s %d/%d exp=%.0f eval=%.0f c=%.1f", r.domain(), r.method(),
            r.solved(), r.instances(), r.expanded(), r.evaluated(), r.cost())).collect(Collectors.joining("; "));
    }

    @Test
    void relaxationHeuristics() {
        List<ClassicalPlanning.Row> rows = ClassicalPlanning.relaxation(ClassicalPlanning.domains(0, 1), 10_000);
        System.out.printf("13.2.1 delete relaxation (budget 1e4): %s%n", brief(rows));
        int ffSolved = 0;
        int blindSolved = 0;
        int total = 0;
        for (ClassicalPlanning.Row r : rows) {
            if (r.method().equals("GBFS h_FF")) {
                ffSolved += r.solved();
                total += r.instances();
            }
            if (r.method().startsWith("BFS blind")) {
                blindSolved += r.solved();
            }
        }
        assertEquals(total, ffSolved, "GBFS h_FF solves every instance");
        assertTrue(blindSolved < total, "blind search runs out of budget somewhere");
        ClassicalPlanning.Row gc = find(rows, "gripper", "GBFS goal-count");
        ClassicalPlanning.Row ff = find(rows, "gripper", "GBFS h_FF");
        assertTrue(ff.solved() == 1 && ff.expanded() <= gc.expanded() || gc.solved() == 0);
    }

    @Test
    void landmarkHeuristics() {
        Map<String, List<ClassicalPlanning.Row>> t = ClassicalPlanning.landmarks(30_000, 1);
        System.out.printf("13.2.2 optimal: %s%n", brief(t.get("optimal")));
        System.out.printf("13.2.2 h(s0)/h*: %s%n", t.get("optimal").stream().filter(r -> r.method().contains("/h*"))
            .map(r -> String.format("%s %s=%.2f", r.domain(), r.method(), r.expanded()))
            .collect(Collectors.joining("; ")));
        System.out.printf("13.2.2 satisficing: %s%n", brief(t.get("satisficing")));
        for (ClassicalPlanning.Row r : t.get("optimal")) {
            if (r.method().equals("LM-cut(s0)/h*")) {
                assertTrue(r.expanded() <= 1 + 1e-9, "LM-cut is admissible");
                double hmax = find(t.get("optimal"), r.domain(), "h_max(s0)/h*").expanded();
                assertTrue(r.expanded() >= hmax - 1e-9, "LM-cut dominates h_max");
            }
        }
        ClassicalPlanning.Row lm = find(t.get("optimal"), "logistics-2c-3 unit", "A* LM-cut");
        ClassicalPlanning.Row hm = find(t.get("optimal"), "logistics-2c-3 unit", "A* h_max");
        assertTrue(lm.solved() == lm.instances() && (hm.solved() < hm.instances() || lm.expanded() < hm.expanded()));
        assertEquals(lm.cost(), hm.solved() == hm.instances() ? hm.cost() : lm.cost(), 1e-9, "both optimal");
    }

    @Test
    void abstractionHeuristics() {
        Map<String, double[]> eight = ClassicalPlanning.eightPuzzle(10, 200, 1);
        System.out.printf("13.2.3 8-puzzle {expansions, optimal share, h>h* share}: %s%n", eight.entrySet().stream()
            .map(e -> String.format("%s=%.0f/%.2f/%.2f", e.getKey(), e.getValue()[0], e.getValue()[1],
                e.getValue()[2])).collect(Collectors.joining("; ")));
        List<ClassicalPlanning.Row> rows = ClassicalPlanning.abstraction(30_000, 1);
        System.out.printf("13.2.3 PDBs (eval column = PDB entries): %s%n", rows.stream().map(r -> String.format(
            "%s/%s %d/%d exp=%.0f size=%.0f", r.domain(), r.method(), r.solved(), r.instances(), r.expanded(),
            r.evaluated())).collect(Collectors.joining("; ")));
        assertEquals(1.0, eight.get("additive PDB 4+4")[1], 1e-9, "the additive PDB is admissible");
        assertEquals(0.0, eight.get("additive PDB 4+4")[2], 1e-9);
        assertTrue(eight.get("additive PDB 4+4")[0] < eight.get("Manhattan")[0], "the PDB beats plain Manhattan");
        ClassicalPlanning.Row canon = find(rows, "gripper", "A* canonical");
        ClassicalPlanning.Row hmax = find(rows, "gripper", "A* h_max");
        assertTrue(canon.solved() == 1 && canon.expanded() < hmax.expanded());
    }
}
