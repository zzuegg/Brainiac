package io.github.zzuegg.brainiac.experiments.symbolic;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertTrue;

class LearnedPlanningTest {

    private static LearnedPlanning.Row find(List<LearnedPlanning.Row> rows, String domain, String method) {
        return rows.stream().filter(r -> r.domain().equals(domain) && r.method().equals(method)).findFirst()
            .orElseThrow();
    }

    private static String brief(List<LearnedPlanning.Row> rows) {
        return rows.stream().map(LearnedPlanning.Row::toString).collect(Collectors.joining("; "));
    }

    @Test
    void wlGooseHeuristics() {
        Map<String, LearnedPlanning.Split> d = LearnedPlanning.domains(1);
        List<LearnedPlanning.Row> rows = LearnedPlanning.heuristics(d, 5_000, true, 1);
        System.out.printf("13.2.4 WL-GOOSE (test 2-4x train size): %s%n", brief(rows));
        int ridge = 0;
        int blind = 0;
        for (String dn : d.keySet()) {
            ridge += find(rows, dn, "WL ridge").solved();
            blind += find(rows, dn, "blind").solved();
            LearnedPlanning.Row v = find(rows, dn, "GP variance train/test");
            assertTrue(v.generated() > v.expanded(), "GP variance grows out of distribution on " + dn);
        }
        assertTrue(ridge > blind, "WL ridge beats blind search in coverage");
        assertTrue(find(rows, "gripper", "WL ridge").trainMillis() < 5_000);
    }

    @Test
    void rankingAndPartialSpace() {
        Map<String, LearnedPlanning.Split> d = LearnedPlanning.domains(1);
        Map<String, LearnedPlanning.Split> partial = new LinkedHashMap<>();
        partial.put("logistics", LearnedPlanning.logistics(1, 2, 8));
        partial.put("ferry-many-cars", new LearnedPlanning.Split(d.get("ferry").train(),
            java.util.List.of(io.github.zzuegg.brainiac.experimental.data.PlanningDomains.ferry(4, 12, 7))));
        List<LearnedPlanning.Row> rows = LearnedPlanning.ranking(d, partial, 2_000, 1);
        System.out.printf("13.2.5 ranking/partial space: %s%n", brief(rows));
        int rank = 0;
        for (String dn : d.keySet()) {
            rank += find(rows, dn, "WL ranking").solved();
        }
        assertTrue(rank >= 4, "the ranking heuristic solves test instances");
        assertTrue(find(rows, "ferry-many-cars", "partial-space action-set ranking").instances() == 1);
    }

    @Test
    void generalisedPlanning() {
        Map<String, LearnedPlanning.Split> d = new LinkedHashMap<>();
        Map<String, LearnedPlanning.Split> all = LearnedPlanning.domains(1);
        d.put("visitall", all.get("visitall"));
        d.put("gripper", all.get("gripper"));
        d.put("logistics", LearnedPlanning.logistics(1, 1, 6));
        List<LearnedPlanning.Row> rows = LearnedPlanning.generalisation(d, 5_000, 1);
        System.out.printf("13.2.6 generalised planning (cost = length / h_FF GBFS length): %s%n", brief(rows));
        LearnedPlanning.Row tm = find(rows, "visitall", "transition model ridge");
        LearnedPlanning.Row rnd = find(rows, "visitall", "random actions");
        assertTrue(tm.solved() >= rnd.solved());
    }
}
