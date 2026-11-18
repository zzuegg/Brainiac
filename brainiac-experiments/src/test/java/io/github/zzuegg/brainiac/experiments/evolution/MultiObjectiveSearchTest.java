package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.experiments.evolution.MultiObjectiveSearch.Problem;
import io.github.zzuegg.brainiac.experiments.evolution.MultiObjectiveSearch.Score;

import org.junit.jupiter.api.Test;

import java.util.EnumMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class MultiObjectiveSearchTest {

    @Test
    void nsga2OnZdtProblems() {
        Map<Problem, Map<String, Score>> all = new EnumMap<>(Problem.class);
        StringBuilder sb = new StringBuilder();
        for (Problem p : Problem.values()) {
            Map<String, Score> m = MultiObjectiveSearch.compare(p, 10_000, 3);
            all.put(p, m);
            sb.append(p).append(": ");
            m.forEach((k, v) -> sb.append(String.format("%s returned %.3f / archive %.3f (spread %.2f), ", k,
                    v.returnedHypervolume(), v.finalHypervolume(), v.spread())));
            sb.append("| ");
        }
        System.out.printf("ZDT, 10k evaluations, hypervolume of the returned set / of the whole archive: %s%n", sb);
        for (Problem p : Problem.values()) {
            Map<String, Score> m = all.get(p);
            assertTrue(m.get("NSGA-II").returnedHypervolume() > m.get("weighted-sum CMA-ES").returnedHypervolume()
                    && m.get("NSGA-II").returnedHypervolume() > m.get("MAP-Elites").returnedHypervolume(),
                    "NSGA-II returns the best front on " + p);
        }
    }
}
