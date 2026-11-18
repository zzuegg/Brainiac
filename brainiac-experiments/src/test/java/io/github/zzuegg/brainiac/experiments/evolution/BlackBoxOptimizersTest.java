package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.experiments.evolution.BlackBoxOptimizers.Function;
import io.github.zzuegg.brainiac.experiments.evolution.BlackBoxOptimizers.Score;

import org.junit.jupiter.api.Test;

import java.util.EnumMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class BlackBoxOptimizersTest {

    @Test
    void populationOptimizersOnBenchmarkFunctions() {
        Map<Function, Map<String, Score>> all = new EnumMap<>(Function.class);
        StringBuilder sb = new StringBuilder();
        for (Function f : Function.values()) {
            Map<String, Score> m = BlackBoxOptimizers.benchmark(f, 10, 20_000, 1, 3);
            all.put(f, m);
            sb.append(f).append(": ");
            m.forEach((k, v) -> sb.append(String.format("%s %.3g (success %.2f), ", k, v.best(), v.success())));
            sb.append("| ");
        }
        System.out.printf("10-D, 20k evaluations, best value (mean of 3 seeds): %s%n", sb);
        for (Function f : Function.values()) {
            Map<String, Score> m = all.get(f);
            assertTrue(m.get("DE").best() < m.get("random").best() && m.get("PSO").best() < m.get("random").best(),
                    "DE and PSO beat random search on " + f);
        }
        assertTrue(all.get(Function.SCHWEFEL).get("DE").best() < all.get(Function.SCHWEFEL).get("CMA-ES").best(),
                "DE beats CMA-ES on the deceptive Schwefel function");
        assertTrue(all.get(Function.RASTRIGIN).get("PSO").best() < all.get(Function.RASTRIGIN).get("CMA-ES").best(),
                "PSO beats restart-free CMA-ES on Rastrigin");
    }
}
