package io.github.zzuegg.brainiac.experiments.exotic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.exotic.ProbabilisticBits.IsingResult;
import io.github.zzuegg.brainiac.experiments.exotic.ProbabilisticBits.IsingSolver;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class IsingSolversTest {

    private static Map<IsingSolver, IsingResult> row(String name, double[][] w, boolean maxCut, int runs,
            int effort) {
        Map<IsingSolver, IsingResult> out = new EnumMap<>(IsingSolver.class);
        StringBuilder line = new StringBuilder();
        for (IsingSolver s : IsingSolver.values()) {
            if (s == IsingSolver.SPECTRAL && (!maxCut || w.length > 256)) {
                continue;
            }
            IsingResult r = ProbabilisticBits.ising(w, maxCut, s, s == IsingSolver.SPECTRAL ? 1 : runs, effort, 1);
            out.put(s, r);
            line.append(String.format("  %s %.3f/%.3f %.0fms", s, r.best(), r.mean(), r.millis()));
        }
        System.out.printf("Ising %s best/mean:%s%n", name, line);
        return out;
    }

    @Test
    void simulatedBifurcationSolvesMaxCut() {
        Map<IsingSolver, IsingResult> dense = row("MaxCut ER n 256 p 0.5", ProbabilisticBits.maxCutGraph(256, 0.5,
                false, 1), true, 4, 500);
        row("MaxCut ER n 512 p 0.02 (sparse)", ProbabilisticBits.maxCutGraph(512, 0.02, false, 2), true, 4, 500);
        row("MaxCut planted n 256 p 0.1", ProbabilisticBits.maxCutGraph(256, 0.1, true, 3), true, 4, 500);
        Map<IsingSolver, IsingResult> sk = row("SK n 256 energy/spin", ProbabilisticBits.sherringtonKirkpatrick(256,
                4), false, 4, 500);
        double bestDense = 0;
        for (IsingResult r : dense.values()) {
            bestDense = Math.max(bestDense, r.best());
        }
        assertTrue(dense.get(IsingSolver.SB_BALLISTIC).best() > 0.99 * bestDense, "bSB is near the best cut");
        assertTrue(sk.get(IsingSolver.SB_DISCRETE).best() < -0.72, "dSB near the SK ground state (about -0.76)");
        assertTrue(sk.get(IsingSolver.SB_DISCRETE).best() <= sk.get(IsingSolver.PBIT_ANNEALING).best(),
                "dSB at least as good as p-bit annealing on SK");
    }
}
