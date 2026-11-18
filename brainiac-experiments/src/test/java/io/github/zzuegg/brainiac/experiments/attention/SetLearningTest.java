package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.attention.SetLearning.Architecture;
import io.github.zzuegg.brainiac.experiments.attention.SetLearning.Task;

import org.junit.jupiter.api.Test;

class SetLearningTest {

    private static double[][] all(Task t, int steps) {
        StringBuilder line = new StringBuilder(t + " at n=8/n=12 (train 3-8):");
        double[][] m = new double[Architecture.values().length][];
        for (Architecture a : Architecture.values()) {
            m[a.ordinal()] = SetLearning.run(t, a, steps, 1);
            line.append(String.format(" %s %.3f/%.3f;", a, m[a.ordinal()][0], m[a.ordinal()][1]));
        }
        System.out.println(line);
        return m;
    }

    @Test
    void maxGeneralisesToLargerSets() {
        double[][] m = all(Task.MAX, 400);
        assertTrue(m[Architecture.DEEP_SETS.ordinal()][1] < m[Architecture.SORTED_MLP.ordinal()][1]
                || m[Architecture.SET_TRANSFORMER.ordinal()][1] < m[Architecture.SORTED_MLP.ordinal()][1],
                "a permutation-invariant encoder keeps the maximum on sets larger than any seen");
    }

    @Test
    void uniqueCount() {
        double[][] m = all(Task.UNIQUE, 700);
        assertTrue(m[Architecture.SORTED_MLP.ordinal()][0] > 0.5, "sorting makes distinct tokens adjacent and easy to count");
        for (Architecture a : Architecture.values()) {
            assertTrue(m[a.ordinal()][1] < m[a.ordinal()][0], a + ": counting does not extrapolate to larger sets");
        }
    }

    @Test
    void amortisedClustering() {
        double[][] m = all(Task.CLUSTERS, 400);
        assertTrue(m[Architecture.SET_TRANSFORMER.ordinal()][0] < 1.0, "the Set Transformer locates both centres");
    }
}
