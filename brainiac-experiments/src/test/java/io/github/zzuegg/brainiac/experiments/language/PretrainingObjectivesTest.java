package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.language.PretrainingObjectives.Objective;

import org.junit.jupiter.api.Test;

class PretrainingObjectivesTest {

    @Test
    void objectivesAtEqualCompute() {
        StringBuilder line = new StringBuilder("Stride probe / 2-token infill exact (steps), equal compute:");
        double[][] r = new double[Objective.values().length][];
        for (Objective o : Objective.values()) {
            r[o.ordinal()] = PretrainingObjectives.run(o, 250, 1);
            line.append(String.format(" %s %.3f/%.3f (%.0f);", o, r[o.ordinal()][0], r[o.ordinal()][1],
                    r[o.ordinal()][2]));
        }
        System.out.println(line);
        double[] span = r[Objective.SPAN_CORRUPTION.ordinal()];
        double[] perm = r[Objective.PERMUTATION_LM.ordinal()];
        assertTrue(span[1] > 0.5, "span corruption learns to write the missing span");
        assertTrue(perm[1] > r[Objective.FIM.ordinal()][1], "the permutation LM infills better than FIM at equal compute");
        assertTrue(span[0] > 0.3 && perm[0] > 0.3, "both encoders carry the stride (chance 0.2)");
    }
}
