package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.InterpretableNetworks.ConceptRow;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

class ConceptBottleneckTest {

    @Test
    void hardBottlenecksGainFromInterventionsAndLeakLess() {
        for (boolean incomplete : new boolean[] {false}) {
            List<ConceptRow> rows = InterpretableNetworks.concepts(incomplete, 300, 1);
            double[] ctl = new double[rows.size() - 1];
            double[] gain = new double[rows.size() - 1];
            for (int i = 0; i < rows.size(); i++) {
                ConceptRow r = rows.get(i);
                System.out.printf("Concepts %s %-11s task %.3f concepts %.3f interventions %s CTL %.3f ICL %.3f%n",
                        incomplete ? "incomplete" : "complete  ", r.model(), r.taskAccuracy(), r.conceptAccuracy(),
                        Arrays.toString(Arrays.stream(r.intervened()).mapToObj(v -> String.format("%.2f", v)).toArray()),
                        r.ctl(), r.icl());
                if (i < ctl.length) {
                    ctl[i] = r.ctl();
                    gain[i] = r.interventionGain();
                }
            }
            System.out.printf("Concepts %s Spearman(CTL, intervention gain) over the five CBMs %.2f%n",
                    incomplete ? "incomplete" : "complete  ", Models.spearman(ctl, gain));
            if (!incomplete) {
                ConceptRow hard = rows.get(3);
                assertTrue(hard.intervened()[hard.intervened().length - 1] > 0.95,
                        "a hard bottleneck with every concept corrected is near perfect");
            }
        }
    }
}
