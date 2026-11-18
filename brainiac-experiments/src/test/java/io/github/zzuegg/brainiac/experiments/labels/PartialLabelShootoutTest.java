package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.WeakLabelShootout.PartialMethod;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class PartialLabelShootoutTest {

    @Test
    void prodenWithThreeDistractors() {
        Map<PartialMethod, Double> rows = new EnumMap<>(PartialMethod.class);
        for (PartialMethod m : PartialMethod.values()) {
            double acc = WeakLabelShootout.partial(m, 3, false, 400, 1);
            rows.put(m, acc);
            System.out.printf("Partial 3 distractors uniform %-22s acc %.3f%n", m, acc);
        }
        assertTrue(rows.get(PartialMethod.PRODEN) > rows.get(PartialMethod.SINGLE_CANDIDATE_ONLY),
                "PRODEN uses the ambiguous rows");
        assertTrue(rows.get(PartialMethod.PRODEN) > rows.get(PartialMethod.SUPERVISED) - 0.15, "PRODEN near supervised");
        double biased = WeakLabelShootout.partial(PartialMethod.PRODEN, 3, true, 400, 1);
        System.out.printf("Partial 3 distractors biased  %-22s acc %.3f%n", PartialMethod.PRODEN, biased);
    }
}
