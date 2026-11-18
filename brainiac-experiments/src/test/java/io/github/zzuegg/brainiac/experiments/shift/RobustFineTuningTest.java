package io.github.zzuegg.brainiac.experiments.shift;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class RobustFineTuningTest {

    @Test
    void wiseFtAndLpFtAgainstFullFineTuning() {
        Map<String, DomainGeneralization.FineTuneScores> r = DomainGeneralization.fineTuning(300, 200,
                new double[] {0.5}, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("  %s %.3f/%.3f", k, v.inDistribution(), v.shifted())));
        System.out.printf("Robust fine-tuning (ID/shifted)%s%n", sb);
        assertTrue(r.get("linear probe").inDistribution() > 0.3, "the probe learns the shapes");
        assertTrue(r.get("full fine-tuning").inDistribution() > 0.5, "fine-tuning learns the shapes");
    }
}
