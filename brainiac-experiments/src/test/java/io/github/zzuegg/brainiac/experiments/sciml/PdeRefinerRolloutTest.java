package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class PdeRefinerRolloutTest {

    @Test
    void refinerRollsOutKuramotoSivashinsky() {
        double[][][] ks = OperatorLearning.kuramotoSivashinsky(20, 32, 40, 0.5, 4);
        List<OperatorLearning.RefinerRow> rows = OperatorLearning.refiner(ks, 4, 3, 300, 1);
        StringBuilder line = new StringBuilder("8.3.7 KS rollouts, corr-time / low-band err / high-band err (ms):");
        rows.forEach(r -> line.append(String.format(" %s %.1f/%.3f/%.3f (%.0f);", r.model(), r.correlationTime(),
                r.lowBandError(), r.highBandError(), r.inferenceMillis())));
        System.out.println(line);
        assertTrue(rows.get(0).correlationTime() > 1, line.toString());
    }
}
