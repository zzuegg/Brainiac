package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.NoisyLabelShootout.Method;
import io.github.zzuegg.brainiac.experiments.labels.NoisyLabelShootout.Noise;
import io.github.zzuegg.brainiac.experiments.labels.NoisyLabelShootout.Result;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class SampleSelectionShootoutTest {

    @Test
    void coTeachingElrAndPruningAtFortyPercentNoise() {
        Map<Method, Result> rows = new EnumMap<>(Method.class);
        for (Method m : new Method[] {Method.CE, Method.CE_EARLY_STOP, Method.COTEACHING, Method.COTEACHING_PLUS,
                Method.EL2N_PRUNE, Method.ELR}) {
            Result r = NoisyLabelShootout.run(m, Noise.SYMMETRIC, 0.4, 600, 1);
            rows.put(m, r);
            RobustLossShootoutTest.print("sym 0.4", r);
        }
        double ce = rows.get(Method.CE).finalAccuracy();
        assertTrue(rows.get(Method.COTEACHING).finalAccuracy() > ce, "co-teaching beats CE");
        assertTrue(rows.get(Method.COTEACHING).splitPrecision() > 0.7, "small-loss rows are mostly clean");
        assertTrue(rows.get(Method.ELR).noisyCorrected() > rows.get(Method.CE).noisyCorrected(),
                "ELR predicts more noisy examples correctly");
        for (Result r : rows.values()) {
            assertTrue(r.bestAccuracy() > 0.2, "every method trains: " + r);
        }
    }
}
