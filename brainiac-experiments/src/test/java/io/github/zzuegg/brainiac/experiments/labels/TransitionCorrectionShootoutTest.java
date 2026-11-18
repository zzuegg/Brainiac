package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.NoisyLabelShootout.Method;
import io.github.zzuegg.brainiac.experiments.labels.NoisyLabelShootout.Noise;
import io.github.zzuegg.brainiac.experiments.labels.NoisyLabelShootout.Result;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class TransitionCorrectionShootoutTest {

    @Test
    void forwardCorrectionUnderPairFlipNoise() {
        Map<Method, Result> rows = new EnumMap<>(Method.class);
        for (Method m : new Method[] {Method.CE, Method.FORWARD_ORACLE, Method.FORWARD_EST, Method.BACKWARD_EST,
                Method.T_REVISION}) {
            Result r = NoisyLabelShootout.run(m, Noise.PAIR_FLIP, 0.4, 600, 1);
            rows.put(m, r);
            RobustLossShootoutTest.print("pair 0.4", r);
        }
        assertTrue(rows.get(Method.FORWARD_ORACLE).finalAccuracy() > rows.get(Method.CE).finalAccuracy() + 0.1,
                "forward correction with the true T beats CE");
        assertTrue(rows.get(Method.FORWARD_EST).transitionError() < 0.1, "the anchor estimate is close");
        for (Method m : new Method[] {Method.CE, Method.FORWARD_EST}) {
            Result r = NoisyLabelShootout.run(m, Noise.INSTANCE, 0.4, 600, 1);
            RobustLossShootoutTest.print("inst 0.4", r);
            assertTrue(r.bestAccuracy() > 0.2, r.toString());
        }
    }
}
