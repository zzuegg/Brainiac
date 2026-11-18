package io.github.zzuegg.brainiac.experiments.shift;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class DomainAdaptationMoonsTest {

    @Test
    void rotatedMoons() {
        Map<DomainAdaptation.Method, DomainAdaptation.Result> r = DomainAdaptation.adapt(DomainAdaptation.Task.MOONS,
                List.of(DomainAdaptation.Method.values()), 300, 1);
        DomainAdaptationTest.print("moons", r);
        assertTrue(r.get(DomainAdaptation.Method.SOURCE_ONLY).sourceAccuracy() > 0.9, "source fit");
        assertTrue(r.get(DomainAdaptation.Method.DANN).targetAccuracy() > r.get(DomainAdaptation.Method.ADABN)
                .targetAccuracy(), "DANN beats AdaBN on the rotation");
    }
}
