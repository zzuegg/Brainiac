package io.github.zzuegg.brainiac.experiments.selfsupervised;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Method;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Result;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class SelfSupervisedBenchmarkTest {

    @ParameterizedTest
    @EnumSource(value = Method.class, names = "IJEPA_FULL", mode = EnumSource.Mode.EXCLUDE)
    void methodLearnsAUsefulRepresentation(Method method) {
        boolean patches = method == Method.MAE || method == Method.IJEPA;
        Result r = SelfSupervisedBenchmark.run(method, patches ? 100 : 150, 1);
        System.out.println(r);
        if (method != Method.DINO) {
            // DINO's loss is not monotone: its teacher sharpens as it learns
            assertTrue(r.lastLoss() < r.firstLoss(), method + " loss falls: " + r);
        }
        assertTrue(r.probeAccuracy() > 1.0f / 6 + 0.15f, method + " probe beats chance: " + r);
    }
}
