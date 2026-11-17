package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.vision.VisionShootout.Architecture;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class VisionShootoutTest {

    @ParameterizedTest
    @EnumSource(Architecture.class)
    void architectureLearnsTheShapes(Architecture a) {
        float acc = VisionShootout.run(a, 150, 32, 3);
        System.out.println(a + ": " + acc);
        assertTrue(acc > 0.6f, a + " test accuracy " + acc);
    }
}
