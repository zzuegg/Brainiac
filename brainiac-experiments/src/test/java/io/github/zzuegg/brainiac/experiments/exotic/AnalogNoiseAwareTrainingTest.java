package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.nn.Model;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class AnalogNoiseAwareTrainingTest {
    @Test
    void noiseAwareTrainingSurvivesNoisyChips() {
        Model aware = AnalogNoiseAwareTraining.model(0.15f, 1);
        Model plain = AnalogNoiseAwareTraining.model(0, 1);
        AnalogNoiseAwareTraining.train(aware, 600, 2);
        AnalogNoiseAwareTraining.train(plain, 600, 2);
        double awareClean = AnalogNoiseAwareTraining.chipAccuracy(aware, 0, 1, 3);
        double plainClean = AnalogNoiseAwareTraining.chipAccuracy(plain, 0, 1, 3);
        double awareNoisy = AnalogNoiseAwareTraining.chipAccuracy(aware, 0.15f, 10, 4);
        double plainNoisy = AnalogNoiseAwareTraining.chipAccuracy(plain, 0.15f, 10, 4);
        System.out.printf("analog: aware %.3f clean / %.3f chips; plain %.3f clean / %.3f chips%n", awareClean,
                awareNoisy, plainClean, plainNoisy);
        assertTrue(awareClean > 0.9 && plainClean > 0.9);
        assertTrue(awareNoisy > plainNoisy + 0.03, "noise-aware should be more robust");
        assertTrue(awareNoisy > 0.85);
    }
}
