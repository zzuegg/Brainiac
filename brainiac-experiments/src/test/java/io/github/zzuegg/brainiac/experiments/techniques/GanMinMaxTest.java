package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.experiments.techniques.GanExperiment.MinMaxMethod;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class GanMinMaxTest {

    @Test
    void gameOptimisersTrainTheGradientPenaltyGanOneToOne() {
        StringBuilder sb = new StringBuilder();
        for (MinMaxMethod m : MinMaxMethod.values()) {
            float[] d = GanExperiment.minMax(m, 300, 2.5e-4f, 1e-3f, 1);
            sb.append(String.format("%s %.3f -> %.3f; ", m, d[0], d[1]));
            assertTrue(Float.isFinite(d[1]) && d[1] < 3 * d[0], m + " stays bounded");
        }
        System.out.printf("WGAN-GP 1:1, 300 joint steps, sliced W1 (spectral-norm reference 0.393): %s%n", sb);
    }
}
