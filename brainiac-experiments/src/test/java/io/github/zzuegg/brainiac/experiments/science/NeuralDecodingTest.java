package io.github.zzuegg.brainiac.experiments.science;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class NeuralDecodingTest {

    @Test
    void decodersRecoverCursorVelocity() {
        Map<String, double[]> r = Neuroscience.decoders(50, new int[] {10, 40, 160}, 30, 100, 1);
        System.out.printf("13.3.16 decoding R2 {train 10, 40, 160 reaches, dropout 30%%, drift, 20 ms bins}: %s%n",
                Lines.of(r));
        assertTrue(r.get("KALMAN")[2] > 0.5, Lines.of(r));
        assertTrue(r.get("KALMAN")[2] > r.get("POPULATION_VECTOR")[2], Lines.of(r));
    }
}
