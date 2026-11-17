package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.experimental.nn.LogicCircuit;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LogicGateExperimentTest {

    @Test
    void learnsAComparatorAndCompilesToAnIdenticalCircuit() {
        LogicGateExperiment exp = LogicGateExperiment.of(new int[] {96, 96, 96, 64},
                LogicGateExperiment.highNibbleGreater(), 4f, 1);
        exp.train(400);
        double[] circuit = exp.circuitAccuracy();
        assertTrue(circuit[0] > 0.9, "discrete circuit accuracy " + circuit[0]);

        // the hardened float network and the bit-op circuit agree on every output bit
        float[] hard = exp.hardenedOutputs();
        LogicCircuit c = LogicCircuit.compile(exp.layers());
        for (int chunk = 0; chunk < 4; chunk++) {
            long[] in = new long[8];
            for (int lane = 0; lane < 64; lane++) {
                for (int b = 0; b < 8; b++) {
                    in[b] |= (long) ((chunk * 64 + lane) >> b & 1) << lane;
                }
            }
            long[] out = c.evaluate(in);
            for (int lane = 0; lane < 64; lane++) {
                for (int j = 0; j < out.length; j++) {
                    assertEquals(hard[(chunk * 64 + lane) * out.length + j], out[j] >>> lane & 1L, 1e-3);
                }
            }
        }
    }

    @Test
    void everyGateMatchesItsRelaxationOnBits() {
        for (int g = 0; g < 16; g++) {
            for (int a = 0; a < 2; a++) {
                for (int b = 0; b < 2; b++) {
                    long bit = LogicCircuit.gate(g, a == 1 ? -1L : 0, b == 1 ? -1L : 0) & 1L;
                    // truth tables: gate id g has output bit (g >> (2a + b)) & 1 in the standard ordering
                    assertEquals((g >> (3 - (2 * a + b))) & 1, bit, "gate " + g + " on " + a + "," + b);
                }
            }
        }
    }
}
