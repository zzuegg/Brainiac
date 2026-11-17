package io.github.zzuegg.brainiac.experiments.generative;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class MemorizationTimescaleTest {

    @Test
    void smallTrainingSetsAreCopiedSoonerAndMore() {
        int[] sizes = {4, 16, 64};
        double[] finalExcess = new double[sizes.length];
        StringBuilder line = new StringBuilder("Memorization (ring of 8 blobs, 2000 steps):");
        for (int k = 0; k < sizes.length; k++) {
            MemorizationTimescale.Curves c = MemorizationTimescale.run(sizes[k], 2000, 200, 1);
            int last = c.steps().length - 1;
            finalExcess[k] = c.excessCopy(last);
            line.append(String.format(" n=%d tau_gen %d tau_mem %d excess copy %.2f (chance %.2f) quality %.2f"
                    + " (real %.2f) early-stop step %d;", sizes[k], c.generalizationTime(1.3), c.memorizationTime(0.1),
                    finalExcess[k], c.realCopyRate(), c.quality()[last], c.realQuality(), c.steps()[c.earlyStop()]));
        }
        System.out.println(line);
        assertTrue(finalExcess[0] > finalExcess[2] + 0.05, "4 points are copied more than 64: " + line);
        assertTrue(finalExcess[0] > 0.1, "4 points are memorized: " + line);
    }
}
