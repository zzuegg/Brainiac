package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class PerceiverIoReversalTest {

    @Test
    void latentCountBoundsWhatPasses() {
        double[] few = PerceiverExperiments.reversal(8, 200, 1);
        double[] many = PerceiverExperiments.reversal(32, 200, 1);
        double[] full = PerceiverExperiments.reversal(0, 200, 1);
        System.out.printf("Reverse 64 tokens (vocab 8), accuracy (MACs): Perceiver IO 8 latents %.3f (%.0f),"
                + " 32 latents %.3f (%.0f); full transformer %.3f (%.0f); 200 steps each%n", few[0], few[1],
                many[0], many[1], full[0], full[1]);
        assertTrue(many[0] > few[0], "more latents carry more of the sequence through the bottleneck");
        assertTrue(full[0] > many[0], "full self-attention solves it, the bottleneck does not");
        assertTrue(many[1] < full[1], "but the bottleneck is cheaper");
    }
}
