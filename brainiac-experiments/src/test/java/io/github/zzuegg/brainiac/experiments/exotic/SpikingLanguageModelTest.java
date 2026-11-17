package io.github.zzuegg.brainiac.experiments.exotic;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class SpikingLanguageModelTest {

    @Test
    void spikingRwkvLearnsToCount() {
        SpikingLanguageModel.Result spiking = SpikingLanguageModel.run(true, 150, 1);
        SpikingLanguageModel.Result dense = SpikingLanguageModel.run(false, 150, 1);
        System.out.printf("SpikeGPT on counting sequences: perplexity %.4f with %.3f of neurons spiking per token; "
                + "non-spiking RWKV %.4f (uniform %d)%n", spiking.perplexity(), spiking.spikeRate(),
                dense.perplexity(), SpikingLanguageModel.VOCAB);
        assertTrue(spiking.perplexity() < 4, "the spiking model learns the language");
        assertTrue(spiking.spikeRate() > 0.01 && spiking.spikeRate() < 0.6, "sparse but alive spikes");
        assertTrue(spiking.perplexity() < 2 * dense.perplexity() + 0.5, "close to the non-spiking model");
    }
}
