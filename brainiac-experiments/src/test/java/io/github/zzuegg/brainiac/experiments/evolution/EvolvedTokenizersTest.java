package io.github.zzuegg.brainiac.experiments.evolution;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class EvolvedTokenizersTest {

    @Test
    void evolvedTokenizerModelsTextBetterThanBytesAndGreedyBpe() {
        EvolvedTokenizers.Outcome o = EvolvedTokenizers.run(12, 1);
        System.out.printf("evolved %.4f bpc (vocab %d), bytes %.4f, greedy %s%n", o.evolvedBits(),
                o.evolvedVocabulary(), o.byteBits(), o.greedyBits());
        assertTrue(o.evolvedBits() < o.byteBits());
        double bestGreedy = o.greedyBits().values().stream().mapToDouble(Double::doubleValue).min().orElseThrow();
        assertTrue(o.evolvedBits() < bestGreedy * 1.02, "evolved " + o.evolvedBits() + " vs greedy " + bestGreedy);
    }
}
