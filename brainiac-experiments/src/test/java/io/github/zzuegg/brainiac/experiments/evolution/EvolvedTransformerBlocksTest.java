package io.github.zzuegg.brainiac.experiments.evolution;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class EvolvedTransformerBlocksTest {

    @Test
    void evolutionFindsABlockAtLeastAsGoodAsTheConventionalOne() {
        EvolvedTransformerBlocks.Outcome o = EvolvedTransformerBlocks.run(3, 6, 80, 1);
        System.out.printf("best %s loss %.3f, baseline %.3f, first median %.3f%n", o.best(), o.bestLoss(),
                o.baselineLoss(), o.firstMedianLoss());
        assertTrue(o.bestLoss() <= o.baselineLoss() + 1e-6);
        assertTrue(o.bestLoss() < o.firstMedianLoss());
    }
}
