package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NeuralCellularAutomatonTest {

    @Test
    void growsTheTargetFromOneCellAndRegrowsAfterDamage() {
        int size = 11;
        NeuralCellularAutomaton nca = NeuralCellularAutomaton.of(NeuralCellularAutomaton.diamond(size, 4), size, 6, 24, 1);
        float untrained = nca.loss(nca.run(nca.seed(1), 16, 3, null));
        for (int i = 0; i < 260; i++) {
            nca.trainStep(4, 12, 18);
        }
        List<float[]> frames = new ArrayList<>();
        Tensor grown = nca.run(nca.seed(1), 16, 3, frames);
        float lossGrown = nca.loss(grown);
        assertTrue(lossGrown < untrained * 0.35, "grown loss " + lossGrown + " vs untrained " + untrained);

        Tensor damaged = nca.damage(grown, 0, 1, 1, 4);
        float lossDamaged = nca.loss(damaged);
        float lossHealed = nca.loss(nca.run(damaged, 20, 4, null));
        float lossKept = nca.loss(nca.run(grown, 20, 4, null));
        System.out.printf("NCA: loss untrained %.4f, grown %.4f; damaged %.4f -> regrown %.4f (undamaged %.4f)%n",
                untrained, lossGrown, lossDamaged, lossHealed, lossKept);
        // at least half of the damage is repaired, relative to an undamaged pattern run as long
        assertTrue(lossHealed - lossKept < 0.5f * (lossDamaged - lossKept),
                "damaged " + lossDamaged + " -> regrown " + lossHealed + " (undamaged " + lossKept + ")");

        assertEquals(16, frames.size());
        assertTrue(nca.gif(frames, 4).length > 500);
    }
}
