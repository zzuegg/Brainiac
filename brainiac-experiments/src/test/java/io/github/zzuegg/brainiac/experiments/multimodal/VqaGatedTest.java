package io.github.zzuegg.brainiac.experiments.multimodal;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class VqaGatedTest {

    @Test
    void bridgeIsScored() {
        FusionShootout.VqaScores s = VqaFixture.run(FusionShootout.Bridge.GATED);
        assertTrue(s.accuracy() >= 0 && s.accuracy() <= 1, "accuracy is a share");
        assertTrue(Double.isFinite(s.textLoss()), "the text loss is finite");
    }
}
