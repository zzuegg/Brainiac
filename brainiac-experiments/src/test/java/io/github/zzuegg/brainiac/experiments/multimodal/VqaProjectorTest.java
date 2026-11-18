package io.github.zzuegg.brainiac.experiments.multimodal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class VqaProjectorTest {

    @Test
    void projectorAgainstTheTextFloorAndDescriptionCeiling() {
        FusionShootout.VqaScores text = VqaFixture.run(FusionShootout.Bridge.TEXT_ONLY);
        FusionShootout.VqaScores described = VqaFixture.run(FusionShootout.Bridge.DESCRIPTION);
        FusionShootout.VqaScores proj = VqaFixture.run(FusionShootout.Bridge.PROJECTOR);
        assertTrue(described.accuracy() > text.accuracy() + 0.1, "the frozen LM answers from a scene description");
        assertEquals(VqaFixture.setup().textLoss(), proj.textLoss(), 1e-6, "the frozen LM is intact");
        assertTrue(proj.accuracy() >= 0 && proj.accuracy() <= 1, "the projector is scored");
    }
}
