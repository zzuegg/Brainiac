package io.github.zzuegg.brainiac.experiments.multimodal;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.multimodal.TwoStageBridgeTrainer.LmMode;

import org.junit.jupiter.api.Test;

/** LLaVA's two stages for the projector bridge, with LoRA on the LM in stage 2. */
class VqaTwoStageTest {

    @Test
    void alignmentThenLoraInstructionTuning() {
        FusionShootout.VqaSetup setup = VqaFixture.setup();
        FusionShootout.TwoStageScores two = FusionShootout.vqaFollowUp(setup, FusionShootout.Bridge.PROJECTOR,
                LmMode.LORA, 100, 150, 1);
        System.out.printf("VQA two-stage: %s (frozen LM text loss %.3f; wave 9 projector 0.650, prior ~0.65,"
                + " description ceiling 0.77)%n", two, setup.textLoss());
        assertTrue(two.accuracy() > 0.3 && two.accuracy() <= 1, two.toString());
        assertTrue(two.alignment() >= 0 && two.alignment() <= 1, two.toString());
        assertTrue(Double.isFinite(two.textLoss()), two.toString());
    }
}
