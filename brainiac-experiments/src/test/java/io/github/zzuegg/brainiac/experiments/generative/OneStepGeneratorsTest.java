package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class OneStepGeneratorsTest {
    static final ToyPoints RING = ToyPoints.ring(4, 1.5, 0.08);

    @Test
    void averageVelocityModelsGenerateInOneStep() {
        int it = 700;
        OneStepGenerators.Score mean = OneStepGenerators.meanFlow(RING, it, 256, 1);
        OneStepGenerators.Score shortcut = OneStepGenerators.shortcut(RING, it, 256, 1);
        OneStepGenerators.Score[] reflow = OneStepGenerators.reflow(RING, it, 256, 1);
        OneStepGenerators.Score consistency = OneStepGenerators.consistency(RING, it, 256, 1);
        System.out.printf("one-step on a 4-blob ring (distance / near share / modes): MeanFlow %.3f/%.2f/%d, "
                + "shortcut %.3f/%.2f/%d; baselines flow matching %.3f/%.2f/%d, reflow %.3f/%.2f/%d, "
                + "consistency %.3f/%.2f/%d%n",
                mean.distance(), mean.nearShare(), mean.modes(), shortcut.distance(), shortcut.nearShare(),
                shortcut.modes(), reflow[0].distance(), reflow[0].nearShare(), reflow[0].modes(),
                reflow[1].distance(), reflow[1].nearShare(), reflow[1].modes(), consistency.distance(),
                consistency.nearShare(), consistency.modes());
        assertTrue(mean.distance() < reflow[0].distance() * 0.6, "MeanFlow beats one-step flow matching");
        assertTrue(shortcut.distance() < reflow[0].distance() * 0.6, "shortcut beats one-step flow matching");
        assertTrue(mean.modes() >= 4 && shortcut.modes() >= 4, "covers the ring");
    }
}
