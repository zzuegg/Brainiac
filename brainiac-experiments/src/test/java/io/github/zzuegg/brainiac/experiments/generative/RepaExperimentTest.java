package io.github.zzuegg.brainiac.experiments.generative;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.nn.Model;
import org.junit.jupiter.api.Test;

class RepaExperimentTest {

    @Test
    void representationAlignmentSpeedsUpDiffusionTraining() {
        int[] checkpoints = {150, 300, 500};
        Model encoder = RepaExperiment.encoder(300, 1);
        double[] plain = RepaExperiment.run(null, checkpoints, 0.5f, 1);
        double[] repa = RepaExperiment.run(encoder, checkpoints, 0.5f, 1);
        System.out.printf("REPA on ToyImages, distance per pixel at 150/300/500 steps: plain %.3f/%.3f/%.3f, "
                + "REPA %.3f/%.3f/%.3f%n", plain[0], plain[1], plain[2], repa[0], repa[1], repa[2]);
        double plainSum = plain[0] + plain[1] + plain[2];
        double repaSum = repa[0] + repa[1] + repa[2];
        assertTrue(repaSum < 1.1 * plainSum, "alignment does not slow training down");
        assertTrue(repa[2] < 0.6, "and the aligned model learns the shapes: " + repa[2]);
    }
}
