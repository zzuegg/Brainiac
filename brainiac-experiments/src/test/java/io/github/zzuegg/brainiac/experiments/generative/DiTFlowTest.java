package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.experimental.nn.DiffusionTransformer;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** A DiT trained by flow matching on the toy shapes, sampled with classifier-free guidance. */
class DiTFlowTest {

    @Test
    void ditGeneratesTheRequestedShapes() {
        FlowMatching fm = FlowMatching.of(DiffusionTransformer.builder(8, 8).patch(4).width(48).depth(2).heads(2)
                .classes(2).seed(1).build(), 6e-3f, 0.15f, 2);
        SplittableRandom r = new SplittableRandom(3);
        for (int i = 0; i < 300; i++) {
            float[][] x = new float[32][];
            int[] l = new int[32];
            for (int b = 0; b < x.length; b++) {
                l[b] = r.nextInt(2);   // horizontal or vertical bars
                x[b] = ToyImages.sample(l[b], r);
            }
            fm.trainStep(x, l);
        }
        int[] labels = new int[16];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = i % 2;
        }
        // strong guidance makes the short-trained DiT commit to the requested class
        double[] guided = TinyDdpmTest.score(fm.sampleHeun(fm.noise(16, 4), labels, 16, 4f, null), labels);
        double[] plain = TinyDdpmTest.score(fm.sampleHeun(fm.noise(16, 4), labels, 16, 1f, null), labels);
        assertTrue(guided[1] >= 0.6 && guided[1] > plain[1],
                "guided samples match their class: " + guided[1] + " (unguided " + plain[1] + ")");
    }
}
