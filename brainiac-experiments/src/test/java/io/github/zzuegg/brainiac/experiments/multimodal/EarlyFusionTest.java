package io.github.zzuegg.brainiac.experiments.multimodal;

import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class EarlyFusionTest {

    @Test
    void oneTransformerReadsAndWritesImages() {
        EarlyFusion e = EarlyFusion.create(1);
        int steps = 300;
        HookedDecoder fused = e.train(EarlyFusion.Mode.BOTH, steps, 2);
        HookedDecoder captioner = e.train(EarlyFusion.Mode.CAPTION, steps / 2, 2);
        HookedDecoder generator = e.train(EarlyFusion.Mode.GENERATE, steps / 2, 2);
        double fusedCaption = e.captionAccuracy(fused);
        double[] fusedGen = e.generationAccuracy(fused, 5, 3);
        double sepCaption = e.captionAccuracy(captioner);
        double[] sepGen = e.generationAccuracy(generator, 5, 3);
        System.out.printf("early fusion, one model %d steps: caption accuracy %.3f, generation accuracy %.3f "
                + "(distance %.3f); separate models %d steps each: caption %.3f, generation %.3f (distance %.3f)%n",
                steps, fusedCaption, fusedGen[0], fusedGen[1], steps / 2, sepCaption, sepGen[0], sepGen[1]);
        assertTrue(fusedCaption > 0.8, "the fused model captions: " + fusedCaption);
        assertTrue(fusedGen[0] > 0.5, "the fused model draws the asked digit: " + fusedGen[0]);
        assertTrue(fusedCaption >= sepCaption - 0.05 && fusedGen[0] >= sepGen[0] - 0.1,
                "one fused model keeps up with two separate ones at equal compute");
    }
}
