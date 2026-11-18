package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.vision.KeypointHeads;
import org.junit.jupiter.api.Test;

class KeypointHeadsTest {

    @Test
    void integralRegressionBeatsDirectAndArgmax() {
        DensePrediction.PoseScore direct = DensePrediction.keypoints(KeypointHeads.Mode.DIRECT, 200, 1);
        DensePrediction.PoseScore argmax = DensePrediction.keypoints(KeypointHeads.Mode.HEATMAP_ARGMAX, 200, 1);
        DensePrediction.PoseScore integral = DensePrediction.keypoints(KeypointHeads.Mode.INTEGRAL, 200, 1);
        System.out.printf("keypoints PCK@2px/mean error/rare-pose error: direct %.3f/%.2f/%.2f, arg-max heatmap"
                + " %.3f/%.2f/%.2f, integral %.3f/%.2f/%.2f%n", direct.pck(), direct.meanError(), direct.rareError(),
                argmax.pck(), argmax.meanError(), argmax.rareError(), integral.pck(), integral.meanError(),
                integral.rareError());
        assertTrue(integral.meanError() < direct.meanError(), "integral regression beats direct regression");
        assertTrue(integral.meanError() < argmax.meanError(), "integral regression beats arg-max");
    }
}
