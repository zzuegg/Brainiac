package io.github.zzuegg.brainiac.experiments.generative;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.generative.HighDimensionalTargets.Target;
import org.junit.jupiter.api.Test;

class HighDimensionalTargetsTest {

    @Test
    void xPredictionSurvivesHighDimensionalPixels() {
        int dim = 512;
        double[] eps = HighDimensionalTargets.run(Target.EPSILON, dim, 96, 400, 10, 1);
        double[] v = HighDimensionalTargets.run(Target.VELOCITY, dim, 96, 400, 10, 1);
        double[] x = HighDimensionalTargets.run(Target.X_SAMPLE_LOSS, dim, 96, 400, 10, 1);
        double[] xv = HighDimensionalTargets.run(Target.X_VELOCITY_LOSS, dim, 96, 400, 10, 1);
        System.out.printf("8x8 images in %d-D, width-96 MLP, distance per pixel (off-manifold energy): "
                + "eps %.3f (%.2f), v %.3f (%.2f), x %.3f (%.2f), x with v-loss %.3f (%.2f)%n", dim, eps[0], eps[1],
                v[0], v[1], x[0], x[1], xv[0], xv[1]);
        assertTrue(xv[0] < 0.8 * eps[0] && x[0] < 0.8 * eps[0], "x-prediction beats epsilon prediction");
    }
}
