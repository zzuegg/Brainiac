package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertTrue;

class RectifiedFlowExperimentTest {
    static final ToyPoints RING = ToyPoints.ring(6, 2, 0.08);

    static RectifiedFlowExperiment trained(int iterations) {
        RectifiedFlowExperiment fm = RectifiedFlowExperiment.of(MlpDenoiser.of(2, RING.modes(), 64, 2, 1), 3e-3f, 0.15f, 2);
        SplittableRandom r = new SplittableRandom(3);
        for (int i = 0; i < iterations; i++) {
            if (i == iterations * 2 / 3) {
                fm.learningRate(5e-4f);
            }
            float[][] x = new float[128][];
            int[] l = new int[128];
            for (int b = 0; b < x.length; b++) {
                l[b] = r.nextInt(RING.modes());
                x[b] = RING.sample(l[b], r);
            }
            fm.trainStep(x, l);
        }
        return fm;
    }

    static double gap(float[][] a, float[][] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += Math.sqrt(Nets.distance2(a[i], b[i]));
        }
        return s / a.length;
    }

    /** The share of points within 0.3 of a blob centre. */
    static double close(float[][] points) {
        int n = 0;
        for (float[] p : points) {
            n += RING.nearest(p)[1] < 0.3 ? 1 : 0;
        }
        return (double) n / points.length;
    }

    static int[] labels(int n) {
        int[] l = new int[n];
        for (int i = 0; i < n; i++) {
            l[i] = i % RING.modes();
        }
        return l;
    }

    @Test
    void learnsTheRingAndHeunBeatsEuler() {
        RectifiedFlowExperiment fm = trained(1000);
        int[] none = labels(64);   // conditional, unguided
        float[][] z = fm.noise(64, 5);
        float[][] reference = fm.sampleHeun(z, none, 64, 1, null);
        double quality = close(reference);
        double untrained = close(trained(0).sampleHeun(z, none, 16, 1, null));
        assertTrue(quality > 0.6 && quality > 3 * untrained, "near a blob: " + quality + " vs untrained " + untrained);
        double euler = gap(fm.sampleEuler(z, none, 4, 1, null), reference);
        double heun = gap(fm.sampleHeun(z, none, 4, 1, null), reference);
        assertTrue(heun < euler, "Heun error " + heun + " should be below Euler " + euler);

        int[] l = labels(60);
        double hit = RING.score(fm.sampleHeun(fm.noise(60, 6), l, 16, 3f, null), l)[1];
        assertTrue(hit > 0.8, "guided samples land in their blob: " + hit);
    }

    @Test
    void reflowStraightensPaths() {
        RectifiedFlowExperiment fm = trained(500);
        int[] none = new int[96];
        java.util.Arrays.fill(none, -1);
        float[][] probe = fm.noise(96, 11);
        double before = gap(fm.sampleEuler(probe, none, 1, 1, null), fm.sampleHeun(probe, none, 32, 1, null));
        for (int round = 0; round < 250; round++) {
            float[][] z = fm.noise(96, 100 + round);
            float[][] x = fm.sampleHeun(z, none, 8, 1, null);
            fm.reflowStep(z, x, none);
        }
        double after = gap(fm.sampleEuler(probe, none, 1, 1, null), fm.sampleHeun(probe, none, 32, 1, null));
        assertTrue(after < before * 0.6, "one-step error " + before + " -> " + after);
    }
}
