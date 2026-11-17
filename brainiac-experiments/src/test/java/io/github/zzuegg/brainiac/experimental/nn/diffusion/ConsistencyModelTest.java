package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ConsistencyModelTest {
    static final io.github.zzuegg.brainiac.experiments.generative.ToyPoints RING =
            io.github.zzuegg.brainiac.experiments.generative.ToyPoints.ring(4, 1, 0.05);

    static ConsistencyModel trained(int iterations) {
        ConsistencyModel cm = ConsistencyModel.of(MlpDenoiser.of(2, RING.modes(), 64, 2, 1), 0.002, 10, 2e-3f, 2);
        SplittableRandom r = new SplittableRandom(3);
        for (int i = 0; i < iterations; i++) {
            float[][] x = new float[128][];
            int[] l = new int[128];
            for (int b = 0; b < x.length; b++) {
                l[b] = r.nextInt(RING.modes());
                x[b] = RING.sample(l[b], r);
            }
            cm.trainStep(x, l, iterations);
        }
        return cm;
    }

    static int[] labels(int n) {
        int[] l = new int[n];
        for (int i = 0; i < n; i++) {
            l[i] = i % RING.modes();
        }
        return l;
    }

    @Test
    void boundaryConditionIsTheIdentity() {
        ConsistencyModel cm = trained(0);
        float[][] x = {{0.3f, -0.7f}, {2f, 1f}};
        float[][] y = cm.denoise(x, 0.002, new int[] {0, 1});
        assertArrayEquals(x[0], y[0], 1e-5f);
        assertArrayEquals(x[1], y[1], 1e-5f);
    }

    @Test
    void generatesInOneStep() {
        ConsistencyModel cm = trained(1000);
        int[] l = labels(64);
        double[] one = RING.score(cm.sampleOneStep(l, 5), l);
        double[] two = RING.score(cm.sampleMultiStep(l, new double[] {0.8}, 5), l);
        double[] base = RING.score(trained(0).sampleOneStep(l, 5), l);
        assertTrue(one[0] < 0.3 && one[0] < base[0] / 2, "one-step distance " + one[0] + " vs untrained " + base[0]);
        assertTrue(one[1] > 0.8, "one-step samples land in their blob: " + one[1]);
        assertTrue(two[0] < 0.35, "two-step distance " + two[0]);
    }
}
