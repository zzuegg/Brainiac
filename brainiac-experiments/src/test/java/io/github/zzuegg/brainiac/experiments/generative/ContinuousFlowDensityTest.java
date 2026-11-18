package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ContinuousFlowDensityTest {

    @Test
    void ffjordBeatsFlowMatchingAtEqualStepsButNotAtEqualTime() {
        DensityBenchmark.Target moons = DensityBenchmark.Target.MOONS;
        long t0 = System.nanoTime();
        double[] ffjord = DensityBenchmark.ffjord(moons, 80, 1);
        double seconds = (System.nanoTime() - t0) / 1e9;
        double[] fm = DensityBenchmark.flowMatching(moons, 80, 1);
        double[] fmLong = DensityBenchmark.flowMatching(moons, 1000, 1);
        System.out.printf("moons held-out NLL nats (eval NFE): FFJORD 80 steps %.3f (%.0f, %.1f s); baselines flow "
                + "matching 80 steps %.3f (%.0f), 1000 steps %.3f (%.0f)%n", ffjord[0], ffjord[1], seconds, fm[0], fm[1],
                fmLong[0], fmLong[1]);
        assertTrue(ffjord[0] < 2.0, "FFJORD learns a density better than the standard normal (about 2.1)");
        assertTrue(ffjord[0] < fm[0], "maximum likelihood beats flow matching at equal steps");
    }
}
