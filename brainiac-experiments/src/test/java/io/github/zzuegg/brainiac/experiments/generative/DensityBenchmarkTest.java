package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.experimental.nn.GlowFlow;
import io.github.zzuegg.brainiac.experimental.nn.MaskedAutoregressiveFlow;
import io.github.zzuegg.brainiac.experimental.nn.SplineFlow;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class DensityBenchmarkTest {
    static final int STEPS = 300;

    @Test
    void splineFlowBeatsRealNvpAndGlowAndMafLearnTheMoons() {
        DensityBenchmark.Target moons = DensityBenchmark.Target.MOONS;
        DensityBenchmark.Target rings = DensityBenchmark.Target.RINGS;
        double glow = DensityBenchmark.train(GlowFlow.of(2, 6, 64, 1), moons, STEPS, 2e-3f, 1);
        double spline = DensityBenchmark.train(SplineFlow.of(2, 4, 8, 3f, 64, 1), moons, STEPS, 2e-3f, 1);
        double maf = DensityBenchmark.train(MaskedAutoregressiveFlow.of(2, 6, 64, 1), moons, STEPS, 2e-3f, 1);
        double nvp = DensityBenchmark.realNvp(moons, STEPS, 1);
        double gmm = DensityBenchmark.gaussianMixture(moons, 8, 1);
        double splineRings = DensityBenchmark.train(SplineFlow.of(2, 4, 8, 3f, 64, 1), rings, STEPS, 2e-3f, 1);
        double nvpRings = DensityBenchmark.realNvp(rings, STEPS, 1);
        double gmmRings = DensityBenchmark.gaussianMixture(rings, 8, 1);
        System.out.printf("held-out NLL nats, moons: Glow %.3f, spline %.3f, MAF %.3f; baselines RealNVP %.3f, "
                + "GMM-8 %.3f | rings: spline %.3f; baselines RealNVP %.3f, GMM-8 %.3f%n", glow, spline, maf, nvp, gmm,
                splineRings, nvpRings, gmmRings);
        // a standard normal scores about 2.1 nats on both targets
        assertTrue(glow < 1.8 && maf < 1.8, "Glow and MAF learn the moons");
        assertTrue(spline < nvp + 0.1, "spline flow at least matches RealNVP on moons");
        assertTrue(splineRings < nvpRings - 0.3, "spline flow beats RealNVP on rings");
    }
}
