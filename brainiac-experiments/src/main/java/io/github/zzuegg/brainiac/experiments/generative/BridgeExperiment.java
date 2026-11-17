package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.experimental.nn.BridgeMatching;
import io.github.zzuegg.brainiac.experimental.nn.MinibatchOptimalTransport;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Transport between two data distributions, two moons to two rings ({@link PointClouds}):
 * {@link BridgeMatching} with a Brownian bridge (stochastic interpolant) and its DSBM
 * refinement by iterative Markovian fitting (wave 6, item 4.15), against rectified flow on
 * independent pairs and its reflow, and against flow matching on minibatch-OT pairs
 * ({@link MinibatchOptimalTransport}). Each run trains the same {@link MlpDenoiser} drift for
 * the same total steps. Scores: the transport cost {@code E|T(x0) - x0|^2} (lower is closer to
 * optimal transport; the minibatch-OT pairing of fresh samples is printed as a reference) and
 * the target fit, the sliced Wasserstein distance between transported points and fresh rings.
 */
@SuppressWarnings("exports")
public final class BridgeExperiment {
    private static final int BATCH = 128;
    private static final int STEPS = 20;

    private BridgeExperiment() {
    }

    /** How pairs are formed in the first round. */
    private enum Coupling {
        INDEPENDENT,
        OPTIMAL
    }

    private static Tensor source(int n, SplittableRandom r) {
        return Nets.matrix(PointClouds.moons(n, 0.05, r));
    }

    private static Tensor target(int n, SplittableRandom r) {
        return Nets.matrix(PointClouds.rings(n, 0.05, r));
    }

    /**
     * Runs one method.
     *
     * @param sigma the bridge diffusion, 0 for rectified flow
     * @param rounds 1 for plain matching, 2 or more for reflow / iterative Markovian fitting
     * @param optimal whether the first round pairs minibatches by optimal transport
     * @param iterations training steps per round
     * @param seed the seed
     * @return {@code {transport cost, sliced Wasserstein to the target}}
     */
    public static double[] run(float sigma, int rounds, boolean optimal, int iterations, long seed) {
        Coupling coupling = optimal ? Coupling.OPTIMAL : Coupling.INDEPENDENT;
        SplittableRandom r = new SplittableRandom(seed);
        Denoiser previous = null;
        Denoiser drift = null;
        for (int round = 0; round < rounds; round++) {
            MlpDenoiser net = MlpDenoiser.of(2, 0, 64, 2, seed + round);
            drift = Nets.unconditional(net);
            Optimizer opt = Optimizer.adam(3e-3f);
            for (int i = 0; i < iterations; i++) {
                if (i == iterations * 3 / 4) {
                    opt.learningRate(1e-3f);
                }
                Tensor x0 = source(BATCH, r);
                Tensor x1;
                if (previous != null) {
                    x1 = BridgeMatching.couple(previous, x0, STEPS, sigma, r)[1];
                } else if (coupling == Coupling.OPTIMAL) {
                    x1 = Nets.matrix(MinibatchOptimalTransport.pair(Nets.rows(x0), Nets.rows(target(BATCH, r))));
                } else {
                    x1 = target(BATCH, r);
                }
                BridgeMatching.loss(drift, x0, x1, sigma, r).backward();
                Nets.clip(net.parameters(), 1f);
                opt.step(net.parameters());
            }
            previous = drift;
        }
        Tensor x0 = source(1024, new SplittableRandom(seed + 77));
        Tensor x1 = BridgeMatching.transport(drift, x0, STEPS, sigma, new SplittableRandom(seed + 78));
        float[][] moved = Nets.rows(x1);
        return new double[] {GenerativeScores.pairedCost(Nets.rows(x0), moved),
            GenerativeScores.slicedWasserstein(moved, Nets.rows(target(1024, new SplittableRandom(seed + 79))), 64,
                    11)};
    }

    /**
     * Every method with the same total training budget.
     *
     * @param iterations total training steps per method
     * @param seed the seed
     * @return per method, {@code {transport cost, sliced Wasserstein}}; {@code "minibatch OT"}
     *     holds the cost of optimally pairing 1024 fresh points (and 0)
     */
    public static Map<String, double[]> compare(int iterations, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("rectified flow", run(0, 1, false, iterations, seed));
        out.put("reflow", run(0, 2, false, iterations / 2, seed));
        out.put("bridge matching", run(0.5f, 1, false, iterations, seed));
        out.put("DSBM (IMF)", run(0.5f, 2, false, iterations / 2, seed));
        out.put("OT flow matching", run(0, 1, true, iterations, seed));
        SplittableRandom r = new SplittableRandom(seed + 90);
        float[][] a = PointClouds.moons(512, 0.05, r);
        float[][] b = PointClouds.rings(512, 0.05, r);
        out.put("minibatch OT", new double[] {GenerativeScores.pairedCost(a, MinibatchOptimalTransport.pair(a, b)),
            0});
        return out;
    }
}
