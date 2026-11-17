package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.ConsistencyModel;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.EquilibriumMatching;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.FlowMatching;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Sample quality against sampling steps on the {@link RectifiedFlowExperiment} ring of
 * {@link ToyPoints}: core flow matching sampled with Euler steps, a {@link ConsistencyModel}
 * (one and few steps) and {@link EquilibriumMatching} sampled by gradient descent with a
 * step size matched to the budget, plus EqM with adaptive stopping. Quality is the mean
 * distance of samples to the nearest blob centre (the blobs have spread 0.08).
 */
@SuppressWarnings("exports")
public final class EquilibriumMatchingExperiment {
    private static final ToyPoints RING = ToyPoints.ring(6, 2, 0.08);

    private EquilibriumMatchingExperiment() {
    }

    private static float[][] batch(SplittableRandom r, int n) {
        float[][] x = new float[n][];
        for (int b = 0; b < n; b++) {
            x[b] = RING.sample(r.nextInt(RING.modes()), r);
        }
        return x;
    }

    /**
     * Trains the three models and samples them at each budget.
     *
     * @param iterations training steps of 128 points for each model
     * @param budgets sampling steps (network evaluations) to compare
     * @param seed the seed
     * @return per method and budget, the mean distance to the nearest centre; the key
     *     {@code "EqM adaptive"} also has an entry {@code "EqM adaptive steps"}
     */
    public static Map<String, Double> run(int iterations, int[] budgets, long seed) {
        int[] none = new int[128];
        Arrays.fill(none, -1);
        MlpDenoiser fmNet = MlpDenoiser.of(2, 0, 64, 2, seed);
        MlpDenoiser eqNet = MlpDenoiser.of(2, 0, 64, 2, seed);
        EquilibriumMatching eqm = EquilibriumMatching.of(4, 0.8f);
        ConsistencyModel cm = ConsistencyModel.of(MlpDenoiser.of(2, 0, 64, 2, seed), 0.002, 10, 2e-3f, seed);
        Optimizer fmOpt = Optimizer.adam(2e-3f);
        Optimizer eqOpt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int i = 0; i < iterations; i++) {
            Tensor x = Nets.matrix(batch(r, 128));
            Var fl = FlowMatching.loss(fmNet.bind(none), x, r);
            fl.backward();
            Nets.clip(fmNet.parameters(), 1f);
            fmOpt.step(fmNet.parameters());
            Var el = eqm.loss(eqNet.bind(none), x, r);
            el.backward();
            Nets.clip(eqNet.parameters(), 1f);
            eqOpt.step(eqNet.parameters());
            cm.trainStep(batch(r, 128), none, iterations);
        }
        int n = 256;
        int[] sampleNone = new int[n];
        Arrays.fill(sampleNone, -1);
        Denoiser fm = fmNet.bind(sampleNone);
        Denoiser eq = eqNet.bind(sampleNone);
        Tensor noise = Nets.gaussian(n, 2, new SplittableRandom(seed + 9));
        Map<String, Double> out = new LinkedHashMap<>();
        for (int steps : budgets) {
            out.put("flow matching " + steps, quality(Nets.rows(FlowMatching.sample(fm, noise, steps))));
            float eta = (float) (Math.min(1.0, 3.0 / steps) / eqm.lambda());
            out.put("EqM " + steps, quality(Nets.rows(eqm.sample(eq, noise, steps, eta, 0, 0).samples())));
            double[] sigmas = new double[steps - 1];
            for (int k = 0; k < sigmas.length; k++) {
                sigmas[k] = 10 * Math.pow(0.002 / 10, (k + 1.0) / steps);
            }
            out.put("consistency " + steps, quality(steps == 1 ? cm.sampleOneStep(sampleNone, seed + 9)
                    : cm.sampleMultiStep(sampleNone, sigmas, seed + 9)));
        }
        EquilibriumMatching.Samples adaptive = eqm.sample(eq, noise, 200, 0.3f / eqm.lambda(), 0.3f, 1e-3f);
        out.put("EqM adaptive", quality(Nets.rows(adaptive.samples())));
        out.put("EqM adaptive steps", adaptive.meanSteps());
        return out;
    }

    private static double quality(float[][] samples) {
        return RING.score(samples, null)[0];
    }
}
