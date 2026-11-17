package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ContinuousFlow;
import io.github.zzuegg.brainiac.experimental.nn.EnergyMatching;
import io.github.zzuegg.brainiac.experimental.nn.OdeSolver;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.EquilibriumMatching;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.FlowMatching;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * {@link EnergyMatching} (wave 6, item 4.17) against core flow matching and
 * {@link EquilibriumMatching} on 2-D data, each trained for the same steps with the same width:
 * sample quality (sliced Wasserstein distance to fresh data) at a fixed number of sampler
 * steps, and a likelihood: the energy's Boltzmann density {@code exp(-V / T)} normalised
 * numerically on a grid (possible in 2-D), with the one temperature {@code T} fitted on a
 * validation batch, against flow matching's exact likelihood through its
 * ODE ({@link ContinuousFlow}). Equilibrium matching learns a gradient field without a
 * potential, so it has no likelihood.
 */
@SuppressWarnings("exports")
public final class EnergyMatchingShootout {
    private static final int BATCH = 128;

    private EnergyMatchingShootout() {
    }

    /**
     * Trains the three models and scores them.
     *
     * @param data draws a batch of 2-D points
     * @param iterations training steps
     * @param steps sampler steps for every method
     * @param seed the seed
     * @return metrics: {@code "<method> SW"} and {@code "<method> NLL"} (nats per point)
     */
    public static Map<String, Double> run(Function<SplittableRandom, float[][]> data, int iterations, int steps,
            long seed) {
        int[] none = new int[BATCH];
        Arrays.fill(none, -1);
        MlpDenoiser fmNet = MlpDenoiser.of(2, 0, 64, 2, seed);
        MlpDenoiser eqNet = MlpDenoiser.of(2, 0, 64, 2, seed);
        EquilibriumMatching eqm = EquilibriumMatching.of(4, 0.8f);
        EnergyMatching enm = EnergyMatching.of(EnergyMatching.mlp(2, 64, 2, seed), 0.8f, 2f, 10, 0.01f);
        Optimizer fmOpt = Optimizer.adam(3e-3f);
        Optimizer eqOpt = Optimizer.adam(3e-3f);
        Optimizer enOpt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int i = 0; i < iterations; i++) {
            if (i == iterations * 3 / 4) {
                fmOpt.learningRate(1e-3f);
                eqOpt.learningRate(1e-3f);
                enOpt.learningRate(1e-3f);
            }
            Tensor x = Nets.matrix(data.apply(r));
            step(FlowMatching.loss(fmNet.bind(none), x, r), fmNet.parameters(), fmOpt);
            step(eqm.loss(eqNet.bind(none), x, r), eqNet.parameters(), eqOpt);
            step(enm.loss(x, r), enm.potential().parameters(), enOpt);
        }
        int n = 1024;
        float[][] real = data.apply(new SplittableRandom(seed + 500));
        float[][] moreReal = new float[n][];
        SplittableRandom rr = new SplittableRandom(seed + 501);
        for (int i = 0; i < n; i += real.length) {
            float[][] b = data.apply(rr);
            System.arraycopy(b, 0, moreReal, i, Math.min(b.length, n - i));
        }
        Tensor noise = Nets.gaussian(n, 2, new SplittableRandom(seed + 9));
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("flow matching SW", sw(Nets.rows(FlowMatching.sample(Nets.unconditional(fmNet), noise, steps)),
                moreReal));
        float eta = (float) (Math.min(1.0, 3.0 / steps) / eqm.lambda());
        out.put("EqM SW", sw(Nets.rows(eqm.sample(Nets.unconditional(eqNet), noise, steps, eta, 0, 0).samples()),
                moreReal));
        out.put("EnM SW", sw(Nets.rows(enm.sample(noise, steps, 0, 0, new SplittableRandom(seed + 3))), moreReal));

        Tensor held = Nets.matrix(moreReal);
        int[] heldNone = new int[n];
        Arrays.fill(heldNone, -1);
        ContinuousFlow flow = ContinuousFlow.of(fmNet.bind(heldNone), fmNet.parameters(), 2,
                OdeSolver.dopri5(1e-4f, 1e-4f), seed);
        out.put("flow matching NLL", (double) -flow.logProb(Var.of(held)).mean().value().item());
        out.put("EnM NLL", gridNll(enm, real, moreReal));
        return out;
    }

    private static void step(Var loss, java.util.List<Var> params, Optimizer opt) {
        loss.backward();
        Nets.clip(params, 1f);
        opt.step(params);
    }

    private static double sw(float[][] samples, float[][] real) {
        return GenerativeScores.slicedWasserstein(samples, real, 64, 11);
    }

    /**
     * {@code -log p} of {@code test} under {@code p ~ exp(-V / T)}, normalised on a 200x200 grid
     * over [-4, 4]^2, with the temperature {@code T} (one scalar, like temperature scaling)
     * chosen on {@code valid}.
     */
    private static double gridNll(EnergyMatching enm, float[][] valid, float[][] test) {
        int g = 200;
        double lo = -4;
        double cell = 8.0 / g;
        float[] grid = new float[g * g * 2];
        for (int i = 0; i < g; i++) {
            for (int j = 0; j < g; j++) {
                grid[2 * (i * g + j)] = (float) (lo + (i + 0.5) * cell);
                grid[2 * (i * g + j) + 1] = (float) (lo + (j + 0.5) * cell);
            }
        }
        float[] e = enm.energy(Tensor.of(Shape.of(g * g, 2), grid)).toArray();
        float[] ev = enm.energy(Nets.matrix(valid)).toArray();
        float[] et = enm.energy(Nets.matrix(test)).toArray();
        double best = Double.MAX_VALUE;
        double bestT = 1;
        for (double t = 0.02; t < 5; t *= 1.15) {
            double v = nll(e, ev, t, cell);
            if (v < best) {
                best = v;
                bestT = t;
            }
        }
        return nll(e, et, bestT, cell);
    }

    private static double nll(float[] grid, float[] points, double temperature, double cell) {
        double min = Double.MAX_VALUE;
        for (float v : grid) {
            min = Math.min(min, v / temperature);
        }
        double z = 0;
        for (float v : grid) {
            z += Math.exp(-(v / temperature - min)) * cell * cell;
        }
        double logZ = Math.log(z) - min;
        double nll = 0;
        for (float v : points) {
            nll += v / temperature + logZ;
        }
        return nll / points.length;
    }
}
