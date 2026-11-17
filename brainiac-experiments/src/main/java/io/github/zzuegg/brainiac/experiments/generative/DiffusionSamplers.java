package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ContinuousFlow;
import io.github.zzuegg.brainiac.experimental.nn.DpmSolver;
import io.github.zzuegg.brainiac.experimental.nn.EdmDenoiser;
import io.github.zzuegg.brainiac.experimental.nn.OdeSolver;
import io.github.zzuegg.brainiac.experimental.nn.ScoreSde;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.nn.diffusion.FlowMatching;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Sample quality against network evaluations (NFE) for continuous-time diffusion samplers on
 * the six-blob {@link ToyPoints} ring, every model the same {@link MlpDenoiser} trained for
 * the same steps: the score SDE ({@link ScoreSde}, wave 6 item 4.6) sampled with
 * predictor-corrector steps and with its probability-flow ODE; EDM ({@link EdmDenoiser}, 4.7)
 * with its Heun sampler and with DPM-Solver++(2M) ({@link DpmSolver}, 4.7); and the baselines
 * DDPM / DDIM (core {@link Diffusion}, cosine schedule, 100 steps, the clean estimate
 * clipped to the data box [-3, 3]) and core flow matching with Heun steps. Quality is the mean
 * distance of samples to the nearest blob centre (blob spread 0.08) and the number of blobs
 * covered. {@link #sdeLikelihood} also scores the VP-SDE's exact likelihood through its
 * probability-flow ODE, for comparison with FFJORD ({@link DensityBenchmark}).
 */
@SuppressWarnings("exports")
public final class DiffusionSamplers {
    /** The target ring. */
    public static final ToyPoints RING = ToyPoints.ring(6, 2, 0.08);
    private static final int BATCH = 128;
    private static final int SAMPLES = 512;
    private static final int DDPM_STEPS = 100;
    private static final Diffusion DDPM = new Diffusion(NoiseSchedule.cosine(DDPM_STEPS), Diffusion.Prediction.NOISE)
            .clipSample(3f);

    /** Trained models, ready to sample. */
    public static final class Models {
        private final Denoiser ddpm;
        private final ScoreSde sde;
        private final EdmDenoiser edm;
        private final Denoiser flow;
        private final int dim;

        Models(Denoiser ddpm, ScoreSde sde, EdmDenoiser edm, Denoiser flow, int dim) {
            this.dim = dim;
            this.ddpm = ddpm;
            this.sde = sde;
            this.edm = edm;
            this.flow = flow;
        }

        /** @return the EDM denoiser, a teacher for distillation */
        public EdmDenoiser edm() {
            return edm;
        }
    }

    private DiffusionSamplers() {
    }

    private static int[] none(int n) {
        int[] l = new int[n];
        Arrays.fill(l, -1);
        return l;
    }

    /**
     * Trains the four models on the ring.
     *
     * @param iterations training steps of 128 points each
     * @param seed the seed
     * @return the models
     */
    public static Models train(int iterations, long seed) {
        return train(r -> PointClouds.blobs(RING, BATCH, r), 2, iterations, seed);
    }

    /**
     * Trains the four models on any 2-D or image data.
     *
     * @param data draws a training batch
     * @param dim the sample dimension
     * @param iterations training steps
     * @param seed the seed
     * @return the models
     */
    public static Models train(Function<SplittableRandom, float[][]> data, int dim, int iterations, long seed) {
        MlpDenoiser ddpmNet = MlpDenoiser.of(dim, 0, 64, 2, seed).timeScale(1f / DDPM_STEPS);
        Optimizer ddpmOpt = Optimizer.adam(3e-3f);
        MlpDenoiser sdeNet = MlpDenoiser.of(dim, 0, 64, 2, seed);
        ScoreSde sde = ScoreSde.vp(Nets.unconditional(sdeNet));
        MlpDenoiser edmNet = MlpDenoiser.of(dim, 0, 64, 2, seed);
        EdmDenoiser edm = EdmDenoiser.of(Nets.unconditional(edmNet), edmNet.parameters(), 1f);
        MlpDenoiser fmNet = MlpDenoiser.of(dim, 0, 64, 2, seed);
        Optimizer sdeOpt = Optimizer.adam(3e-3f);
        Optimizer edmOpt = Optimizer.adam(3e-3f);
        Optimizer fmOpt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int[] unlabelled = none(BATCH);
        for (int i = 0; i < iterations; i++) {
            if (i == iterations * 3 / 4) {
                ddpmOpt.learningRate(1e-3f);
                sdeOpt.learningRate(1e-3f);
                edmOpt.learningRate(1e-3f);
                fmOpt.learningRate(1e-3f);
            }
            float[][] batch = data.apply(r);
            Tensor x = Nets.matrix(batch);
            step(DDPM.loss(ddpmNet.bind(unlabelled), x, r), ddpmNet.parameters(), ddpmOpt);
            step(sde.loss(x, r), sdeNet.parameters(), sdeOpt);
            step(edm.loss(x, r), edmNet.parameters(), edmOpt);
            step(FlowMatching.loss(fmNet.bind(unlabelled), x, r), fmNet.parameters(), fmOpt);
        }
        return new Models(Nets.unconditional(ddpmNet), sde, edm, Nets.unconditional(fmNet), dim);
    }

    private static void step(Var loss, java.util.List<Var> params, Optimizer opt) {
        loss.backward();
        Nets.clip(params, 1f);
        opt.step(params);
    }

    /**
     * Samples every method at about the given number of network evaluations.
     *
     * @param models trained models
     * @param nfe the evaluation budget, even
     * @param seed the noise seed
     * @return per method, the samples
     */
    public static Map<String, float[][]> sample(Models models, int nfe, long seed) {
        Tensor noise = Nets.gaussian(SAMPLES, models.dim, new SplittableRandom(seed));
        Map<String, float[][]> out = new LinkedHashMap<>();
        out.put("DDIM", Nets.rows(DDPM.sample(models.ddpm, noise, nfe, 0, new SplittableRandom(seed))));
        out.put("SDE PC", Nets.rows(models.sde.samplePredictorCorrector(noise, nfe / 2, 1, 0.16,
                new SplittableRandom(seed + 1))));
        out.put("SDE ODE", Nets.rows(models.sde.sampleOde(noise, nfe / 2 + 1, true)));
        out.put("EDM Heun", Nets.rows(models.edm.sample(noise, EdmDenoiser.karrasSigmas(nfe / 2 + 1, 0.002, 20, 7),
                true)));
        out.put("EDM DPM-Solver++", Nets.rows(DpmSolver.sample(models.edm, noise,
                EdmDenoiser.karrasSigmas(nfe, 0.002, 20, 7))));
        out.put("flow matching Heun", Nets.rows(FlowMatching.sample(models.flow, noise, nfe / 2,
                FlowMatching.Solver.HEUN, null)));
        return out;
    }

    /**
     * DDPM's ancestral sampler over all 100 steps, the reference at high NFE.
     *
     * @param models trained models
     * @param seed the noise seed
     * @return the samples
     */
    public static float[][] sampleDdpm(Models models, long seed) {
        Tensor noise = Nets.gaussian(SAMPLES, models.dim, new SplittableRandom(seed));
        return Nets.rows(DDPM.sample(models.ddpm, noise, DDPM_STEPS, 1, new SplittableRandom(seed + 1)));
    }

    /**
     * Mean distance to the nearest ring blob, and blobs covered by at least 5% of samples.
     *
     * @param samples points
     * @return {@code {distance, blobs}}
     */
    public static double[] quality(float[][] samples) {
        int[] counts = new int[RING.modes()];
        double dist = 0;
        for (float[] p : samples) {
            double[] n = RING.nearest(p);
            if (n[0] < 0) {
                dist += 10;   // a diverged (NaN) sample
                continue;
            }
            dist += Math.min(10, n[1]);
            counts[(int) n[0]]++;
        }
        int modes = (int) Arrays.stream(counts).filter(c -> c >= 0.05 * samples.length).count();
        return new double[] {dist / samples.length, modes};
    }

    /**
     * Trains a VP score SDE on a 2-D target and returns its exact held-out negative
     * log-likelihood through the probability-flow ODE (adaptive Dormand-Prince).
     *
     * @param target the data
     * @param iterations training steps
     * @param seed the seed
     * @return {@code {nats per point, function evaluations}}
     */
    public static double[] sdeLikelihood(DensityBenchmark.Target target, int iterations, long seed) {
        MlpDenoiser net = MlpDenoiser.of(2, 0, 64, 2, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        ScoreSde train = ScoreSde.vp(Nets.unconditional(net));
        for (int i = 0; i < iterations; i++) {
            if (i == iterations * 3 / 4) {
                opt.learningRate(1e-3f);
            }
            step(train.loss(Nets.matrix(DensityBenchmark.data(target, BATCH, r)), r), net.parameters(), opt);
        }
        int n = 1024;
        ContinuousFlow flow = ContinuousFlow.of(train.probabilityFlow(), net.parameters(), 2,
                OdeSolver.dopri5(1e-4f, 1e-4f), seed);
        Tensor held = Nets.matrix(DensityBenchmark.data(target, n, new SplittableRandom(seed + 1000)));
        double nll = -flow.logProb(Var.of(held)).mean().value().item();
        return new double[] {nll, flow.evaluations()};
    }
}
