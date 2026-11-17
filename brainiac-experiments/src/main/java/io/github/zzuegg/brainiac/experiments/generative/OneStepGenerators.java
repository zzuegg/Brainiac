package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.ConsistencyModel;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MeanFlow;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * One-step generation on the {@link ToyPoints} ring, four ways at the same number of
 * training steps: MeanFlow and shortcut models ({@link MeanFlow}), which learn average
 * velocities so one step of size 1 is accurate by construction; flow matching straightened
 * by reflow ({@link RectifiedFlowExperiment}); and consistency training
 * ({@link ConsistencyModel}). Each is scored on one-step samples from the same noise: the
 * mean distance to the nearest blob centre and the share of points within 0.3 of one.
 */
@SuppressWarnings("exports")
public final class OneStepGenerators {
    private static final int BATCH = 128;

    /**
     * One method's score.
     *
     * @param method the method
     * @param distance the mean distance of one-step samples to the nearest blob centre
     * @param nearShare the share of one-step samples within 0.3 of a centre
     * @param modes how many blobs received at least 5% of the samples
     */
    public record Score(String method, double distance, double nearShare, int modes) {
    }

    private OneStepGenerators() {
    }

    private static float[][] batch(ToyPoints ring, SplittableRandom r) {
        float[][] x = new float[BATCH][];
        for (int b = 0; b < BATCH; b++) {
            x[b] = ring.sample(r.nextInt(ring.modes()), r);
        }
        return x;
    }

    /**
     * Scores generated points.
     *
     * @param ring the target distribution
     * @param method the method name
     * @param points samples
     * @return the score
     */
    public static Score score(ToyPoints ring, String method, float[][] points) {
        int[] counts = new int[ring.modes()];
        int near = 0;
        double dist = 0;
        for (float[] p : points) {
            double[] n = ring.nearest(p);
            dist += n[1];
            near += n[1] < 0.3 ? 1 : 0;
            counts[(int) n[0]]++;
        }
        int modes = (int) Arrays.stream(counts).filter(c -> c >= 0.05 * points.length).count();
        return new Score(method, dist / points.length, (double) near / points.length, modes);
    }

    private static float[][] trainAverageVelocity(ToyPoints ring, int iterations, long seed,
            Function<Tensor, Var> loss, MeanFlow.Network net, int samples) {
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 7);
        for (int i = 0; i < iterations; i++) {
            if (i == iterations * 3 / 4) {
                opt.learningRate(1e-3f);
            }
            Var l = loss.apply(Nets.matrix(batch(ring, r)));
            l.backward();
            Nets.clip(net.parameters(), 10f);
            opt.step(net.parameters());
        }
        return Nets.rows(MeanFlow.sample(net, Nets.gaussian(samples, 2, new SplittableRandom(seed + 99)), 1));
    }

    /**
     * MeanFlow, one step.
     *
     * @param ring the data
     * @param iterations training steps
     * @param samples how many points to generate
     * @param seed the seed
     * @return the score
     */
    public static Score meanFlow(ToyPoints ring, int iterations, int samples, long seed) {
        MeanFlow.Network net = MeanFlow.mlp(2, 64, 2, seed);
        SplittableRandom r = new SplittableRandom(seed);
        return score(ring, "MeanFlow", trainAverageVelocity(ring, iterations, seed,
                x -> MeanFlow.meanFlowLoss(net, x, 0.5f, 1f, r), net, samples));
    }

    /**
     * A shortcut model, one step.
     *
     * @param ring the data
     * @param iterations training steps
     * @param samples how many points to generate
     * @param seed the seed
     * @return the score
     */
    public static Score shortcut(ToyPoints ring, int iterations, int samples, long seed) {
        MeanFlow.Network net = MeanFlow.mlp(2, 64, 2, seed);
        SplittableRandom r = new SplittableRandom(seed);
        return score(ring, "shortcut", trainAverageVelocity(ring, iterations, seed,
                x -> MeanFlow.shortcutLoss(net, x, 0.75f, 6, r), net, samples));
    }

    private static int[] none(int n) {
        int[] l = new int[n];
        Arrays.fill(l, -1);
        return l;
    }

    /**
     * Flow matching for two thirds of the steps, then reflow on its own 8-step Heun samples;
     * also returns the one-step score before reflow.
     *
     * @param ring the data
     * @param iterations training steps in all
     * @param samples how many points to generate
     * @param seed the seed
     * @return {@code {before reflow, after reflow}}
     */
    public static Score[] reflow(ToyPoints ring, int iterations, int samples, long seed) {
        RectifiedFlowExperiment fm = RectifiedFlowExperiment.of(MlpDenoiser.of(2, 0, 64, 2, seed), 3e-3f, 0, seed + 1);
        SplittableRandom r = new SplittableRandom(seed + 7);
        int flowSteps = iterations * 2 / 3;
        int[] unlabelled = none(BATCH);
        for (int i = 0; i < flowSteps; i++) {
            fm.trainStep(batch(ring, r), unlabelled);
        }
        float[][] probe = fm.noise(samples, seed + 99);
        int[] probeLabels = none(samples);
        Score before = score(ring, "flow matching", fm.sampleEuler(probe, probeLabels, 1, 1, null));
        fm.learningRate(1e-3f);
        for (int i = flowSteps; i < iterations; i++) {
            float[][] z = fm.noise(BATCH, seed + 1000 + i);
            fm.reflowStep(z, fm.sampleHeun(z, unlabelled, 8, 1, null), unlabelled);
        }
        Score after = score(ring, "reflow", fm.sampleEuler(probe, probeLabels, 1, 1, null));
        return new Score[] {before, after};
    }

    /**
     * Consistency training, one step.
     *
     * @param ring the data
     * @param iterations training steps
     * @param samples how many points to generate
     * @param seed the seed
     * @return the score
     */
    public static Score consistency(ToyPoints ring, int iterations, int samples, long seed) {
        ConsistencyModel cm = ConsistencyModel.of(MlpDenoiser.of(2, 0, 64, 2, seed), 0.002, 10, 2e-3f, seed + 1);
        SplittableRandom r = new SplittableRandom(seed + 7);
        int[] unlabelled = none(BATCH);
        for (int i = 0; i < iterations; i++) {
            cm.trainStep(batch(ring, r), unlabelled, iterations);
        }
        return score(ring, "consistency", cm.sampleOneStep(none(samples), seed + 99));
    }
}
