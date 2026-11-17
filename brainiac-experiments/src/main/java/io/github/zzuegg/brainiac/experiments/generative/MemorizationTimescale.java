package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * When does a diffusion model start to memorize? Bonnaire et al., "Why Diffusion Models
 * Don't Memorize: The Role of Implicit Dynamical Regularization in Training"
 * (<a href="https://arxiv.org/abs/2505.17638">2505.17638</a>, NeurIPS 2025 best paper)
 * find two timescales: samples become good after a time {@code tau_gen} that does not
 * depend on the training-set size {@code n}, and the model starts copying training points
 * only after {@code tau_mem}, which grows in proportion to {@code n}. Between the two lies
 * a window of generalization that widens with more data, and stopping there avoids
 * memorization.
 *
 * <p>Here: the {@link TinyDdpm} setting on {@link ToyPoints} (a ring of 8 blobs), a training
 * set of {@code n} points, and checkpoints along training at which samples are scored for
 * quality (mean distance to the nearest blob centre, against the same for real points) and
 * for copying (the share of samples whose nearest training point is at least three times
 * closer than the second nearest, the criterion of the memorization literature), measured
 * against the same rate for fresh real points, which copy by chance when {@code n} is small.
 * The held-out denoising loss is also tracked, to compare its minimum with the timescales.
 */
@SuppressWarnings("exports")
public final class MemorizationTimescale {
    private static final int STEPS = 100;
    private static final int WIDTH = 128;
    private static final ToyPoints DATA = ToyPoints.ring(8, 2, 0.3);

    private MemorizationTimescale() {
    }

    /**
     * One training run's curves.
     *
     * @param steps the checkpoint steps
     * @param quality mean distance of samples to the nearest blob centre at each checkpoint
     * @param copyRate share of samples copying a training point at each checkpoint
     * @param validation held-out denoising loss at each checkpoint
     * @param realQuality the quality measure for real points, the target
     * @param realCopyRate the copy rate of fresh real points against the training set, the
     *     share that counts as copying by chance
     */
    public record Curves(int[] steps, double[] quality, double[] copyRate, double[] validation, double realQuality,
            double realCopyRate) {

        /**
         * @param tolerance how close to the real quality counts as generalizing
         * @return the first checkpoint whose quality is within {@code tolerance} times the
         *     real one, or -1
         */
        public int generalizationTime(double tolerance) {
            for (int i = 0; i < steps.length; i++) {
                if (quality[i] < tolerance * realQuality) {
                    return steps[i];
                }
            }
            return -1;
        }

        /**
         * @param i a checkpoint index
         * @return the copy rate above chance at checkpoint {@code i}, averaged with its
         *     neighbours to smooth sampling noise
         */
        public double excessCopy(int i) {
            double sum = 0;
            int count = 0;
            for (int j = Math.max(0, i - 1); j <= Math.min(steps.length - 1, i + 1); j++) {
                sum += copyRate[j] - realCopyRate;
                count++;
            }
            return sum / count;
        }

        /**
         * @param threshold the copy rate above chance that counts as memorizing
         * @return the first checkpoint whose smoothed excess copy rate reaches
         *     {@code threshold}, or -1
         */
        public int memorizationTime(double threshold) {
            for (int i = 0; i < steps.length; i++) {
                if (excessCopy(i) >= threshold) {
                    return steps[i];
                }
            }
            return -1;
        }

        /** @return the checkpoint with the lowest held-out loss */
        public int earlyStop() {
            int best = 0;
            for (int i = 1; i < steps.length; i++) {
                if (validation[i] < validation[best]) {
                    best = i;
                }
            }
            return best;
        }
    }

    /**
     * Trains on {@code n} points and records the curves.
     *
     * @param n the training-set size
     * @param iterations training steps
     * @param every steps between checkpoints
     * @param seed the seed
     * @return the curves
     */
    public static Curves run(int n, int iterations, int every, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] train = new float[n][];
        for (int i = 0; i < n; i++) {
            train[i] = DATA.sample(r.nextInt(DATA.modes()), r);
        }
        float[][] held = new float[256][];
        for (int i = 0; i < held.length; i++) {
            held[i] = DATA.sample(i % DATA.modes(), r);
        }
        double realQuality = DATA.score(held, null)[0];
        MlpDenoiser net = MlpDenoiser.of(2, 0, WIDTH, 3, seed).timeScale(1f / STEPS);
        Diffusion d = new Diffusion(NoiseSchedule.cosine(STEPS), Diffusion.Prediction.NOISE).clipSample(3f);
        Optimizer opt = Optimizer.adam(2e-3f);
        int batch = Math.min(n, 64);
        int[] none = new int[batch];
        Arrays.fill(none, -1);
        int[] heldNone = new int[held.length];
        Arrays.fill(heldNone, -1);
        int checkpoints = iterations / every;
        int[] steps = new int[checkpoints];
        double[] quality = new double[checkpoints];
        double[] copy = new double[checkpoints];
        double[] validation = new double[checkpoints];
        for (int c = 0; c < checkpoints; c++) {
            for (int s = 0; s < every; s++) {
                float[][] x = new float[batch][];
                for (int b = 0; b < batch; b++) {
                    x[b] = train[r.nextInt(n)];
                }
                Var loss = d.loss(net.bind(none), Nets.matrix(x), r);
                loss.backward();
                Nets.clip(net.parameters(), 1f);
                opt.step(net.parameters());
            }
            steps[c] = (c + 1) * every;
            validation[c] = d.loss(net.bind(heldNone), Nets.matrix(held), new SplittableRandom(seed + 77)).value()
                    .item();
            float[][] samples = sample(d, net, seed);
            quality[c] = DATA.score(samples, null)[0];
            copy[c] = copyRate(samples, train);
        }
        double realCopy = copyRate(Arrays.copyOf(held, 200), train);
        return new Curves(steps, quality, copy, validation, realQuality, realCopy);
    }

    private static float[][] sample(Diffusion d, MlpDenoiser net, long seed) {
        int[] none = new int[200];
        Arrays.fill(none, -1);
        Tensor noise = Nets.gaussian(none.length, 2, new SplittableRandom(seed + 99));
        return Nets.rows(d.sample(net.bind(none), noise, 50, 0, new SplittableRandom(seed + 98)));
    }

    /**
     * The share of samples that copy a training point: nearest training point at least three
     * times closer than the second nearest.
     *
     * @param samples generated points
     * @param train the training set (at least two points)
     * @return the copy rate
     */
    public static double copyRate(float[][] samples, float[][] train) {
        int copies = 0;
        for (float[] s : samples) {
            double first = Double.MAX_VALUE;
            double second = Double.MAX_VALUE;
            for (float[] t : train) {
                double d = Math.sqrt(Nets.distance2(s, t));
                if (d < first) {
                    second = first;
                    first = d;
                } else if (d < second) {
                    second = d;
                }
            }
            copies += first < second / 3 ? 1 : 0;
        }
        return (double) copies / samples.length;
    }
}
