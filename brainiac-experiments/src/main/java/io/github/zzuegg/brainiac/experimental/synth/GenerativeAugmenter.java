package io.github.zzuegg.brainiac.experimental.synth;

import io.github.zzuegg.brainiac.experimental.nn.diffusion.ConditionalDenoiser;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Class-conditional diffusion as a source of extra training data (Azizi et al., "Synthetic Data
 * from Diffusion Models Improves ImageNet Classification", TMLR 2023,
 * <a href="https://arxiv.org/abs/2304.08466">arXiv:2304.08466</a>).
 * <ul>
 *   <li>an {@link MlpDenoiser} predicting the noise under a cosine schedule, trained on the real
 *       training set with label dropout so that the same network is also the unconditional model;</li>
 *   <li>sampling per class with classifier-free guidance {@link Denoiser.Conditional#guided} at a
 *       chosen weight (1 is the conditional model alone; higher trades diversity for fidelity),
 *       by deterministic DDIM;</li>
 *   <li>{@link #mix}: real rows plus synthetic rows at a chosen synthetic share.</li>
 * </ul>
 * Features are standardised per column before training and mapped back after sampling; the clean
 * sample predicted at each sampling step is clamped to three standard deviations.
 */
public final class GenerativeAugmenter {

    /**
     * A labelled training set.
     *
     * @param x rows
     * @param y labels
     */
    public record Labelled(float[][] x, int[] y) {
    }

    private final int dimension;
    private final int classes;
    private final MlpDenoiser net;
    private final Diffusion diffusion;
    private final Optimizer optimizer;
    private double[] mean;
    private double[] scale;

    /**
     * Creates an untrained generator.
     *
     * @param dimension features per row
     * @param classes the number of classes
     * @param width the denoiser width
     * @param depth the denoiser depth
     * @param steps diffusion training steps, for example 100
     * @param lr the Adam rate
     * @param seed the initialisation seed
     */
    public GenerativeAugmenter(int dimension, int classes, int width, int depth, int steps, float lr, long seed) {
        this.dimension = dimension;
        this.classes = classes;
        this.net = MlpDenoiser.of(dimension, classes, width, depth, seed).timeScale(1f / steps);
        this.diffusion = new Diffusion(NoiseSchedule.cosine(steps), Diffusion.Prediction.NOISE).clipSample(3f);
        this.optimizer = Optimizer.adam(lr);
    }

    /**
     * Trains on real rows.
     *
     * @param x rows
     * @param y labels
     * @param iterations Adam steps
     * @param batch rows per step
     * @param labelDrop the label dropout rate, for example 0.2
     * @param seed the seed
     * @return the final loss
     */
    public double fit(float[][] x, int[] y, int iterations, int batch, double labelDrop, long seed) {
        mean = new double[dimension];
        scale = new double[dimension];
        for (float[] row : x) {
            for (int j = 0; j < dimension; j++) {
                mean[j] += row[j] / x.length;
            }
        }
        for (float[] row : x) {
            for (int j = 0; j < dimension; j++) {
                scale[j] += (row[j] - mean[j]) * (row[j] - mean[j]) / x.length;
            }
        }
        for (int j = 0; j < dimension; j++) {
            scale[j] = Math.sqrt(Math.max(1e-6, scale[j]));
        }
        SplittableRandom r = new SplittableRandom(seed);
        double last = 0;
        for (int it = 0; it < iterations; it++) {
            float[] v = new float[batch * dimension];
            int[] labels = new int[batch];
            for (int b = 0; b < batch; b++) {
                int i = r.nextInt(x.length);
                labels[b] = y[i];
                for (int j = 0; j < dimension; j++) {
                    v[b * dimension + j] = (float) ((x[i][j] - mean[j]) / scale[j]);
                }
            }
            var loss = diffusion.loss(net.bind(ConditionalDenoiser.dropLabels(labels, labelDrop, r)),
                    Tensor.of(Shape.of(batch, dimension), v), r);
            loss.backward();
            optimizer.step(net.parameters());
            last = loss.value().item();
        }
        return last;
    }

    /**
     * Samples rows of given classes.
     *
     * @param labels the class of each row
     * @param guidance the classifier-free guidance weight; 1 is the conditional model alone
     * @param samplingSteps DDIM steps
     * @param seed the seed
     * @return the rows, in the original feature scale
     */
    public float[][] sample(int[] labels, float guidance, int samplingSteps, long seed) {
        int n = labels.length;
        int[] none = new int[n];
        Arrays.fill(none, -1);
        Tensor cond = ConditionalDenoiser.condition(labels);
        Denoiser model = guidance == 1f ? net.given(cond)
                : net.guided(cond, ConditionalDenoiser.condition(none), guidance);
        SplittableRandom r = new SplittableRandom(seed);
        Tensor noise = Tensor.normal(Shape.of(n, dimension), 0, 1, r);
        float[] out = diffusion.sample(model, noise, samplingSteps, 0f, r).toArray();
        float[][] rows = new float[n][dimension];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dimension; j++) {
                rows[i][j] = (float) (out[i * dimension + j] * scale[j] + mean[j]);
            }
        }
        return rows;
    }

    /**
     * Samples the same number of rows for every class, class-major.
     *
     * @param perClass rows per class
     * @param guidance the guidance weight
     * @param samplingSteps DDIM steps
     * @param seed the seed
     * @return the rows and their labels
     */
    public Labelled samplePerClass(int perClass, float guidance, int samplingSteps, long seed) {
        int[] labels = new int[classes * perClass];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = i / perClass;
        }
        return new Labelled(sample(labels, guidance, samplingSteps, seed), labels);
    }

    /**
     * All real rows plus synthetic rows, so that synthetic rows make up a given share of the
     * result (as far as there are enough of them), shuffled.
     *
     * @param real the real rows
     * @param synthetic the synthetic rows
     * @param syntheticShare the share of synthetic rows in the result, in {@code [0, 1)}
     * @param seed the shuffle seed
     * @return the mixed set
     */
    public static Labelled mix(Labelled real, Labelled synthetic, double syntheticShare, long seed) {
        int n = real.y().length;
        int want = (int) Math.round(n * syntheticShare / Math.max(1e-9, 1 - syntheticShare));
        int extra = Math.min(want, synthetic.y().length);
        SplittableRandom r = new SplittableRandom(seed);
        int[] pick = new int[synthetic.y().length];
        for (int i = 0; i < pick.length; i++) {
            pick[i] = i;
        }
        for (int i = pick.length - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = pick[i];
            pick[i] = pick[j];
            pick[j] = t;
        }
        float[][] x = new float[n + extra][];
        int[] y = new int[n + extra];
        for (int i = 0; i < n; i++) {
            x[i] = real.x()[i];
            y[i] = real.y()[i];
        }
        for (int i = 0; i < extra; i++) {
            x[n + i] = synthetic.x()[pick[i]];
            y[n + i] = synthetic.y()[pick[i]];
        }
        for (int i = x.length - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            float[] tx = x[i];
            x[i] = x[j];
            x[j] = tx;
            int ty = y[i];
            y[i] = y[j];
            y[j] = ty;
        }
        return new Labelled(x, y);
    }

    @Override
    public String toString() {
        return "GenerativeAugmenter(" + dimension + " features, " + classes + " classes)";
    }
}
