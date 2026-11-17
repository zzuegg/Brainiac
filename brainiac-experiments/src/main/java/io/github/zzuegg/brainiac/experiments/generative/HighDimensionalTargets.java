package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.XPrediction;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * The {@link DiffusionTargets} comparison moved to a high-dimensional pixel space, the
 * setting of JiT ({@link XPrediction}): the 64-pixel {@link ToyImages} are embedded in
 * {@code D} dimensions by a fixed random orthonormal map, so the data lie on a 64-dimensional
 * subspace (the "manifold") of a much bigger space, and the denoiser is an MLP narrower than
 * {@code D}. At a fixed budget, epsilon prediction, velocity prediction, x-prediction with
 * the sample loss and x-prediction with the velocity loss are trained and sampled with the
 * same deterministic steps. Samples are mapped back to 64 pixels and scored by the mean
 * squared distance per pixel to the nearest real image; the energy they put outside the
 * data subspace is reported too.
 */
@SuppressWarnings("exports")
public final class HighDimensionalTargets {
    private static final int STEPS = 100;

    /** The training targets compared. */
    public enum Target {
        /** Core noise prediction. */
        EPSILON,
        /** Core velocity prediction. */
        VELOCITY,
        /** x-prediction, loss on x (core sample prediction). */
        X_SAMPLE_LOSS,
        /** x-prediction, loss on the implied velocity (JiT). */
        X_VELOCITY_LOSS
    }

    private HighDimensionalTargets() {
    }

    /**
     * A random {@code dimension x 64} matrix with orthonormal columns (Gram-Schmidt).
     *
     * @param dimension the embedding dimension, at least 64
     * @param seed the seed
     * @return the columns, {@code [64][dimension]}
     */
    static float[][] basis(int dimension, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] q = new double[ToyImages.PIXELS][dimension];
        for (int c = 0; c < q.length; c++) {
            for (int i = 0; i < dimension; i++) {
                q[c][i] = r.nextGaussian();
            }
            for (int p = 0; p < c; p++) {
                double dot = 0;
                for (int i = 0; i < dimension; i++) {
                    dot += q[c][i] * q[p][i];
                }
                for (int i = 0; i < dimension; i++) {
                    q[c][i] -= dot * q[p][i];
                }
            }
            double n = 0;
            for (double v : q[c]) {
                n += v * v;
            }
            n = Math.sqrt(n);
            for (int i = 0; i < dimension; i++) {
                q[c][i] /= n;
            }
        }
        float[][] out = new float[q.length][dimension];
        for (int c = 0; c < q.length; c++) {
            for (int i = 0; i < dimension; i++) {
                out[c][i] = (float) q[c][i];
            }
        }
        return out;
    }

    private static float[] lift(float[] image, float[][] basis) {
        float[] out = new float[basis[0].length];
        for (int c = 0; c < image.length; c++) {
            for (int i = 0; i < out.length; i++) {
                out[i] += image[c] * basis[c][i];
            }
        }
        return out;
    }

    private static float[] project(float[] x, float[][] basis) {
        float[] out = new float[basis.length];
        for (int c = 0; c < basis.length; c++) {
            double s = 0;
            for (int i = 0; i < x.length; i++) {
                s += x[i] * basis[c][i];
            }
            out[c] = (float) s;
        }
        return out;
    }

    /**
     * Trains one target and scores its samples.
     *
     * @param target the training target
     * @param dimension the embedding dimension D
     * @param width the denoiser width (narrower than D is the interesting case)
     * @param iterations training steps of 64 images
     * @param samplingSteps DDIM steps
     * @param seed the seed
     * @return {@code {distance per pixel to the nearest real image, share of sample energy off the data subspace}}
     */
    public static double[] run(Target target, int dimension, int width, int iterations, int samplingSteps,
            long seed) {
        float[][] basis = basis(dimension, 7);
        NoiseSchedule schedule = NoiseSchedule.cosine(STEPS);
        MlpDenoiser net = MlpDenoiser.of(dimension, 0, width, 2, seed).timeScale(1f / STEPS);
        Diffusion.Prediction core = switch (target) {
            case EPSILON -> Diffusion.Prediction.NOISE;
            case VELOCITY -> Diffusion.Prediction.VELOCITY;
            default -> Diffusion.Prediction.SAMPLE;
        };
        Diffusion diffusion = new Diffusion(schedule, core).clipSample(2f);   // data stay within +-1
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int[] none = new int[64];
        Arrays.fill(none, -1);
        Denoiser model = net.bind(none);
        for (int i = 0; i < iterations; i++) {
            float[][] x = new float[64][];
            for (int b = 0; b < x.length; b++) {
                x[b] = lift(ToyImages.sample(r.nextInt(ToyImages.CLASSES), r), basis);
            }
            Var loss = target == Target.X_VELOCITY_LOSS
                    ? XPrediction.loss(schedule, model, Nets.matrix(x), XPrediction.LossSpace.VELOCITY, r)
                    : diffusion.loss(model, Nets.matrix(x), r);
            loss.backward();
            Nets.clip(net.parameters(), 1f);
            opt.step(net.parameters());
        }
        int[] all = new int[128];
        Arrays.fill(all, -1);
        Tensor noise = Nets.gaussian(all.length, dimension, new SplittableRandom(seed + 99));
        float[][] samples = Nets.rows(diffusion.sample(net.bind(all), noise, samplingSteps, 0,
                new SplittableRandom(seed + 2)));
        double d = 0;
        double off = 0;
        for (float[] s : samples) {
            float[] img = project(s, basis);
            d += ToyImages.nearest(img)[1] / ToyImages.PIXELS;
            double total = 0;
            double inside = 0;
            for (float v : s) {
                total += v * v;
            }
            for (float v : img) {
                inside += v * v;
            }
            off += (total - inside) / Math.max(1e-9, total);
        }
        return new double[] {d / samples.length, off / samples.length};
    }
}
