package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * x-prediction with a loss in another space, as in "Back to Basics: Let Denoising Generative
 * Models Denoise" (JiT, Li and He, <a href="https://arxiv.org/abs/2511.13720">2511.13720</a>).
 * The network always outputs the clean sample {@code x}; the loss can still be taken on the
 * noise or the velocity implied by that output (core {@code Diffusion.Prediction.SAMPLE}
 * fixes it to the sample).
 *
 * <p>The argument is the manifold assumption: clean data lie on a low-dimensional manifold,
 * noise and velocity do not. In high-dimensional pixel space, a network much narrower than
 * the data (a large-patch plain transformer, an MLP with a bottleneck) can represent
 * {@code x} but not the full-dimensional noise, so epsilon- and v-prediction collapse while
 * x-prediction keeps working. The implied quantities at noise level
 * {@code x_t = a x + b eps} are {@code eps = (x_t - a x) / b} and {@code v = a eps - b x},
 * so their squared errors are the sample error times {@code a^2 / b^2} and {@code 1 / b^2};
 * {@code b} is floored at {@link #MIN_NOISE} to keep the weight finite near clean data.
 *
 * <p>Sample with {@link #sampler}, the core sampler for a sample-predicting model.
 */
public final class XPrediction {
    /** The floor on the noise level {@code b} in the loss weights. */
    public static final double MIN_NOISE = 0.05;

    /** Where the loss of the x-predicting network is measured. */
    public enum LossSpace {
        /** On the sample itself (core {@code Prediction.SAMPLE}). */
        SAMPLE,
        /** On the implied noise. */
        NOISE,
        /** On the implied velocity, JiT's default. */
        VELOCITY
    }

    private XPrediction() {
    }

    /**
     * The training loss: a random step per example, noise added, the network's clean
     * estimate compared in the chosen space.
     *
     * @param schedule the noise schedule
     * @param model the network, predicting the clean sample; it receives core
     *     {@code Diffusion}'s step index as time
     * @param x0 clean data, batch first
     * @param space where to measure the loss
     * @param random the source of steps and noise
     * @return the loss, a scalar
     */
    public static Var loss(NoiseSchedule schedule, Denoiser model, Tensor x0, LossSpace space,
            RandomGenerator random) {
        int batch = x0.shape().dim(0);
        int[] steps = new int[batch];
        float[] time = new float[batch];
        float[] w = new float[batch];
        for (int i = 0; i < batch; i++) {
            steps[i] = random.nextInt(schedule.steps());
            time[i] = steps[i];
            double ab = schedule.alphaBar(steps[i]);
            double b2 = Math.max(MIN_NOISE * MIN_NOISE, 1 - ab);
            w[i] = (float) switch (space) {
                case SAMPLE -> 1;
                case NOISE -> ab / b2;
                case VELOCITY -> 1 / b2;
            };
        }
        Diffusion d = new Diffusion(schedule, Diffusion.Prediction.SAMPLE);
        Tensor noise = Tensor.normal(x0.shape(), 0, 1, random).to(x0.device()).to(x0.dtype());
        Tensor xt = d.noisy(x0, noise, steps);
        Var out = model.apply(Var.of(xt), Tensor.of(Shape.of(batch), time, x0.device()));
        int[] dims = new int[x0.shape().rank()];
        java.util.Arrays.fill(dims, 1);
        dims[0] = batch;
        return out.sub(Var.of(x0)).square().mul(Var.of(Tensor.of(Shape.of(dims), w, x0.device()))).mean();
    }

    /**
     * The core sampler for an x-predicting network.
     *
     * @param schedule the noise schedule it was trained with
     * @param clip the range to clamp the predicted samples to, 0 for none
     * @return the sampler
     */
    public static Diffusion sampler(NoiseSchedule schedule, float clip) {
        Diffusion d = new Diffusion(schedule, Diffusion.Prediction.SAMPLE);
        return clip > 0 ? d.clipSample(clip) : d;
    }
}
