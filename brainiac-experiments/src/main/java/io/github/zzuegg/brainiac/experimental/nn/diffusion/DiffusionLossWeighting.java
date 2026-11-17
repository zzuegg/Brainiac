package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Per-timestep loss weighting for core {@link Diffusion}, which itself averages the squared
 * error uniformly over steps. The same draw of steps and noise as {@link Diffusion#loss},
 * but each example's error is scaled by a weight that depends on its signal-to-noise ratio
 * {@code SNR(t) = alphaBar / (1 - alphaBar)} and on what the network predicts
 * ({@link Diffusion.Prediction}: noise, clean sample or velocity).
 *
 * <p><b>Min-SNR-gamma</b> (Hang et al., <a href="https://arxiv.org/abs/2303.09556">2303.09556</a>)
 * treats denoising at different noise levels as tasks that fight over the gradient: the
 * nearly clean steps are easy but dominate with uniform weights. It clamps the equivalent
 * clean-sample weight at {@code min(SNR, gamma)}, which in the other parameterizations is
 * {@code min(SNR, gamma) / SNR} for noise prediction and {@code min(SNR, gamma) / (SNR + 1)}
 * for velocity prediction. With velocity prediction (Salimans and Ho,
 * <a href="https://arxiv.org/abs/2202.00512">2202.00512</a>, core {@code Prediction.VELOCITY})
 * this is the recipe of many current image models.
 */
public final class DiffusionLossWeighting {
    private final float gamma;

    private DiffusionLossWeighting(float gamma) {
        this.gamma = gamma;
    }

    /** @return the plain, uniform weighting of core {@link Diffusion#loss} */
    public static DiffusionLossWeighting uniform() {
        return new DiffusionLossWeighting(Float.NaN);
    }

    /**
     * Min-SNR-gamma weighting.
     *
     * @param gamma the clamp, 5 in the paper
     * @return the weighting
     */
    public static DiffusionLossWeighting minSnr(float gamma) {
        if (!(gamma > 0)) {
            throw new IllegalArgumentException("gamma must be positive");
        }
        return new DiffusionLossWeighting(gamma);
    }

    /**
     * The weight of one example.
     *
     * @param prediction what the network predicts
     * @param alphaBar the cumulative signal level of its step
     * @return the weight; 1 for the uniform weighting
     */
    public double weight(Diffusion.Prediction prediction, double alphaBar) {
        if (Float.isNaN(gamma)) {
            return 1;
        }
        double snr = alphaBar / Math.max(1e-12, 1 - alphaBar);
        double clamped = Math.min(snr, gamma);
        return switch (prediction) {
            case NOISE -> clamped / snr;
            case SAMPLE -> clamped;
            case VELOCITY -> clamped / (snr + 1);
        };
    }

    /**
     * The weighted training loss for a batch.
     *
     * @param diffusion the process (schedule and prediction target)
     * @param model the denoiser
     * @param x0 clean data, batch first
     * @param random the source of steps and noise
     * @return the loss, a scalar
     */
    public Var loss(Diffusion diffusion, Denoiser model, Tensor x0, RandomGenerator random) {
        NoiseSchedule schedule = diffusion.schedule();
        int batch = x0.shape().dim(0);
        int[] steps = new int[batch];
        float[] time = new float[batch];
        float[] a = new float[batch];
        float[] b = new float[batch];
        float[] w = new float[batch];
        for (int i = 0; i < batch; i++) {
            steps[i] = random.nextInt(schedule.steps());
            time[i] = steps[i];
            double ab = schedule.alphaBar(steps[i]);
            a[i] = (float) Math.sqrt(ab);
            b[i] = (float) Math.sqrt(1 - ab);
            w[i] = (float) weight(diffusion.prediction(), ab);
        }
        Tensor noise = Tensor.normal(x0.shape(), 0, 1, random).to(x0.device()).to(x0.dtype());
        Tensor xt = diffusion.noisy(x0, noise, steps);
        Tensor target = switch (diffusion.prediction()) {
            case NOISE -> noise;
            case SAMPLE -> x0;
            case VELOCITY -> noise.mul(perExample(a, x0)).sub(x0.mul(perExample(b, x0)));
        };
        Var out = model.apply(Var.of(xt), Tensor.of(Shape.of(batch), time, x0.device()));
        return out.sub(Var.of(target)).square().mul(Var.of(perExample(w, x0))).mean();
    }

    private static Tensor perExample(float[] values, Tensor like) {
        int[] dims = new int[like.shape().rank()];
        java.util.Arrays.fill(dims, 1);
        dims[0] = values.length;
        return Tensor.of(Shape.of(dims), values.clone(), like.device()).to(like.dtype());
    }
}
