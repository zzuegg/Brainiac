package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Flow matching with straight paths (Lipman et al., 2023; Liu et al., rectified flow): data
 * at time 0, noise at time 1, {@code x_t = (1 - t) x0 + t noise}; the {@link Denoiser} learns
 * the velocity {@code noise - x0}, and {@link #sample} follows it back from noise with Euler
 * steps.
 */
public final class FlowMatching {

    private FlowMatching() {
    }

    /**
     * The training loss for a batch: a uniform time per example and the mean squared error of
     * the predicted velocity.
     *
     * @param model the velocity model
     * @param x0 clean data, batch first
     * @param random the source of times and noise
     * @return the loss, a scalar
     */
    public static Var loss(Denoiser model, Tensor x0, RandomGenerator random) {
        int batch = x0.shape().dim(0);
        double[] t = new double[batch];
        double[] keep = new double[batch];
        float[] time = new float[batch];
        for (int i = 0; i < batch; i++) {
            t[i] = random.nextDouble();
            keep[i] = 1 - t[i];
            time[i] = (float) t[i];
        }
        Tensor noise = Tensor.normal(x0.shape(), 0, 1, random).to(x0.device()).to(x0.dtype());
        Tensor xt = x0.mul(Diffusion.perExample(keep, x0)).add(noise.mul(Diffusion.perExample(t, x0)));
        Var v = model.apply(Var.of(xt), Tensor.of(Shape.of(batch), time, x0.device()));
        return v.sub(Var.of(noise.sub(x0))).square().mean();
    }

    /**
     * Turns standard normal noise into data with {@code steps} Euler steps from time 1 to 0.
     *
     * @param model the velocity model
     * @param shape the samples' shape, batch first
     * @param steps the number of steps
     * @param random the source of the noise
     * @param device where to compute
     * @return the samples
     */
    public static Tensor sample(Denoiser model, Shape shape, int steps, RandomGenerator random, Device device) {
        return sample(model, Tensor.normal(shape, 0, 1, random).to(device), steps);
    }

    /**
     * Follows the velocity from given noise at time 1 to time 0.
     *
     * @param model the velocity model
     * @param noise the start, batch first
     * @param steps the number of Euler steps
     * @return the samples
     */
    public static Tensor sample(Denoiser model, Tensor noise, int steps) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be positive, got " + steps);
        }
        int batch = noise.shape().dim(0);
        Tensor x = noise;
        float dt = 1f / steps;
        for (int i = 0; i < steps; i++) {
            float[] time = new float[batch];
            java.util.Arrays.fill(time, 1 - i * dt);
            Tensor v = model.apply(Var.of(x), Tensor.of(Shape.of(batch), time, x.device())).value();
            x = x.sub(v.mul(dt));
        }
        return x;
    }
}
