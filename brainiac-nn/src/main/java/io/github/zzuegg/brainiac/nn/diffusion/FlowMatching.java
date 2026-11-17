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
        return sample(model, noise, steps, Solver.EULER, null);
    }

    /** How {@link #sample(Denoiser, Tensor, int, Solver, java.util.List)} steps along the velocity. */
    public enum Solver {
        /** One model call per step. */
        EULER,
        /** Heun's method: two calls per step, second-order accurate, so fewer steps do. */
        HEUN
    }

    /**
     * Follows the velocity from given noise at time 1 to time 0.
     *
     * @param model the velocity model
     * @param noise the start, batch first
     * @param steps the number of steps
     * @param solver how to step
     * @param trajectory if not {@code null}, receives the start and the state after every
     *     step
     * @return the samples
     */
    public static Tensor sample(Denoiser model, Tensor noise, int steps, Solver solver,
            java.util.List<Tensor> trajectory) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be positive, got " + steps);
        }
        Tensor x = noise;
        if (trajectory != null) {
            trajectory.add(x);
        }
        float dt = 1f / steps;
        for (int i = 0; i < steps; i++) {
            float t = 1 - i * dt;
            Tensor v = velocity(model, x, t);
            if (solver == Solver.HEUN) {
                Tensor guess = x.sub(v.mul(dt));
                v = v.add(velocity(model, guess, t - dt)).mul(0.5f);
            }
            x = x.sub(v.mul(dt));
            if (trajectory != null) {
                trajectory.add(x);
            }
        }
        return x;
    }

    private static Tensor velocity(Denoiser model, Tensor x, float t) {
        int batch = x.shape().dim(0);
        float[] time = new float[batch];
        java.util.Arrays.fill(time, Math.max(0, t));
        return model.apply(Var.of(x), Tensor.of(Shape.of(batch), time, x.device())).value();
    }
}
