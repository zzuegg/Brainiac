package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.FlowMatching;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Additions to core {@link FlowMatching}, with the same conventions (time 0 is data,
 * time 1 is noise, {@code x_t = (1 - t) x0 + t noise}, the model predicts {@code noise - x0}):
 *
 * <ul>
 * <li>{@link #sampleHeun}: Heun's second-order method (explicit trapezoid), two model calls
 *     per step but much smaller error than Euler at few steps.</li>
 * <li>{@link #sampleEuler} with a trajectory, for animations.</li>
 * <li>{@link #couplingLoss}: flow matching on <em>given</em> (noise, data) pairs. With pairs
 *     produced by the model itself this is the <em>reflow</em> of Rectified Flow (Liu et al.,
 *     <a href="https://arxiv.org/abs/2209.03003">2209.03003</a>): the couplings no longer
 *     cross, the learned paths straighten, and one Euler step gets close to many.</li>
 * </ul>
 */
public final class RectifiedFlow {
    private RectifiedFlow() {
    }

    private static Tensor time(int batch, float t, Tensor like) {
        float[] f = new float[batch];
        java.util.Arrays.fill(f, t);
        return Tensor.of(Shape.of(batch), f, like.device());
    }

    /**
     * Follows the velocity from noise (time 1) to data (time 0) with Euler steps.
     *
     * @param model the velocity model
     * @param noise the start
     * @param steps the number of steps
     * @param trajectory receives the state after every step, or {@code null}
     * @return the samples
     */
    public static Tensor sampleEuler(Denoiser model, Tensor noise, int steps, List<Tensor> trajectory) {
        int batch = noise.shape().dim(0);
        Tensor x = noise;
        float dt = 1f / steps;
        for (int i = 0; i < steps; i++) {
            Tensor v = model.apply(Var.of(x), time(batch, 1 - i * dt, x)).value();
            x = x.sub(v.mul(dt));
            if (trajectory != null) {
                trajectory.add(x);
            }
        }
        return x;
    }

    /**
     * Follows the velocity from noise to data with Heun's method.
     *
     * @param model the velocity model
     * @param noise the start
     * @param steps the number of steps (two model calls each)
     * @param trajectory receives the state after every step, or {@code null}
     * @return the samples
     */
    public static Tensor sampleHeun(Denoiser model, Tensor noise, int steps, List<Tensor> trajectory) {
        int batch = noise.shape().dim(0);
        Tensor x = noise;
        float dt = 1f / steps;
        for (int i = 0; i < steps; i++) {
            float t = 1 - i * dt;
            Tensor v1 = model.apply(Var.of(x), time(batch, t, x)).value();
            Tensor predicted = x.sub(v1.mul(dt));
            Tensor v2 = model.apply(Var.of(predicted), time(batch, t - dt, x)).value();
            x = x.sub(v1.add(v2).mul(dt / 2));
            if (trajectory != null) {
                trajectory.add(x);
            }
        }
        return x;
    }

    /**
     * Flow matching on given couplings: a uniform time per pair and the squared error of the
     * predicted velocity {@code noise - data} at {@code (1 - t) data + t noise}.
     *
     * @param model the velocity model
     * @param noise the noise end of each pair
     * @param data the data end of each pair
     * @param random the source of times
     * @return the loss, a scalar
     */
    public static Var couplingLoss(Denoiser model, Tensor noise, Tensor data, RandomGenerator random) {
        int batch = data.shape().dim(0);
        float[] t = new float[batch];
        float[] keep = new float[batch];
        for (int i = 0; i < batch; i++) {
            t[i] = (float) random.nextDouble();
            keep[i] = 1 - t[i];
        }
        int[] dims = new int[data.shape().rank()];
        java.util.Arrays.fill(dims, 1);
        dims[0] = batch;
        Tensor xt = data.mul(Tensor.of(Shape.of(dims), keep, data.device()))
                .add(noise.mul(Tensor.of(Shape.of(dims), t, data.device())));
        Var v = model.apply(Var.of(xt), Tensor.of(Shape.of(batch), t, data.device()));
        return v.sub(Var.of(noise.sub(data))).square().mean();
    }
}
