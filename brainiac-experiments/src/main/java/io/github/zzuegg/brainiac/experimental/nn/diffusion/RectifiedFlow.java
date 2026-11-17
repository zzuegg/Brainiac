package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.FlowMatching;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * An addition to core {@link FlowMatching}, with the same conventions (time 0 is data,
 * time 1 is noise, {@code x_t = (1 - t) x0 + t noise}, the model predicts {@code noise - x0}):
 * {@link #couplingLoss}, flow matching on <em>given</em> (noise, data) pairs. With pairs
 * produced by the model itself this is the <em>reflow</em> of Rectified Flow (Liu et al.,
 * <a href="https://arxiv.org/abs/2209.03003">2209.03003</a>): the couplings no longer cross,
 * the learned paths straighten, and one Euler step gets close to many. Sampling, with Euler
 * or Heun steps and a trajectory, is core {@link FlowMatching#sample(Denoiser, Tensor, int,
 * FlowMatching.Solver, List)}.
 */
public final class RectifiedFlow {
    private RectifiedFlow() {
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
