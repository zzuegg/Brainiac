package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.List;

/**
 * A network that looks at a noisy sample, the noise level and an optional class label and
 * predicts something about the clean sample: the added noise (DDPM), a velocity (flow
 * matching) or the clean sample itself (consistency models). The training objectives and
 * samplers built on it (DDPM, flow matching, consistency models) only see this interface,
 * so an {@link MlpDenoiser} and a {@link DiffusionTransformer} are interchangeable.
 *
 * <p>Labels are class ids, or {@code -1} for "no label". Training with some labels dropped
 * to {@code -1} makes the same network both conditional and unconditional, which is what
 * classifier-free guidance needs (Ho and Salimans, <a href="https://arxiv.org/abs/2207.12598">2207.12598</a>).
 */
public interface Denoiser {

    /**
     * Runs the network.
     *
     * @param x noisy samples {@code [batch, dimension]}
     * @param t the noise level or time of each sample, in {@code [0, 1]}
     * @param labels class id per sample, {@code -1} for unconditional
     * @return the prediction, {@code [batch, dimension]}
     */
    Var forward(Var x, float[] t, int[] labels);

    /** @return the trainable parameters */
    List<Var> parameters();

    /** @return the size of one sample */
    int dimension();

    /** @return the number of classes it is conditioned on, 0 for unconditional only */
    int classes();
}
