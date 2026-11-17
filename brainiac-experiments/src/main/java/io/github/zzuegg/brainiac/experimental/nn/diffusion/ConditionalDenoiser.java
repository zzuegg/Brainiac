package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * A class-conditional {@link Denoiser}: it also sees a class label per example, with
 * {@code -1} meaning "no label". Training with a share of labels dropped to {@code -1}
 * makes one network both conditional and unconditional, which is what classifier-free
 * guidance needs (Ho and Salimans, <a href="https://arxiv.org/abs/2207.12598">2207.12598</a>).
 *
 * <p>It is a core {@link Denoiser.Conditional} whose condition is the label of each example
 * as a number. {@link #bind} turns it into a plain core {@link Denoiser} for fixed labels
 * ({@link Denoiser.Conditional#given}), so it trains with core {@code Diffusion.loss} /
 * {@code FlowMatching.loss} and samples with their samplers; {@link #guided} does the same
 * with core guidance ({@link Denoiser.Conditional#guided}).
 */
public interface ConditionalDenoiser extends Denoiser.Conditional {

    /**
     * Runs the network.
     *
     * @param noisy the noisy data, {@code [batch, dimension]}
     * @param time per example, as the core {@link Denoiser} receives it
     * @param labels class per example, {@code -1} for none
     * @return the prediction, the shape of {@code noisy}
     */
    Var apply(Var noisy, Tensor time, int[] labels);

    /**
     * Runs the network with the labels given as a tensor, as core conditional denoisers
     * are called.
     *
     * @param noisy the noisy data
     * @param time per example
     * @param condition the class of each example as a number {@code [batch]}, {@code -1} for none
     * @return the prediction
     */
    @Override
    default Var apply(Var noisy, Tensor time, Tensor condition) {
        float[] c = condition.toArray();
        int[] labels = new int[c.length];
        for (int i = 0; i < c.length; i++) {
            labels[i] = Math.round(c[i]);
        }
        return apply(noisy, time, labels);
    }

    /**
     * @param labels class per example, {@code -1} for none
     * @return the labels as a core condition tensor {@code [batch]}
     */
    static Tensor condition(int[] labels) {
        float[] f = new float[labels.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = labels[i];
        }
        return Tensor.of(Shape.of(f.length), f);
    }

    /** @return the trainable parameters */
    List<Var> parameters();

    /** @return the size of one example */
    int dimension();

    /** @return the number of classes, 0 for unconditional only */
    int classes();

    /**
     * Fixes the labels.
     *
     * @param labels class per example, {@code -1} for none
     * @return a core denoiser
     */
    default Denoiser bind(int[] labels) {
        return given(condition(labels));
    }

    /**
     * Classifier-free guidance: {@code uncond + w (cond - uncond)}, for sampling. {@code w = 1}
     * is plain conditional sampling, larger values trade diversity for fidelity to the label.
     *
     * @param labels class per example
     * @param weight the guidance weight w
     * @return a core denoiser
     */
    default Denoiser guided(int[] labels, float weight) {
        if (weight == 1f) {
            return bind(labels);
        }
        int[] none = new int[labels.length];
        java.util.Arrays.fill(none, -1);
        return guided(condition(labels), condition(none), weight);
    }

    /**
     * Copies labels, replacing each by {@code -1} with a given probability (label dropout
     * for guidance training).
     *
     * @param labels the labels
     * @param rate the share to drop
     * @param random the random source
     * @return the new labels
     */
    static int[] dropLabels(int[] labels, double rate, java.util.random.RandomGenerator random) {
        int[] out = labels.clone();
        for (int i = 0; i < out.length; i++) {
            if (random.nextDouble() < rate) {
                out[i] = -1;
            }
        }
        return out;
    }
}
