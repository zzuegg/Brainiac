package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/** A model that looks at noisy data and the noise level and predicts what training asked for. */
@FunctionalInterface
public interface Denoiser {

    /**
     * @param noisy the noisy data, batch first
     * @param time per example, {@code [batch]} float32: the step for {@link Diffusion} (0 to
     *     steps - 1), the time in [0, 1] for {@link FlowMatching}; {@code [batch, tokens]} when
     *     trained with a step per token ({@link Diffusion#loss(Denoiser, Tensor, int[][],
     *     java.util.random.RandomGenerator)})
     * @return the prediction, the shape of {@code noisy}
     */
    Var apply(Var noisy, Tensor time);

    /**
     * A denoiser that also sees a condition per example, such as a class label or an
     * embedding of a prompt.
     */
    @FunctionalInterface
    interface Conditional {

        /**
         * @param noisy the noisy data, batch first
         * @param time per example, as for {@link Denoiser#apply}
         * @param condition per example, batch first
         * @return the prediction, the shape of {@code noisy}
         */
        Var apply(Var noisy, Tensor time, Tensor condition);

        /**
         * This model with a fixed condition, for {@link Diffusion#loss} on a batch with its
         * labels, or for sampling one class.
         *
         * @param condition per example, batch first
         * @return the denoiser
         */
        default Denoiser given(Tensor condition) {
            return (noisy, time) -> apply(noisy, time, condition);
        }

        /**
         * Classifier-free guidance (Ho and Salimans, 2022): {@code uncond + scale * (cond -
         * uncond)}, where the unconditional prediction uses a "no condition" value the model
         * was also trained on (by dropping conditions at random).
         *
         * @param condition per example
         * @param none the "no condition" value, shaped like {@code condition}
         * @param scale the guidance; 1 is the conditional model alone
         * @return the guided denoiser, for sampling
         */
        default Denoiser guided(Tensor condition, Tensor none, float scale) {
            return (noisy, time) -> {
                Var cond = apply(noisy, time, condition);
                Var uncond = apply(noisy, time, none);
                return uncond.add(cond.sub(uncond).mul(scale));
            };
        }
    }
}
