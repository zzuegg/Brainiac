package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;

/**
 * Autoguidance (Karras et al., "Guiding a Diffusion Model with a Bad Version of Itself",
 * <a href="https://arxiv.org/abs/2406.02507">2406.02507</a>, NeurIPS 2024 runner-up):
 * {@code bad + w (good - bad)}, where {@code bad} is a smaller or less-trained model of the
 * <em>same</em> task and conditioning, instead of the unconditional model of
 * classifier-free guidance.
 *
 * <p>Why: CFG's unconditional model differs from the conditional one in <em>what</em> it
 * models, so its guidance direction also pushes samples towards the class centre, which
 * improves precision at the price of diversity (recall). A weaker version of the same
 * model makes the same kind of errors, only bigger, so extrapolating away from it mainly
 * corrects errors (samples pulled off the data manifold) and keeps the spread of the data.
 * Works for unconditional models too, where CFG cannot be used. Any core {@link Denoiser},
 * any prediction type.
 */
public final class Autoguidance {
    private Autoguidance() {
    }

    /**
     * Guides a model with a weaker version of itself.
     *
     * @param good the main model
     * @param bad the weaker model with the same inputs and prediction type
     * @param weight the guidance weight w; 1 is the main model alone
     * @return the guided denoiser, for sampling
     */
    public static Denoiser guide(Denoiser good, Denoiser bad, float weight) {
        if (weight == 1f) {
            return good;
        }
        return (noisy, time) -> {
            var g = good.apply(noisy, time);
            var b = bad.apply(noisy, time);
            return b.add(g.sub(b).mul(weight));
        };
    }
}
