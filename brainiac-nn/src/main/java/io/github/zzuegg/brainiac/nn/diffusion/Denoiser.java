package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/** A model that looks at noisy data and the noise level and predicts what training asked for. */
@FunctionalInterface
public interface Denoiser {

    /**
     * @param noisy the noisy data, batch first
     * @param time per example, {@code [batch]} float32: the step for {@link Diffusion} (0 to
     *     steps - 1), the time in [0, 1] for {@link FlowMatching}
     * @return the prediction, the shape of {@code noisy}
     */
    Var apply(Var noisy, Tensor time);
}
