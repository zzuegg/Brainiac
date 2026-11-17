package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;

/**
 * Two small tricks that keep the output logits of large language models in check during
 * training:
 * <ul>
 * <li>{@link #zLoss}: the auxiliary z-loss of PaLM (Chowdhery et al. 2022,
 *     <a href="https://arxiv.org/abs/2204.02311">arXiv:2204.02311</a>; also ST-MoE's router
 *     z-loss), {@code weight * mean(log(Z)^2)} with {@code Z} the softmax normalizer, which
 *     pulls {@code log Z} towards 0 and prevents the logit drift behind late loss spikes;</li>
 * <li>{@link #softCap}: logit soft-capping of Gemma 2 (Gemma Team 2024,
 *     <a href="https://arxiv.org/abs/2408.00118">arXiv:2408.00118</a>), {@code c tanh(x / c)},
 *     which bounds attention or output logits smoothly.</li>
 * </ul>
 */
public final class LogitStabilizers {

    private LogitStabilizers() {
    }

    /**
     * @param logits {@code [..., vocab]}
     * @param weight the coefficient, for example 1e-4
     * @return the z-loss, a scalar
     */
    public static Var zLoss(Var logits, float weight) {
        int axis = logits.shape().rank() - 1;
        Var max = Var.of(logits.value().max(axis, true));
        Var lse = logits.sub(max).exp().sum(axis, true).log().add(max);
        return lse.square().mean().mul(weight);
    }

    /**
     * @param logits any values
     * @param cap the bound {@code c}, for example 30
     * @return {@code cap * tanh(logits / cap)}
     */
    public static Var softCap(Var logits, float cap) {
        return logits.div(cap).tanh().mul(cap);
    }
}
