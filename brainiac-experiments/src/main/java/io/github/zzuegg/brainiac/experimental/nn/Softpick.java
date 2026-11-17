package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Softpick, a rectified softmax whose weights need not sum to one:
 *
 * <pre>
 * softpick(x)_i = relu(exp(x_i) - 1) / (sum_j |exp(x_j) - 1| + eps)
 * </pre>
 *
 * <p>Scores at or below zero get exactly zero weight, and a row whose scores are all negative
 * attends to nothing at all. A head that has nothing to say therefore no longer has to park
 * its attention on a "sink" token (usually the first one), which is what makes softmax
 * transformers grow attention sinks and the massive activations that come with them; both
 * hurt low-bit quantization. The denominator still counts the negative scores, so they keep
 * receiving gradient. It is computed stably by scaling numerator and denominator with
 * {@code exp(-m)}, {@code m = max(0, max_j x_j)}.
 *
 * <p>Use {@link #apply} as a drop-in for the softmax of any attention, or {@link #of} for a
 * whole multi-head attention layer ({@link RefinedAttention} with
 * {@link RefinedAttention.Weighting#SOFTPICK}).
 *
 * <p>Paper: Zuhri et al., "Softpick: No Attention Sink, No Massive Activations with Rectified
 * Softmax", <a href="https://arxiv.org/abs/2504.20966">arXiv:2504.20966</a>.
 */
public final class Softpick {

    private static final float EPS = 1e-6f;

    private Softpick() {
    }

    /**
     * Softpick over the last axis, with masked-out entries getting weight 0 and no say in
     * the denominator.
     *
     * @param scores the scores {@code [..., n]}
     * @param keep a 0/1 mask broadcast to the scores, or {@code null} to keep all
     * @return the weights, each in {@code [0, 1)}, summing to at most 1 per row
     */
    public static Var apply(Var scores, Tensor keep) {
        Var x = keep == null ? scores : scores.mask(keep);
        Var m = x.max(-1, true).maximum(0f).detach();
        Var shift = m.neg().exp();                                   // exp(-m), the scaled "1"
        Var d = x.sub(m).exp().sub(shift);
        if (keep != null) {
            d = d.mask(keep);
        }
        return d.relu().div(d.abs().sum(-1, true).add(shift.mul(EPS)));
    }

    /**
     * A causal or bidirectional multi-head softpick attention layer for
     * {@code Model.sequential().add(...)}.
     *
     * @param heads the number of heads
     * @param causal whether attention is causal
     * @param rotaryBase the RoPE base, or 0 for no positions
     * @return the spec
     */
    public static LayerSpec of(int heads, boolean causal, float rotaryBase) {
        return RefinedAttention.of(heads).causal(causal).rotary(rotaryBase)
                .weighting(RefinedAttention.Weighting.SOFTPICK);
    }
}
