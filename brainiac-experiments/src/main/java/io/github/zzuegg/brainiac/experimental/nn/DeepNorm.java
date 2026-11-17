package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;

/**
 * DeepNorm's constants for an encoder-only (or decoder-only) stack of {@code N} blocks:
 * residuals are scaled up by {@code alpha = (2N)^(1/4)} inside post-norm,
 * {@code LN(alpha x + f(x))}, and the feed-forward, value and output projections start scaled
 * down by {@code beta = (8N)^(-1/4)}. Together they bound the size of each update to the
 * residual stream, which let post-norm transformers (better final quality than pre-norm)
 * train stably at 1,000 layers. Used by {@link TunableTransformerBlock} with
 * {@link TunableTransformerBlock.Residual#DEEP_NORM}.
 *
 * <p>Paper: Wang et al., "DeepNet: Scaling Transformers to 1,000 Layers" (2022),
 * <a href="https://arxiv.org/abs/2203.00555">arXiv:2203.00555</a>.
 */
public final class DeepNorm {

    private DeepNorm() {
    }

    /**
     * The residual scale.
     *
     * @param layers the number of blocks {@code N}
     * @return {@code (2N)^(1/4)}
     */
    public static float alpha(int layers) {
        return (float) Math.pow(2.0 * layers, 0.25);
    }

    /**
     * The initialization scale.
     *
     * @param layers the number of blocks {@code N}
     * @return {@code (8N)^(-1/4)}
     */
    public static float beta(int layers) {
        return (float) Math.pow(8.0 * layers, -0.25);
    }

    /**
     * Multiplies a weight in place, as DeepNorm does at initialization.
     *
     * @param weight the weight
     * @param factor the factor, usually {@link #beta(int)}
     */
    public static void scale(Var weight, float factor) {
        weight.assign(weight.value().mul(factor));
    }
}
