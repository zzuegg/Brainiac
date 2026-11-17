package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Sigmoid attention: every query-key score goes through an element-wise sigmoid instead of a
 * row softmax, {@code w_ij = sigmoid(q_i k_j / sqrt(d) + b)}, with the bias
 * {@code b = -log n} for {@code n} keys. Without row normalization there is no competition
 * between keys (a query can attend to several keys fully, or to none), no row-wise
 * reduction in the kernel, and no sink token; the {@code -log n} bias keeps the total
 * attention mass near one at initialization, which the paper found essential for stable
 * training.
 *
 * <p>Use {@link #weights} as a drop-in for a softmax, or {@link #of} for a whole layer
 * ({@link RefinedAttention} with {@link RefinedAttention.Weighting#SIGMOID}).
 *
 * <p>Paper: Ramapuram et al., "Theory, Analysis, and Best Practices for Sigmoid
 * Self-Attention", <a href="https://arxiv.org/abs/2409.04431">arXiv:2409.04431</a>.
 */
public final class SigmoidAttention {

    private SigmoidAttention() {
    }

    /**
     * Sigmoid attention weights.
     *
     * @param scores scaled scores {@code [..., queries, keys]}
     * @param keep a 0/1 mask broadcast to the scores, or {@code null}
     * @param keys the sequence length {@code n} of the {@code -log n} bias
     * @return the weights, each in {@code (0, 1)} where kept and 0 elsewhere
     */
    public static Var weights(Var scores, Tensor keep, int keys) {
        Var w = scores.sub((float) Math.log(Math.max(1, keys))).sigmoid();
        return keep == null ? w : w.mask(keep);
    }

    /**
     * A multi-head sigmoid attention layer for {@code Model.sequential().add(...)}.
     *
     * @param heads the number of heads
     * @param causal whether attention is causal
     * @param rotaryBase the RoPE base, or 0 for no positions
     * @return the spec
     */
    public static LayerSpec of(int heads, boolean causal, float rotaryBase) {
        return RefinedAttention.of(heads).causal(causal).rotary(rotaryBase)
                .weighting(RefinedAttention.Weighting.SIGMOID);
    }
}
