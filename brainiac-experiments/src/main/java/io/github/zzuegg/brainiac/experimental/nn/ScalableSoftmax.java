package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Scalable-Softmax (SSMax): multiplies the attention logits of a query that sees {@code n}
 * keys by {@code s log n}, with a learned per-head {@code s}. The largest softmax weight of
 * plain attention shrinks like {@code 1/n} as the context grows, so attention "fades" at
 * lengths beyond training; with the {@code log n} factor a logit gap keeps the same weight at
 * any length, which helps length generalization and long-context retrieval. Llama 4's
 * attention temperature tuning on its NoPE layers is the same idea.
 *
 * <p>Use {@link #scale} on any {@code [batch, heads, queries, keys]} logits, or set
 * {@link RefinedAttention.Spec#scalableSoftmax} on a layer.
 *
 * <p>Paper: Nakanishi, "Scalable-Softmax Is Superior for Attention",
 * <a href="https://arxiv.org/abs/2501.19399">arXiv:2501.19399</a>.
 */
public final class ScalableSoftmax {

    /** The initial {@code s}: {@code 1 / log 16}, so the scale is 1 at 16 keys. */
    public static final float INITIAL_GAIN = (float) (1 / Math.log(16));

    private ScalableSoftmax() {
    }

    /**
     * {@code log n_i} per query: {@code n_i} is the number of keys query {@code i} sees.
     *
     * @param queries the queries
     * @param keys all keys (including always-visible prefixes)
     * @param prefix always-visible leading keys
     * @param causal whether query {@code i} sees only {@code prefix + i + 1} keys
     * @param window a sliding window, or 0
     * @param device the device
     * @return {@code [queries, 1]}
     */
    public static Tensor logCounts(int queries, int keys, int prefix, boolean causal, int window, Device device) {
        float[] l = new float[queries];
        for (int i = 0; i < queries; i++) {
            int n = causal ? prefix + i + 1 : keys;
            if (window > 0) {
                n = Math.min(n, prefix + window);
            }
            l[i] = (float) Math.log(Math.max(n, 1));
        }
        return Tensor.of(Shape.of(queries, 1), l, device);
    }

    /**
     * Scales logits by {@code s_h log n_i}.
     *
     * @param logits {@code [batch, heads, queries, keys]}
     * @param gain the per-head {@code s}, {@code [heads]}
     * @param logCounts {@code [queries, 1]} from {@link #logCounts}
     * @return the scaled logits
     */
    public static Var scale(Var logits, Var gain, Tensor logCounts) {
        int heads = gain.shape().dim(0);
        return logits.mul(gain.reshape(heads, 1, 1).mul(logCounts));
    }
}
