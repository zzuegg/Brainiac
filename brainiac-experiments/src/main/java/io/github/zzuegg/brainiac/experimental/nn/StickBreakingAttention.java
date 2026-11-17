package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Stick-breaking attention: a query walks back from itself through the visible keys, and
 * each key {@code j} breaks off a share {@code beta_ij = sigmoid(z_ij)} of the stick that the
 * nearer keys left over:
 *
 * <pre>
 * A_ij = beta_ij * prod_{j &lt; k &lt;= i} (1 - beta_ik)
 * </pre>
 *
 * <p>Nearer keys get the first pick, so recency is built in and no position encoding is
 * needed; the weights sum to at most one and the remainder is attention to nothing. It is
 * computed in log space, {@code log A_ij = -softplus(-z_ij) - sum_{k > j} softplus(z_ik)},
 * where the suffix sum is a product with a constant triangular matrix, so it is a few
 * plain autograd ops. The query attends to itself too (the diagonal comes first).
 *
 * <p>Use {@link #weights} on causal scores, or {@link #of} for a whole layer
 * ({@link RefinedAttention} with {@link RefinedAttention.Weighting#STICK_BREAKING}).
 *
 * <p>Paper: Tan et al., "Scaling Stick-Breaking Attention: An Efficient Implementation and
 * In-depth Study", <a href="https://arxiv.org/abs/2410.17980">arXiv:2410.17980</a>.
 */
public final class StickBreakingAttention {

    private StickBreakingAttention() {
    }

    /**
     * Stick-breaking weights over the last axis.
     *
     * @param scores scaled scores {@code [..., queries, keys]}
     * @param keep a {@code [queries, keys]} 0/1 mask of visible keys (causal, possibly
     *     windowed); keys are broken in order from the last visible one backwards
     * @return the weights, summing to at most 1 per row
     */
    public static Var weights(Var scores, Tensor keep) {
        int keys = scores.shape().dim(-1);
        float[] u = new float[keys * keys];
        for (int k = 0; k < keys; k++) {
            for (int j = 0; j < k; j++) {
                u[k * keys + j] = 1;                 // U[k, j] = 1 for k > j
            }
        }
        Tensor later = Tensor.of(Shape.of(keys, keys), u, scores.value().device());
        Var z = scores.mask(keep);
        Var remaining = z.softplus().mask(keep).matmul(later);   // sum_{k > j, visible} -log(1 - beta_ik)
        Var logBeta = z.neg().softplus().neg();
        return logBeta.sub(remaining).exp().mask(keep);
    }

    /**
     * A causal multi-head stick-breaking attention layer without position encoding, for
     * {@code Model.sequential().add(...)}.
     *
     * @param heads the number of heads
     * @return the spec
     */
    public static LayerSpec of(int heads) {
        return RefinedAttention.of(heads).causal(true).weighting(RefinedAttention.Weighting.STICK_BREAKING);
    }
}
