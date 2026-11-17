package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Selective attention: a parameter-free rule that lets a token mask later attention to
 * earlier tokens it has made irrelevant (a new value of a variable makes the old one
 * useless). Token {@code k} "selects" earlier token {@code j} by
 * {@code S_kj = relu(logit_kj)} of one designated head (head 0); the selections accumulate
 * over the queries, and every head of every later query {@code i} subtracts them from its
 * logits:
 *
 * <pre>
 * F_ij = sum_{k &lt; i} S_kj,   logits_ij -= F_ij
 * </pre>
 *
 * <p>No token masks itself, and the first token is never masked. Since {@code F} only grows
 * along the queries, a key whose penalty has passed a threshold can be dropped from the KV
 * cache for good; {@link #keptContext} measures how small the context becomes.
 *
 * <p>Use {@link #penalty} with any attention, or {@link #of} for a whole layer
 * ({@link RefinedAttention} with {@link RefinedAttention.Spec#selective}).
 *
 * <p>Paper: Leviathan, Kalman and Matias, "Selective Attention Improves Transformer",
 * <a href="https://arxiv.org/abs/2410.02703">arXiv:2410.02703</a>.
 */
public final class SelectiveAttention {

    private SelectiveAttention() {
    }

    /**
     * The accumulated selection penalty.
     *
     * @param logits head 0's attention logits {@code [batch, 1, t, t]} (or {@code [batch, t, t]})
     * @return {@code F}, same shape, to subtract from every head's logits
     */
    public static Var penalty(Var logits) {
        Shape s = logits.shape();
        int t = s.dim(-1);
        float[] m = new float[t * t];
        for (int k = 0; k < t; k++) {
            for (int j = 1; j < k; j++) {
                m[k * t + j] = 1;                    // only earlier tokens, never the first, never itself
            }
        }
        Var sel = logits.relu().mask(Tensor.of(Shape.of(t, t), m, logits.value().device()));
        Var cumulative = sel.cumsum(-2);             // sum_{k <= i}
        int[] zeros = s.toArray();
        zeros[zeros.length - 2] = 1;
        return Var.concat(-2, Var.of(Tensor.zeros(Shape.of(zeros))), cumulative.slice(-2, 0, t - 1));
    }

    /**
     * How much context survives pruning: for each query, the share of its visible keys whose
     * penalty is still below {@code threshold}, averaged over queries and batch.
     *
     * @param penalty a penalty from {@link #penalty} (as a tensor)
     * @param threshold the pruning threshold
     * @return the mean kept share of the causal context, in {@code (0, 1]}
     */
    public static double keptContext(Tensor penalty, float threshold) {
        int t = penalty.shape().dim(-1);
        float[] f = penalty.toArray();
        int rows = f.length / (t * t);
        double sum = 0;
        for (int r = 0; r < rows; r++) {
            for (int i = 0; i < t; i++) {
                int kept = 0;
                for (int j = 0; j <= i; j++) {
                    kept += f[(r * t + i) * t + j] < threshold ? 1 : 0;
                }
                sum += kept / (double) (i + 1);
            }
        }
        return sum / (rows * (double) t);
    }

    /**
     * A causal multi-head softmax attention layer with selective attention.
     *
     * @param heads the number of heads
     * @param rotaryBase the RoPE base, or 0 for none
     * @return the spec
     */
    public static LayerSpec of(int heads, float rotaryBase) {
        return RefinedAttention.of(heads).causal(true).rotary(rotaryBase).selective(true);
    }
}
