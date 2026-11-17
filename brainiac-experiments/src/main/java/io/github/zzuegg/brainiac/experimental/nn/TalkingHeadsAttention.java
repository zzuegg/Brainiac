package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Talking-heads attention: the heads exchange information through two learned
 * {@code [heads, heads]} mixing matrices, one applied to the attention logits before the
 * softmax and one to the attention weights after it. Plain multi-head attention keeps every
 * head isolated; mixing lets each head's pattern be a learned combination of all heads'
 * scores for a negligible {@code 2 h^2} extra parameters.
 *
 * <p>Use {@link #mix} on any {@code [batch, heads, queries, keys]} tensor, or {@link #of} for
 * a whole layer ({@link RefinedAttention} with {@link RefinedAttention.Spec#talkingHeads}).
 *
 * <p>Paper: Shazeer et al., "Talking-Heads Attention",
 * <a href="https://arxiv.org/abs/2003.02436">arXiv:2003.02436</a>.
 */
public final class TalkingHeadsAttention {

    private TalkingHeadsAttention() {
    }

    /**
     * Mixes the head axis: {@code out[b, g, i, j] = sum_h x[b, h, i, j] P[h, g]}.
     *
     * @param x {@code [batch, heads, queries, keys]}
     * @param mixing {@code [heads, heads]}
     * @return the mixed tensor, same shape
     */
    public static Var mix(Var x, Var mixing) {
        return x.permute(0, 2, 3, 1).matmul(mixing).permute(0, 3, 1, 2);
    }

    /**
     * A mixing matrix initialized to the identity (plain multi-head attention) plus small
     * noise.
     *
     * @param heads the number of heads
     * @param noise the noise scale
     * @param random the random source
     * @return the parameter
     */
    public static Var identityMixing(int heads, float noise, RandomGenerator random) {
        float[] m = Tensor.uniform(Shape.of(heads, heads), -noise, noise, random).toArray();
        for (int h = 0; h < heads; h++) {
            m[h * heads + h] += 1;
        }
        return Var.param(Tensor.of(Shape.of(heads, heads), m));
    }

    /**
     * A causal or bidirectional talking-heads softmax attention layer.
     *
     * @param heads the number of heads
     * @param causal whether attention is causal
     * @param rotaryBase the RoPE base, or 0 for none
     * @return the spec
     */
    public static LayerSpec of(int heads, boolean causal, float rotaryBase) {
        return RefinedAttention.of(heads).causal(causal).rotary(rotaryBase).talkingHeads(true);
    }
}
