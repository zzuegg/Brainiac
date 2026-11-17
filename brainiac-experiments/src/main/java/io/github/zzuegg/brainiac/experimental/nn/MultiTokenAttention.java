package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Multi-token attention (MTA): one attention weight in standard attention depends on a
 * single query and a single key, so finding "the block that contains both A and B" needs
 * one token to carry both. MTA convolves the attention logits so neighbouring queries,
 * keys and heads share evidence:
 *
 * <ul>
 *   <li><b>key-query convolution</b>: before the softmax, each head's logit matrix is
 *       convolved with a learned {@code queryKernel x keyKernel} kernel, over the current and
 *       earlier queries and keys centred on the key. Logits of hidden (future) keys are zeroed
 *       before the convolution and masked after it, so nothing leaks;</li>
 *   <li><b>head mixing</b>: after the softmax, the weights of the heads in a group are mixed by
 *       a learned matrix (a convolution over the head axis).</li>
 * </ul>
 * Both start at the identity, so the layer starts as plain multi-head attention.
 *
 * <p>Paper: Golovneva et al., "Multi-Token Attention",
 * <a href="https://arxiv.org/abs/2504.00927">arXiv:2504.00927</a>.
 */
public final class MultiTokenAttention implements Layer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final int queryKernel;
    private final int keyKernel;
    private final float rotaryBase;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var kernel;
    private final Var headMix;

    /**
     * Creates the layer (causal).
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads the heads
     * @param queryKernel the kernel length over queries (current and earlier ones)
     * @param keyKernel the kernel length over keys, centred (odd)
     * @param headMixing whether to mix heads after the softmax
     * @param rotaryBase the RoPE base, or 0 for none
     * @param random the random source
     */
    public MultiTokenAttention(int features, int heads, int queryKernel, int keyKernel, boolean headMixing,
            float rotaryBase, RandomGenerator random) {
        if (features % heads != 0 || queryKernel < 1 || keyKernel < 1 || keyKernel % 2 == 0) {
            throw new IllegalArgumentException("need heads dividing features, kernels >= 1 and an odd key kernel");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.queryKernel = queryKernel;
        this.keyKernel = keyKernel;
        this.rotaryBase = rotaryBase;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.wo = MixerOps.weight(random, features, features);
        float[] k = new float[heads * queryKernel * keyKernel];
        for (int h = 0; h < heads; h++) {
            for (int i = 0; i < queryKernel * keyKernel; i++) {
                k[h * queryKernel * keyKernel + i] = (float) (random.nextDouble() * 0.02 - 0.01);
            }
            k[h * queryKernel * keyKernel + keyKernel / 2] += 1f;   // identity tap: query offset 0, key offset 0
        }
        this.kernel = Var.param(Tensor.of(Shape.of(heads, queryKernel * keyKernel), k));
        this.headMix = headMixing ? TalkingHeadsAttention.identityMixing(heads, 0.01f, random) : null;
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}: kernels 2 x 3, head
     * mixing, RoPE.
     *
     * @param heads the heads
     * @return the spec
     */
    public static LayerSpec of(int heads) {
        return of(heads, 2, 3, true);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}, with RoPE.
     *
     * @param heads the heads
     * @param queryKernel the kernel length over queries
     * @param keyKernel the odd kernel length over keys
     * @param headMixing whether to mix heads after the softmax
     * @return the spec
     */
    public static LayerSpec of(int heads, int queryKernel, int keyKernel, boolean headMixing) {
        return MixerOps.sameShape((in, r) -> new MultiTokenAttention(in.dim(-1), heads, queryKernel, keyKernel,
                headMixing, 10000f, r));
    }

    /** Shifts {@code x [b, h, t, t]} by {@code by} along an axis (2 or 3), filling with zeros. */
    private static Var shift(Var x, int axis, int by) {
        if (by == 0) {
            return x;
        }
        int[] dims = x.shape().toArray();
        int n = dims[axis];
        int m = Math.min(Math.abs(by), n);
        dims[axis] = m;
        Var zeros = Var.of(Tensor.zeros(Shape.of(dims)));
        if (m == n) {
            return zeros;
        }
        return by > 0 ? Var.concat(axis, zeros, x.slice(axis, 0, n - m))
                : Var.concat(axis, x.slice(axis, m, n), zeros);
    }

    /**
     * The key-query convolution of masked logits: {@code out[i, j] = sum_{a, c} K[a, c] A[i - a, j - c + keyKernel / 2]}.
     *
     * @param logits {@code [b, h, t, t]}, zero at hidden keys
     * @return {@code [b, h, t, t]}
     */
    Var convolve(Var logits) {
        Var out = null;
        for (int a = 0; a < queryKernel; a++) {
            Var rows = shift(logits, 2, a);
            for (int c = 0; c < keyKernel; c++) {
                int tap = a * keyKernel + c;
                Var w = kernel.slice(1, tap, tap + 1).reshape(1, heads, 1, 1);
                Var part = shift(rows, 3, c - keyKernel / 2).mul(w);
                out = out == null ? part : out.add(part);
            }
        }
        return out;
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "MultiTokenAttention");
        int t = input.shape().dim(1);
        Var q = MixerOps.split(input.matmul(wq), heads);
        Var k = MixerOps.split(input.matmul(wk), heads);
        Var v = MixerOps.split(input.matmul(wv), heads);
        if (rotaryBase > 0) {
            q = Functional.rotary(q, rotaryBase, 0);
            k = Functional.rotary(k, rotaryBase, 0);
        }
        Tensor keep = RefinementOps.keep(t, t, 0, 0, true, 0, input.value().device());
        Var logits = q.matmul(k.transpose()).mul((float) (1 / Math.sqrt(headSize))).mul(keep);
        Var w = MixerOps.softmax(convolve(logits).add(RefinementOps.additive(keep)));
        if (headMix != null) {
            w = TalkingHeadsAttention.mix(w, headMix);
        }
        return MixerOps.merge(w.matmul(v)).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(wq, wk, wv, wo, kernel));
        if (headMix != null) {
            p.add(headMix);
        }
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "MultiTokenAttention(" + features + ", heads=" + heads + ", kernel=" + queryKernel + "x" + keyKernel
                + (headMix != null ? ", head mixing" : "") + ")";
    }
}
