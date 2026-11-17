package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A Kimi-Linear-style hybrid stack: mostly linear-attention layers (Kimi Delta Attention, a
 * {@link GatedDeltaNet} with channel-wise gates) with a full softmax attention layer after
 * every {@code ratio} of them (3:1 in the paper). The linear layers carry most of the
 * sequence mixing with a constant-size state; the periodic full-attention layers restore
 * exact retrieval over the whole context. Only the full layers keep a KV cache, so the cache
 * shrinks by about {@code ratio / (ratio + 1)} (75% at 3:1). The full layers use no positional
 * encoding (NoPE): the delta-rule layers already supply order information.
 *
 * <p>Every block is pre-norm: {@code x + mixer(norm x)}, then {@code x + mlp(norm x)}. Input and
 * output are {@code [batch, steps, features]}.
 *
 * <p>Paper: Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture",
 * <a href="https://arxiv.org/abs/2510.26692">arXiv:2510.26692</a>.
 */
public final class HybridLinearStack implements Layer {

    private final int features;
    private final List<Layer> mixers = new ArrayList<>();
    private final List<Var[]> mlps = new ArrayList<>();
    private final boolean[] full;

    /**
     * Creates the stack.
     *
     * @param features the width
     * @param heads attention heads in every layer
     * @param blocks the number of blocks
     * @param ratio linear layers per full-attention layer, for example 3
     * @param random the random source
     */
    public HybridLinearStack(int features, int heads, int blocks, int ratio, SplittableRandom random) {
        this.features = features;
        this.full = new boolean[blocks];
        for (int i = 0; i < blocks; i++) {
            full[i] = (i + 1) % (ratio + 1) == 0;
            mixers.add(full[i] ? new GatedAttention(features, heads, true, 0f, random)
                    : new GatedDeltaNet(features, heads, GatedDeltaNet.Gating.CHANNEL, 8, random));
            mlps.add(new Var[] {MixerOps.weight(random, features, 2 * features),
                MixerOps.weight(random, 2 * features, features)});
        }
    }

    /**
     * Describes the stack for {@code Model.sequential().add(...)}.
     *
     * @param heads attention heads
     * @param blocks the number of blocks
     * @param ratio linear layers per full-attention layer
     * @return the spec
     */
    public static LayerSpec of(int heads, int blocks, int ratio) {
        return MixerOps.sameShape((in, r) -> new HybridLinearStack(in.dim(-1), heads, blocks, ratio, r));
    }

    /** @return for each block, whether it is full attention */
    public boolean[] fullAttentionLayers() {
        return full.clone();
    }

    /**
     * Returns the share of KV cache kept compared with an all-attention stack of the same
     * depth: the full-attention layers' share.
     *
     * @return a number in {@code [0, 1]}
     */
    public double kvCacheFraction() {
        int n = 0;
        for (boolean f : full) {
            n += f ? 1 : 0;
        }
        return n / (double) full.length;
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "HybridLinearStack");
        Var x = input;
        for (int i = 0; i < mixers.size(); i++) {
            x = x.add(mixers.get(i).forward(MixerOps.rmsNorm(x)));
            Var[] w = mlps.get(i);
            x = x.add(MixerOps.mlp(MixerOps.rmsNorm(x), w[0], w[1]));
        }
        return x;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int i = 0; i < mixers.size(); i++) {
            p.addAll(mixers.get(i).parameters());
            p.addAll(List.of(mlps.get(i)));
        }
        return p;
    }

    @Override
    public String toString() {
        return "HybridLinearStack(" + features + ", blocks=" + full.length + ", kv=" + kvCacheFraction() + ")";
    }
}
