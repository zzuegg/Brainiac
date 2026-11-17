package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A Hymba-style hybrid: inside every layer, attention heads and Mamba (selective state
 * space) heads read the same normalized input in parallel, and their normalized outputs are
 * averaged with learned per-channel scales,
 *
 * <pre>
 * m = (beta_a * rms(attn(n)) + beta_s * rms(mamba(n))) / 2,   n = rms(x)
 * </pre>
 *
 * <p>followed by a SwiGLU feed-forward network. Attention gives high-resolution recall, the
 * SSM a cheap summary of the whole context, and both are available in every layer instead
 * of alternating between layers as in sequential hybrids (Jamba, Kimi Linear's 3:1 stack,
 * {@link HybridLinearStack}). Learnable <b>meta tokens</b> are prepended to every sequence:
 * they give attention somewhere to put unneeded mass (instead of a sink token) and hold
 * learned task-independent knowledge; they are removed from the output.
 *
 * <p>Input and output are {@code [batch, steps, features]}, causal. The attention heads use
 * no position encoding (the SSM supplies order).
 *
 * <p>Paper: Dong et al., "Hymba: A Hybrid-head Architecture for Small Language Models",
 * <a href="https://arxiv.org/abs/2411.13676">arXiv:2411.13676</a>.
 */
public final class ParallelHybridBlock implements Layer {

    private final int features;
    private final int metaTokens;
    private final Var meta;
    private final List<Layer> attention = new ArrayList<>();
    private final List<Layer> ssm = new ArrayList<>();
    private final List<GluFeedForward> ffns = new ArrayList<>();
    private final List<Var[]> scales = new ArrayList<>();

    /**
     * Creates the stack.
     *
     * @param features the width
     * @param heads the attention heads per layer
     * @param blocks the number of layers
     * @param state the SSM state size per channel
     * @param metaTokens the number of learned meta tokens, 0 for none
     * @param random the random source
     */
    public ParallelHybridBlock(int features, int heads, int blocks, int state, int metaTokens,
            RandomGenerator random) {
        this.features = features;
        this.metaTokens = metaTokens;
        this.meta = metaTokens > 0 ? TechniqueOps.normal(random, 0.5f, 1, metaTokens, features) : null;
        for (int i = 0; i < blocks; i++) {
            attention.add(new RefinedAttention(RefinedAttention.of(heads).causal(true).settings(features), random));
            ssm.add(new MambaLayer(features, features, state, random));
            ffns.add(new GluFeedForward(GluFeedForward.Kind.SWIGLU, features, 2 * features, random));
            scales.add(new Var[] {RefinementOps.ones(features), RefinementOps.ones(features),
                RefinementOps.ones(features), RefinementOps.ones(features)});
        }
    }

    /**
     * Describes the stack for {@code Model.sequential().add(...)}.
     *
     * @param heads the attention heads per layer
     * @param blocks the number of layers
     * @param metaTokens the number of meta tokens
     * @return the spec
     */
    public static LayerSpec of(int heads, int blocks, int metaTokens) {
        return MixerOps.sameShape((in, r) -> new ParallelHybridBlock(in.dim(-1), heads, blocks, 8, metaTokens, r));
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "ParallelHybridBlock");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var x = meta == null ? input : Var.concat(1, meta.expand(Shape.of(b, metaTokens, features)), input);
        for (int i = 0; i < attention.size(); i++) {
            Var[] g = scales.get(i);
            Var n = RefinementOps.rms(x, g[0]);
            Var a = MixerOps.rmsNorm(attention.get(i).forward(n)).mul(g[1]);
            Var s = MixerOps.rmsNorm(ssm.get(i).forward(n)).mul(g[2]);
            x = x.add(a.add(s).mul(0.5f));
            x = x.add(ffns.get(i).forward(RefinementOps.rms(x, g[3])));
        }
        return meta == null ? x : x.slice(1, metaTokens, metaTokens + t);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        if (meta != null) {
            p.add(meta);
        }
        for (int i = 0; i < attention.size(); i++) {
            p.addAll(List.of(scales.get(i)));
            p.addAll(attention.get(i).parameters());
            p.addAll(ssm.get(i).parameters());
            p.addAll(ffns.get(i).parameters());
        }
        return p;
    }

    @Override
    public String toString() {
        return "ParallelHybridBlock(" + features + ", blocks=" + attention.size() + ", meta=" + metaTokens + ")";
    }
}
