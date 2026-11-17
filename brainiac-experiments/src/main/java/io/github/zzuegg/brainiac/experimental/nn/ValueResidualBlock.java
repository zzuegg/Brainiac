package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A causal transformer stack with value residual learning (ResFormer): every layer after the
 * first mixes its attention values with the values of the first layer,
 *
 * <pre>
 * v_l = lambda_l v_l + (1 - lambda_l) v_1
 * </pre>
 *
 * <p>with a learned {@code lambda_l} per layer, starting at 0.5. Deep stacks tend to
 * over-smooth: the token representations, and so the values later layers can read, grow
 * alike. The shortcut from the first layer's values keeps the token-level information
 * available at every depth, at the cost of one extra value tensor; the paper reports the
 * loss of a model with 10–15% more parameters. With {@code valueResidual = false} the stack
 * is a plain pre-norm transformer of the same shape, the baseline.
 *
 * <p>Every layer is {@code x + attn(rms x)}, then {@code x + swiglu(rms x)}, with RoPE.
 * Input and output are {@code [batch, steps, features]}.
 *
 * <p>Paper: Zhou et al., "Value Residual Learning For Alleviating Attention Concentration In
 * Transformers", <a href="https://arxiv.org/abs/2410.17897">arXiv:2410.17897</a>.
 */
public final class ValueResidualBlock implements Layer {

    private final int features;
    private final int heads;
    private final boolean valueResidual;
    private final float rotaryBase;
    private final List<Var[]> attention = new ArrayList<>();
    private final List<GluFeedForward> ffns = new ArrayList<>();
    private final List<Var> lambdas = new ArrayList<>();

    /**
     * Creates the stack.
     *
     * @param features the width
     * @param heads the attention heads
     * @param depth the number of layers
     * @param valueResidual whether to mix in the first layer's values
     * @param rotaryBase the RoPE base, or 0 for none
     * @param random the random source
     */
    public ValueResidualBlock(int features, int heads, int depth, boolean valueResidual, float rotaryBase,
            RandomGenerator random) {
        this.features = features;
        this.heads = heads;
        this.valueResidual = valueResidual;
        this.rotaryBase = rotaryBase;
        for (int l = 0; l < depth; l++) {
            attention.add(new Var[] {MixerOps.weight(random, features, features),
                MixerOps.weight(random, features, features), MixerOps.weight(random, features, features),
                MixerOps.weight(random, features, features), RefinementOps.ones(features),
                RefinementOps.ones(features)});
            ffns.add(new GluFeedForward(GluFeedForward.Kind.SWIGLU, features, 2 * features, random));
            lambdas.add(MixerOps.filled(0.5f, 1));
        }
    }

    /**
     * Describes the stack for {@code Model.sequential().add(...)}.
     *
     * @param heads the attention heads
     * @param depth the number of layers
     * @param valueResidual whether to use value residual learning
     * @return the spec
     */
    public static LayerSpec of(int heads, int depth, boolean valueResidual) {
        return MixerOps.sameShape((in, r) -> new ValueResidualBlock(in.dim(-1), heads, depth, valueResidual,
                10000f, r));
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "ValueResidualBlock");
        Var x = input;
        Var first = null;
        for (int l = 0; l < attention.size(); l++) {
            Var[] w = attention.get(l);
            Var n = RefinementOps.rms(x, w[4]);
            Var q = MixerOps.split(n.matmul(w[0]), heads);
            Var k = MixerOps.split(n.matmul(w[1]), heads);
            Var v = MixerOps.split(n.matmul(w[2]), heads);
            if (l == 0) {
                first = v;
            } else if (valueResidual) {
                Var lambda = lambdas.get(l);
                v = v.mul(lambda).add(first.mul(lambda.neg().add(1f)));
            }
            x = x.add(MixerOps.merge(RefinementOps.causalAttention(q, k, v, rotaryBase, 0)).matmul(w[3]));
            x = x.add(ffns.get(l).forward(RefinementOps.rms(x, w[5])));
        }
        return x;
    }

    /** @return the learned mixing weight {@code lambda_l} of every layer (layer 0 unused) */
    public float[] lambdas() {
        float[] out = new float[lambdas.size()];
        for (int l = 0; l < out.length; l++) {
            out[l] = lambdas.get(l).value().item();
        }
        return out;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int l = 0; l < attention.size(); l++) {
            p.addAll(List.of(attention.get(l)));
            p.addAll(ffns.get(l).parameters());
            if (valueResidual && l > 0) {
                p.add(lambdas.get(l));
            }
        }
        return p;
    }

    @Override
    public String toString() {
        return "ValueResidualBlock(" + features + ", depth=" + attention.size() + (valueResidual ? ", resformer" : "")
                + ")";
    }
}
