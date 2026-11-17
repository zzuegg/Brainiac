package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Causal linear attention: the softmax kernel {@code exp(q . k)} is replaced by a feature
 * map {@code phi(q) . phi(k)} with {@code phi(x) = elu(x) + 1}, so attention can be computed
 * either in parallel ({@code (phi(Q) phi(K)^T masked) V}, like a transformer, for training)
 * or as an RNN with a fixed-size state {@code S_t = S_{t-1} + phi(k_t) v_t^T},
 * {@code z_t = z_{t-1} + phi(k_t)} and {@code o_t = phi(q_t) S_t / (phi(q_t) . z_t)}, for
 * constant-memory inference. Both forms give the same result; {@link #recurrent(Var)}
 * computes the second one.
 *
 * <p>The base of the whole family of linear-time sequence mixers (RetNet, GLA, DeltaNet,
 * Mamba-2, RWKV); here without decay or gating.
 *
 * <p>Paper: Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers
 * with Linear Attention", <a href="https://arxiv.org/abs/2006.16236">arXiv:2006.16236</a>.
 */
public final class LinearAttention implements Layer {

    private final int features;
    private final int heads;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;

    /**
     * Creates the layer.
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads the number of heads
     * @param random the random source
     */
    public LinearAttention(int features, int heads, RandomGenerator random) {
        if (features % heads != 0) {
            throw new IllegalArgumentException("features " + features + " not divisible by heads " + heads);
        }
        this.features = features;
        this.heads = heads;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.wo = MixerOps.weight(random, features, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param heads the number of heads
     * @return the spec
     */
    public static LayerSpec of(int heads) {
        return MixerOps.sameShape((in, r) -> new LinearAttention(in.dim(-1), heads, r));
    }

    /** elu(x) + 1 = x + 1 for x > 0, exp(x) otherwise. */
    private static Var phi(Var x) {
        return x.relu().add(x.minimum(0f).exp());
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "LinearAttention");
        int t = input.shape().dim(1);
        Var q = phi(MixerOps.split(input.matmul(wq), heads));
        Var k = phi(MixerOps.split(input.matmul(wk), heads));
        Var v = MixerOps.split(input.matmul(wv), heads);
        Var a = q.matmul(k.transpose()).mul(MixerOps.lowerOnes(t));   // [b, h, t, t]
        Var o = a.matmul(v).div(a.sum(-1, true).add(1e-6f));
        return MixerOps.merge(o).matmul(wo);
    }

    /**
     * Computes the same output step by step with the recurrent state.
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, steps, features]}
     */
    public Var recurrent(Var input) {
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        int d = features / heads;
        Var q = phi(MixerOps.split(input.matmul(wq), heads));
        Var k = phi(MixerOps.split(input.matmul(wk), heads));
        Var v = MixerOps.split(input.matmul(wv), heads);
        Var state = Var.of(Tensor.zeros(Shape.of(b, heads, d, d)));
        Var norm = Var.of(Tensor.zeros(Shape.of(b, heads, d, 1)));
        List<Var> out = new ArrayList<>();
        for (int s = 0; s < t; s++) {
            Var ks = k.slice(2, s, s + 1);   // [b, h, 1, d]
            Var vs = v.slice(2, s, s + 1);
            Var qs = q.slice(2, s, s + 1);
            state = state.add(ks.transpose().matmul(vs));
            norm = norm.add(ks.transpose());
            out.add(qs.matmul(state).div(qs.matmul(norm).add(1e-6f)));
        }
        return MixerOps.merge(Var.concat(2, out.toArray(Var[]::new))).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, wo);
    }

    @Override
    public String toString() {
        return "LinearAttention(" + features + ", heads=" + heads + ")";
    }
}
