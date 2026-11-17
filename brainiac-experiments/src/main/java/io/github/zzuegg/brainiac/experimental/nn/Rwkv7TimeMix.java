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
 * The RWKV-7 "Goose" time mix: an RNN with a matrix state per head updated by a
 * generalized delta rule with vector-valued decay and in-context learning rate:
 *
 * <pre>
 * S_t = (diag(w_t) - (a_t . kh_t) kh_t^T) S_{t-1} + kt_t v_t^T
 * o_t = r_t^T S_t + (r_t . (k_t . u)) v_t
 * </pre>
 *
 * <ul>
 *   <li>{@code w_t = exp(-e^-0.5 sigmoid(...))}: per-channel decay in {@code (0.545, 1)},
 *       from a low-rank projection of the token;</li>
 *   <li>{@code a_t = sigmoid(...)}: the in-context learning rate, per channel;</li>
 *   <li>{@code kh_t}: the normalized "removal" key; {@code kt_t = k . (1 + (a - 1) . alpha)}
 *       the "replacement" key;</li>
 *   <li>the last term is the "bonus" that lets the current token be read directly.</li>
 * </ul>
 *
 * <p>Every projection reads a token-shifted input {@code x + mu (x_{t-1} - x)}. The output is
 * group-normalized per head, gated by a low-rank sigmoid gate and projected. Unlike plain
 * decaying linear attention, the delta-rule term removes the old value stored under a key
 * before writing a new one, and with a non-diagonal transition RWKV-7 can track state beyond
 * what diagonal SSMs can express.
 *
 * <p>Input and output are {@code [batch, steps, features]}; computed recurrently.
 *
 * <p>Paper: Peng et al., "RWKV-7 'Goose' with Expressive Dynamic State Evolution",
 * <a href="https://arxiv.org/abs/2503.14456">arXiv:2503.14456</a>.
 */
public final class Rwkv7TimeMix implements Layer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final Var mu;
    private final Var wr;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var w0;
    private final Var w1;
    private final Var w2;
    private final Var a0;
    private final Var a1;
    private final Var a2;
    private final Var g1;
    private final Var g2;
    private final Var removal;
    private final Var replacement;
    private final Var bonus;

    /**
     * Creates the time mix.
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads the number of heads
     * @param rank the rank of the low-rank decay, learning-rate and gate projections
     * @param random the random source
     */
    public Rwkv7TimeMix(int features, int heads, int rank, RandomGenerator random) {
        if (features % heads != 0) {
            throw new IllegalArgumentException("features " + features + " not divisible by heads " + heads);
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.mu = Var.param(Tensor.uniform(Shape.of(6, features), 0.2f, 0.8f, random));
        this.wr = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.wo = MixerOps.weight(random, features, features);
        this.w0 = MixerOps.filled(-1f, features);
        this.w1 = MixerOps.uniform(random, 0.1f, features, rank);
        this.w2 = MixerOps.uniform(random, 0.1f, rank, features);
        this.a0 = MixerOps.filled(0f, features);
        this.a1 = MixerOps.uniform(random, 0.1f, features, rank);
        this.a2 = MixerOps.uniform(random, 0.1f, rank, features);
        this.g1 = MixerOps.uniform(random, 0.3f, features, rank);
        this.g2 = MixerOps.weight(random, rank, features);
        this.removal = MixerOps.filled(0.85f, features);
        this.replacement = MixerOps.filled(1f, features);
        this.bonus = MixerOps.uniform(random, 0.1f, features);
    }

    /**
     * Describes the time mix for {@code Model.sequential().add(...)}.
     *
     * @param heads the number of heads
     * @param rank the low-rank projection size
     * @return the spec
     */
    public static LayerSpec of(int heads, int rank) {
        return MixerOps.sameShape((in, r) -> new Rwkv7TimeMix(in.dim(-1), heads, rank, r));
    }

    /**
     * The per-step quantities, each {@code [b, heads, t, headSize]}: r, w, replacement key,
     * v, a, normalized removal key, and k.
     */
    Var[] mixInputs(Var x) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        Var prev = Var.concat(1, Var.of(Tensor.zeros(Shape.of(b, 1, features))), x.slice(1, 0, t - 1));
        Var delta = prev.sub(x);
        Var[] xs = new Var[6];
        for (int i = 0; i < 6; i++) {
            xs[i] = x.add(delta.mul(mu.slice(0, i, i + 1)));
        }
        Var r = xs[0].matmul(wr);
        Var w = xs[1].matmul(w1).tanh().matmul(w2).add(w0).sigmoid().mul((float) -Math.exp(-0.5)).exp();
        Var k = xs[2].matmul(wk);
        Var v = xs[3].matmul(wv);
        Var a = xs[4].matmul(a1).matmul(a2).add(a0).sigmoid();
        Var kappa = MixerOps.unit(MixerOps.split(k.mul(removal), heads));
        Var kt = k.mul(a.sub(1f).mul(replacement).add(1f));
        return new Var[] {MixerOps.split(r, heads), MixerOps.split(w, heads), MixerOps.split(kt, heads),
            MixerOps.split(v, heads), MixerOps.split(a, heads), kappa, MixerOps.split(k, heads), xs[5]};
    }

    /**
     * The state readout {@code r^T S_t} for every step, before the bonus, norm and gate.
     *
     * @param p the result of {@link #mixInputs}
     * @return {@code [b, heads, t, headSize]}
     */
    Var wkv(Var[] p) {
        int b = p[0].shape().dim(0);
        int t = p[0].shape().dim(2);
        int d = headSize;
        Var state = Var.of(Tensor.zeros(Shape.of(b, heads, d, d)));
        List<Var> out = new ArrayList<>();
        for (int s = 0; s < t; s++) {
            Var r = p[0].slice(2, s, s + 1);           // [b, h, 1, d]
            Var w = p[1].slice(2, s, s + 1).transpose();   // [b, h, d, 1]
            Var kt = p[2].slice(2, s, s + 1);
            Var v = p[3].slice(2, s, s + 1);
            Var a = p[4].slice(2, s, s + 1);
            Var kh = p[5].slice(2, s, s + 1);
            // S = diag(w) S - (a . kh)(kh^T S) + kt v^T
            state = state.mul(w).sub(a.mul(kh).transpose().matmul(kh.matmul(state))).add(kt.transpose().matmul(v));
            out.add(r.matmul(state));
        }
        return Var.concat(2, out.toArray(Var[]::new));
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "Rwkv7TimeMix");
        Var[] p = mixInputs(input);
        Var o = wkv(p);
        Var u = MixerOps.split(bonus.reshape(1, 1, features), heads);
        Var direct = p[0].mul(p[6]).mul(u).sum(-1, true).mul(p[3]);
        o = o.sub(o.mean(-1, true));
        o = MixerOps.rmsNorm(o).add(direct);
        Var gate = p[7].matmul(g1).sigmoid().matmul(g2);
        return MixerOps.merge(o).mul(gate).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        return List.of(mu, wr, wk, wv, wo, w0, w1, w2, a0, a1, a2, g1, g2, removal, replacement, bonus);
    }

    @Override
    public String toString() {
        return "Rwkv7TimeMix(" + features + ", heads=" + heads + ")";
    }
}
