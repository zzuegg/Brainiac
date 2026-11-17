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
 * The mLSTM of xLSTM: an LSTM whose cell is a matrix memory written with a covariance
 * update, with exponential input gates stabilized in log space:
 *
 * <pre>
 * C_t = f_t C_{t-1} + i_t k_t v_t^T      n_t = f_t n_{t-1} + i_t k_t
 * h_t = o_t . (C_t^T q_t) / max(|n_t . q_t|, exp(-m_t))
 * </pre>
 *
 * <p>with {@code i_t = exp(i~_t)}, {@code f_t = sigmoid(f~_t)} per head. Exponentials
 * overflow, so the recurrence carries a stabilizer {@code m_t = max(log f_t + m_{t-1}, i~_t)}
 * and uses {@code exp(i~ - m)} and {@code exp(log f + m_{t-1} - m_t)} instead: the ratios, and
 * so the output, are unchanged. The exponential input gate can sharply overwrite memory,
 * which sigmoid gates cannot ("revising storage decisions").
 *
 * <p>No hidden-to-hidden connections, so it parallelizes: {@link #forward(Var)} computes
 * {@code H = (Q K^T / sqrt(d) . D) V / normalizer} with
 * {@code D_ts = exp(F_t - F_s + i~_s - m_t)} for {@code s <= t} ({@code F} the cumulative
 * log forget gates, {@code m_t} the row maximum), and {@link #recurrent(Var)} runs the
 * stabilized recurrence; they agree.
 *
 * <p>Paper: Beck et al., "xLSTM: Extended Long Short-Term Memory",
 * <a href="https://arxiv.org/abs/2405.04517">arXiv:2405.04517</a>.
 */
public final class MLstmLayer implements Layer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wi;
    private final Var bi;
    private final Var wf;
    private final Var bf;
    private final Var wo;
    private final Var out;

    /**
     * Creates the layer.
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads the number of heads
     * @param random the random source
     */
    public MLstmLayer(int features, int heads, RandomGenerator random) {
        if (features % heads != 0) {
            throw new IllegalArgumentException("features " + features + " not divisible by heads " + heads);
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.wi = MixerOps.uniform(random, 0.1f, features, heads);
        this.bi = MixerOps.filled(0f, heads);
        this.wf = MixerOps.uniform(random, 0.1f, features, heads);
        this.bf = MixerOps.filled(3f, heads);
        this.wo = MixerOps.weight(random, features, features);
        this.out = MixerOps.weight(random, features, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param heads the number of heads
     * @return the spec
     */
    public static LayerSpec of(int heads) {
        return MixerOps.sameShape((in, r) -> new MLstmLayer(in.dim(-1), heads, r));
    }

    /** q, k (scaled), v [b,h,t,d], log input gate i~ [b,h,t], log forget gate [b,h,t]. */
    private Var[] project(Var x) {
        Var q = MixerOps.split(x.matmul(wq), heads);
        Var k = MixerOps.split(x.matmul(wk), heads).mul((float) (1 / Math.sqrt(headSize)));
        Var v = MixerOps.split(x.matmul(wv), heads);
        Var iLog = x.matmul(wi).add(bi).permute(0, 2, 1);
        Var fLog = x.matmul(wf).add(bf).neg().softplus().neg().permute(0, 2, 1);   // log sigmoid
        return new Var[] {q, k, v, iLog, fLog};
    }

    private Var finish(Var x, Var h) {
        return MixerOps.merge(h).mul(x.matmul(wo).sigmoid()).matmul(out);
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "MLstmLayer");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var[] p = project(input);
        Var cum = p[4].cumsum(-1);   // F_t
        Tensor lower = MixerOps.lowerOnes(t);
        Var logD = cum.reshape(b, heads, t, 1).sub(cum.reshape(b, heads, 1, t)).add(p[3].reshape(b, heads, 1, t))
                .mul(lower).add(lower.sub(1f).mul(1e9f));            // -1e9 above the diagonal
        Var m = logD.max(-1, true);                                  // [b, h, t, 1]
        Var weights = p[0].matmul(p[1].transpose()).mul(logD.sub(m).exp());
        Var norm = weights.sum(-1, true).abs().maximum(m.neg().exp());
        return finish(input, weights.matmul(p[2]).div(norm));
    }

    /**
     * Computes the output with the stabilized recurrence.
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, steps, features]}
     */
    public Var recurrent(Var input) {
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        int d = headSize;
        Var[] p = project(input);
        Var c = Var.of(Tensor.zeros(Shape.of(b, heads, d, d)));
        Var n = Var.of(Tensor.zeros(Shape.of(b, heads, d, 1)));
        Var m = Var.of(Tensor.full(Shape.of(b, heads, 1, 1), -1e30f));
        List<Var> hs = new ArrayList<>();
        for (int s = 0; s < t; s++) {
            Var q = p[0].slice(2, s, s + 1);
            Var k = p[1].slice(2, s, s + 1);
            Var v = p[2].slice(2, s, s + 1);
            Var iLog = p[3].slice(2, s, s + 1).reshape(b, heads, 1, 1);
            Var fLog = p[4].slice(2, s, s + 1).reshape(b, heads, 1, 1);
            Var mNew = fLog.add(m).maximum(iLog);
            Var ig = iLog.sub(mNew).exp();
            Var fg = fLog.add(m).sub(mNew).exp();
            c = c.mul(fg).add(k.transpose().matmul(v).mul(ig));
            n = n.mul(fg).add(k.transpose().mul(ig));
            m = mNew;
            Var norm = q.matmul(n).abs().maximum(m.neg().exp());
            hs.add(q.matmul(c).div(norm));
        }
        return finish(input, Var.concat(2, hs.toArray(Var[]::new)));
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, wi, bi, wf, bf, wo, out);
    }

    @Override
    public String toString() {
        return "MLstmLayer(" + features + ", heads=" + heads + ")";
    }
}
