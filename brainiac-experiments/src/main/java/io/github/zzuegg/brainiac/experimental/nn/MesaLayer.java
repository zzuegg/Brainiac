package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * The MesaNet layer (von Oswald et al., "MesaNet: sequence modeling by locally optimal
 * test-time training", <a href="https://arxiv.org/abs/2506.05233">arXiv:2506.05233</a>): instead
 * of one gradient step on the in-context regression loss per token (DeltaNet, TTT), every token
 * <em>solves</em> the regularized least-squares problem over everything seen so far,
 *
 * <pre>
 * W_t = argmin_W  sum_(s &lt;= t) gamma_(s..t) beta_s ||W k_s - v_s||^2 + lambda ||W||^2
 *     = G_t (H_t + lambda I)^(-1),
 * H_t = gamma_t H_(t-1) + beta_t k_t k_t^T,   G_t = gamma_t G_(t-1) + beta_t v_t k_t^T,
 * o_t = W_t q_t = G_t z_t,   (H_t + lambda I) z_t = q_t
 * </pre>
 *
 * <p>with input-dependent forget gates {@code gamma} and input gates {@code beta}. The linear
 * system is solved per token by a fixed number of <em>conjugate gradient</em> steps, as in the
 * paper, unrolled so training differentiates through the solver. The two {@code d x d}
 * statistics are the whole state, so the cost per token is constant, as in linear attention,
 * but the read-out is the optimal in-context regressor rather than one step towards it.
 * Input and output {@code [batch, steps, features]}.
 */
public final class MesaLayer implements Layer {
    private final int features;
    private final int heads;
    private final int headSize;
    private final int cgSteps;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wGamma;
    private final Var gammaBias;
    private final Var wBeta;
    private final Var lambdaRaw;
    private final Var wo;

    /**
     * Creates the layer.
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads heads
     * @param cgSteps conjugate-gradient steps per token
     * @param random the random source
     */
    public MesaLayer(int features, int heads, int cgSteps, RandomGenerator random) {
        if (features % heads != 0) {
            throw new IllegalArgumentException("features " + features + " not divisible by heads " + heads);
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.cgSteps = cgSteps;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.wGamma = MixerOps.uniform(random, 0.01f, features, heads);
        this.gammaBias = MixerOps.filled(3f, heads);
        this.wBeta = MixerOps.weight(random, features, heads);
        this.lambdaRaw = MixerOps.filled(0f, heads);
        this.wo = MixerOps.weight(random, features, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param heads heads
     * @param cgSteps conjugate-gradient steps per token
     * @return the spec
     */
    public static LayerSpec of(int heads, int cgSteps) {
        return MixerOps.sameShape((in, r) -> new MesaLayer(in.dim(-1), heads, cgSteps, r));
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "MesaLayer");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        int d = headSize;
        Var q = MixerOps.unit(MixerOps.split(input.matmul(wq), heads));                  // [b, h, t, d]
        Var k = MixerOps.unit(MixerOps.split(input.matmul(wk), heads));
        Var v = MixerOps.split(input.matmul(wv).activation(ActivationOp.SILU, 0), heads);
        Var gamma = input.matmul(wGamma).add(gammaBias).sigmoid().permute(0, 2, 1);      // [b, h, t]
        Var beta = input.matmul(wBeta).sigmoid().permute(0, 2, 1);
        Var lambda = lambdaRaw.softplus().add(0.05f).reshape(1, heads, 1, 1);
        Var h = Var.of(Tensor.zeros(Shape.of(b, heads, d, d)));
        Var g = Var.of(Tensor.zeros(Shape.of(b, heads, d, d)));
        List<Var> out = new ArrayList<>();
        for (int s = 0; s < t; s++) {
            Var qs = q.slice(2, s, s + 1).transpose();                  // [b, h, d, 1]
            Var ks = k.slice(2, s, s + 1).transpose();
            Var vs = v.slice(2, s, s + 1).transpose();
            Var gs = gamma.slice(2, s, s + 1).reshape(b, heads, 1, 1);
            Var bs = beta.slice(2, s, s + 1).reshape(b, heads, 1, 1);
            h = h.mul(gs).add(ks.matmul(ks.transpose()).mul(bs));
            g = g.mul(gs).add(vs.matmul(ks.transpose()).mul(bs));
            Var z = conjugateGradient(h, lambda, qs, cgSteps);
            out.add(g.matmul(z).transpose());                            // [b, h, 1, d]
        }
        Var o = Var.concat(2, out.toArray(Var[]::new));
        return MixerOps.merge(MixerOps.rmsNorm(o)).matmul(wo);
    }

    /**
     * Solves {@code (H + lambda I) z = q} by unrolled conjugate gradient from {@code z = 0}.
     *
     * @param h symmetric positive semi-definite {@code [..., d, d]}
     * @param lambda the ridge, broadcast against {@code [..., 1, 1]}
     * @param q right-hand sides {@code [..., d, 1]}
     * @param steps iterations
     * @return {@code z}, differentiable in {@code h}, {@code lambda} and {@code q}
     */
    public static Var conjugateGradient(Var h, Var lambda, Var q, int steps) {
        Var z = q.mul(0f);
        Var r = q;
        Var p = q;
        Var rr = r.square().sum(-2, true);
        for (int i = 0; i < steps; i++) {
            Var ap = h.matmul(p).add(p.mul(lambda));
            Var alpha = rr.div(p.mul(ap).sum(-2, true).add(1e-8f));
            z = z.add(p.mul(alpha));
            r = r.sub(ap.mul(alpha));
            Var rrNew = r.square().sum(-2, true);
            p = r.add(p.mul(rrNew.div(rr.add(1e-8f))));
            rr = rrNew;
        }
        return z;
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, wGamma, gammaBias, wBeta, lambdaRaw, wo);
    }

    @Override
    public String toString() {
        return "MesaLayer(" + features + ", heads=" + heads + ", cg=" + cgSteps + ")";
    }
}
