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
 * Manifold-constrained hyper-connections (mHC): a residual connection widened to
 * {@code n} parallel residual streams. Around a wrapped sub-layer {@code f}:
 *
 * <pre>
 * x_in = H_pre X                       (mix the n streams into the layer input)
 * X'   = H_res X + H_post^T f(x_in)    (mix streams, write the output back into each)
 * </pre>
 *
 * <p>with {@code X} of shape {@code [batch, steps, n, features]}. {@code H_pre} and
 * {@code H_post} are non-negative (sigmoid), and {@code H_res} is projected onto the
 * Birkhoff polytope of doubly stochastic matrices by Sinkhorn-Knopp iterations. A product
 * of doubly stochastic matrices is doubly stochastic again, so however deep the stack, the
 * streams' mixing keeps the mean signal and cannot blow up: this restores the identity
 * mapping property that unconstrained hyper-connections lose. All three maps have a static
 * part and a token-dependent part computed from the RMS-normalized streams, scaled by
 * small learned gates.
 *
 * <p>Use {@link #expand} to turn {@code [b, t, d]} into {@code n} streams before the first
 * layer and {@link #collapse} to sum them after the last.
 *
 * <p>Papers: Xie et al., "mHC: Manifold-Constrained Hyper-Connections",
 * <a href="https://arxiv.org/abs/2512.24880">arXiv:2512.24880</a>; Zhu et al.,
 * "Hyper-Connections", <a href="https://arxiv.org/abs/2409.19606">arXiv:2409.19606</a>.
 */
public final class ManifoldHyperConnection implements Layer {

    private final Layer inner;
    private final int streams;
    private final int features;
    private final int sinkhornIterations;
    private final Var staticPre;
    private final Var staticPost;
    private final Var staticRes;
    private final Var dynPre;
    private final Var dynPost;
    private final Var dynRes;
    private final Var gatePre;
    private final Var gatePost;
    private final Var gateRes;

    /**
     * Wraps a layer.
     *
     * @param inner the sub-layer, mapping {@code [b, t, features]} to the same shape
     * @param streams the number of residual streams {@code n}
     * @param features the width of one stream
     * @param random the random source
     */
    public ManifoldHyperConnection(Layer inner, int streams, int features, RandomGenerator random) {
        this.inner = inner;
        this.streams = streams;
        this.features = features;
        this.sinkhornIterations = 20;
        // the layer reads stream 0 at first, writes to all streams, and H_res starts near identity
        float[] pre = new float[streams];
        java.util.Arrays.fill(pre, -4f);
        pre[0] = 4f;
        this.staticPre = Var.param(Tensor.of(Shape.of(streams), pre));
        this.staticPost = MixerOps.filled(0f, streams);
        float[] res = new float[streams * streams];
        for (int i = 0; i < streams; i++) {
            for (int j = 0; j < streams; j++) {
                res[i * streams + j] = i == j ? 4f : 0f;
            }
        }
        this.staticRes = Var.param(Tensor.of(Shape.of(streams, streams), res));
        this.dynPre = MixerOps.uniform(random, 0.01f, streams * features, streams);
        this.dynPost = MixerOps.uniform(random, 0.01f, streams * features, streams);
        this.dynRes = MixerOps.uniform(random, 0.01f, streams * features, streams * streams);
        this.gatePre = MixerOps.filled(0.01f, 1);
        this.gatePost = MixerOps.filled(0.01f, 1);
        this.gateRes = MixerOps.filled(0.01f, 1);
    }

    /**
     * Projects positive matrices onto doubly stochastic ones: {@code exp(logits)}, then
     * alternately normalizes rows and columns. Differentiable.
     *
     * @param logits {@code [..., n, n]}
     * @param iterations the number of row-and-column passes, for example 20
     * @return a matrix whose rows and columns each sum to about 1
     */
    public static Var sinkhorn(Var logits, int iterations) {
        Var m = logits.sub(logits.max(-1, true).max(-2, true)).exp();
        for (int i = 0; i < iterations; i++) {
            m = m.div(m.sum(-1, true));
            m = m.div(m.sum(-2, true));
        }
        return m;
    }

    /**
     * Copies a sequence into {@code n} identical residual streams.
     *
     * @param x {@code [b, t, d]}
     * @param streams {@code n}
     * @return {@code [b, t, n, d]}
     */
    public static Var expand(Var x, int streams) {
        Shape s = x.shape();
        return x.reshape(s.dim(0), s.dim(1), 1, s.dim(2)).expand(Shape.of(s.dim(0), s.dim(1), streams, s.dim(2)));
    }

    /**
     * Sums the residual streams back into one sequence.
     *
     * @param x {@code [b, t, n, d]}
     * @return {@code [b, t, d]}
     */
    public static Var collapse(Var x) {
        return x.sum(2, false);
    }

    /**
     * Returns the residual mixing matrices.
     *
     * @param streamsInput {@code [b, t, n, d]}
     * @return {@code [b, t, n, n]}, doubly stochastic
     */
    public Tensor residualMixing(Var streamsInput) {
        return maps(streamsInput)[2].value();
    }

    /** {pre [b,t,1,n], post [b,t,n,1], res [b,t,n,n]}. */
    private Var[] maps(Var x) {
        Shape s = x.shape();
        int b = s.dim(0);
        int t = s.dim(1);
        Var flat = MixerOps.rmsNorm(x.reshape(b, t, streams * features));
        Var pre = flat.matmul(dynPre).mul(gatePre).add(staticPre).sigmoid().reshape(b, t, 1, streams);
        Var post = flat.matmul(dynPost).mul(gatePost).add(staticPost).sigmoid().mul(2f).reshape(b, t, streams, 1);
        Var resLogits = flat.matmul(dynRes).mul(gateRes).reshape(b, t, streams, streams).add(staticRes);
        return new Var[] {pre, post, sinkhorn(resLogits, sinkhornIterations)};
    }

    /**
     * Describes a hyper-connection around a sub-layer for {@code Model.sequential().add(...)};
     * the input is {@code [steps, streams, features]}.
     *
     * @param inner the spec of the wrapped sub-layer, which sees {@code [steps, features]}
     * @return the spec
     */
    public static LayerSpec of(LayerSpec inner) {
        return MixerOps.sameShape((in, r) -> new ManifoldHyperConnection(
                inner.create(Shape.of(in.dim(0), in.dim(2)), r), in.dim(1), in.dim(2), r));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Shape s = input.shape();
        if (s.rank() != 4 || s.dim(2) != streams || s.dim(3) != features) {
            throw new IllegalArgumentException("ManifoldHyperConnection expects [batch, steps, " + streams + ", "
                    + features + "], got " + s);
        }
        int b = s.dim(0);
        int t = s.dim(1);
        Var[] m = maps(input);
        Var layerIn = m[0].matmul(input).reshape(b, t, features);
        Var out = inner.forward(layerIn, training).reshape(b, t, 1, features);
        return m[2].matmul(input).add(m[1].mul(out));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(staticPre, staticPost, staticRes, dynPre, dynPost, dynRes,
                gatePre, gatePost, gateRes));
        p.addAll(inner.parameters());
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "ManifoldHyperConnection(" + inner + ", streams=" + streams + ")";
    }
}
