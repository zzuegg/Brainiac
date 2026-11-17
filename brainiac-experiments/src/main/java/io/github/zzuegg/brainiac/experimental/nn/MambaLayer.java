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
 * A Mamba block: a selective state-space model whose parameters depend on the input. For
 * every inner channel {@code d} and state index {@code n},
 *
 * <pre>
 * h_t = exp(delta_t A) h_{t-1} + delta_t B_t x_t,     y_t = C_t . h_t + D x_t
 * delta_t = softplus(x_t W_delta + b),  B_t = x_t W_B,  C_t = x_t W_C,  A = -exp(A_log)
 * </pre>
 *
 * <p>Because {@code delta}, {@code B} and {@code C} come from the current token, the model
 * chooses what to write into its state and what to ignore (a large {@code delta} resets the
 * state towards the current input, a small one keeps it), which lets a linear-time RNN do
 * content-based tasks such as selective copying. The block around the SSM: an input
 * projection to {@code x} and a gate {@code z}, a short causal depthwise convolution and SiLU
 * on {@code x}, the SSM, {@code y * silu(z)} and an output projection. {@code A} uses the
 * S4D-real initialisation {@code A[n] = -(n + 1)}.
 *
 * <p>Two computations of the same scan:
 * <ul>
 *   <li>{@link #recurrent(Var)}: the selective scan, token by token (constant memory);</li>
 *   <li>{@link #forward(Var)}: the parallel form
 *       {@code y_t = sum_{s <= t} sum_n C_t[n] B_s[n] exp(A[d,n] (cs_t - cs_s)) delta_s x_s}
 *       with {@code cs} the cumulative sum of {@code delta}, materialized over
 *       {@code [steps, steps]} (quadratic, matmul-friendly, for short training sequences).</li>
 * </ul>
 *
 * <p>Paper: Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
 * <a href="https://arxiv.org/abs/2312.00752">arXiv:2312.00752</a>.
 */
public final class MambaLayer implements Layer {

    private final int features;
    private final int inner;
    private final int state;
    private final Var inProj;
    private final Var conv;
    private final Var wDelta;
    private final Var deltaBias;
    private final Var wB;
    private final Var wC;
    private final Var aLog;
    private final Var skip;
    private final Var outProj;

    /**
     * Creates the block.
     *
     * @param features the model width
     * @param inner the inner (expanded) width, usually {@code 2 * features}
     * @param state the state size {@code N} per channel
     * @param random the random source
     */
    public MambaLayer(int features, int inner, int state, RandomGenerator random) {
        this.features = features;
        this.inner = inner;
        this.state = state;
        this.inProj = MixerOps.weight(random, features, 2 * inner);
        this.conv = MixerOps.uniform(random, 0.5f, 4, inner);
        this.wDelta = MixerOps.uniform(random, 0.1f, inner, inner);
        float[] db = new float[inner];
        for (int i = 0; i < inner; i++) {
            // delta initialised log-uniformly in [0.001, 0.1], through the inverse softplus
            double dt = Math.exp(Math.log(0.001) + random.nextDouble() * (Math.log(0.1) - Math.log(0.001)));
            db[i] = (float) (dt + Math.log(-Math.expm1(-dt)));
        }
        this.deltaBias = Var.param(Tensor.of(Shape.of(inner), db));
        this.wB = MixerOps.weight(random, inner, state);
        this.wC = MixerOps.weight(random, inner, state);
        float[] a = new float[inner * state];
        for (int d = 0; d < inner; d++) {
            for (int n = 0; n < state; n++) {
                a[d * state + n] = (float) Math.log(n + 1);
            }
        }
        this.aLog = Var.param(Tensor.of(Shape.of(inner, state), a));
        this.skip = MixerOps.filled(1f, inner);
        this.outProj = MixerOps.weight(random, inner, features);
    }

    /**
     * Describes the block for {@code Model.sequential().add(...)}.
     *
     * @param expand the inner width as a multiple of the model width
     * @param state the state size per channel
     * @return the spec
     */
    public static LayerSpec of(int expand, int state) {
        return MixerOps.sameShape((in, r) -> new MambaLayer(in.dim(-1), expand * in.dim(-1), state, r));
    }

    /** {x [b,t,D], z, delta [b,t,D], B [b,t,N], C [b,t,N]}. */
    private Var[] project(Var input) {
        Var xz = input.matmul(inProj);
        Var x = MixerOps.causalDepthwise(xz.slice(2, 0, inner), conv).activation(ActivationOp.SILU, 0);
        Var z = xz.slice(2, inner, 2 * inner);
        Var delta = x.matmul(wDelta).add(deltaBias).softplus();
        return new Var[] {x, z, delta, x.matmul(wB), x.matmul(wC)};
    }

    private Var finish(Var x, Var z, Var y) {
        return y.add(x.mul(skip)).mul(z.activation(ActivationOp.SILU, 0)).matmul(outProj);
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "MambaLayer");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var[] p = project(input);
        Var x = p[0];
        Var delta = p[2];
        Var a = aLog.exp().neg();                                                     // [D, N]
        Var cs = delta.permute(0, 2, 1).cumsum(-1);    // [b, D, t]
        Var gap = cs.reshape(b, inner, t, 1).sub(cs.reshape(b, inner, 1, t))
                .mul(MixerOps.lowerOnes(t)).permute(0, 2, 3, 1);                     // [b, t, s, D]
        Var decay = gap.reshape(b, t, t, inner, 1).mul(a).exp()
                .mul(MixerOps.lowerOnes(t).reshape(t, t, 1, 1));                      // [b, t, s, D, N]
        Var cb = p[4].reshape(b, t, 1, 1, state).mul(p[3].reshape(b, 1, t, 1, state)); // C_t[n] B_s[n]
        Var u = delta.mul(x).reshape(b, 1, t, inner, 1);                              // delta_s x_s
        Var y = decay.mul(cb).mul(u).sum(-1, false).sum(2, false);                   // [b, t, D]
        return finish(x, p[1], y);
    }

    /**
     * Computes the output with the recurrent selective scan.
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, steps, features]}
     */
    public Var recurrent(Var input) {
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var[] p = project(input);
        Var a = aLog.exp().neg();
        Var h = Var.of(Tensor.zeros(Shape.of(b, inner, state)));
        List<Var> ys = new ArrayList<>();
        for (int s = 0; s < t; s++) {
            Var dt = p[2].slice(1, s, s + 1).reshape(b, inner, 1);
            Var xs = p[0].slice(1, s, s + 1).reshape(b, inner, 1);
            Var bs = p[3].slice(1, s, s + 1).reshape(b, 1, state);
            Var cs = p[4].slice(1, s, s + 1).reshape(b, 1, state);
            h = h.mul(dt.mul(a).exp()).add(dt.mul(xs).mul(bs));
            ys.add(h.mul(cs).sum(-1, false).reshape(b, 1, inner));
        }
        return finish(p[0], p[1], Var.concat(1, ys.toArray(Var[]::new)));
    }

    @Override
    public List<Var> parameters() {
        return List.of(inProj, conv, wDelta, deltaBias, wB, wC, aLog, skip, outProj);
    }

    @Override
    public String toString() {
        return "MambaLayer(" + features + ", inner=" + inner + ", state=" + state + ")";
    }
}
