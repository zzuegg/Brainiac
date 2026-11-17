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
 * Gated DeltaNet: linear attention whose matrix state is both decayed (a gate, for fast
 * forgetting) and edited by the delta rule (for precise overwriting of one association):
 *
 * <pre>
 * S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T,   o_t = S_t^T q_t
 * </pre>
 *
 * <p>with L2-normalized keys and queries, {@code alpha_t = exp(-exp(a) softplus(x w + b))} and
 * {@code beta_t = sigmoid(x w')}. Before reading, the delta rule removes what the state
 * currently stores for {@code k_t} ({@code beta = 1} replaces it), so a fixed-size state does
 * associative recall far better than plain linear attention. q, k and v pass a short causal
 * depthwise convolution and SiLU; the output is RMS-normalized and gated, as in the paper.
 *
 * <p>Two computations of the same recurrence:
 * <ul>
 *   <li>{@link #recurrent(Var)}: token by token, constant memory (inference);</li>
 *   <li>{@link #forward(Var)} with scalar gating: chunkwise-parallel. Within a chunk the
 *       state contributions are written as {@code S_t = Gamma_t S_0 + sum_i (Gamma_t/Gamma_i) k_i u_i^T};
 *       the pseudo-values {@code U} solve the unit lower-triangular system
 *       {@code (I + diag(beta) A) U = diag(beta)(V - diag(Gamma) K S_0)} with
 *       {@code A_ti = (Gamma_t/Gamma_i) k_t . k_i} for {@code i < t} (the gated UT transform),
 *       inverted exactly by the nilpotent product {@code (I - L)(I + L^2)(I + L^4)...}; then
 *       {@code O = diag(Gamma) Q S_0 + (Q K^T . D) U}. Everything inside a chunk is matmuls.</li>
 * </ul>
 *
 * <p>{@link Gating#CHANNEL} makes the decay a vector per key channel (Kimi Delta Attention):
 * {@code S_t = (I - beta k k^T) Diag(alpha_t) S_{t-1} + beta k v^T}; it is computed recurrently.
 *
 * <p>Papers: Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule",
 * <a href="https://arxiv.org/abs/2412.06464">arXiv:2412.06464</a>; Kimi Team, "Kimi Linear",
 * <a href="https://arxiv.org/abs/2510.26692">arXiv:2510.26692</a>.
 */
public final class GatedDeltaNet implements Layer {

    /** The shape of the forget gate. */
    public enum Gating {
        /** One decay per head and step (Gated DeltaNet). */
        SCALAR,
        /** One decay per head, key channel and step (Kimi Delta Attention). */
        CHANNEL
    }

    private final int features;
    private final int heads;
    private final int headSize;
    private final Gating gating;
    private final int chunk;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var convQ;
    private final Var convK;
    private final Var convV;
    private final Var wAlpha;
    private final Var alphaBias;
    private final Var alphaLog;
    private final Var wBeta;
    private final Var wGate;
    private final Var wo;

    /**
     * Creates the layer.
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads the number of heads
     * @param gating scalar (Gated DeltaNet) or channel-wise (Kimi Delta Attention) decay
     * @param chunk the chunk length of the parallel form
     * @param random the random source
     */
    public GatedDeltaNet(int features, int heads, Gating gating, int chunk, RandomGenerator random) {
        if (features % heads != 0) {
            throw new IllegalArgumentException("features " + features + " not divisible by heads " + heads);
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.gating = gating;
        this.chunk = chunk;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.convQ = convInit(features, random);
        this.convK = convInit(features, random);
        this.convV = convInit(features, random);
        int gates = gating == Gating.SCALAR ? heads : features;
        this.wAlpha = MixerOps.uniform(random, 0.01f, features, gates);
        this.alphaBias = MixerOps.filled(-2f, gates);
        this.alphaLog = MixerOps.filled(0f, gates);
        this.wBeta = MixerOps.weight(random, features, heads);
        this.wGate = MixerOps.weight(random, features, features);
        this.wo = MixerOps.weight(random, features, features);
    }

    /** A short causal convolution that starts close to the identity. */
    private static Var convInit(int features, RandomGenerator random) {
        float[] w = new float[4 * features];
        for (int i = 0; i < w.length; i++) {
            w[i] = (i < features ? 1f : 0f) + (float) (random.nextDouble() * 0.2 - 0.1);
        }
        return Var.param(Tensor.of(Shape.of(4, features), w));
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param heads the number of heads
     * @param gating the gate shape
     * @param chunk the chunk length of the parallel form
     * @return the spec
     */
    public static LayerSpec of(int heads, Gating gating, int chunk) {
        return MixerOps.sameShape((in, r) -> new GatedDeltaNet(in.dim(-1), heads, gating, chunk, r));
    }

    /** Projected q, k, v [b, h, t, d], log alpha [b, h, t] or [b, h, t, d], beta [b, h, t]. */
    private Var[] project(Var x) {
        Var q = MixerOps.unit(MixerOps.split(MixerOps.causalDepthwise(x.matmul(wq), convQ).activation(
                io.github.zzuegg.brainiac.tensor.spi.ActivationOp.SILU, 0), heads));
        Var k = MixerOps.unit(MixerOps.split(MixerOps.causalDepthwise(x.matmul(wk), convK).activation(
                io.github.zzuegg.brainiac.tensor.spi.ActivationOp.SILU, 0), heads));
        Var v = MixerOps.split(MixerOps.causalDepthwise(x.matmul(wv), convV).activation(
                io.github.zzuegg.brainiac.tensor.spi.ActivationOp.SILU, 0), heads);
        Var logAlpha = x.matmul(wAlpha).add(alphaBias).softplus().mul(alphaLog.exp()).neg();
        logAlpha = gating == Gating.SCALAR ? logAlpha.permute(0, 2, 1) : MixerOps.split(logAlpha, heads);
        Var beta = x.matmul(wBeta).sigmoid().permute(0, 2, 1);
        return new Var[] {q, k, v, logAlpha, beta};
    }

    private Var finish(Var x, Var o) {
        Var gate = x.matmul(wGate).activation(io.github.zzuegg.brainiac.tensor.spi.ActivationOp.SILU, 0);
        return MixerOps.merge(MixerOps.rmsNorm(o)).mul(gate).matmul(wo);
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "GatedDeltaNet");
        return gating == Gating.SCALAR ? chunkwise(input) : recurrent(input);
    }

    /**
     * Computes the output token by token with the recurrent state.
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, steps, features]}
     */
    public Var recurrent(Var input) {
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        int d = headSize;
        Var[] p = project(input);
        Var state = Var.of(Tensor.zeros(Shape.of(b, heads, d, d)));
        List<Var> out = new ArrayList<>();
        for (int s = 0; s < t; s++) {
            Var q = p[0].slice(2, s, s + 1);   // [b, h, 1, d]
            Var k = p[1].slice(2, s, s + 1);
            Var v = p[2].slice(2, s, s + 1);
            Var beta = p[4].slice(2, s, s + 1).reshape(b, heads, 1, 1);
            Var kt = k.transpose();            // [b, h, d, 1]
            if (gating == Gating.SCALAR) {
                Var alpha = p[3].slice(2, s, s + 1).exp().reshape(b, heads, 1, 1);
                state = state.mul(alpha);
            } else {
                Var alpha = p[3].slice(2, s, s + 1).exp().transpose();   // [b, h, d, 1]: rows of S
                state = state.mul(alpha);
            }
            // delta rule: S - beta k (k^T S) + beta k v^T = S + beta k (v - k^T S)
            Var predicted = k.matmul(state);   // [b, h, 1, d]
            state = state.add(kt.matmul(v.sub(predicted)).mul(beta));
            out.add(q.matmul(state));
        }
        return finish(input, Var.concat(2, out.toArray(Var[]::new)));
    }

    /**
     * Computes the output with the chunkwise-parallel algorithm (scalar gating only).
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, steps, features]}
     */
    public Var chunkwise(Var input) {
        if (gating != Gating.SCALAR) {
            throw new IllegalStateException("the chunkwise form needs scalar gating");
        }
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        int d = headSize;
        Var[] p = project(input);
        Var state = Var.of(Tensor.zeros(Shape.of(b, heads, d, d)));
        List<Var> out = new ArrayList<>();
        for (int start = 0; start < t; start += chunk) {
            int end = Math.min(t, start + chunk);
            int c = end - start;
            Var q = p[0].slice(2, start, end);
            Var k = p[1].slice(2, start, end);
            Var v = p[2].slice(2, start, end);
            Var beta = p[4].slice(2, start, end).reshape(b, heads, c, 1);
            Var logGamma = p[3].slice(2, start, end).matmul(Var.of(MixerOps.cumsumMatrix(c)));   // [b, h, c]
            Var gamma = logGamma.exp().reshape(b, heads, c, 1);
            Var decay = MixerOps.segmentDecay(logGamma);                                         // [b, h, c, c]
            Tensor strict = MixerOps.lowerOnes(c).sub(Tensor.of(Shape.of(c, c), identity(c)));
            Var lower = k.matmul(k.transpose()).mul(decay).mul(strict).mul(beta);               // diag(beta) A
            Var inverse = unitLowerInverse(lower, c);
            Var u = inverse.matmul(v.sub(k.mul(gamma).matmul(state)).mul(beta));                 // [b, h, c, d]
            out.add(q.mul(gamma).matmul(state).add(q.matmul(k.transpose()).mul(decay).matmul(u)));
            Var last = logGamma.slice(2, c - 1, c);                                              // [b, h, 1]
            Var toEnd = last.sub(logGamma).exp().reshape(b, heads, c, 1);
            state = state.mul(last.exp().reshape(b, heads, 1, 1)).add(k.mul(toEnd).transpose().matmul(u));
        }
        return finish(input, out.size() == 1 ? out.getFirst() : Var.concat(2, out.toArray(Var[]::new)));
    }

    private static float[] identity(int c) {
        float[] m = new float[c * c];
        for (int i = 0; i < c; i++) {
            m[i * c + i] = 1;
        }
        return m;
    }

    /** {@code (I + L)^{-1}} for strictly lower-triangular {@code L}: {@code (I - L)(I + L^2)(I + L^4)...}. */
    static Var unitLowerInverse(Var lower, int c) {
        Tensor eye = Tensor.of(Shape.of(c, c), identity(c));
        Var inverse = lower.neg().add(eye);
        Var power = lower;
        for (int reach = 2; reach < c; reach *= 2) {
            power = power.matmul(power);
            inverse = inverse.matmul(power.add(eye));
        }
        return inverse;
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, convQ, convK, convV, wAlpha, alphaBias, alphaLog, wBeta, wGate, wo);
    }

    @Override
    public String toString() {
        return "GatedDeltaNet(" + features + ", heads=" + heads + ", " + gating + ")";
    }
}
