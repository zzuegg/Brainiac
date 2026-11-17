package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Kimi Delta Attention (KDA), the linear attention of Kimi Linear: Gated DeltaNet with a
 * <b>channel-wise</b> forget gate, so every key dimension of the matrix state has its own
 * decay rate and the state behaves like a bank of memories with different horizons:
 *
 * <pre>
 * S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T,    o_t = S_t^T q_t
 * </pre>
 *
 * <p>As in the paper, the decay comes from a <b>low-rank</b> projection,
 * {@code log alpha_t = -exp(A_h) softplus(x W_down W_up + b)}, and the output passes a
 * per-head RMS norm and a low-rank sigmoid gate before {@code W_o}; q, k and v pass a short
 * causal convolution and SiLU, and q and k are L2-normalized.
 *
 * <p>{@link #forward} uses the chunkwise-parallel algorithm for channel-wise decays (the
 * diagonal-plus-low-rank form): inside a chunk, all pairwise decays {@code exp(G_t - G_i)}
 * are formed per channel (exponents at most 0, so it never overflows), the delta-rule
 * pseudo-values solve a unit lower-triangular system (inverted by a nilpotent product), and
 * the state is carried between chunks. {@link #recurrent} computes the same token by token.
 * {@code GatedDeltaNet} with {@code Gating.CHANNEL} has the channel-wise recurrence but runs it
 * sequentially and gates with a full-rank projection.
 *
 * <p>Paper: Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture",
 * <a href="https://arxiv.org/abs/2510.26692">arXiv:2510.26692</a>.
 */
public final class KimiDeltaAttention implements Layer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final int chunk;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var convQ;
    private final Var convK;
    private final Var convV;
    private final Var gateDown;
    private final Var gateUp;
    private final Var gateBias;
    private final Var aLog;
    private final Var wBeta;
    private final Var outDown;
    private final Var outUp;
    private final Var wo;

    /**
     * Creates the layer.
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads the heads
     * @param rank the rank of the gate projections
     * @param chunk the chunk length of the parallel form
     * @param random the random source
     */
    public KimiDeltaAttention(int features, int heads, int rank, int chunk, RandomGenerator random) {
        if (features % heads != 0 || rank < 1 || chunk < 1) {
            throw new IllegalArgumentException("need heads dividing features, rank >= 1 and chunk >= 1");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.chunk = chunk;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.convQ = DeltaRuleOps.convInit(features, random);
        this.convK = DeltaRuleOps.convInit(features, random);
        this.convV = DeltaRuleOps.convInit(features, random);
        this.gateDown = MixerOps.weight(random, features, rank);
        this.gateUp = MixerOps.uniform(random, 0.01f, rank, features);
        float[] bias = new float[features];
        for (int i = 0; i < features; i++) {
            // spread the initial decay rates over the channels: from slow to fast forgetting
            bias[i] = -4f + 4f * (i % headSize) / Math.max(1, headSize - 1);
        }
        this.gateBias = Var.param(io.github.zzuegg.brainiac.tensor.Tensor.of(Shape.of(features), bias));
        this.aLog = MixerOps.filled(0f, heads, 1, 1);
        this.wBeta = MixerOps.weight(random, features, heads);
        this.outDown = MixerOps.weight(random, features, rank);
        this.outUp = MixerOps.weight(random, rank, features);
        this.wo = MixerOps.weight(random, features, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param heads the heads
     * @param rank the gate rank
     * @param chunk the chunk length
     * @return the spec
     */
    public static LayerSpec of(int heads, int rank, int chunk) {
        return MixerOps.sameShape((in, r) -> new KimiDeltaAttention(in.dim(-1), heads, rank, chunk, r));
    }

    /** q, k, v, log alpha [b, h, t, d] and beta [b, h, t]. */
    private Var[] project(Var x) {
        Var q = MixerOps.unit(MixerOps.split(DeltaRuleOps.convSilu(x, wq, convQ), heads));
        Var k = MixerOps.unit(MixerOps.split(DeltaRuleOps.convSilu(x, wk, convK), heads));
        Var v = MixerOps.split(DeltaRuleOps.convSilu(x, wv, convV), heads);
        Var f = x.matmul(gateDown).matmul(gateUp).add(gateBias);
        Var logAlpha = MixerOps.split(f.softplus(), heads).mul(aLog.exp()).neg();
        Var beta = x.matmul(wBeta).sigmoid().permute(0, 2, 1);
        return new Var[] {q, k, v, logAlpha, beta};
    }

    private Var finish(Var x, Var o) {
        Var gate = x.matmul(outDown).matmul(outUp).sigmoid();
        return MixerOps.merge(MixerOps.rmsNorm(o)).mul(gate).matmul(wo);
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "KimiDeltaAttention");
        Var[] p = project(input);
        return finish(input, DeltaRuleOps.chunkwise(p[0], p[1], p[2], p[3], p[4], chunk));
    }

    /**
     * The same output computed token by token with the recurrent state.
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, steps, features]}
     */
    public Var recurrent(Var input) {
        MixerOps.requireSequence(input, features, "KimiDeltaAttention");
        Var[] p = project(input);
        return finish(input, DeltaRuleOps.recurrent(p[0], p[1], p[2], p[3], p[4]));
    }

    /**
     * The per-channel decays {@code alpha} of an input, for inspection.
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, heads, steps, headSize]} values in {@code (0, 1]}
     */
    public io.github.zzuegg.brainiac.tensor.Tensor decays(Var input) {
        return project(input)[3].value().exp();
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, convQ, convK, convV, gateDown, gateUp, gateBias, aLog, wBeta, outDown, outUp, wo);
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "KimiDeltaAttention(" + features + ", heads=" + heads + ")";
    }
}
