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
 * Log-linear attention: between linear attention (one fixed-size state for the whole past)
 * and softmax attention (one entry per token) there is a hierarchy of {@code O(log T)}
 * states. The past of step {@code t} is split by a Fenwick tree into buckets of sizes
 * {@code 1, 2, 4, ...} (bucket {@code l} holds the keys {@code s} whose highest bit
 * differing from {@code t} is bit {@code l - 1}; bucket 0 is {@code t} itself), recent tokens
 * in small buckets and old ones in large ones. Each bucket is a linear-attention state, and
 * the query mixes the buckets with learned, data-dependent level weights
 * {@code lambda_t^l = softplus(x_t W_lambda)}:
 *
 * <pre>
 * o_t = sum_l lambda_t^l q_t^T sum_{s in bucket_l(t)} exp(G_t - G_s) k_s v_s^T
 * </pre>
 *
 * <p>Here the base is gated linear attention with a scalar decay per head ({@code G} its
 * cumulative log, as in Mamba-2), L2-normalized queries and keys, and an RMS-normed, SiLU-gated
 * output. With {@code hierarchical = false} every level weight is the same, which reduces
 * the layer to that gated linear attention (one state), the natural baseline.
 *
 * <p>{@link #forward} is the parallel form {@code O = (Q K^T o D o H) V}, with {@code H} the
 * level-weight matrix; {@link #recurrent} keeps only the {@code O(log T)} bucket states and
 * merges them like a binary counter (when moving to {@code t + 1}, buckets 0 to {@code c}
 * merge into bucket {@code c + 1}, where {@code c} counts the trailing ones of {@code t}).
 *
 * <p>Paper: Guo et al., "Log-Linear Attention", <a href="https://arxiv.org/abs/2506.04761">arXiv:2506.04761</a>.
 */
public final class LogLinearAttention implements Layer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final int levels;
    private final boolean hierarchical;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var convQ;
    private final Var convK;
    private final Var convV;
    private final Var wAlpha;
    private final Var alphaBias;
    private final Var wLevel;
    private final Var levelBias;
    private final Var wGate;
    private final Var wo;

    /**
     * Creates the layer.
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads the heads
     * @param maxSteps the longest sequence (fixes the number of levels, {@code 1 + ceil(log2 maxSteps)})
     * @param hierarchical whether the level weights are learned (log-linear) or all equal (linear)
     * @param random the random source
     */
    public LogLinearAttention(int features, int heads, int maxSteps, boolean hierarchical, RandomGenerator random) {
        if (features % heads != 0 || maxSteps < 1) {
            throw new IllegalArgumentException("need heads dividing features and maxSteps >= 1");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.levels = 2 + 32 - Integer.numberOfLeadingZeros(Math.max(1, maxSteps - 1));
        this.hierarchical = hierarchical;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.convQ = DeltaRuleOps.convInit(features, random);
        this.convK = DeltaRuleOps.convInit(features, random);
        this.convV = DeltaRuleOps.convInit(features, random);
        this.wAlpha = MixerOps.uniform(random, 0.01f, features, heads);
        this.alphaBias = MixerOps.filled(-4f, heads);
        this.wLevel = hierarchical ? MixerOps.uniform(random, 0.01f, features, heads * levels) : null;
        this.levelBias = hierarchical ? MixerOps.filled(0.5413f, heads * levels) : null;   // softplus = 1
        this.wGate = MixerOps.weight(random, features, features);
        this.wo = MixerOps.weight(random, features, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}; the input length sets the
     * levels (longer inputs share the top level).
     *
     * @param heads the heads
     * @param hierarchical log-linear (true) or plain gated linear attention (false)
     * @return the spec
     */
    public static LayerSpec of(int heads, boolean hierarchical) {
        return MixerOps.sameShape((in, r) -> new LogLinearAttention(in.dim(-1), heads, Math.max(64, in.dim(0)),
                hierarchical, r));
    }

    /**
     * The Fenwick level of key {@code s} for query {@code t}: 0 for {@code s = t}, otherwise
     * one more than the highest bit in which they differ.
     *
     * @param t the query step
     * @param s the key step, at most {@code t}
     * @return the level
     */
    public static int level(int t, int s) {
        return s == t ? 0 : 32 - Integer.numberOfLeadingZeros(t ^ s);
    }

    /** q, k, v [b, h, t, d], cumulative log decay [b, h, t], level weights [b, h, t, L] (or null). */
    private Var[] project(Var x) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        Var q = MixerOps.unit(MixerOps.split(DeltaRuleOps.convSilu(x, wq, convQ), heads));
        Var k = MixerOps.unit(MixerOps.split(DeltaRuleOps.convSilu(x, wk, convK), heads));
        Var v = MixerOps.split(DeltaRuleOps.convSilu(x, wv, convV), heads);
        Var logAlpha = x.matmul(wAlpha).add(alphaBias).softplus().neg().permute(0, 2, 1);   // [b, h, t]
        Var lambda = null;
        if (hierarchical) {
            lambda = x.matmul(wLevel).add(levelBias).softplus().reshape(b, t, heads, levels).permute(0, 2, 1, 3);
        }
        return new Var[] {q, k, v, logAlpha.cumsum(-1), lambda, logAlpha};
    }

    private Var finish(Var x, Var o) {
        Var gate = x.matmul(wGate).activation(io.github.zzuegg.brainiac.tensor.spi.ActivationOp.SILU, 0);
        return MixerOps.merge(MixerOps.rmsNorm(o)).mul(gate).matmul(wo);
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "LogLinearAttention");
        int t = input.shape().dim(1);
        Var[] p = project(input);
        Var scores = p[0].matmul(p[1].transpose()).mul(MixerOps.segmentDecay(p[3]));   // [b, h, t, t], causal
        if (hierarchical) {
            Var h = null;
            for (int l = 0; l < levels; l++) {
                float[] m = new float[t * t];
                boolean any = false;
                for (int i = 0; i < t; i++) {
                    for (int j = 0; j <= i; j++) {
                        if (Math.min(level(i, j), levels - 1) == l) {
                            m[i * t + j] = 1;
                            any = true;
                        }
                    }
                }
                if (any) {
                    Var part = p[4].slice(3, l, l + 1).mul(Tensor.of(Shape.of(t, t), m));
                    h = h == null ? part : h.add(part);
                }
            }
            scores = scores.mul(h);
        }
        return finish(input, scores.matmul(p[2]));
    }

    /**
     * The same output from {@code O(log T)} bucket states per head, merged like a binary
     * counter.
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, steps, features]}
     */
    public Var recurrent(Var input) {
        MixerOps.requireSequence(input, features, "LogLinearAttention");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        int d = headSize;
        Var[] p = project(input);
        Var[] buckets = new Var[levels];
        List<Var> out = new ArrayList<>();
        for (int s = 0; s < t; s++) {
            Var alpha = p[5].slice(2, s, s + 1).exp().reshape(b, heads, 1, 1);
            if (s > 0) {
                int c = Integer.numberOfTrailingZeros(~(s - 1));   // trailing ones of s - 1
                int target = Math.min(c + 1, levels - 1);
                Var merged = buckets[target];
                for (int l = 0; l <= Math.min(c, levels - 1); l++) {
                    if (buckets[l] != null && l != target) {
                        merged = merged == null ? buckets[l] : merged.add(buckets[l]);
                        buckets[l] = null;
                    }
                }
                buckets[target] = merged;
                for (int l = 0; l < levels; l++) {
                    if (buckets[l] != null) {
                        buckets[l] = buckets[l].mul(alpha);
                    }
                }
            }
            Var ks = p[1].slice(2, s, s + 1);
            buckets[0] = ks.transpose().matmul(p[2].slice(2, s, s + 1));       // [b, h, d, d]
            Var q = p[0].slice(2, s, s + 1);
            Var o = Var.of(Tensor.zeros(Shape.of(b, heads, 1, d)));
            for (int l = 0; l < levels; l++) {
                if (buckets[l] != null) {
                    Var read = q.matmul(buckets[l]);
                    o = o.add(hierarchical ? read.mul(p[4].slice(2, s, s + 1).slice(3, l, l + 1)) : read);
                }
            }
            out.add(o);
        }
        return finish(input, Var.concat(2, out.toArray(Var[]::new)));
    }

    /** @return the number of levels (bucket states per head) */
    public int levels() {
        return levels;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(wq, wk, wv, convQ, convK, convV, wAlpha, alphaBias, wGate, wo));
        if (hierarchical) {
            p.add(wLevel);
            p.add(levelBias);
        }
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "LogLinearAttention(" + features + ", heads=" + heads + (hierarchical ? ", levels=" + levels
                : ", linear") + ")";
    }
}
