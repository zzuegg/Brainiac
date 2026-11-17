package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * The lightning indexer of DeepSeek Sparse Attention: a few tiny ReLU heads score how useful
 * each earlier token (or compressed block) is for a query,
 *
 * <pre>
 * I(t, s) = sum_j w_{t,j} ReLU(q_{t,j} . k_s),   q = x_t W_q,  k = x_s W_k (one key head),  w = x_t W_w
 * </pre>
 *
 * <p>so that the main attention only has to look at the top-k entries. With few heads, a
 * small dimension and FP8 queries and keys ({@link Fp8Format#E4M3}, straight-through), scoring
 * every past token costs a fraction of real attention. The indexer is trained by
 * distillation: {@link #distillationLoss} is the KL divergence from the main attention's
 * head-summed, renormalized weights to {@code softmax(I)}. Its input is detached, so this
 * loss trains only the indexer.
 *
 * <p>Paper: DeepSeek-AI, "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models",
 * <a href="https://arxiv.org/abs/2512.02556">arXiv:2512.02556</a>.
 */
public final class LightningIndexer {

    private final int heads;
    private final int dim;
    private final boolean lowPrecision;
    private final Var wq;
    private final Var wk;
    private final Var ww;

    /**
     * Creates an indexer.
     *
     * @param features the input width of queries and keys
     * @param heads the indexer heads
     * @param dim the indexer head dimension
     * @param lowPrecision whether to round queries and keys to FP8 (E4M3)
     * @param random the random source
     */
    public LightningIndexer(int features, int heads, int dim, boolean lowPrecision, RandomGenerator random) {
        this(features, features, heads, dim, lowPrecision, random);
    }

    /**
     * Creates an indexer whose keys have another width than its queries (for example
     * compressed key/value blocks).
     *
     * @param queryFeatures the query input width
     * @param keyFeatures the key input width
     * @param heads the indexer heads
     * @param dim the indexer head dimension
     * @param lowPrecision whether to round queries and keys to FP8 (E4M3)
     * @param random the random source
     */
    public LightningIndexer(int queryFeatures, int keyFeatures, int heads, int dim, boolean lowPrecision,
            RandomGenerator random) {
        if (heads < 1 || dim < 1) {
            throw new IllegalArgumentException("heads and dim must be positive");
        }
        this.heads = heads;
        this.dim = dim;
        this.lowPrecision = lowPrecision;
        this.wq = MixerOps.weight(random, queryFeatures, heads * dim);
        this.wk = MixerOps.weight(random, keyFeatures, dim);
        this.ww = MixerOps.weight(random, queryFeatures, heads);
    }

    /**
     * Scores every key for every query.
     *
     * @param queries {@code [batch, queries, features]}
     * @param keys {@code [batch, keys, keyFeatures]}
     * @return {@code [batch, queries, keys]}
     */
    public Var scores(Var queries, Var keys) {
        int b = queries.shape().dim(0);
        int tq = queries.shape().dim(1);
        int tk = keys.shape().dim(1);
        Var q = queries.matmul(wq).reshape(b, tq, heads, dim).permute(0, 2, 1, 3);   // [b, H, tq, d]
        Var k = keys.matmul(wk).reshape(b, 1, tk, dim);
        if (lowPrecision) {
            q = Fp8Format.E4M3.fakeQuantize(q);
            k = Fp8Format.E4M3.fakeQuantize(k);
        }
        Var s = q.matmul(k.transpose()).mul((float) (1 / Math.sqrt(dim))).relu();   // [b, H, tq, tk]
        Var w = queries.matmul(ww).permute(0, 2, 1).reshape(b, heads, tq, 1);
        return s.mul(w).sum(1, false);
    }

    /**
     * The distillation loss {@code KL(p || softmax(I))} over the kept entries, averaged over
     * queries.
     *
     * @param scores indexer scores {@code [batch, queries, keys]}
     * @param attention the main attention's weights {@code [batch, heads, queries, keys]}
     * @param keep 1 where an entry takes part (visible and, after warm-up, selected),
     *     broadcastable to {@code [batch, queries, keys]}
     * @return the scalar loss
     */
    public static Var distillationLoss(Var scores, Tensor attention, Tensor keep) {
        Shape s = scores.shape();
        Tensor k = keep.expand(s);
        Tensor p = attention.sum(1, false).mul(k);
        p = p.div(p.sum(-1, true).add(1e-9f));
        return SparseSelectOps.distillation(scores, p, k);
    }

    /** @return the indexer's parameters */
    public List<Var> parameters() {
        return List.of(wq, wk, ww);
    }

    @Override
    public String toString() {
        return "LightningIndexer(heads=" + heads + ", dim=" + dim + (lowPrecision ? ", fp8" : "") + ")";
    }
}
