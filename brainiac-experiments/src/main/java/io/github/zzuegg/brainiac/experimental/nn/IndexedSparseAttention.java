package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * DeepSeek Sparse Attention (DSA): causal multi-head attention in which each query attends
 * only to the {@code topK} past tokens that a {@link LightningIndexer} ranks highest. The
 * indexer scores all past tokens cheaply (few ReLU heads, FP8), the expensive attention then
 * costs {@code O(T k)} instead of {@code O(T^2)}.
 *
 * <p>Training follows the paper's two stages, counted in training forward passes so the
 * layer drops into any trainer:
 * <ol>
 *   <li><b>dense warm-up</b> ({@code warmupSteps}): attention stays dense, and the indexer
 *       learns to imitate it by KL distillation (its input is detached, so the loss only
 *       trains the indexer);</li>
 *   <li><b>sparse training</b>: attention is restricted to the indexer's top-k, and the
 *       indexer keeps distilling the main attention over the selected set.</li>
 * </ol>
 * The distillation loss, times {@code klWeight}, is the layer's {@link #auxiliaryLoss()}.
 * Outside training the layer always runs sparse. {@link #lastAttended()} reports the mean
 * number of keys a query looked at.
 *
 * <p>This is a reference implementation: selection masks the full score matrix; a kernel
 * would gather the chosen keys only.
 *
 * <p>Paper: DeepSeek-AI, "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models",
 * <a href="https://arxiv.org/abs/2512.02556">arXiv:2512.02556</a>.
 */
public final class IndexedSparseAttention implements Layer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final int topK;
    private final int warmupSteps;
    private final float klWeight;
    private final float rotaryBase;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final LightningIndexer indexer;
    private long trainingPasses;
    private Var lastIndexLoss;
    private double lastAttended;

    /**
     * Creates the layer.
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads the attention heads
     * @param topK keys per query after warm-up
     * @param warmupSteps training forward passes with dense attention
     * @param klWeight the weight of the indexer's distillation loss
     * @param rotaryBase the RoPE base, or 0 for none
     * @param random the random source
     */
    public IndexedSparseAttention(int features, int heads, int topK, int warmupSteps, float klWeight,
            float rotaryBase, RandomGenerator random) {
        if (features % heads != 0 || topK < 1) {
            throw new IllegalArgumentException("need heads dividing features and topK >= 1");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.topK = topK;
        this.warmupSteps = warmupSteps;
        this.klWeight = klWeight;
        this.rotaryBase = rotaryBase;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.wo = MixerOps.weight(random, features, features);
        this.indexer = new LightningIndexer(features, 2, Math.max(4, headSize / 2), true, random);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}: RoPE, KL weight 1.
     *
     * @param heads the attention heads
     * @param topK keys per query
     * @param warmupSteps dense warm-up passes
     * @return the spec
     */
    public static LayerSpec of(int heads, int topK, int warmupSteps) {
        return MixerOps.sameShape((in, r) -> new IndexedSparseAttention(in.dim(-1), heads, topK, warmupSteps, 1f,
                10000f, r));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        MixerOps.requireSequence(input, features, "IndexedSparseAttention");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var q = MixerOps.split(input.matmul(wq), heads);
        Var k = MixerOps.split(input.matmul(wk), heads);
        Var v = MixerOps.split(input.matmul(wv), heads);
        if (rotaryBase > 0) {
            q = Functional.rotary(q, rotaryBase, 0);
            k = Functional.rotary(k, rotaryBase, 0);
        }
        Var detached = input.detach();
        Var index = indexer.scores(detached, detached);                  // [b, t, t]
        float[] visible = SparseSelectOps.causal(t);
        boolean dense = training && trainingPasses < warmupSteps;
        float[] keep;
        if (dense) {
            keep = new float[b * t * t];
            for (int n = 0; n < b; n++) {
                System.arraycopy(visible, 0, keep, n * t * t, t * t);
            }
        } else {
            keep = SparseSelectOps.topKeep(index.value().toArray(), visible, null, b * t, t, topK);
        }
        lastAttended = SparseSelectOps.meanKept(keep, b * t);
        Tensor mask = SparseSelectOps.perBatch(keep, b, t, t);
        Var scores = q.matmul(k.transpose()).mul((float) (1 / Math.sqrt(headSize)));
        Var w = MixerOps.softmax(scores.add(RefinementOps.additive(mask)));
        if (training) {
            trainingPasses++;
            lastIndexLoss = LightningIndexer.distillationLoss(index, w.value(), mask.reshape(b, t, t))
                    .mul(klWeight);
        } else {
            lastIndexLoss = null;
        }
        return MixerOps.merge(w.matmul(v)).matmul(wo);
    }

    /** @return the indexer's distillation loss of the last training pass, weighted; {@code null} otherwise */
    @Override
    public Var auxiliaryLoss() {
        return lastIndexLoss;
    }

    /** @return the mean number of keys each query attended to in the last pass */
    public double lastAttended() {
        return lastAttended;
    }

    /** @return the training forward passes seen so far */
    public long trainingPasses() {
        return trainingPasses;
    }

    /** @return the indexer */
    public LightningIndexer indexer() {
        return indexer;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(wq, wk, wv, wo));
        p.addAll(indexer.parameters());
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "IndexedSparseAttention(" + features + ", heads=" + heads + ", top=" + topK + ")";
    }
}
