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
 * Compressed attention of DeepSeek-V4: the KV cache holds one entry per block of
 * {@code ratio} tokens instead of one per token, plus a short sliding window of raw entries.
 *
 * <ul>
 *   <li>Every token writes one shared key/value vector {@code c = x W_kv} (multi-query, the
 *       same vector serves as key and value). A block of {@code ratio} tokens is compressed
 *       by <b>gated pooling</b>: per channel, a softmax over the block's positions of
 *       {@code x W_z + B_pos} weighs the block's vectors.</li>
 *   <li>{@link Mode#CSA} (compressed sparse attention, small ratio such as 4): a
 *       {@link LightningIndexer} scores the compressed entries, and each query attends to its
 *       {@code topK} best complete blocks; the indexer distils the main attention (its loss
 *       is the {@link #auxiliaryLoss()}).</li>
 *   <li>{@link Mode#HCA} (heavily compressed attention, large ratio such as 128): each query
 *       attends densely to every complete compressed block.</li>
 *   <li>Both add a sliding window over the last {@code window} raw entries (covering the
 *       block in progress) and a learned per-head sink logit, all in one softmax.</li>
 * </ul>
 *
 * <p>V4 alternates CSA and HCA layers ({@link #alternating}). The cache grows by
 * {@code kvDim / ratio} numbers per token ({@link #cachedPerToken()}), against
 * {@code 2 * features} for multi-head attention.
 *
 * <p>Source: DeepSeek-AI, DeepSeek-V4 technical report, summarized in
 * <a href="https://huggingface.co/blog/deepseekv4">huggingface.co/blog/deepseekv4</a>.
 */
public final class CompressedBlockAttention implements Layer {

    /** Which compression. */
    public enum Mode {
        /** Compressed sparse attention: small blocks, top-k selection by an indexer. */
        CSA,
        /** Heavily compressed attention: large blocks, dense over all of them. */
        HCA
    }

    private final int features;
    private final int heads;
    private final int kvDim;
    private final Mode mode;
    private final int ratio;
    private final int topK;
    private final int window;
    private final Var wq;
    private final Var wkv;
    private final Var wz;
    private final Var positionBias;
    private final Var sink;
    private final Var wo;
    private final LightningIndexer indexer;
    private Var lastIndexLoss;
    private double lastAttended;

    /**
     * Creates the layer.
     *
     * @param features the model width
     * @param heads the query heads
     * @param kvDim the width of the shared key/value vector (and of every query head)
     * @param mode CSA or HCA
     * @param ratio tokens per compressed block
     * @param topK blocks per query (CSA only)
     * @param window the sliding window of raw entries
     * @param random the random source
     */
    public CompressedBlockAttention(int features, int heads, int kvDim, Mode mode, int ratio, int topK, int window,
            RandomGenerator random) {
        if (ratio < 1 || window < 1 || heads < 1 || kvDim < 1 || (mode == Mode.CSA && topK < 1)) {
            throw new IllegalArgumentException("ratio, window, heads, kvDim (and topK for CSA) must be positive");
        }
        this.features = features;
        this.heads = heads;
        this.kvDim = kvDim;
        this.mode = mode;
        this.ratio = ratio;
        this.topK = topK;
        this.window = window;
        this.wq = MixerOps.weight(random, features, heads * kvDim);
        this.wkv = MixerOps.weight(random, features, kvDim);
        this.wz = MixerOps.uniform(random, 0.01f, features, kvDim);
        this.positionBias = MixerOps.filled(0f, ratio, kvDim);
        this.sink = MixerOps.filled(0f, heads);
        this.wo = MixerOps.weight(random, heads * kvDim, features);
        this.indexer = mode == Mode.CSA ? new LightningIndexer(features, kvDim, 2, Math.max(4, kvDim / 2), true,
                random) : null;
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param heads the query heads
     * @param kvDim the shared key/value width
     * @param mode CSA or HCA
     * @param ratio tokens per block
     * @param topK blocks per query (CSA)
     * @param window the sliding window
     * @return the spec
     */
    public static LayerSpec of(int heads, int kvDim, Mode mode, int ratio, int topK, int window) {
        return MixerOps.sameShape((in, r) -> new CompressedBlockAttention(in.dim(-1), heads, kvDim, mode, ratio, topK,
                window, r));
    }

    /**
     * V4's hybrid: {@code depth} pre-norm blocks ({@link ResidualMixerBlock}) alternating CSA
     * (even layers) and HCA (odd layers).
     *
     * @param depth the number of blocks
     * @param heads the query heads
     * @param kvDim the shared key/value width
     * @param csaRatio the CSA block length
     * @param topK CSA blocks per query
     * @param hcaRatio the HCA block length
     * @param window the sliding window
     * @return the spec
     */
    public static LayerSpec alternating(int depth, int heads, int kvDim, int csaRatio, int topK, int hcaRatio,
            int window) {
        LayerSpec[] blocks = new LayerSpec[depth];
        for (int l = 0; l < depth; l++) {
            blocks[l] = ResidualMixerBlock.of(l % 2 == 0 ? of(heads, kvDim, Mode.CSA, csaRatio, topK, window)
                    : of(heads, kvDim, Mode.HCA, hcaRatio, 0, window));
        }
        return LayerChain.of(blocks);
    }

    /**
     * Gated pooling of the complete blocks.
     *
     * @param x the input {@code [b, t, features]}
     * @param c the key/value vectors {@code [b, t, kvDim]}
     * @return {@code [b, blocks, kvDim]}, or {@code null} without a complete block
     */
    private Var compress(Var x, Var c) {
        int b = x.shape().dim(0);
        int nb = x.shape().dim(1) / ratio;
        if (nb == 0) {
            return null;
        }
        Var z = x.slice(1, 0, nb * ratio).matmul(wz).reshape(b, nb, ratio, kvDim).add(positionBias);
        Var w = MixerOps.softmax(z.permute(0, 1, 3, 2)).permute(0, 1, 3, 2);   // softmax over positions
        return c.slice(1, 0, nb * ratio).reshape(b, nb, ratio, kvDim).mul(w).sum(2, false);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        MixerOps.requireSequence(input, features, "CompressedBlockAttention");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        int h = heads;
        float scale = (float) (1 / Math.sqrt(kvDim));
        Var q = MixerOps.split(input.matmul(wq), h);                              // [b, h, t, c]
        Var c = input.matmul(wkv);                                                // [b, t, c]
        Var blocks = compress(input, c);
        int nb = blocks == null ? 0 : blocks.shape().dim(1);
        // keys: [compressed blocks | raw tokens], one softmax, plus a sink logit
        float[] visible = new float[t * (nb + t)];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < nb; j++) {
                visible[i * (nb + t) + j] = (j + 1) * ratio <= i + 1 ? 1 : 0;
            }
            for (int j = 0; j < t; j++) {
                visible[i * (nb + t) + nb + j] = j <= i && j > i - window ? 1 : 0;
            }
        }
        float[] keep = new float[b * t * (nb + t)];
        Var index = null;
        if (mode == Mode.CSA && nb > 0) {
            index = indexer.scores(input.detach(), blocks.detach());              // [b, t, nb]
            float[] blockVisible = new float[t * nb];
            for (int i = 0; i < t; i++) {
                System.arraycopy(visible, i * (nb + t), blockVisible, i * nb, nb);
            }
            float[] chosen = SparseSelectOps.topKeep(index.value().toArray(), blockVisible, null, b * t, nb, topK);
            for (int r = 0; r < b * t; r++) {
                System.arraycopy(chosen, r * nb, keep, r * (nb + t), nb);
                System.arraycopy(visible, (r % t) * (nb + t) + nb, keep, r * (nb + t) + nb, t);
            }
        } else {
            for (int n = 0; n < b; n++) {
                System.arraycopy(visible, 0, keep, n * t * (nb + t), t * (nb + t));
            }
        }
        lastAttended = SparseSelectOps.meanKept(keep, b * t);
        Var kv = blocks == null ? c : Var.concat(1, blocks, c);                  // [b, nb + t, c]
        Var kvh = kv.reshape(b, 1, nb + t, kvDim);
        Var scores = q.matmul(kvh.transpose()).mul(scale);                        // [b, h, t, nb + t]
        Tensor mask = Tensor.of(Shape.of(b, 1, t, nb + t), keep);
        scores = Var.concat(-1, scores.add(RefinementOps.additive(mask)),
                sink.reshape(1, h, 1, 1).expand(Shape.of(b, h, t, 1)));
        Var w = MixerOps.softmax(scores).slice(-1, 0, nb + t);
        if (training && index != null) {
            Tensor blockWeights = w.value().slice(-1, 0, nb);
            Tensor blockKeep = mask.slice(-1, 0, nb).reshape(b, t, nb);
            lastIndexLoss = LightningIndexer.distillationLoss(index, blockWeights, blockKeep);
        } else {
            lastIndexLoss = null;
        }
        return MixerOps.merge(w.matmul(kvh)).matmul(wo);
    }

    /** @return the CSA indexer's distillation loss of the last training pass, or {@code null} */
    @Override
    public Var auxiliaryLoss() {
        return lastIndexLoss;
    }

    /** @return the mean number of entries (blocks and window tokens) each query attended to last */
    public double lastAttended() {
        return lastAttended;
    }

    /** @return cached numbers per token in the long run: {@code kvDim / ratio} (the window is constant) */
    public double cachedPerToken() {
        return kvDim / (double) ratio;
    }

    /** @return cached numbers of the sliding window, independent of the context length */
    public int windowCache() {
        return window * kvDim;
    }

    /** @return CSA or HCA */
    public Mode mode() {
        return mode;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(wq, wkv, wz, positionBias, sink, wo));
        if (indexer != null) {
            p.addAll(indexer.parameters());
        }
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "CompressedBlockAttention(" + mode + ", ratio=" + ratio + ", kv=" + kvDim + ", window=" + window + ")";
    }
}
