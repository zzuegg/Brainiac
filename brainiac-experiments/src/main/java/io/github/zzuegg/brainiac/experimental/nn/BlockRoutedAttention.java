package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * MoBA, mixture of block attention: the keys are cut into blocks of {@code block} tokens, and
 * a parameter-free gate routes each query, per head, to the {@code topBlocks - 1} past blocks
 * whose mean-pooled key has the highest dot product with the query, plus its own block
 * (causally masked inside). Attention then runs only over the routed blocks, so the cost per
 * query is {@code topBlocks * block} keys whatever the context length, and the layer can
 * switch between full and sparse attention without new parameters (set
 * {@code topBlocks} to cover every block for full attention).
 *
 * <p>The routing itself has no gradient, just like MoE top-k without the gate values; the
 * attention inside the chosen blocks trains the keys that the gate pools. This reference
 * implementation masks the full score matrix.
 *
 * <p>Paper: Lu et al., "MoBA: Mixture of Block Attention for Long-Context LLMs",
 * <a href="https://arxiv.org/abs/2502.13189">arXiv:2502.13189</a>.
 */
public final class BlockRoutedAttention implements Layer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final int block;
    private final int topBlocks;
    private final float rotaryBase;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private double lastAttended;

    /**
     * Creates the layer.
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads the attention heads
     * @param block the block length
     * @param topBlocks blocks per query, its own block included
     * @param rotaryBase the RoPE base, or 0 for none
     * @param random the random source
     */
    public BlockRoutedAttention(int features, int heads, int block, int topBlocks, float rotaryBase,
            RandomGenerator random) {
        if (features % heads != 0 || block < 1 || topBlocks < 1) {
            throw new IllegalArgumentException("need heads dividing features, block >= 1 and topBlocks >= 1");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.block = block;
        this.topBlocks = topBlocks;
        this.rotaryBase = rotaryBase;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.wo = MixerOps.weight(random, features, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}, with RoPE.
     *
     * @param heads the attention heads
     * @param block the block length
     * @param topBlocks blocks per query, its own included
     * @return the spec
     */
    public static LayerSpec of(int heads, int block, int topBlocks) {
        return MixerOps.sameShape((in, r) -> new BlockRoutedAttention(in.dim(-1), heads, block, topBlocks, 10000f,
                r));
    }

    /**
     * The routing mask: which keys every query of every head may see.
     *
     * @param q queries {@code [b, h, t, d]}
     * @param k keys {@code [b, h, t, d]}
     * @return a 0/1 mask {@code [b * h * t * t]}
     */
    float[] route(Tensor q, Tensor k) {
        int b = q.shape().dim(0);
        int t = q.shape().dim(2);
        int d = headSize;
        int blocks = (t + block - 1) / block;
        float[] qa = q.toArray();
        float[] ka = k.toArray();
        int rows = b * heads;
        float[] keep = new float[rows * t * t];
        float[] gate = new float[blocks];
        float[] mean = new float[blocks * d];
        Integer[] order = new Integer[blocks];
        for (int r = 0; r < rows; r++) {
            java.util.Arrays.fill(mean, 0f);
            for (int j = 0; j < t; j++) {
                int blk = j / block;
                for (int c = 0; c < d; c++) {
                    mean[blk * d + c] += ka[(r * t + j) * d + c];
                }
            }
            for (int i = 0; i < t; i++) {
                int current = i / block;
                int n = 0;
                for (int blk = 0; blk < current; blk++) {
                    float s = 0;
                    for (int c = 0; c < d; c++) {
                        s += qa[(r * t + i) * d + c] * mean[blk * d + c];
                    }
                    gate[blk] = s / block;
                    order[n++] = blk;
                }
                java.util.Arrays.sort(order, 0, n, (x, y) -> Float.compare(gate[y], gate[x]));
                int base = (r * t + i) * t;
                for (int j = current * block; j <= i; j++) {
                    keep[base + j] = 1;
                }
                for (int s = 0; s < Math.min(topBlocks - 1, n); s++) {
                    int blk = order[s];
                    for (int j = blk * block; j < (blk + 1) * block; j++) {
                        keep[base + j] = 1;
                    }
                }
            }
        }
        return keep;
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "BlockRoutedAttention");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var q = MixerOps.split(input.matmul(wq), heads);
        Var k = MixerOps.split(input.matmul(wk), heads);
        Var v = MixerOps.split(input.matmul(wv), heads);
        if (rotaryBase > 0) {
            q = Functional.rotary(q, rotaryBase, 0);
            k = Functional.rotary(k, rotaryBase, 0);
        }
        float[] keep = route(q.value(), k.value());
        lastAttended = SparseSelectOps.meanKept(keep, b * heads * t);
        Tensor mask = Tensor.of(Shape.of(b, heads, t, t), keep);
        Var scores = q.matmul(k.transpose()).mul((float) (1 / Math.sqrt(headSize)));
        Var w = MixerOps.softmax(scores.add(RefinementOps.additive(mask)));
        return MixerOps.merge(w.matmul(v)).matmul(wo);
    }

    /** @return the mean number of keys a query attended to in the last pass (per head) */
    public double lastAttended() {
        return lastAttended;
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, wo);
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "BlockRoutedAttention(" + features + ", heads=" + heads + ", block=" + block + ", top=" + topBlocks
                + ")";
    }
}
