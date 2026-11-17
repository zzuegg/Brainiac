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
 * Native Sparse Attention (NSA): causal attention through three branches whose outputs a
 * learned sigmoid gate mixes per head and step:
 *
 * <ol>
 *   <li><b>compressed</b>: every block of {@code block} keys and values is compressed by a
 *       learned linear map (with intra-block positions) into one coarse token; each query
 *       attends to all complete blocks before it, for a cheap global view;</li>
 *   <li><b>selected</b>: the compressed attention probabilities, summed over heads, rank
 *       the blocks; each query attends at full resolution to its {@code topBlocks} best
 *       blocks (the first and the current block always included);</li>
 *   <li><b>sliding window</b>: the last {@code window} tokens, for local patterns.</li>
 * </ol>
 *
 * <p>A learned "null" compressed token is always visible, so queries before the first
 * complete block still have something to attend to. This is a reference implementation:
 * the selected and window branches use masks over the full {@code [steps, steps]} score
 * matrix; a fused kernel would gather only the chosen blocks, which is where NSA's speed
 * comes from. The selection itself is not differentiable; gradients reach the ranking
 * through the compressed branch, as in the paper.
 *
 * <p>Paper: Yuan et al., "Native Sparse Attention: Hardware-Aligned and Natively Trainable
 * Sparse Attention", <a href="https://arxiv.org/abs/2502.11089">arXiv:2502.11089</a>.
 */
public final class NativeSparseAttention implements Layer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final int block;
    private final int topBlocks;
    private final int window;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var compressK;
    private final Var compressV;
    private final Var blockPosition;
    private final Var nullKey;
    private final Var wg;
    private final Var bg;

    /**
     * Creates the layer.
     *
     * @param features the model width, a multiple of {@code heads}
     * @param heads the number of heads
     * @param block the compression and selection block length
     * @param topBlocks how many blocks each query selects
     * @param window the sliding window length
     * @param random the random source
     */
    public NativeSparseAttention(int features, int heads, int block, int topBlocks, int window,
            RandomGenerator random) {
        if (features % heads != 0) {
            throw new IllegalArgumentException("features " + features + " not divisible by heads " + heads);
        }
        if (block < 1 || topBlocks < 1 || window < 1) {
            throw new IllegalArgumentException("block, topBlocks and window must be positive");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.block = block;
        this.topBlocks = topBlocks;
        this.window = window;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.wo = MixerOps.weight(random, features, features);
        this.compressK = MixerOps.weight(random, block * headSize, headSize);
        this.compressV = MixerOps.weight(random, block * headSize, headSize);
        this.blockPosition = MixerOps.uniform(random, 0.1f, block * headSize);
        this.nullKey = MixerOps.uniform(random, 0.1f, headSize);
        this.wg = MixerOps.weight(random, features, 3 * heads);
        this.bg = MixerOps.filled(0f, 3 * heads);
    }

    /** Intermediate results of one forward pass. */
    private record Parts(Var q, Var k, Var v, Var compressed, int[][][] selected) {
    }

    /**
     * Returns the blocks each query selects.
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch][step][]} block indices
     */
    public int[][][] selectedBlocks(Var input) {
        return parts(input).selected();
    }

    private Parts parts(Var x) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        Var q = MixerOps.split(x.matmul(wq), heads);
        Var k = MixerOps.split(x.matmul(wk), heads);
        Var v = MixerOps.split(x.matmul(wv), heads);
        float scale = (float) (1 / Math.sqrt(headSize));
        int nb = t / block;
        Var nullK = nullKey.reshape(1, 1, 1, headSize).expand(Shape.of(b, heads, 1, headSize));
        Var nullV = Var.of(Tensor.zeros(Shape.of(b, heads, 1, headSize)));
        Var kc = nullK;
        Var vc = nullV;
        if (nb > 0) {
            Var kb = k.slice(2, 0, nb * block).reshape(b, heads, nb, block * headSize).add(blockPosition);
            Var vb = v.slice(2, 0, nb * block).reshape(b, heads, nb, block * headSize).add(blockPosition);
            kc = Var.concat(2, nullK, kb.matmul(compressK));
            vc = Var.concat(2, nullV, vb.matmul(compressV));
        }
        float[] mask = new float[t * (nb + 1)];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < nb; j++) {
                if ((j + 1) * block - 1 > i) {
                    mask[i * (nb + 1) + j + 1] = -1e9f;
                }
            }
        }
        Var pc = MixerOps.softmax(q.matmul(kc.transpose()).mul(scale).add(Tensor.of(Shape.of(t, nb + 1), mask)));
        Var compressed = pc.matmul(vc);
        // rank blocks by compressed attention, summed over heads
        float[] importance = pc.value().sum(1, false).toArray();   // [b, t, nb + 1]
        int blocks = (t + block - 1) / block;
        int[][][] selected = new int[b][t][];
        for (int n = 0; n < b; n++) {
            for (int i = 0; i < t; i++) {
                int current = i / block;
                boolean[] chosen = new boolean[blocks];
                chosen[current] = true;
                chosen[0] = true;
                int count = current == 0 ? 1 : 2;
                while (count < Math.min(topBlocks, current + 1)) {
                    int best = -1;
                    for (int j = 0; j < current; j++) {
                        if (!chosen[j] && (best < 0 || importance[(n * t + i) * (nb + 1) + j + 1]
                                > importance[(n * t + i) * (nb + 1) + best + 1])) {
                            best = j;
                        }
                    }
                    chosen[best] = true;
                    count++;
                }
                int[] list = new int[count];
                int c = 0;
                for (int j = 0; j < blocks; j++) {
                    if (chosen[j]) {
                        list[c++] = j;
                    }
                }
                selected[n][i] = list;
            }
        }
        return new Parts(q, k, v, compressed, selected);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}, reading the width from the
     * input {@code [steps, features]}.
     *
     * @param heads the number of heads
     * @param block the block length
     * @param topBlocks how many blocks each query selects
     * @param window the sliding window length
     * @return the spec
     */
    public static LayerSpec of(int heads, int block, int topBlocks, int window) {
        return MixerOps.sameShape((in, r) -> new NativeSparseAttention(in.dim(-1), heads, block, topBlocks, window, r));
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "NativeSparseAttention");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Parts p = parts(input);
        float scale = (float) (1 / Math.sqrt(headSize));
        float[] sel = new float[b * t * t];
        java.util.Arrays.fill(sel, -1e9f);
        for (int n = 0; n < b; n++) {
            for (int i = 0; i < t; i++) {
                for (int blk : p.selected()[n][i]) {
                    for (int j = blk * block; j < Math.min(t, (blk + 1) * block) && j <= i; j++) {
                        sel[(n * t + i) * t + j] = 0;
                    }
                }
            }
        }
        float[] win = new float[t * t];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < t; j++) {
                win[i * t + j] = j <= i && j > i - window ? 0 : -1e9f;
            }
        }
        Var selected = Functional.attention(p.q(), p.k(), p.v(), Tensor.of(Shape.of(b, 1, t, t), sel), scale, 0, null);
        Var local = Functional.attention(p.q(), p.k(), p.v(), Tensor.of(Shape.of(t, t), win), scale, 0, null);
        Var g = input.matmul(wg).add(bg).sigmoid().reshape(b, t, heads, 3).permute(0, 2, 1, 3);   // [b, h, t, 3]
        Var o = p.compressed().mul(g.slice(3, 0, 1))
                .add(selected.mul(g.slice(3, 1, 2)))
                .add(local.mul(g.slice(3, 2, 3)));
        return MixerOps.merge(o).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, wo, compressK, compressV, blockPosition, nullKey, wg, bg);
    }

    @Override
    public String toString() {
        return "NativeSparseAttention(" + features + ", heads=" + heads + ", block=" + block + ", top="
                + topBlocks + ", window=" + window + ")";
    }
}
