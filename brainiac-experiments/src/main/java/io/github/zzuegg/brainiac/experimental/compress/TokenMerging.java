package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.PatchEmbedding;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A small pre-norm Vision Transformer with token reduction between the attention and the MLP
 * of every block. The reductions are:
 * <ul>
 * <li>Token merging, ToMe (Bolya et al., ICLR 2023,
 *     <a href="https://arxiv.org/abs/2210.09461">arXiv:2210.09461</a>). Bipartite soft matching
 *     on the attention keys (averaged over the heads): the patch tokens alternate between sets A
 *     and B, and every A token finds its most cosine-similar B token. The {@code r} best pairs
 *     are merged by size-weighted averaging. Proportional attention then adds {@code log s} to
 *     the logits, so a merged token keeps the weight of the tokens it stands for. The class
 *     token is never merged.</li>
 * <li>Token pruning by class-token attention (the EViT/DynamicViT-style baseline): it keeps the
 *     tokens that the class token attends to most.</li>
 * </ul>
 *
 * <p>Both reductions are training-free. The merge is a constant matrix applied with an ordinary
 * batched matmul, so the reduced network can also be fine-tuned. The core
 * {@code VisionTransformer} keeps its blocks private and has no attention-bias hook, so this
 * class builds its own blocks from {@link PatchEmbedding} and {@link LayerNorm}.
 */
public final class TokenMerging implements Layer {

    /** Token reduction between attention and MLP. */
    public enum Mode {
        /** No reduction. */
        NONE,
        /** ToMe bipartite soft matching with proportional attention. */
        MERGE,
        /** Keep the tokens with the highest class-token attention. */
        PRUNE
    }

    private final int channels;
    private final int side;
    private final int patch;
    private final int dim;
    private final int heads;
    private final int classes;
    private final PatchEmbedding embedding;
    private final List<LayerNorm> norms = new ArrayList<>();
    private final List<Var[]> blocks = new ArrayList<>();
    private final LayerNorm finalNorm;
    private final Var head;
    private final Var headBias;
    private Mode mode = Mode.NONE;
    private int perLayer;

    /**
     * @param channels image channels
     * @param side image side
     * @param patch the patch side
     * @param dim the token width
     * @param depth the number of blocks
     * @param heads attention heads
     * @param classes the classes of the head
     * @param seed the seed
     */
    public TokenMerging(int channels, int side, int patch, int dim, int depth, int heads, int classes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.channels = channels;
        this.side = side;
        this.patch = patch;
        this.dim = dim;
        this.heads = heads;
        this.classes = classes;
        this.embedding = new PatchEmbedding(channels, side, side, patch, dim, true, true, r);
        for (int i = 0; i < depth; i++) {
            norms.add(LayerNorm.of(dim));
            norms.add(LayerNorm.of(dim));
            blocks.add(new Var[] {weight(dim, dim, r), weight(dim, dim, r), weight(dim, dim, r), weight(dim, dim, r),
                weight(dim, 2 * dim, r), Var.param(Tensor.zeros(Shape.of(2 * dim))), weight(2 * dim, dim, r),
                Var.param(Tensor.zeros(Shape.of(dim)))});
        }
        this.finalNorm = LayerNorm.of(dim);
        this.head = weight(dim, classes, r);
        this.headBias = Var.param(Tensor.zeros(Shape.of(classes)));
    }

    private static Var weight(int in, int out, SplittableRandom r) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0, (float) Math.sqrt(1.0 / in), r));
    }

    /**
     * @param m the reduction
     * @param r tokens removed per block
     * @return this
     */
    public TokenMerging reduce(Mode m, int r) {
        this.mode = m;
        this.perLayer = m == Mode.NONE ? 0 : r;
        return this;
    }

    /** @return a copy with the same weights and reduction */
    public TokenMerging copy() {
        TokenMerging c = new TokenMerging(channels, side, patch, dim, blocks.size(), heads, classes, 0);
        List<Var> from = parameters();
        List<Var> to = c.parameters();
        for (int i = 0; i < from.size(); i++) {
            to.get(i).assign(from.get(i).value());
        }
        return c.reduce(mode, perLayer);
    }

    @Override
    public Var forward(Var images) {
        Var x = embedding.forward(images);
        int b = x.shape().dim(0);
        int n = x.shape().dim(1);
        float[] size = new float[b * n];
        Arrays.fill(size, 1f);
        int dh = dim / heads;
        for (int l = 0; l < blocks.size(); l++) {
            Var[] w = blocks.get(l);
            Var h = norms.get(2 * l).forward(x);
            Var k = h.matmul(w[1]);
            Var q = heads(h.matmul(w[0]), b, n, dh);
            Var kh = heads(k, b, n, dh);
            Var v = heads(h.matmul(w[2]), b, n, dh);
            Var logits = q.matmul(kh.permute(0, 1, 3, 2)).mul((float) (1 / Math.sqrt(dh)));
            if (mode == Mode.MERGE) {
                float[] logSize = new float[b * n];
                for (int i = 0; i < logSize.length; i++) {
                    logSize[i] = (float) Math.log(size[i]);
                }
                logits = logits.add(Tensor.of(Shape.of(b, 1, 1, n), logSize));
            }
            Var att = logits.sub(logits.logsumexp(-1, true)).exp();
            Var o = att.matmul(v).permute(0, 2, 1, 3).reshape(b, n, dim).matmul(w[3]);
            x = x.add(o);
            int r = Math.min(perLayer, (n - 1) / 2);
            if (r > 0) {
                int keep = n - r;
                float[] mix = new float[b * keep * n];
                float[] newSize = new float[b * keep];
                float[] kv = k.value().toArray();
                float[] av = mode == Mode.PRUNE ? att.value().toArray() : null;
                for (int s = 0; s < b; s++) {
                    int[][] groups = mode == Mode.MERGE ? bipartiteMatch(kv, s * n * dim, n, dim, r)
                            : pruneByClass(av, s, n, r);
                    for (int g = 0; g < keep; g++) {
                        float total = 0;
                        for (int t : groups[g]) {
                            total += size[s * n + t];
                        }
                        for (int t : groups[g]) {
                            mix[(s * keep + g) * n + t] = size[s * n + t] / total;
                        }
                        newSize[s * keep + g] = mode == Mode.MERGE ? total : 1f;
                    }
                }
                x = Var.of(Tensor.of(Shape.of(b, keep, n), mix)).matmul(x);
                size = newSize;
                n = keep;
            }
            Var m = norms.get(2 * l + 1).forward(x).matmul(w[4]).add(w[5]).relu().matmul(w[6]).add(w[7]);
            x = x.add(m);
        }
        Var cls = finalNorm.forward(x).slice(1, 0, 1).reshape(b, dim);
        return cls.matmul(head).add(headBias);
    }

    private Var heads(Var t, int b, int n, int dh) {
        return t.reshape(b, n, heads, dh).permute(0, 2, 1, 3);
    }

    /**
     * Bipartite soft matching of one sample's tokens; token 0 (the class token) stays alone.
     *
     * @param keys the keys, {@code n x dim} from {@code offset}
     * @param offset the sample's offset
     * @param n tokens
     * @param dim key width
     * @param r pairs to merge
     * @return the {@code n - r} output tokens as groups of input tokens
     */
    public static int[][] bipartiteMatch(float[] keys, int offset, int n, int dim, int r) {
        double[] norm = new double[n];
        for (int t = 0; t < n; t++) {
            double s = 0;
            for (int j = 0; j < dim; j++) {
                s += keys[offset + t * dim + j] * keys[offset + t * dim + j];
            }
            norm[t] = Math.sqrt(s) + 1e-12;
        }
        List<Integer> a = new ArrayList<>();
        List<Integer> bSet = new ArrayList<>();
        for (int t = 1; t < n; t++) {
            ((t - 1) % 2 == 0 ? a : bSet).add(t);
        }
        int[] best = new int[a.size()];
        double[] score = new double[a.size()];
        for (int i = 0; i < a.size(); i++) {
            score[i] = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < bSet.size(); j++) {
                double c = 0;
                int p = offset + a.get(i) * dim;
                int q = offset + bSet.get(j) * dim;
                for (int e = 0; e < dim; e++) {
                    c += keys[p + e] * keys[q + e];
                }
                c /= norm[a.get(i)] * norm[bSet.get(j)];
                if (c > score[i]) {
                    score[i] = c;
                    best[i] = j;
                }
            }
        }
        Integer[] order = new Integer[a.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (x, y) -> Double.compare(score[y], score[x]));
        boolean[] merged = new boolean[a.size()];
        for (int i = 0; i < r; i++) {
            merged[order[i]] = true;
        }
        List<List<Integer>> groups = new ArrayList<>();
        groups.add(new ArrayList<>(List.of(0)));
        for (int i = 0; i < a.size(); i++) {
            if (!merged[i]) {
                groups.add(new ArrayList<>(List.of(a.get(i))));
            }
        }
        int base = groups.size();
        for (int j : bSet) {
            groups.add(new ArrayList<>(List.of(j)));
        }
        for (int i = 0; i < a.size(); i++) {
            if (merged[i]) {
                groups.get(base + best[i]).add(a.get(i));
            }
        }
        int[][] out = new int[groups.size()][];
        for (int g = 0; g < out.length; g++) {
            out[g] = groups.get(g).stream().mapToInt(Integer::intValue).toArray();
        }
        return out;
    }

    private int[][] pruneByClass(float[] att, int s, int n, int r) {
        double[] score = new double[n];
        for (int h = 0; h < heads; h++) {
            int row = ((s * heads + h) * n) * n;
            for (int t = 1; t < n; t++) {
                score[t] += att[row + t];
            }
        }
        Integer[] order = new Integer[n - 1];
        for (int i = 0; i < n - 1; i++) {
            order[i] = i + 1;
        }
        Arrays.sort(order, (x, y) -> Double.compare(score[y], score[x]));
        int[] kept = new int[n - 1 - r];
        for (int i = 0; i < kept.length; i++) {
            kept[i] = order[i];
        }
        Arrays.sort(kept);
        int[][] out = new int[n - r][];
        out[0] = new int[] {0};
        for (int i = 0; i < kept.length; i++) {
            out[i + 1] = new int[] {kept[i]};
        }
        return out;
    }

    /**
     * Multiply-adds per image at the current reduction: projections, attention and MLP of every
     * block plus the patch embedding and head (the merge matrices are not counted).
     *
     * @return the multiply-adds
     */
    public long macs() {
        int n = (side / patch) * (side / patch) + 1;
        long macs = (long) (n - 1) * channels * patch * patch * dim + (long) dim * classes;
        for (int l = 0; l < blocks.size(); l++) {
            macs += 4L * n * dim * dim + 2L * n * n * dim;
            n -= Math.min(perLayer, (n - 1) / 2);
            macs += 4L * n * dim * dim;
        }
        return macs;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(embedding.parameters());
        for (int l = 0; l < blocks.size(); l++) {
            p.addAll(norms.get(2 * l).parameters());
            p.addAll(norms.get(2 * l + 1).parameters());
            p.addAll(List.of(blocks.get(l)));
        }
        p.addAll(finalNorm.parameters());
        p.add(head);
        p.add(headBias);
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(classes);
    }

    @Override
    public String toString() {
        return "TokenMerging(" + mode + ", r=" + perLayer + ")";
    }
}
