package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * RETRO chunked cross-attention (Borgeaud et al. 2022, <a href="https://arxiv.org/abs/2112.04426">arXiv:2112.04426</a>):
 * the sequence is split into chunks of {@code m} tokens, and each chunk has retrieved
 * neighbours (already encoded). Query {@code i} attends to the neighbours of chunk
 * {@code (i - m + 1) / m}: the neighbours of a chunk are read from its last token on, so a
 * token never sees neighbours retrieved with tokens after it, and the first {@code m - 1}
 * queries read nothing. The result is added to the input (a residual block).
 *
 * <p>Inputs: hidden states {@code [batch, steps, features]} with {@code steps = chunks * m}
 * and neighbours {@code [batch, chunks, tokens, features]}, where {@code tokens} is the
 * neighbour tokens of a chunk (all its neighbours, each with its continuation, flattened).
 * Output {@code [batch, steps, features]}.
 */
public final class ChunkedCrossAttention implements MultiInputLayer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final int chunk;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;

    /**
     * Creates the layer.
     *
     * @param features the features of hidden states and neighbours
     * @param heads the heads
     * @param headSize the size per head
     * @param chunk the tokens per chunk {@code m}
     * @param random the random source for the weights
     */
    public ChunkedCrossAttention(int features, int heads, int headSize, int chunk, SplittableRandom random) {
        if (features < 1 || heads < 1 || headSize < 1 || chunk < 1) {
            throw new IllegalArgumentException("sizes must be positive");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = headSize;
        this.chunk = chunk;
        int hd = heads * headSize;
        wq = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wk = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wv = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wo = Var.param(Init.XAVIER.create(Shape.of(hd, features), hd, features, random));
    }

    /**
     * Describes the layer for graph builders.
     *
     * @param heads the heads
     * @param headSize the size per head
     * @param chunk the tokens per chunk
     * @return the spec
     */
    public static MultiInputLayer.Spec spec(int heads, int headSize, int chunk) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return inputs.getFirst();
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                return new ChunkedCrossAttention(inputs.getFirst().dim(-1), heads, headSize, chunk, random);
            }
        };
    }

    /**
     * The chunk whose neighbours query {@code i} reads.
     *
     * @param i the query position
     * @param m the chunk size
     * @return the chunk, or -1 for the first {@code m - 1} positions
     */
    static int chunkOf(int i, int m) {
        return i < m - 1 ? -1 : (i - m + 1) / m;
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException(this + " takes hidden states and neighbours");
        }
        Var hidden = inputs.get(0);
        Var neighbours = inputs.get(1);
        Shape hs = hidden.shape();
        Shape ns = neighbours.shape();
        if (hs.rank() != 3 || hs.dim(2) != features || ns.rank() != 4 || ns.dim(3) != features
                || ns.dim(0) != hs.dim(0)) {
            throw new IllegalArgumentException(this + " expects [b, T, " + features + "] and [b, C, N, " + features
                    + "], got " + hs + " and " + ns);
        }
        int n = hs.dim(0);
        int steps = hs.dim(1);
        int chunks = ns.dim(1);
        int tokens = ns.dim(2);
        if (chunks * chunk < steps) {
            throw new IllegalArgumentException("need " + Math.ceilDiv(steps, chunk) + " chunks of neighbours, got "
                    + chunks);
        }
        int keys = chunks * tokens;
        float[] bias = new float[steps * keys];
        float[] live = new float[steps];
        for (int i = 0; i < steps; i++) {
            int c = chunkOf(i, chunk);
            live[i] = c < 0 ? 0 : 1;
            for (int j = 0; j < keys; j++) {
                bias[i * keys + j] = c < 0 || j / tokens == c ? 0 : -1e9f;
            }
        }
        var device = hidden.value().device();
        Var flat = neighbours.reshape(n, keys, features);
        Var q = hidden.matmul(wq).reshape(n, steps, heads, headSize).permute(0, 2, 1, 3);
        Var k = flat.matmul(wk).reshape(n, keys, heads, headSize).permute(0, 2, 1, 3);
        Var v = flat.matmul(wv).reshape(n, keys, heads, headSize).permute(0, 2, 1, 3);
        Var att = Functional.attention(q, k, v, Tensor.of(Shape.of(steps, keys), bias, device),
                (float) (1 / Math.sqrt(headSize)), 0, null);
        Var out = att.permute(0, 2, 1, 3).reshape(n, steps, heads * headSize).matmul(wo);
        return hidden.add(out.mul(Tensor.of(Shape.of(steps, 1), live, device)));
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return inputs.getFirst();
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, wo);
    }

    @Override
    public String toString() {
        return "ChunkedCrossAttention(" + heads + " heads, chunk " + chunk + ")";
    }
}
