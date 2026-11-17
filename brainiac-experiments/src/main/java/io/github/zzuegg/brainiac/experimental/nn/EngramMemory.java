package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Engram-style conditional memory (DeepSeek 2026, "Conditional Memory via Scalable Lookup",
 * <a href="https://arxiv.org/abs/2601.07372">arXiv:2601.07372</a>): an O(1) lookup of the
 * n-grams ending at each position in large hashed embedding tables, a sparsity axis next to
 * mixture-of-experts. For each n-gram order and each of several hash functions, the last
 * {@code n} token ids are hashed to a row of that order's table; the rows are summed per
 * order and the orders concatenated, then projected to {@code dimension}. Static, local
 * knowledge (names, phrases, idioms) is looked up instead of recomputed, freeing the layers
 * for reasoning; the tables are plain arrays that can live in host memory.
 *
 * <p>As a {@link Layer} it maps token ids {@code [batch, steps]} to n-gram embeddings
 * {@code [batch, steps, dimension]}. {@link #fuse} adds them to a hidden state through the
 * paper's context-aware gate {@code alpha = sigmoid(<h W_k, e> / sqrt(d))}, so the model can
 * ignore a lookup that a hash collision made irrelevant.
 */
public final class EngramMemory implements Layer {

    private final int[] orders;
    private final int hashes;
    private final int rows;
    private final int tableSize;
    private final int dimension;
    private final List<Var> tables = new ArrayList<>();
    private final Var projection;
    private final Var gateKey;

    /**
     * Creates the memory.
     *
     * @param orders the n-gram orders, such as {@code {2, 3}}
     * @param hashes the hash functions per order (their rows are summed, which reduces the harm of collisions)
     * @param rows the rows per table
     * @param tableSize the features per row
     * @param dimension the output features
     * @param random the random source for the weights
     */
    public EngramMemory(int[] orders, int hashes, int rows, int tableSize, int dimension, SplittableRandom random) {
        if (orders.length == 0 || hashes < 1 || rows < 1) {
            throw new IllegalArgumentException("need orders, hashes and rows");
        }
        this.orders = orders.clone();
        this.hashes = hashes;
        this.rows = rows;
        this.tableSize = tableSize;
        this.dimension = dimension;
        for (int i = 0; i < orders.length; i++) {
            tables.add(Var.param(Tensor.normal(Shape.of(rows, tableSize), 0, 0.1f, random)));
        }
        int in = orders.length * tableSize;
        projection = Var.param(Init.XAVIER.create(Shape.of(in, dimension), in, dimension, random));
        gateKey = Var.param(Init.XAVIER.create(Shape.of(dimension, dimension), dimension, dimension, random));
    }

    /**
     * Describes the memory for model builders, on token ids {@code [steps]}.
     *
     * @param orders the n-gram orders
     * @param hashes the hash functions per order
     * @param rows the rows per table
     * @param dimension the output features (also the table width)
     * @return the spec
     */
    public static LayerSpec spec(int[] orders, int hashes, int rows, int dimension) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), dimension);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new EngramMemory(orders, hashes, rows, dimension, dimension, random);
            }
        };
    }

    /**
     * The table row of an n-gram for one hash function.
     *
     * @param ids the token ids
     * @param start the first position of the sequence (earlier positions count as padding)
     * @param end the n-gram's last position
     * @param n the order
     * @param hash which hash function
     * @param rows the table rows
     * @return the row
     */
    static int row(float[] ids, int start, int end, int n, int hash, int rows) {
        long h = 0x9E3779B97F4A7C15L * (hash + 1) + n;
        for (int i = end - n + 1; i <= end; i++) {
            long id = i < start ? -1 : (long) ids[i];
            h ^= id + 0x632BE59BD9B4E019L + (h << 6) + (h >>> 2);
            h *= 0xBF58476D1CE4E5B9L;
            h ^= h >>> 31;
        }
        return (int) Math.floorMod(h, (long) rows);
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        float[] ids = input.value().toArray();
        List<Var> parts = new ArrayList<>();
        for (int o = 0; o < orders.length; o++) {
            Var sum = null;
            for (int hsh = 0; hsh < hashes; hsh++) {
                float[] idx = new float[b * t];
                for (int s = 0; s < b; s++) {
                    for (int i = 0; i < t; i++) {
                        idx[s * t + i] = row(ids, s * t, s * t + i, orders[o], hsh, rows);
                    }
                }
                Var e = tables.get(o).gather(0, Tensor.of(Shape.of(b, t), idx, input.value().device()));
                sum = sum == null ? e : sum.add(e);
            }
            parts.add(sum);
        }
        Var cat = parts.size() == 1 ? parts.get(0) : Var.concat(-1, parts.toArray(Var[]::new));
        return cat.matmul(projection);
    }

    /**
     * Adds the n-gram memory to a hidden state through a context-aware gate:
     * {@code h + alpha e} with {@code alpha = sigmoid(<h W_k, e> / sqrt(d))} per position.
     *
     * @param hidden {@code [batch, steps, dimension]}
     * @param ids the token ids {@code [batch, steps]}
     * @return the fused hidden state
     */
    public Var fuse(Var hidden, Var ids) {
        Var e = forward(ids);
        Var alpha = hidden.matmul(gateKey).mul(e).sum(-1, true).mul((float) (1 / Math.sqrt(dimension))).sigmoid();
        return hidden.add(e.mul(alpha));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(tables);
        p.add(projection);
        p.add(gateKey);
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), dimension);
    }

    @Override
    public String toString() {
        return "EngramMemory(orders " + java.util.Arrays.toString(orders) + ", " + hashes + " hashes, " + rows
                + " rows)";
    }
}
