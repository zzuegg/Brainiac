package io.github.zzuegg.brainiac.experimental.parallel;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A transformer block split over {@code t} ranks the Megatron-LM way, optionally with sequence
 * parallelism.
 *
 * <p>Tensor parallelism: the attention is split by heads (each rank holds the query, key and
 * value columns of {@code heads / t} heads and the matching rows of the output projection) and
 * the MLP by hidden units (column-parallel first matrix, row-parallel second matrix). Each
 * half then needs one all-reduce forward (operator {@code g}: sum the partial outputs) and one
 * backward (operator {@code f}: sum the input gradients), four per block per step. Layer norms
 * and residuals stay replicated on every rank.
 *
 * <p>Sequence parallelism (Korthikanti et al.) additionally shards the layer-norm and residual
 * regions along the sequence: the block's input and output are {@code [rows / t, width]} per
 * rank, each all-reduce becomes a reduce-scatter (leaving the region) plus an all-gather
 * (entering it), the same bytes, and the replicated activations shrink by {@code t}. The layer
 * norm gains' gradients are then partial sums, all-reduced by {@link Shard#syncNormGradients}.
 *
 * <p>Rows are sequence-major ({@code [sequence, batch]} flattened), so a sequence shard is a
 * contiguous block of rows. With one rank the block is the ordinary single-device block, which
 * is the reference for the exactness check.
 *
 * <p>Papers: Shoeybi et al., "Megatron-LM", <a href="https://arxiv.org/abs/1909.08053">arXiv:1909.08053</a>;
 * Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models",
 * <a href="https://arxiv.org/abs/2205.05198">arXiv:2205.05198</a>.
 */
public final class TensorParallelBlock {

    private final int width;
    private final int heads;
    private final int hidden;
    private final boolean causal;
    private final Tensor wq;
    private final Tensor wk;
    private final Tensor wv;
    private final Tensor wo;
    private final Tensor w1;
    private final Tensor w2;

    /**
     * Creates the full weights.
     *
     * @param width the model width
     * @param heads attention heads
     * @param hidden the MLP width
     * @param causal causal attention
     * @param random initialises the weights
     */
    public TensorParallelBlock(int width, int heads, int hidden, boolean causal, RandomGenerator random) {
        if (width % heads != 0) {
            throw new IllegalArgumentException("width must divide into heads");
        }
        this.width = width;
        this.heads = heads;
        this.hidden = hidden;
        this.causal = causal;
        float a = (float) (1 / Math.sqrt(width));
        float h = (float) (1 / Math.sqrt(hidden));
        this.wq = Tensor.normal(Shape.of(width, width), 0, a, random);
        this.wk = Tensor.normal(Shape.of(width, width), 0, a, random);
        this.wv = Tensor.normal(Shape.of(width, width), 0, a, random);
        this.wo = Tensor.normal(Shape.of(width, width), 0, a, random);
        this.w1 = Tensor.normal(Shape.of(width, hidden), 0, a, random);
        this.w2 = Tensor.normal(Shape.of(hidden, width), 0, h, random);
    }

    /**
     * Creates one rank's shard.
     *
     * @param rank the rank
     * @param collectives the group, whose size {@code t} divides the heads and the hidden width
     * @param sequenceParallel whether norm and residual regions are sharded along the sequence
     * @return the shard
     */
    public Shard shard(int rank, Collectives collectives, boolean sequenceParallel) {
        int t = collectives.ranks();
        if (heads % t != 0 || hidden % t != 0) {
            throw new IllegalArgumentException("heads and hidden must divide over " + t + " ranks");
        }
        return new Shard(rank, collectives, sequenceParallel);
    }

    /**
     * Parameters one rank holds.
     *
     * @param t ranks
     * @return the parameter count
     */
    public long parametersPerRank(int t) {
        return (4L * width * width + 2L * width * hidden) / t + 2L * width;
    }

    /**
     * Activation floats one rank keeps for the backward pass of one block, following the
     * accounting of Korthikanti et al.: the norm and residual regions ({@code rows x width} each:
     * two norm inputs, two norm outputs, the attention-output residual) are replicated under
     * tensor parallelism and divided by {@code t} under sequence parallelism; queries, keys,
     * values, attention scores and the MLP hidden layer are always divided by {@code t}.
     *
     * @param sequence the sequence length
     * @param batch the batch size
     * @param t ranks
     * @param sequenceParallel whether the norm regions are sharded
     * @return floats per rank
     */
    public long activationFloatsPerRank(int sequence, int batch, int t, boolean sequenceParallel) {
        long rows = (long) sequence * batch;
        long replicated = 5 * rows * width;
        long split = 4 * rows * width + (long) batch * heads * sequence * sequence + 2 * rows * hidden;
        return (sequenceParallel ? replicated / t : replicated) + split / t;
    }

    private static Tensor columns(Tensor w, int rank, int t) {
        int c = w.shape().dim(1) / t;
        return w.slice(1, rank * c, (rank + 1) * c).copy();
    }

    private static Tensor rows(Tensor w, int rank, int t) {
        int r = w.shape().dim(0) / t;
        return w.slice(0, rank * r, (rank + 1) * r).copy();
    }

    /** One rank's part of the block; a {@link Layer} over that rank's rows. */
    public final class Shard implements Layer {

        private final int rank;
        private final Collectives c;
        private final boolean sp;
        private final int t;
        private final Var q;
        private final Var k;
        private final Var v;
        private final Var o;
        private final Var up;
        private final Var down;
        private final Var gain1;
        private final Var gain2;

        Shard(int rank, Collectives c, boolean sp) {
            this.rank = rank;
            this.c = c;
            this.sp = sp;
            this.t = c.ranks();
            this.q = Var.param(columns(wq, rank, t));
            this.k = Var.param(columns(wk, rank, t));
            this.v = Var.param(columns(wv, rank, t));
            this.o = Var.param(rows(wo, rank, t));
            this.up = Var.param(columns(w1, rank, t));
            this.down = Var.param(rows(w2, rank, t));
            this.gain1 = Var.param(Tensor.ones(Shape.of(width)));
            this.gain2 = Var.param(Tensor.ones(Shape.of(width)));
        }

        /** Megatron's f: identity forward, all-reduce of the gradient backward. */
        private Var f(Var x) {
            return Var.custom(in -> in[0], (in, out, g) -> new Tensor[] {c.allReduce(rank, g)}, x);
        }

        /** Megatron's g: all-reduce forward, identity backward. */
        private Var g(Var x) {
            return Var.custom(in -> c.allReduce(rank, in[0]), (in, out, g) -> new Tensor[] {g}, x);
        }

        /** Entering a tensor-parallel region under sequence parallelism: all-gather rows. */
        private Var gather(Var x) {
            return Var.custom(in -> c.allGather(rank, in[0]), (in, out, g) -> new Tensor[] {c.reduceScatter(rank, g)},
                    x);
        }

        /** Leaving it: reduce-scatter rows. */
        private Var scatter(Var x) {
            return Var.custom(in -> c.reduceScatter(rank, in[0]), (in, out, g) -> new Tensor[] {c.allGather(rank, g)},
                    x);
        }

        private Var enter(Var x) {
            return sp ? gather(x) : f(x);
        }

        private Var leave(Var x) {
            return sp ? scatter(x) : g(x);
        }

        private Var norm(Var x, Var gain) {
            Var mean = x.mean(1, true);
            Var centred = x.sub(mean);
            return centred.div(centred.square().mean(1, true).add(1e-5f).sqrt()).mul(gain);
        }

        /**
         * Forward over {@code [rows, width]} (sequence-major); under sequence parallelism this
         * rank's contiguous {@code rows / t} of them. The batch is inferred with
         * {@link #batch(int)}.
         */
        @Override
        public Var forward(Var x) {
            int rows = x.shape().dim(0) * (sp ? t : 1);
            int b = batch;
            int s = rows / b;
            int hl = heads / t;
            int dh = width / heads;
            int local = width / t;
            Var n1 = enter(norm(x, gain1));
            Var qh = n1.matmul(q).reshape(s, b, hl, dh).permute(1, 2, 0, 3);
            Var kh = n1.matmul(k).reshape(s, b, hl, dh).permute(1, 2, 0, 3);
            Var vh = n1.matmul(v).reshape(s, b, hl, dh).permute(1, 2, 0, 3);
            Var a = Functional.attention(qh, kh, vh, causal, (float) (1 / Math.sqrt(dh)));
            Var attn = a.permute(2, 0, 1, 3).reshape(rows, local).matmul(o);
            Var h = x.add(leave(attn));
            Var n2 = enter(norm(h, gain2));
            Var mlp = n2.matmul(up).relu().matmul(down);
            return h.add(leave(mlp));
        }

        private int batch = 1;

        /**
         * Sets the batch size, so the rows can be split into sequence and batch.
         *
         * @param b the batch size
         * @return this shard
         */
        public Shard batch(int b) {
            this.batch = b;
            return this;
        }

        /**
         * Under sequence parallelism, sums the norm gains' partial gradients over the ranks
         * (counted traffic); without it they are already complete.
         */
        public void syncNormGradients() {
            if (!sp) {
                return;
            }
            for (Var gv : List.of(gain1, gain2)) {
                gv.setGrad(c.allReduce(rank, gv.grad()));
            }
        }

        @Override
        public List<Var> parameters() {
            return List.of(q, k, v, o, up, down, gain1, gain2);
        }

        /** @return whether parameter {@code i} is split by columns (true) or rows (false); norms are neither */
        int splitAxis(int i) {
            return switch (i) {
                case 0, 1, 2, 4 -> 1;
                case 3, 5 -> 0;
                default -> -1;
            };
        }
    }

    /**
     * Compares a shard's gradients with the matching slices of the single-device gradients.
     *
     * @param shard a shard
     * @param reference the one-rank shard holding the full gradients
     * @return the largest absolute difference
     */
    public static double maxGradientDifference(Shard shard, Shard reference) {
        double worst = 0;
        List<Var> mine = shard.parameters();
        List<Var> full = reference.parameters();
        for (int i = 0; i < mine.size(); i++) {
            Tensor g = full.get(i).grad();
            int axis = shard.splitAxis(i);
            if (axis == 1) {
                g = columns(g, shard.rank, shard.t);
            } else if (axis == 0) {
                g = rows(g, shard.rank, shard.t);
            }
            float[] a = mine.get(i).grad().toArray();
            float[] b = g.toArray();
            for (int j = 0; j < a.length; j++) {
                worst = Math.max(worst, Math.abs(a[j] - b[j]));
            }
        }
        return worst;
    }
}
