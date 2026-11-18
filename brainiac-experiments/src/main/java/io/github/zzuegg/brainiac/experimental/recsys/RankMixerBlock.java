package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ReluRoutedMoe;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * One RankMixer block (Zhu et al., ByteDance 2025, <a href="https://arxiv.org/abs/2507.15551">
 * arXiv:2507.15551</a>), a hardware-friendly replacement for self-attention in ranking models, as
 * a {@link Layer} on {@code [batch, tokens, dim]} feature tokens (the token count must equal the
 * head count, as in the paper).
 * <ol>
 *   <li><em>Multi-head token mixing</em>, parameter-free: every token is split into {@code T}
 *       heads and head {@code h} of every token is gathered into new token {@code h}; then
 *       residual and layer norm. This exchanges information across feature subspaces with a
 *       reshape instead of a {@code T x T} attention map.</li>
 *   <li><em>Per-token FFN</em>: a separate two-layer GELU-free (ReLU) feed-forward network for
 *       each token, so heterogeneous feature subspaces do not share parameters; then residual and
 *       layer norm.</li>
 * </ol>
 * With {@link #sparse} the per-token FFNs become sparse mixtures of experts with ReLU routing
 * and an adaptive L1 penalty on the number of active experts ({@link ReluRoutedMoe}), trained
 * densely and executed sparsely. {@link Ranker} stacks the tokenizer, blocks, mean pooling and a
 * logit head into a complete click model.
 */
public final class RankMixerBlock implements Layer {

    private final int tokens;
    private final int dim;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private final List<ReluRoutedMoe> experts;
    private final LayerNorm norm1;
    private final LayerNorm norm2;

    private RankMixerBlock(int tokens, int dim, int expansion, int sparseExperts, SplittableRandom random) {
        if (dim % tokens != 0) {
            throw new IllegalArgumentException("dim must be divisible by the token count");
        }
        this.tokens = tokens;
        this.dim = dim;
        int hidden = expansion * dim;
        if (sparseExperts > 0) {
            experts = new ArrayList<>();
            for (int t = 0; t < tokens; t++) {
                experts.add(ReluRoutedMoe.of(dim, sparseExperts, hidden / sparseExperts * 2, 2, random));
            }
            w1 = null;
            b1 = null;
            w2 = null;
            b2 = null;
        } else {
            experts = null;
            w1 = Var.param(Tensor.normal(Shape.of(tokens, dim, hidden), 0, (float) Math.sqrt(2.0 / dim), random));
            b1 = Var.param(Tensor.zeros(Shape.of(tokens, 1, hidden)));
            w2 = Var.param(Tensor.normal(Shape.of(tokens, hidden, dim), 0, (float) Math.sqrt(1.0 / hidden), random));
            b2 = Var.param(Tensor.zeros(Shape.of(tokens, 1, dim)));
        }
        norm1 = LayerNorm.of(dim);
        norm2 = LayerNorm.of(dim);
    }

    /**
     * A dense block.
     *
     * @param tokens feature tokens (and heads)
     * @param dim token width, divisible by {@code tokens}
     * @param expansion the per-token FFN's hidden width as a multiple of {@code dim}
     * @param random the initializer
     * @return the block
     */
    public static RankMixerBlock dense(int tokens, int dim, int expansion, SplittableRandom random) {
        return new RankMixerBlock(tokens, dim, expansion, 0, random);
    }

    /**
     * A block whose per-token FFNs are ReLU-routed sparse mixtures of experts.
     *
     * @param tokens feature tokens
     * @param dim token width
     * @param expansion total hidden width (summed over experts) as a multiple of {@code dim}, times 2
     * @param experts experts per token
     * @param random the initializer
     * @return the block
     */
    public static RankMixerBlock sparse(int tokens, int dim, int expansion, int experts, SplittableRandom random) {
        return new RankMixerBlock(tokens, dim, expansion, experts, random);
    }

    /**
     * @param expansion FFN expansion
     * @return a spec for model builders on {@code [tokens, dim]} inputs
     */
    public static LayerSpec spec(int expansion) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return dense(input.dim(-2), input.dim(-1), expansion, random);
            }
        };
    }

    /**
     * The parameter-free token mixing: head {@code h} of every token becomes new token {@code h}.
     *
     * @param x {@code [batch, tokens, dim]}
     * @return the mixed tokens, same shape
     */
    public Var mix(Var x) {
        int b = x.shape().dim(0);
        int part = dim / tokens;
        return x.reshape(b, tokens, tokens, part).permute(0, 2, 1, 3).reshape(b, tokens, dim);
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        Var h = norm1.forward(input.add(mix(input)));
        Var ffn;
        if (experts == null) {
            Var t = h.permute(1, 0, 2);
            ffn = t.matmul(w1).add(b1).relu().matmul(w2).add(b2).permute(1, 0, 2);
        } else {
            List<Var> parts = new ArrayList<>();
            for (int k = 0; k < tokens; k++) {
                parts.add(experts.get(k).forward(h.slice(1, k, k + 1).reshape(b, dim)).reshape(b, 1, dim));
            }
            ffn = Var.concat(1, parts.toArray(Var[]::new));
        }
        return norm2.forward(h.add(ffn));
    }

    @Override
    public Var auxiliaryLoss() {
        if (experts == null) {
            return null;
        }
        Var total = null;
        for (ReluRoutedMoe m : experts) {
            Var a = m.auxiliaryLoss();
            if (a != null) {
                total = total == null ? a : total.add(a);
            }
        }
        return total;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        if (experts == null) {
            p.addAll(List.of(w1, b1, w2, b2));
        } else {
            for (ReluRoutedMoe m : experts) {
                p.addAll(m.parameters());
            }
        }
        p.addAll(norm1.parameters());
        p.addAll(norm2.parameters());
        return p;
    }

    /** @return the mean number of experts active per token in the last forward pass (sparse blocks) */
    public double activeExperts() {
        if (experts == null) {
            return Double.NaN;
        }
        double s = 0;
        int c = 0;
        for (ReluRoutedMoe m : experts) {
            for (int v : m.expertsPerToken()) {
                s += v;
                c++;
            }
        }
        return c == 0 ? 0 : s / c;
    }

    @Override
    public String toString() {
        return "RankMixerBlock(" + tokens + " tokens x " + dim + (experts == null ? "" : ", sparse") + ")";
    }

    /**
     * A complete RankMixer click model on {@code [batch, fields * categories]} one-hot rows: field
     * embeddings are concatenated, split into {@code tokens} equal groups and projected to
     * {@code dim} each (one projection per token), passed through the blocks, mean-pooled and
     * mapped to one logit.
     */
    public static final class Ranker implements Layer {
        private final int fields;
        private final int categories;
        private final int tokens;
        private final int dim;
        private final Var table;
        private final Var projection;
        private final List<RankMixerBlock> blocks = new ArrayList<>();
        private final Var head;
        private final Var headBias;

        /**
         * Creates the model.
         *
         * @param fields categorical fields, divisible by {@code tokens}
         * @param categories categories per field
         * @param embed field embedding width
         * @param tokens feature tokens
         * @param dim token width, divisible by {@code tokens}
         * @param depth blocks
         * @param expansion FFN expansion
         * @param sparseExperts experts per token, or 0 for dense FFNs
         * @param random the initializer
         */
        public Ranker(int fields, int categories, int embed, int tokens, int dim, int depth, int expansion,
                int sparseExperts, SplittableRandom random) {
            if (fields % tokens != 0) {
                throw new IllegalArgumentException("fields must divide into tokens");
            }
            this.fields = fields;
            this.categories = categories;
            this.tokens = tokens;
            this.dim = dim;
            this.table = RecsysOps.normal(random, 0.1f, fields * categories, embed);
            int group = fields / tokens * embed;
            this.projection = Var.param(Tensor.normal(Shape.of(tokens, group, dim), 0,
                    (float) Math.sqrt(1.0 / group), random));
            for (int l = 0; l < depth; l++) {
                blocks.add(sparseExperts > 0 ? sparse(tokens, dim, expansion, sparseExperts, random)
                        : dense(tokens, dim, expansion, random));
            }
            this.head = RecsysOps.matrix(random, dim, 1);
            this.headBias = RecsysOps.zeros(1);
        }

        @Override
        public Var forward(Var input) {
            int b = input.shape().dim(0);
            Var e = FieldAttention.embedFields(input, table, fields, categories);
            int group = fields / tokens * e.shape().dim(2);
            Var x = e.reshape(b, tokens, group).permute(1, 0, 2).matmul(projection).permute(1, 0, 2);
            for (RankMixerBlock block : blocks) {
                x = block.forward(x);
            }
            return x.mean(1, false).matmul(head).add(headBias);
        }

        @Override
        public Var auxiliaryLoss() {
            Var total = null;
            for (RankMixerBlock block : blocks) {
                Var a = block.auxiliaryLoss();
                if (a != null) {
                    total = total == null ? a : total.add(a);
                }
            }
            return total;
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(table, projection));
            for (RankMixerBlock block : blocks) {
                p.addAll(block.parameters());
            }
            p.add(head);
            p.add(headBias);
            return p;
        }

        /** @return the blocks */
        public List<RankMixerBlock> blocks() {
            return List.copyOf(blocks);
        }

        @Override
        public int outputSize(int inputSize) {
            return 1;
        }

        @Override
        public String toString() {
            return "RankMixer(" + tokens + " tokens x " + dim + ", " + blocks.size() + " blocks)";
        }
    }
}
