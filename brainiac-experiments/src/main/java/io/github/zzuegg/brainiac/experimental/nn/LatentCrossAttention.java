package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Cross-attention from a (usually small) set of queries to a (usually large) context, with the
 * residual, layer-norm and feed-forward wrapping of a transformer block: the multihead
 * attention block {@code MAB(Q, X) = LN(H + FFN(H))}, {@code H = LN(Q + Attention(Q, X))} of
 * the Set Transformer. The queries are either a second input or, with {@code latents > 0}, a
 * learned latent array shared by every example; that makes it
 *
 * <ul>
 *   <li>pooling by multihead attention (PMA, {@code k} seed vectors attend over a set),</li>
 *   <li>the inducing points of ISAB ({@code m} latents summarise a set in {@code O(n m)}),</li>
 *   <li>the Perceiver's latent bottleneck and Perceiver IO's output queries.</li>
 * </ul>
 *
 * <p>The attention itself is core {@link MultiHeadAttention} with a separate key/value input.
 * Inputs are {@code [batch, steps, features]}; as a {@link MultiInputLayer} it takes
 * {@code (queries, context)}, or just {@code context} when it has latents.
 *
 * <p>Papers: Lee et al., "Set Transformer" (ICML 2019),
 * <a href="https://arxiv.org/abs/1810.00825">arXiv:1810.00825</a>; Jaegle et al., "Perceiver"
 * (ICML 2021), <a href="https://arxiv.org/abs/2103.03206">arXiv:2103.03206</a>.
 */
public final class LatentCrossAttention implements MultiInputLayer {

    private final int width;
    private final int contextFeatures;
    private final Var latents;
    private final MultiInputLayer attention;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    private final Var up;
    private final Var upBias;
    private final Var down;
    private final Var downBias;

    /**
     * Creates the block.
     *
     * @param width the query (and output) width
     * @param contextFeatures the context width
     * @param heads the attention heads; must divide {@code width}
     * @param latents the number of learned queries, or 0 to take queries as the first input
     * @param random the source of initial weights
     */
    public LatentCrossAttention(int width, int contextFeatures, int heads, int latents, RandomGenerator random) {
        SplittableRandom r = new SplittableRandom(random.nextLong());
        this.width = width;
        this.contextFeatures = contextFeatures;
        this.latents = latents > 0 ? Var.param(Tensor.normal(Shape.of(1, latents, width), 0, 0.5f, r)) : null;
        this.attention = MultiHeadAttention.of(heads).create(List.of(Shape.of(1, width), Shape.of(1,
                contextFeatures)), r);
        this.norm1 = LayerNorm.of(width);
        this.norm2 = LayerNorm.of(width);
        this.up = ClassicOps.matrix(r, width, 2 * width);
        this.upBias = ClassicOps.zeros(2 * width);
        this.down = ClassicOps.matrix(r, 2 * width, width);
        this.downBias = ClassicOps.zeros(width);
    }

    /**
     * Describes a block with learned queries for model builders: {@code [steps, c] -> [latents, c]}.
     *
     * @param latents the learned queries
     * @param heads the attention heads
     * @return the spec
     */
    public static LayerSpec latents(int latents, int heads) {
        return ClassicOps.spec(s -> Shape.of(latents, s.dim(1)),
                (s, r) -> new LatentCrossAttention(s.dim(1), s.dim(1), heads, latents, r));
    }

    /**
     * Describes a block that takes {@code (queries, context)} for graph builders.
     *
     * @param heads the attention heads
     * @return the spec
     */
    public static MultiInputLayer.Spec cross(int heads) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return inputs.get(0);
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                return new LatentCrossAttention(inputs.get(0).dim(1), inputs.get(1).dim(1), heads, 0, random);
            }
        };
    }

    /**
     * Attends from queries to a context.
     *
     * @param queries {@code [batch, n, width]}
     * @param context {@code [batch, m, contextFeatures]}
     * @param training whether training
     * @return {@code [batch, n, width]}
     */
    public Var attend(Var queries, Var context, boolean training) {
        Var h = norm1.forward(queries.add(attention.forward(List.of(queries, context), training)));
        Var f = Activation.GELU.forward(h.matmul(up).add(upBias)).matmul(down).add(downBias);
        return norm2.forward(h.add(f));
    }

    /**
     * The learned queries for a batch.
     *
     * @param batch the batch size
     * @return {@code [batch, latents, width]}
     */
    public Var queries(int batch) {
        if (latents == null) {
            throw new IllegalStateException("this block takes its queries as an input");
        }
        return latents.expand(Shape.of(batch, latents.shape().dim(1), width));
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        if (latents != null) {
            Var context = inputs.get(inputs.size() - 1);
            return attend(queries(context.shape().dim(0)), context, training);
        }
        if (inputs.size() != 2) {
            throw new IllegalArgumentException(this + " takes (queries, context), got " + inputs.size() + " inputs");
        }
        return attend(inputs.get(0), inputs.get(1), training);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return latents != null ? Shape.of(latents.shape().dim(1), width) : inputs.get(0);
    }

    @Override
    public Shape outputShape(Shape input) {
        return outputShape(List.of(input));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        if (latents != null) {
            p.add(latents);
        }
        p.addAll(attention.parameters());
        p.addAll(norm1.parameters());
        p.addAll(norm2.parameters());
        p.addAll(List.of(up, upBias, down, downBias));
        return p;
    }

    /** @return the number of learned queries, 0 if queries are an input */
    public int latentCount() {
        return latents == null ? 0 : latents.shape().dim(1);
    }

    @Override
    public String toString() {
        return "LatentCrossAttention(" + width + " <- " + contextFeatures
                + (latents != null ? ", " + latents.shape().dim(1) + " latents" : "") + ")";
    }
}
