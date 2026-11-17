package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A Vision Transformer: {@link PatchEmbedding}, a stack of pre-norm transformer blocks
 * ({@link TunableTransformerBlock}, so every normalization and position trick is available,
 * including 2-D RoPE), a final LayerNorm, and pooling by the class token or the mean of the
 * patch tokens, then an optional linear classification head. Images
 * {@code [batch, channels, height, width]} in, {@code [batch, classes]} logits (or
 * {@code [batch, dim]} features without a head) out.
 *
 * <pre>{@code
 * Model m = Model.sequential().input(1, 16, 16)
 *     .add(VisionTransformer.of(4, 32, 2, 4).classes(10))
 *     .build();
 * }</pre>
 *
 * <p>Paper: Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020),
 * <a href="https://arxiv.org/abs/2010.11929">arXiv:2010.11929</a>.
 */
public final class VisionTransformer implements Layer {

    private final Spec spec;
    private final PatchEmbedding embedding;
    private final List<TunableTransformerBlock> blocks = new ArrayList<>();
    private final LayerNorm norm;
    private final Var head;
    private final Var headBias;

    private VisionTransformer(Spec spec, Shape image, RandomGenerator random) {
        this.spec = spec;
        boolean rope = spec.rope2d;
        embedding = new PatchEmbedding(image.dim(0), image.dim(1), image.dim(2), spec.patch, spec.dim, spec.classToken,
                !rope, random);
        TunableTransformerBlock.Spec b = spec.block;
        if (rope) {
            b = b.position(PositionalAttention.Position.ROTARY_2D)
                    .grid(embedding.gridHeight(), embedding.gridWidth(), spec.classToken ? 1 : 0);
        }
        for (int i = 0; i < spec.depth; i++) {
            blocks.add(b.dropPath(DropPath.linearRate(i, spec.depth, spec.dropPath)).build(spec.dim, random));
        }
        norm = LayerNorm.of(spec.dim);
        head = spec.classes > 0 ? TechniqueOps.weight(random, spec.dim, spec.classes) : null;
        headBias = spec.classes > 0 ? TechniqueOps.filled(0f, spec.classes) : null;
    }

    /**
     * Describes a ViT with pre-LayerNorm blocks and a GELU MLP of 4x width.
     *
     * @param patch the patch side
     * @param dim the token width
     * @param depth the number of blocks
     * @param heads attention heads per block
     * @return the spec
     */
    public static Spec of(int patch, int dim, int depth, int heads) {
        return new Spec(patch, dim, depth, heads);
    }

    /** An immutable description; every setter returns a new spec. */
    public static final class Spec implements LayerSpec {

        private final int patch;
        private final int dim;
        private final int depth;
        private int classes;
        private boolean classToken = true;
        private boolean rope2d;
        private float dropPath;
        private TunableTransformerBlock.Spec block;

        private Spec(int patch, int dim, int depth, int heads) {
            this.patch = patch;
            this.dim = dim;
            this.depth = depth;
            this.block = TunableTransformerBlock.of(heads);
        }

        private Spec copy() {
            Spec s = new Spec(patch, dim, depth, 1);
            s.classes = classes;
            s.classToken = classToken;
            s.rope2d = rope2d;
            s.dropPath = dropPath;
            s.block = block;
            return s;
        }

        /**
         * @param n the classes of the linear head; 0 for features only
         * @return a new spec
         */
        public Spec classes(int n) {
            Spec s = copy();
            s.classes = n;
            return s;
        }

        /**
         * @param on {@code true} to pool by a class token (default), {@code false} for the mean
         *     of patch tokens without a class token
         * @return a new spec
         */
        public Spec classToken(boolean on) {
            Spec s = copy();
            s.classToken = on;
            return s;
        }

        /**
         * @param on whether to use 2-D rotary positions instead of learned absolute ones
         * @return a new spec
         */
        public Spec rope2d(boolean on) {
            Spec s = copy();
            s.rope2d = on;
            return s;
        }

        /**
         * @param maxRate the stochastic-depth rate of the last block (linear in depth)
         * @return a new spec
         */
        public Spec dropPath(float maxRate) {
            Spec s = copy();
            s.dropPath = maxRate;
            return s;
        }

        /**
         * @param b the block description (heads, norm, residual, feed-forward...)
         * @return a new spec
         */
        public Spec block(TunableTransformerBlock.Spec b) {
            Spec s = copy();
            s.block = b;
            return s;
        }

        @Override
        public Shape outputShape(Shape input) {
            return Shape.of(classes > 0 ? classes : dim);
        }

        @Override
        public Layer create(Shape input, SplittableRandom random) {
            return new VisionTransformer(this, input, random);
        }

        /**
         * Creates the network for an image shape.
         *
         * @param channels image channels
         * @param height image height
         * @param width image width
         * @param random the random source
         * @return the network
         */
        public VisionTransformer build(int channels, int height, int width, RandomGenerator random) {
            return new VisionTransformer(this, Shape.of(channels, height, width), random);
        }
    }

    /**
     * The token sequence after the final norm.
     *
     * @param images {@code [batch, channels, height, width]}
     * @param training whether to apply stochastic depth
     * @return {@code [batch, tokens, dim]}
     */
    public Var tokens(Var images, boolean training) {
        Var x = embedding.forward(images);
        for (TunableTransformerBlock b : blocks) {
            x = b.forward(x, training);
        }
        return norm.forward(x);
    }

    /**
     * The pooled image representation, the input of the head (and of linear probes).
     *
     * @param images {@code [batch, channels, height, width]}
     * @param training whether to apply stochastic depth
     * @return {@code [batch, dim]}
     */
    public Var features(Var images, boolean training) {
        Var x = tokens(images, training);
        if (spec.classToken) {
            return x.slice(1, 0, 1).reshape(x.shape().dim(0), spec.dim);
        }
        return x.mean(1, false);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Var f = features(input, training);
        return head == null ? f : f.matmul(head).add(headBias);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(head == null ? spec.dim : spec.classes);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(embedding.parameters());
        for (TunableTransformerBlock b : blocks) {
            p.addAll(b.parameters());
        }
        p.addAll(norm.parameters());
        if (head != null) {
            p.add(head);
            p.add(headBias);
        }
        return p;
    }

    @Override
    public String toString() {
        return "VisionTransformer(patch " + spec.patch + ", dim " + spec.dim + ", depth " + spec.depth + ")";
    }
}
