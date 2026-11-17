package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Turns images {@code [batch, channels, height, width]} into a sequence of patch tokens
 * {@code [batch, (1 +) patches, dim]}, the input stage of a Vision Transformer: the image is
 * cut into non-overlapping {@code p x p} patches, each flattened and linearly projected (the
 * same as a stride-{@code p} convolution), optionally preceded by a learned class token, plus
 * learned absolute position embeddings (which can be left out when attention uses 2-D RoPE).
 *
 * <p>Paper: Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020),
 * <a href="https://arxiv.org/abs/2010.11929">arXiv:2010.11929</a>.
 */
public final class PatchEmbedding implements Layer {

    private final int channels;
    private final int height;
    private final int width;
    private final int patch;
    private final int dim;
    private final Var projection;
    private final Var bias;
    private final Var classToken;
    private final Var positions;

    /**
     * Creates the layer.
     *
     * @param channels image channels
     * @param height image height, a multiple of {@code patch}
     * @param width image width, a multiple of {@code patch}
     * @param patch the patch side
     * @param dim the token width
     * @param classToken whether to prepend a learned class token
     * @param learnedPositions whether to add learned position embeddings
     * @param random the random source
     */
    public PatchEmbedding(int channels, int height, int width, int patch, int dim, boolean classToken,
            boolean learnedPositions, RandomGenerator random) {
        if (height % patch != 0 || width % patch != 0) {
            throw new IllegalArgumentException("patch " + patch + " must divide the image " + height + "x" + width);
        }
        this.channels = channels;
        this.height = height;
        this.width = width;
        this.patch = patch;
        this.dim = dim;
        this.projection = TechniqueOps.weight(random, channels * patch * patch, dim);
        this.bias = Var.param(Tensor.zeros(Shape.of(dim)));
        this.classToken = classToken ? TechniqueOps.normal(random, 0.02f, 1, 1, dim) : null;
        int tokens = patches() + (classToken ? 1 : 0);
        this.positions = learnedPositions ? TechniqueOps.normal(random, 0.02f, tokens, dim) : null;
    }

    /**
     * Describes the layer for {@code Model.sequential().input(c, h, w).add(...)}.
     *
     * @param patch the patch side
     * @param dim the token width
     * @param classToken whether to prepend a class token
     * @return the spec
     */
    public static LayerSpec spec(int patch, int dim, boolean classToken) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(1) / patch * (input.dim(2) / patch) + (classToken ? 1 : 0), dim);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new PatchEmbedding(input.dim(0), input.dim(1), input.dim(2), patch, dim, classToken, true,
                        random);
            }
        };
    }

    /** @return the number of patches */
    public int patches() {
        return height / patch * (width / patch);
    }

    /** @return the patch grid height */
    public int gridHeight() {
        return height / patch;
    }

    /** @return the patch grid width */
    public int gridWidth() {
        return width / patch;
    }

    /**
     * Cuts images into flattened patches, without projecting them (the regression target of
     * MAE).
     *
     * @param images {@code [batch, channels, height, width]}
     * @param patch the patch side
     * @return {@code [batch, patches, channels * patch * patch]}
     */
    public static Var patchify(Var images, int patch) {
        int b = images.shape().dim(0);
        int c = images.shape().dim(1);
        int gh = images.shape().dim(2) / patch;
        int gw = images.shape().dim(3) / patch;
        return images.reshape(b, c, gh, patch, gw, patch).permute(0, 2, 4, 1, 3, 5)
                .reshape(b, gh * gw, c * patch * patch);
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 4 || s.dim(1) != channels || s.dim(2) != height || s.dim(3) != width) {
            throw new IllegalArgumentException(this + " expects [batch, " + channels + ", " + height + ", " + width
                    + "], got " + s);
        }
        int b = s.dim(0);
        Var tokens = patchify(input, patch).matmul(projection).add(bias);
        if (classToken != null) {
            tokens = Var.concat(1, classToken.expand(Shape.of(b, 1, dim)), tokens);
        }
        return positions == null ? tokens : tokens.add(positions);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(patches() + (classToken != null ? 1 : 0), dim);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(projection, bias));
        if (classToken != null) {
            p.add(classToken);
        }
        if (positions != null) {
            p.add(positions);
        }
        return p;
    }

    /** @return the learned position embeddings {@code [tokens, dim]}, or {@code null} */
    public Var positions() {
        return positions;
    }

    @Override
    public String toString() {
        return "PatchEmbedding(" + channels + "x" + height + "x" + width + ", patch " + patch + " -> " + dim + ")";
    }
}
