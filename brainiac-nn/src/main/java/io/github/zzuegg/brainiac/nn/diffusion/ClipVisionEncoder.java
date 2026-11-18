package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * CLIP's vision transformer: the image cut into square patches, a class token in front,
 * learned position embeddings, a layer norm, then pre-norm {@link TransformerBlock}s. It
 * returns the hidden states of its last block, without CLIP's post layer norm or projection,
 * so that loading all but the last block gives the penultimate states that video models
 * such as Wan 2.1's image-to-video condition on.
 *
 * <p>As a layer it maps pixels {@code [batch, 3, size, size]}, normalized with CLIP's mean
 * and standard deviation, to {@code [batch, 1 + patches, features]}.
 */
public final class ClipVisionEncoder implements Layer {

    /** CLIP's pixel mean per channel. */
    public static final float[] MEAN = {0.48145466f, 0.4578275f, 0.40821073f};

    /** CLIP's pixel standard deviation per channel. */
    public static final float[] STD = {0.26862954f, 0.26130258f, 0.27577711f};

    /**
     * The shape of the encoder.
     *
     * @param imageSize the side of the square input in pixels (224)
     * @param patch the side of a patch (14 for ViT-H/14)
     * @param features the width
     * @param layers the number of transformer blocks
     * @param heads the attention heads
     * @param ffnSize the feed-forward width
     * @param activation the feed-forward activation: {@link Activation#GELU} for OpenCLIP,
     *     {@link Activation#QUICK_GELU} for OpenAI's CLIP
     */
    public record Settings(int imageSize, int patch, int features, int layers, int heads, int ffnSize,
            Activation activation) {

        /** Checks the values. */
        public Settings {
            Objects.requireNonNull(activation, "activation");
            if (patch < 1 || imageSize % patch != 0 || features < 1 || layers < 1 || heads < 1
                    || features % heads != 0 || ffnSize < 1) {
                throw new IllegalArgumentException("invalid CLIP vision encoder shape");
            }
        }

        /** @return patches per side */
        public int grid() {
            return imageSize / patch;
        }

        /** @return the number of states: the class token and the patches */
        public int tokens() {
            return 1 + grid() * grid();
        }

        /**
         * @param count the number of blocks to keep
         * @return this shape with fewer blocks, as if the last ones were not there
         */
        public Settings withLayers(int count) {
            return new Settings(imageSize, patch, features, count, heads, ffnSize, activation);
        }
    }

    private final Settings settings;
    private final Dense patchEmbedding;
    private final Var classEmbedding;
    private final Var positionEmbedding;
    private final LayerNorm preNorm;
    private final List<TransformerBlock> blocks = new ArrayList<>();

    /**
     * Creates an encoder with fresh weights.
     *
     * @param settings its shape
     * @param random the source of initial weights
     */
    public ClipVisionEncoder(Settings settings, SplittableRandom random) {
        this.settings = Objects.requireNonNull(settings, "settings");
        int d = settings.features();
        int in = 3 * settings.patch() * settings.patch();
        patchEmbedding = new Dense(in, d, Activation.IDENTITY, Init.XAVIER, random);
        classEmbedding = Var.param(Tensor.normal(Shape.of(d), 0, 0.02f, random));
        positionEmbedding = Var.param(Tensor.normal(Shape.of(settings.tokens(), d), 0, 0.02f, random));
        preNorm = LayerNorm.of(d);
        TransformerBlock.Settings block = new TransformerBlock.Settings(d, 0, settings.heads(), settings.heads(),
                settings.ffnSize(), settings.activation(), false, false, true, false, 0, false, 0, true, false);
        for (int i = 0; i < settings.layers(); i++) {
            blocks.add(new TransformerBlock(block, random));
        }
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var pixels, boolean training) {
        Shape s = pixels.shape();
        int size = settings.imageSize();
        if (s.rank() != 4 || s.dim(1) != 3 || s.dim(2) != size || s.dim(3) != size) {
            throw new IllegalArgumentException(this + " takes pixels [batch, 3, " + size + ", " + size + "], got " + s);
        }
        int batch = s.dim(0);
        int p = settings.patch();
        int g = settings.grid();
        int d = settings.features();
        Var x = Parts.step(() -> {
            // the patch convolution (kernel = stride = patch, no bias) as one matrix product
            Var patches = pixels.reshape(batch, 3, g, p, g, p).permute(0, 2, 4, 1, 3, 5)
                    .reshape(batch, g * g, 3 * p * p);
            Var embedded = patchEmbedding.forward(patches);
            Var cls = classEmbedding.reshape(1, 1, d).expand(Shape.of(batch, 1, d));
            Var all = Var.concat(1, cls, embedded).add(positionEmbedding.reshape(1, settings.tokens(), d));
            return preNorm.forward(all);
        });
        for (TransformerBlock b : blocks) {
            Var in = x;
            x = Parts.step(() -> b.forward(List.of(in), training));
            Parts.free(in, pixels);
        }
        return x;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), settings.tokens(), settings.features());
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(patchEmbedding.parameters());
        p.add(classEmbedding);
        p.add(positionEmbedding);
        p.addAll(preNorm.parameters());
        for (TransformerBlock b : blocks) {
            p.addAll(b.parameters());
        }
        return p;
    }

    /** @return the type the weights are held in */
    public DType dtype() {
        return positionEmbedding.value().dtype();
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    /** @return the patch embedding, {@code [3 * patch * patch, features]} over (channel, row, column) */
    public Dense patchEmbedding() {
        return patchEmbedding;
    }

    /** @return the class token's embedding {@code [features]} */
    public Var classEmbedding() {
        return classEmbedding;
    }

    /** @return the position embeddings {@code [tokens, features]} */
    public Var positionEmbedding() {
        return positionEmbedding;
    }

    /** @return the layer norm before the blocks */
    public LayerNorm preNorm() {
        return preNorm;
    }

    /** @return the transformer blocks */
    public List<TransformerBlock> blocks() {
        return List.copyOf(blocks);
    }

    @Override
    public String toString() {
        return "ClipVisionEncoder(" + settings.features() + ", " + settings.layers() + " layers)";
    }
}
