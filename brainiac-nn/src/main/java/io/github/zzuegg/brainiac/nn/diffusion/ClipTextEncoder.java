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
 * CLIP's text transformer: token and position embeddings, causal pre-norm
 * {@link TransformerBlock}s and a final layer norm, with the hidden state at each prompt's
 * end token as its pooled embedding (optionally projected). With pretrained weights it is
 * the text encoder of Stable Diffusion (OpenAI's CLIP ViT-L/14, OpenCLIP ViT-H and ViT-bigG);
 * with fresh weights it trains like any other layer.
 *
 * <p>As a layer it maps token ids {@code [batch, tokens]} (as numbers) to the last hidden
 * states {@code [batch, tokens, features]}; {@link #encode} also returns the penultimate
 * layer and the pooled embedding.
 */
public final class ClipTextEncoder implements Layer {

    /**
     * The shape of the encoder.
     *
     * @param vocabulary the number of tokens
     * @param features the width
     * @param layers the number of transformer blocks
     * @param heads the attention heads
     * @param ffnSize the feed-forward width
     * @param positions the longest prompt in tokens (77)
     * @param activation the feed-forward activation: {@link Activation#QUICK_GELU} for OpenAI's
     *     CLIP, {@link Activation#GELU} for OpenCLIP
     * @param epsilon the final layer norm's epsilon
     * @param endToken the end-of-text token, whose hidden state is pooled; a negative value
     *     pools at the largest id, as CLIP's end token is the last in its vocabulary
     * @param projection the width of the pooled embedding's projection, or 0 for none
     */
    public record Settings(int vocabulary, int features, int layers, int heads, int ffnSize, int positions,
            Activation activation, float epsilon, int endToken, int projection) {

        /** Checks the values. */
        public Settings {
            Objects.requireNonNull(activation, "activation");
            if (vocabulary < 1 || features < 1 || layers < 1 || heads < 1 || features % heads != 0 || ffnSize < 1
                    || positions < 1 || projection < 0) {
                throw new IllegalArgumentException("invalid CLIP text encoder shape");
            }
        }

        /**
         * This shape with fewer layers, as if the last ones were not there; Stable Diffusion
         * 2 conditions on the penultimate layer this way.
         *
         * @param count the number of layers to keep
         * @return the settings
         */
        public Settings withLayers(int count) {
            return new Settings(vocabulary, features, count, heads, ffnSize, positions, activation, epsilon,
                    endToken, projection);
        }
    }

    /**
     * What the encoder returns for a batch of prompts.
     *
     * @param last the last layer after the final layer norm, {@code [batch, tokens, features]}
     * @param penultimate the second-to-last layer's output (no final norm), or {@code null}
     *     with a single layer
     * @param pooled the final-normed state at each prompt's end token, projected if the model
     *     has a projection, {@code [batch, width]}
     */
    public record Output(Var last, Var penultimate, Var pooled) {
    }

    private final Settings settings;
    private final Var tokenEmbedding;
    private final Var positionEmbedding;
    private final List<TransformerBlock> blocks = new ArrayList<>();
    private final LayerNorm finalNorm;
    private final Dense projection;

    /**
     * Creates an encoder with fresh weights.
     *
     * @param settings its shape
     * @param random the source of initial weights
     */
    public ClipTextEncoder(Settings settings, SplittableRandom random) {
        this.settings = Objects.requireNonNull(settings, "settings");
        int d = settings.features();
        tokenEmbedding = Var.param(Tensor.normal(Shape.of(settings.vocabulary(), d), 0, 0.02f, random));
        positionEmbedding = Var.param(Tensor.normal(Shape.of(settings.positions(), d), 0, 0.01f, random));
        TransformerBlock.Settings block = new TransformerBlock.Settings(d, 0, settings.heads(), settings.heads(),
                settings.ffnSize(), settings.activation(), false, false, true, true, 0, false, 0, true, false);
        for (int i = 0; i < settings.layers(); i++) {
            blocks.add(new TransformerBlock(block, random));
        }
        finalNorm = new LayerNorm(new int[] {d}, settings.epsilon(), true, true, false);
        projection = settings.projection() > 0 ? new Dense(d, settings.projection(), Activation.IDENTITY,
                Init.XAVIER.create(Shape.of(d, settings.projection()), d, settings.projection(), random),
                Tensor.zeros(Shape.of(settings.projection()))) : null;
    }

    /**
     * Encodes token ids.
     *
     * @param ids {@code [batch, tokens]} token ids as numbers, at most {@link Settings#positions()} per row
     * @param training whether the model is training
     * @return the hidden states and pooled embedding
     */
    public Output encode(Var ids, boolean training) {
        Shape s = ids.shape();
        if (s.rank() != 2 || s.dim(1) > settings.positions()) {
            throw new IllegalArgumentException(this + " expects ids [batch, up to " + settings.positions()
                    + "], got " + s);
        }
        int batch = s.dim(0);
        int n = s.dim(1);
        int[] end = ends(ids.value().toArray(), batch, n);
        Tensor index = ids.value().to(DType.FLOAT32);
        Var x = Parts.step(() -> tokenEmbedding.gather(0, index).add(positionEmbedding.slice(0, 0, n)));
        Var penultimate = null;
        for (int i = 0; i < blocks.size(); i++) {
            TransformerBlock b = blocks.get(i);
            Var in = x;
            x = Parts.step(() -> b.forward(List.of(in), training));
            if (in != penultimate) {
                Parts.free(in);
            }
            if (i == blocks.size() - 2) {
                penultimate = x;
            }
        }
        Var h = x;
        Var last = Parts.step(() -> finalNorm.forward(h));
        if (h != penultimate) {
            Parts.free(h);
        }
        Var pooled = Parts.step(() -> {
            float[] rows = new float[batch];
            for (int b = 0; b < batch; b++) {
                rows[b] = b * n + end[b];
            }
            Var p = last.reshape(batch * n, settings.features())
                    .gather(0, Tensor.of(Shape.of(batch), rows, last.value().device()));
            return projection == null ? p : projection.forward(p);
        });
        return new Output(last, penultimate, pooled);
    }

    /** Each row's end token: the first {@code endToken}, or the largest id when it is negative. */
    private int[] ends(float[] ids, int batch, int n) {
        int[] end = new int[batch];
        for (int b = 0; b < batch; b++) {
            int best = 0;
            for (int i = 0; i < n; i++) {
                float id = ids[b * n + i];
                if (settings.endToken() < 0 ? id > ids[b * n + best] : id == settings.endToken()) {
                    best = i;
                    if (settings.endToken() >= 0) {
                        break;
                    }
                }
            }
            end[b] = best;
        }
        return end;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Output o = encode(input, training);
        return o.last();
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), settings.features());
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(tokenEmbedding, positionEmbedding));
        for (TransformerBlock b : blocks) {
            p.addAll(b.parameters());
        }
        p.addAll(finalNorm.parameters());
        if (projection != null) {
            p.addAll(projection.parameters());
        }
        return p;
    }

    /** @return the type the weights are held in */
    public DType dtype() {
        return tokenEmbedding.value().dtype();
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    /** @return the token embeddings {@code [vocabulary, features]} */
    public Var tokenEmbedding() {
        return tokenEmbedding;
    }

    /** @return the position embeddings {@code [positions, features]} */
    public Var positionEmbedding() {
        return positionEmbedding;
    }

    /** @return the transformer blocks */
    public List<TransformerBlock> blocks() {
        return List.copyOf(blocks);
    }

    /** @return the final layer norm */
    public LayerNorm finalNorm() {
        return finalNorm;
    }

    /** @return the projection of the pooled embedding, or {@code null} */
    public Dense projection() {
        return projection;
    }

    @Override
    public String toString() {
        return "ClipTextEncoder(" + settings.features() + ", " + settings.layers() + " layers)";
    }
}
