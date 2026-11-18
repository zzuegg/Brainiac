package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.GroupNorm;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A transformer over the pixels of an image ({@code Transformer2DModel} in diffusers): group
 * norm, a projection to tokens, {@link TransformerBlock}s that attend to each other and, with
 * a context, to the context (such as a prompt's embeddings), and a projection back, added to
 * the input. The blocks use a GELU-gated feed-forward network (GEGLU) four times as wide.
 *
 * <p>Its inputs are the images {@code [batch, channels, h, w]} and, if
 * {@link Settings#contextFeatures()} is positive, the context {@code [batch, tokens,
 * contextFeatures]}.
 */
public final class SpatialTransformer implements MultiInputLayer {

    /**
     * The shape of the transformer.
     *
     * @param channels the image channels, also the transformer width
     * @param heads the attention heads; must divide {@code channels}
     * @param depth the number of transformer blocks
     * @param contextFeatures the width of the context to cross-attend to, or 0 for none
     * @param groups group-norm groups
     * @param linearProjection whether to project with dense layers on tokens rather than 1x1
     *     convolutions on the image (Stable Diffusion 2 and XL)
     */
    public record Settings(int channels, int heads, int depth, int contextFeatures, int groups,
            boolean linearProjection) {

        /** Checks the values. */
        public Settings {
            if (channels < 1 || heads < 1 || channels % heads != 0 || depth < 1 || contextFeatures < 0
                    || groups < 1 || channels % groups != 0) {
                throw new IllegalArgumentException("a spatial transformer needs heads and groups that divide "
                        + channels + " channels and at least one block");
            }
        }
    }

    private final Settings settings;
    private final GroupNorm norm;
    private final Layer projectIn;
    private final List<TransformerBlock> blocks;
    private final Layer projectOut;

    /**
     * Creates a transformer with fresh weights.
     *
     * @param settings its shape
     * @param random the source of initial weights
     */
    public SpatialTransformer(Settings settings, SplittableRandom random) {
        this.settings = Objects.requireNonNull(settings, "settings");
        int c = settings.channels();
        norm = new GroupNorm(settings.groups(), c, 1e-6f, true);
        projectIn = settings.linearProjection() ? Parts.dense(c, c, Activation.IDENTITY, random)
                : Parts.conv1(c, c, random);
        blocks = new ArrayList<>();
        TransformerBlock.Settings block = new TransformerBlock.Settings(c, settings.contextFeatures(),
                settings.heads(), settings.heads(), 4 * c, Activation.GELU, true, false, true, false, 0, false, 0,
                true, false);
        for (int i = 0; i < settings.depth(); i++) {
            blocks.add(new TransformerBlock(block, random));
        }
        projectOut = settings.linearProjection() ? Parts.dense(c, c, Activation.IDENTITY, random)
                : Parts.conv1(c, c, random);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var x = inputs.get(0);
        Var context = settings.contextFeatures() > 0 ? inputs.get(1) : null;
        Shape s = x.shape();
        int h = s.dim(2);
        int w = s.dim(3);
        // without gradients, each step frees what it made (see Parts.step)
        Var t = Parts.step(() -> {
            Var n = norm.forward(x);
            return settings.linearProjection() ? projectIn.forward(Parts.tokens(n))
                    : Parts.tokens(projectIn.forward(n));
        });
        for (TransformerBlock b : blocks) {
            Var in = t;
            t = Parts.step(() -> b.forward(context == null ? List.of(in) : List.of(in, context), training));
            Parts.free(in);
        }
        Var tokens = t;
        Var y = Parts.step(() -> (settings.linearProjection() ? Parts.image(projectOut.forward(tokens), h, w)
                : projectOut.forward(Parts.image(tokens, h, w))).add(x));
        Parts.free(tokens);
        return y;
    }

    @Override
    public Var forward(Var input) {
        return forward(List.of(input), false);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return inputs.get(0);
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(norm.parameters());
        p.addAll(projectIn.parameters());
        for (TransformerBlock b : blocks) {
            p.addAll(b.parameters());
        }
        p.addAll(projectOut.parameters());
        return p;
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    /** @return the group norm */
    public GroupNorm norm() {
        return norm;
    }

    /** @return the projection in: a {@link Dense} or a 1x1 {@link Conv} */
    public Layer projectIn() {
        return projectIn;
    }

    /** @return the transformer blocks */
    public List<TransformerBlock> blocks() {
        return List.copyOf(blocks);
    }

    /** @return the projection out: a {@link Dense} or a 1x1 {@link Conv} */
    public Layer projectOut() {
        return projectOut;
    }

    @Override
    public String toString() {
        return "SpatialTransformer(" + settings.channels() + ", " + settings.heads() + " heads, depth "
                + settings.depth() + (settings.contextFeatures() > 0 ? ", context " + settings.contextFeatures() : "")
                + ")";
    }
}
