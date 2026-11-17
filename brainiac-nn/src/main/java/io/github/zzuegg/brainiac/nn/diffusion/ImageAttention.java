package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.GroupNorm;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Self-attention between the pixels of an image {@code [batch, channels, h, w]}: group norm,
 * attention over the pixels as tokens, added to the input. The middle of Stable Diffusion's
 * VAE uses it with one head.
 */
public final class ImageAttention implements Layer {

    /**
     * The shape of the layer.
     *
     * @param channels the image channels
     * @param heads the attention heads; must divide {@code channels}
     * @param groups group-norm groups
     * @param epsilon the group norm's epsilon
     */
    public record Settings(int channels, int heads, int groups, float epsilon) {

        /** Checks the values. */
        public Settings {
            if (channels < 1 || heads < 1 || channels % heads != 0 || groups < 1 || channels % groups != 0) {
                throw new IllegalArgumentException("heads and groups must divide " + channels + " channels");
            }
        }
    }

    private final Settings settings;
    private final GroupNorm norm;
    private final MultiHeadAttention attention;

    /**
     * Creates the layer with fresh weights.
     *
     * @param settings its shape
     * @param random the source of initial weights
     */
    public ImageAttention(Settings settings, SplittableRandom random) {
        this.settings = Objects.requireNonNull(settings, "settings");
        int c = settings.channels();
        int d = c / settings.heads();
        norm = new GroupNorm(settings.groups(), c, settings.epsilon(), true);
        attention = new MultiHeadAttention(new MultiHeadAttention.Settings(c, c, settings.heads(), settings.heads(),
                d, d, c, true, false, 0, false, 0, 0, false), Init.XAVIER, random);
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        Var t = Parts.tokens(norm.forward(input));
        return Parts.image(attention.forward(List.of(t), false), s.dim(2), s.dim(3)).add(input);
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(norm.parameters());
        p.addAll(attention.parameters());
        return p;
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    @Override
    public String toString() {
        return "ImageAttention(" + settings.channels() + ", " + settings.heads() + " heads)";
    }
}
