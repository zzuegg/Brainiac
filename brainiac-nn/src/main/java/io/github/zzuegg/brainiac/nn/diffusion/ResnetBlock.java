package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.GroupNorm;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * The residual block of diffusion UNets and VAEs on images {@code [batch, channels, h, w]}:
 * group norm, SiLU and a 3x3 convolution, twice, with a time embedding added in between,
 * plus the input (through a 1x1 convolution when the width changes).
 *
 * <p>Its inputs are the images and, if {@link Settings#timeFeatures()} is positive, the time
 * embedding {@code [batch, timeFeatures]} (for example from {@link TimestepEmbedding}); wire
 * it with {@code Node.merge}. Without a time embedding it also works as a single-input layer.
 */
public final class ResnetBlock implements MultiInputLayer {

    /**
     * The shape of the block.
     *
     * @param channels the input channels
     * @param filters the output channels
     * @param timeFeatures the width of the time embedding, or 0 for none
     * @param groups group-norm groups; must divide both widths
     * @param epsilon the group norms' epsilon
     * @param dropout the dropout rate before the second convolution while training
     */
    public record Settings(int channels, int filters, int timeFeatures, int groups, float epsilon, float dropout) {

        /** Checks the values. */
        public Settings {
            if (channels < 1 || filters < 1 || timeFeatures < 0 || groups < 1 || channels % groups != 0
                    || filters % groups != 0) {
                throw new IllegalArgumentException(groups + " groups must divide " + channels + " and " + filters
                        + " channels");
            }
            if (!(dropout >= 0 && dropout < 1)) {
                throw new IllegalArgumentException("dropout must be in [0, 1), got " + dropout);
            }
        }
    }

    private final Settings settings;
    private final GroupNorm norm1;
    private final Conv conv1;
    private final Dense timeProjection;
    private final GroupNorm norm2;
    private final Conv conv2;
    private final Conv shortcut;
    private final SplittableRandom random;

    /**
     * Creates a block with fresh weights.
     *
     * @param settings its shape
     * @param random the source of initial weights and dropout
     */
    public ResnetBlock(Settings settings, SplittableRandom random) {
        this.settings = Objects.requireNonNull(settings, "settings");
        int c = settings.channels();
        int f = settings.filters();
        norm1 = new GroupNorm(settings.groups(), c, settings.epsilon(), true);
        conv1 = Parts.conv3(c, f, random);
        timeProjection = settings.timeFeatures() > 0
                ? Parts.dense(settings.timeFeatures(), f, Activation.IDENTITY, random) : null;
        norm2 = new GroupNorm(settings.groups(), f, settings.epsilon(), true);
        conv2 = Parts.conv3(f, f, random);
        shortcut = c != f ? Parts.conv1(c, f, random) : null;
        this.random = random.split();
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var x = inputs.get(0);
        Var h = conv1.forward(Parts.silu(norm1.forward(x)));
        if (timeProjection != null) {
            if (inputs.size() < 2) {
                throw new IllegalArgumentException(this + " needs a time embedding as its second input");
            }
            Var e = timeProjection.forward(Parts.silu(inputs.get(1)));
            h = h.add(e.reshape(e.shape().dim(0), settings.filters(), 1, 1));
        }
        h = Parts.silu(norm2.forward(h));
        if (training && settings.dropout() > 0) {
            Tensor keep = Tensor.random(h.shape(), random.nextLong(), h.value().device())
                    .greater(Tensor.full(Shape.of(), settings.dropout(), h.value().device()));
            h = h.mul(keep.div(1 - settings.dropout()).to(h.value().dtype()));
        }
        h = conv2.forward(h);
        return (shortcut == null ? x : shortcut.forward(x)).add(h);
    }

    @Override
    public Var forward(Var input) {
        return forward(List.of(input), false);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return outputShape(inputs.get(0));
    }

    @Override
    public Shape outputShape(Shape input) {
        if (input.rank() != 3 || input.dim(0) != settings.channels()) {
            throw new IllegalArgumentException(this + " expects [" + settings.channels() + ", h, w], got " + input);
        }
        return Shape.of(settings.filters(), input.dim(1), input.dim(2));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(norm1.parameters());
        p.addAll(conv1.parameters());
        if (timeProjection != null) {
            p.addAll(timeProjection.parameters());
        }
        p.addAll(norm2.parameters());
        p.addAll(conv2.parameters());
        if (shortcut != null) {
            p.addAll(shortcut.parameters());
        }
        return p;
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    /** @return the first group norm */
    public GroupNorm norm1() {
        return norm1;
    }

    /** @return the first convolution */
    public Conv conv1() {
        return conv1;
    }

    /** @return the projection of the time embedding, or {@code null} */
    public Dense timeProjection() {
        return timeProjection;
    }

    /** @return the second group norm */
    public GroupNorm norm2() {
        return norm2;
    }

    /** @return the second convolution */
    public Conv conv2() {
        return conv2;
    }

    /** @return the 1x1 convolution of the skip path, or {@code null} when the width stays */
    public Conv shortcut() {
        return shortcut;
    }

    @Override
    public String toString() {
        return "ResnetBlock(" + settings.channels() + " -> " + settings.filters()
                + (settings.timeFeatures() > 0 ? ", time " + settings.timeFeatures() : "") + ")";
    }
}
