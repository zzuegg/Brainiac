package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Turns a diffusion step (or any number, such as an image size) into a vector a network can
 * condition on: sines and cosines of the number at geometrically spaced frequencies, then
 * two dense layers with SiLU between them, as diffusion UNets embed their timestep.
 *
 * <pre>{@code
 * TimestepEmbedding time = new TimestepEmbedding(TimestepEmbedding.Settings.of(64, 256), random);
 * Var embedding = time.forward(Var.of(steps));   // steps [batch] -> [batch, 256]
 * }</pre>
 */
public final class TimestepEmbedding implements Layer {

    /**
     * The shape of the embedding.
     *
     * @param frequencies how many sines and cosines (even); the first dense layer's input
     * @param features the width of the embedding
     * @param flipSinToCos whether the cosines come first, as in Stable Diffusion
     * @param freqShift the frequency shift: frequency {@code i} is {@code 10000^(-i / (half - shift))}
     */
    public record Settings(int frequencies, int features, boolean flipSinToCos, int freqShift) {

        /** Checks the values. */
        public Settings {
            if (frequencies < 2 || frequencies % 2 != 0 || features < 1) {
                throw new IllegalArgumentException("frequencies must be even and positive, got " + frequencies
                        + " and " + features + " features");
            }
        }

        /**
         * The settings Stable Diffusion uses: cosines first, no shift.
         *
         * @param frequencies how many sines and cosines
         * @param features the width of the embedding
         * @return the settings
         */
        public static Settings of(int frequencies, int features) {
            return new Settings(frequencies, features, true, 0);
        }
    }

    private final Settings settings;
    private final Dense linear1;
    private final Dense linear2;

    /**
     * Creates an embedding with fresh weights.
     *
     * @param settings its shape
     * @param random the source of initial weights
     */
    public TimestepEmbedding(Settings settings, SplittableRandom random) {
        this.settings = Objects.requireNonNull(settings, "settings");
        linear1 = Parts.dense(settings.frequencies(), settings.features(), Activation.SILU, random);
        linear2 = Parts.dense(settings.features(), settings.features(), Activation.IDENTITY, random);
    }

    /**
     * The sinusoidal embedding of each value, without the dense layers.
     *
     * @param values {@code [n]} (or {@code [n, 1]}), any number format
     * @param dim the width, even
     * @param flipSinToCos whether cosines come first
     * @param freqShift the frequency shift
     * @return {@code [n, dim]} in float32, on the device of {@code values}
     */
    public static Tensor sinusoids(Tensor values, int dim, boolean flipSinToCos, int freqShift) {
        int n = Math.toIntExact(values.size());
        int half = dim / 2;
        float[] freqs = new float[half];
        for (int j = 0; j < half; j++) {
            freqs[j] = (float) Math.exp(-Math.log(10000) * j / (half - freqShift));
        }
        Tensor args = values.to(DType.FLOAT32).reshape(n, 1)
                .mul(Tensor.of(Shape.of(1, half), freqs, values.device()));
        Tensor sin = args.sin();
        Tensor cos = args.cos();
        return flipSinToCos ? Tensor.concat(1, cos, sin) : Tensor.concat(1, sin, cos);
    }

    /**
     * Embeds a batch of steps.
     *
     * @param input the steps, {@code [batch]} or {@code [batch, 1]}; no gradient flows to them
     * @return {@code [batch, features]}
     */
    @Override
    public Var forward(Var input) {
        DType type = linear1.weight().value().dtype();
        Tensor e = sinusoids(input.value(), settings.frequencies(), settings.flipSinToCos(), settings.freqShift());
        return linear2.forward(linear1.forward(Var.of(e.to(type))));
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(settings.features());
    }

    @Override
    public int outputSize(int inputSize) {
        return settings.features();
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(linear1.parameters());
        p.addAll(linear2.parameters());
        return p;
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    /** @return the first dense layer, with SiLU */
    public Dense linear1() {
        return linear1;
    }

    /** @return the second dense layer */
    public Dense linear2() {
        return linear2;
    }

    @Override
    public String toString() {
        return "TimestepEmbedding(" + settings.frequencies() + " -> " + settings.features() + ")";
    }
}
