package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.FourierFeatures;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A neural radiance field (NeRF, Mildenhall et al., 2020): an MLP that maps a 3-D point and a
 * viewing direction to a volume density and a colour. The point goes through an encoder (a
 * {@link FourierFeatures} positional encoding, a {@link HashGridEncoding}, or nothing) and a
 * two-layer trunk; the density depends on the point only, while the colour head also sees the
 * encoded direction, so view-dependent shading is possible but geometry is not. The output is
 * raw: column 0 is the pre-softplus density, columns 1-3 the pre-sigmoid colour, as
 * {@link VolumeRenderer} expects.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2003.08934">arXiv:2003.08934</a>.
 */
public final class RadianceField implements Layer {

    private final Layer encoder;
    private final FourierFeatures direction;
    private final Dense trunk1;
    private final Dense trunk2;
    private final Dense density;
    private final Dense colour1;
    private final Dense colour2;

    /**
     * Creates a field.
     *
     * @param encoder the point encoder, or {@code null} for raw coordinates
     * @param encodedWidth the width the encoder gives (3 without one)
     * @param directionOctaves the positional-encoding octaves of the direction
     * @param width the trunk width
     * @param random the source of initial weights
     */
    public RadianceField(Layer encoder, int encodedWidth, int directionOctaves, int width, RandomGenerator random) {
        this.encoder = encoder;
        this.direction = FourierFeatures.positional(3, directionOctaves, true);
        int dirWidth = 3 + 6 * directionOctaves;
        this.trunk1 = new Dense(encodedWidth, width, Activation.RELU, Init.HE, random);
        this.trunk2 = new Dense(width, width, Activation.RELU, Init.HE, random);
        this.density = new Dense(width, 1, Activation.IDENTITY, Init.XAVIER, random);
        this.colour1 = new Dense(width + dirWidth, width / 2, Activation.RELU, Init.HE, random);
        this.colour2 = new Dense(width / 2, 3, Activation.IDENTITY, Init.XAVIER, random);
    }

    /**
     * A field behind a NeRF positional encoding of the point.
     *
     * @param octaves the octaves of the point encoding, 0 for raw coordinates
     * @param directionOctaves the octaves of the direction encoding
     * @param width the trunk width
     * @param random the source of initial weights
     * @return the field
     */
    public static RadianceField positional(int octaves, int directionOctaves, int width, RandomGenerator random) {
        Layer enc = octaves == 0 ? null : FourierFeatures.positional(3, octaves, true);
        return new RadianceField(enc, 3 + 6 * octaves, directionOctaves, width, random);
    }

    /**
     * Evaluates the field.
     *
     * @param input {@code [n, 6]}: the point, then the unit view direction
     * @return {@code [n, 4]}: raw density, then raw colour
     */
    @Override
    public Var forward(Var input) {
        Var point = input.slice(1, 0, 3);
        Var dir = input.slice(1, 3, 6);
        Var h = trunk2.forward(trunk1.forward(encoder == null ? point : encoder.forward(point)));
        Var c = colour2.forward(colour1.forward(Var.concat(1, h, direction.forward(dir))));
        return Var.concat(1, density.forward(h), c);
    }

    @Override
    public int outputSize(int inputSize) {
        return 4;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        if (encoder != null) {
            p.addAll(encoder.parameters());
        }
        for (Dense d : List.of(trunk1, trunk2, density, colour1, colour2)) {
            p.addAll(d.parameters());
        }
        return p;
    }

    @Override
    public String toString() {
        return "RadianceField(" + (encoder == null ? "raw" : encoder) + ")";
    }
}
