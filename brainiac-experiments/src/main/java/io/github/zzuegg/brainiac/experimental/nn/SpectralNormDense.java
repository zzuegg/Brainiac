package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A dense layer whose weight is spectrally normalized on every forward pass (see
 * {@link SpectralNorm}): a drop-in for {@code Dense} in GAN critics and Lipschitz networks.
 */
public final class SpectralNormDense implements Layer {

    private final int inputs;
    private final int units;
    private final Activation activation;
    private final Var weight;
    private final Var bias;
    private final SpectralNorm norm;

    /**
     * Creates the layer.
     *
     * @param inputs the input width
     * @param units the output width
     * @param activation the activation
     * @param random the random source
     */
    public SpectralNormDense(int inputs, int units, Activation activation, RandomGenerator random) {
        this.inputs = inputs;
        this.units = units;
        this.activation = activation.bind(units);
        this.weight = TechniqueOps.weight(random, inputs, units);
        this.bias = Var.param(Tensor.zeros(Shape.of(units)));
        this.norm = new SpectralNorm(inputs, 1, random);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param units the output width
     * @param activation the activation
     * @return the spec
     */
    public static LayerSpec spec(int units, Activation activation) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = units;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new SpectralNormDense(input.dim(input.rank() - 1), units, activation, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return activation.forward(input.matmul(norm.normalize(weight)).add(bias));
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(weight, bias));
        p.addAll(activation.parameters());
        return p;
    }

    /** @return the largest singular value of the raw weight, as last estimated */
    public float sigma() {
        return norm.lastSigma();
    }

    @Override
    public String toString() {
        return "SpectralNormDense(" + inputs + " -> " + units + ")";
    }
}
