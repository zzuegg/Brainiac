package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A SIREN layer (Sitzmann et al., 2020): {@code sin(omega0 * (x W + b))}. The first layer
 * draws weights from {@code U(-1/in, 1/in)}, so {@code omega0 x W} spans several periods over
 * the input range; later layers draw from {@code U(-sqrt(6/in)/omega0, sqrt(6/in)/omega0)},
 * which keeps the pre-activations arcsine-distributed through depth. A sine network is smooth,
 * and every derivative of it is again a sine network, so derivative-based losses such as the
 * eikonal term of {@link SignedDistanceField} see well-behaved gradients.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2006.09661">arXiv:2006.09661</a>.
 */
public final class SineLayer implements Layer {

    private final int inputs;
    private final int outputs;
    private final float omega;
    private final Var weight;
    private final Var bias;

    /**
     * Creates the layer.
     *
     * @param inputs the input width
     * @param outputs the output width
     * @param omega the frequency factor omega0, 30 in the paper
     * @param first whether this is the first layer (which sets the initialisation)
     * @param random the source of initial weights
     */
    public SineLayer(int inputs, int outputs, float omega, boolean first, RandomGenerator random) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.omega = omega;
        double bound = first ? 1.0 / inputs : Math.sqrt(6.0 / inputs) / omega;
        float[] w = new float[inputs * outputs];
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) (bound * (2 * random.nextDouble() - 1));
        }
        float[] b = new float[outputs];
        double biasBound = 1.0 / Math.sqrt(inputs);
        for (int i = 0; i < outputs; i++) {
            b[i] = (float) (biasBound * (2 * random.nextDouble() - 1) / omega);
        }
        this.weight = Var.param(Tensor.of(Shape.of(inputs, outputs), w));
        this.bias = Var.param(Tensor.of(Shape.of(outputs), b));
    }

    /**
     * Describes the layer for model builders.
     *
     * @param outputs the output width
     * @param omega the frequency factor
     * @param first whether this is the first layer
     * @return the spec
     */
    public static LayerSpec spec(int outputs, float omega, boolean first) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), outputs);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new SineLayer(input.dim(-1), outputs, omega, first, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return input.matmul(weight).add(bias).mul(omega).sin();
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public String toString() {
        return "SineLayer(" + inputs + " -> " + outputs + ", omega " + omega + ")";
    }
}
