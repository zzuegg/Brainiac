package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A dense layer with weight normalisation: each output unit's weight vector is
 * reparameterised as {@code w = g * v / |v|}, decoupling its length {@code g} from its
 * direction {@code v}. Gradient descent on {@code (g, v)} is better conditioned than on
 * {@code w}, without any dependence on the batch (unlike batch norm), so it suits small
 * batches, RNNs and RL. A drop-in for core {@code Dense}.
 *
 * <p>Paper: Salimans and Kingma, "Weight Normalization: A Simple Reparameterization to
 * Accelerate Training of Deep Neural Networks" (NeurIPS 2016),
 * <a href="https://arxiv.org/abs/1602.07868">arXiv:1602.07868</a>.
 */
public final class WeightNormDense implements Layer {

    private final int inputs;
    private final int units;
    private final Activation activation;
    private final Var direction;
    private final Var length;
    private final Var bias;

    /**
     * Creates the layer; {@code g} starts at each column's initial norm, so the initial
     * function equals a plainly initialised dense layer.
     *
     * @param inputs the input features
     * @param units the output features
     * @param activation the activation
     * @param random the source of initial weights
     */
    public WeightNormDense(int inputs, int units, Activation activation, RandomGenerator random) {
        this.inputs = inputs;
        this.units = units;
        this.activation = activation;
        this.direction = ClassicOps.matrix(random, inputs, units);
        this.length = Var.param(direction.value().square().sum(0, false).sqrt());
        this.bias = ClassicOps.zeros(units);
    }

    /**
     * Describes the layer for model builders.
     *
     * @param units the output features
     * @param activation the activation
     * @return the spec
     */
    public static LayerSpec spec(int units, Activation activation) {
        return ClassicOps.spec(s -> ClassicOps.withLast(s, units),
                (s, r) -> new WeightNormDense(s.dim(s.rank() - 1), units, activation, r));
    }

    /** @return the effective weight {@code g v / |v|}, {@code [inputs, units]} */
    public Var weight() {
        return direction.mul(length.div(direction.square().sum(0, true).add(1e-12f).sqrt()));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().dim(input.shape().rank() - 1) != inputs) {
            throw new IllegalArgumentException(this + " expects [*, " + inputs + "], got " + input.shape());
        }
        return activation.forward(input.matmul(weight()).add(bias), training);
    }

    @Override
    public List<Var> parameters() {
        return List.of(direction, length, bias);
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }

    @Override
    public String toString() {
        return "WeightNormDense(" + inputs + " -> " + units + ")";
    }
}
