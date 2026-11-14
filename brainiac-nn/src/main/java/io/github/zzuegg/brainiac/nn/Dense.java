package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A fully connected layer: {@code activation(input x weight + bias)}, with weight shape
 * {@code [inputSize, units]} and bias shape {@code [units]}.
 */
public final class Dense implements Layer {

    private final int inputSize;
    private final int units;
    private final Activation activation;
    private final Var weight;
    private final Var bias;

    /**
     * Creates a layer with initialized weights and zero bias.
     *
     * @param inputSize the number of input features
     * @param units the number of outputs
     * @param activation the activation applied to the outputs
     * @param init how weights are initialized
     * @param random the random source for the initializer
     */
    public Dense(int inputSize, int units, Activation activation, Init init, RandomGenerator random) {
        this(inputSize, units, activation,
                init.create(Shape.of(inputSize, units), inputSize, units, random),
                Tensor.zeros(Shape.of(units)));
    }

    /**
     * Creates a layer with given weights, for example loaded from a file.
     *
     * @param inputSize the number of input features
     * @param units the number of outputs
     * @param activation the activation applied to the outputs
     * @param weight the weights, shape {@code [inputSize, units]}
     * @param bias the bias, shape {@code [units]}
     */
    public Dense(int inputSize, int units, Activation activation, Tensor weight, Tensor bias) {
        if (inputSize < 1 || units < 1) {
            throw new IllegalArgumentException("Dense needs at least 1 input and 1 unit, got " + inputSize + " -> " + units);
        }
        if (!weight.shape().equals(Shape.of(inputSize, units)) || !bias.shape().equals(Shape.of(units))) {
            throw new IllegalArgumentException("Dense(" + inputSize + " -> " + units + ") needs weight [" + inputSize
                    + ", " + units + "] and bias [" + units + "], got " + weight.shape() + " and " + bias.shape());
        }
        this.inputSize = inputSize;
        this.units = units;
        this.activation = activation;
        this.weight = Var.param(weight);
        this.bias = Var.param(bias);
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 2 || s.dim(1) != inputSize) {
            throw new IllegalArgumentException(
                    "Dense(" + units + ") expects input [*, " + inputSize + "], got " + s);
        }
        return activation.forward(input.matmul(weight).add(bias));
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }

    /** @return the weight parameter, shape {@code [inputSize, units]} */
    public Var weight() {
        return weight;
    }

    /** @return the bias parameter, shape {@code [units]} */
    public Var bias() {
        return bias;
    }

    /** @return the activation */
    public Activation activation() {
        return activation;
    }

    @Override
    public String toString() {
        return "Dense(" + inputSize + " -> " + units + ", " + activation + ")";
    }
}
