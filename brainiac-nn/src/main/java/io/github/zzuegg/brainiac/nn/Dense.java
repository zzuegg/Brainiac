package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
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
    private volatile Capture capture;

    /**
     * Receives, for each backward pass, what second-order optimizers such as K-FAC and
     * Shampoo collect per layer.
     */
    @FunctionalInterface
    public interface Capture {

        /**
         * Called during {@code backward()}.
         *
         * @param input the layer's input {@code [*, inputSize]} of that forward pass
         * @param preActivationGradient the gradient of {@code input x weight + bias},
         *     {@code [*, units]}
         */
        void accept(Tensor input, Tensor preActivationGradient);
    }

    /**
     * Creates a layer with initialized weights and zero bias.
     *
     * @param inputSize the number of input features
     * @param units the number of outputs
     * @param activation the activation applied to the outputs; a learnable one is bound
     *     to {@code units}, so this layer gets its own copy of its parameters
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
        this.activation = Objects.requireNonNull(activation, "activation").bind(units);
        this.weight = Var.param(weight);
        this.bias = Var.param(bias);
    }

    /**
     * Describes a dense layer whose input width is taken from the last dimension of its
     * input. On inputs with more dimensions, such as sequences {@code [steps, features]},
     * the same weights apply at every position.
     *
     * @param units the number of outputs
     * @param activation the activation
     * @param init the weight initializer
     * @return the spec
     */
    public static LayerSpec spec(int units, Activation activation, Init init) {
        if (units < 1) {
            throw new IllegalArgumentException("units must be at least 1, got " + units);
        }
        Objects.requireNonNull(activation, "activation");
        Objects.requireNonNull(init, "init");
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] dims = input.toArray();
                dims[dims.length - 1] = units;
                return Shape.of(dims);
            }

            @Override
            public Layer create(Shape input, java.util.SplittableRandom random) {
                return new Dense(input.dim(-1), units, activation, init, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Shape s = input.shape();
        if (s.rank() < 2 || s.dim(-1) != inputSize) {
            throw new IllegalArgumentException(
                    "Dense(" + units + ") expects input [*, " + inputSize + "], got " + s);
        }
        Var pre = input.matmul(weight).add(bias);
        Capture c = capture;
        if (c != null && pre.requiresGrad()) {
            Tensor x = input.value();
            pre.onGradient(g -> {
                c.accept(x, g);
                return null;
            });
        }
        return activation.forward(pre, training);
    }

    /**
     * Captures the layer's inputs and pre-activation gradients on each later forward and
     * backward pass, or stops with {@code null}.
     *
     * @param capture what receives them, or {@code null}
     */
    public void capture(Capture capture) {
        this.capture = capture;
    }

    @Override
    public List<Var> parameters() {
        List<Var> params = new ArrayList<>(List.of(weight, bias));
        params.addAll(activation.parameters());
        return params;
    }

    @Override
    public List<Var> buffers() {
        return activation.buffers();
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

    /** @return the activation, bound to this layer's width */
    public Activation activation() {
        return activation;
    }

    @Override
    public String toString() {
        return "Dense(" + inputSize + " -> " + units + ", " + activation + ")";
    }
}
