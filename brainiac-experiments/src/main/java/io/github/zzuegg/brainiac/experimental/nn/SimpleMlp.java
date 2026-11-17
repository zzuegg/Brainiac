package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * A multi-layer perceptron whose weights are plain {@link Var} parameters that callers can
 * read and update layer by layer: the backbone that local learning rules (feedback
 * alignment, target propagation, synthetic gradients, ...) and their backpropagation
 * baselines share. Layer {@code l} computes {@code h_{l+1} = f(h_l W_l + b_l)}, with weights
 * {@code [in, out]}; the last layer is linear and gives logits.
 */
public final class SimpleMlp {

    private final int[] sizes;
    private final Var[] weights;
    private final Var[] biases;
    private final UnaryOperator<Var> activation;

    private SimpleMlp(int[] sizes, Var[] weights, Var[] biases, UnaryOperator<Var> activation) {
        this.sizes = sizes;
        this.weights = weights;
        this.biases = biases;
        this.activation = activation;
    }

    /**
     * A network with tanh hidden layers.
     *
     * @param seed the random seed
     * @param sizes the layer sizes, input first, at least two
     * @return the network
     */
    public static SimpleMlp tanh(long seed, int... sizes) {
        return of(seed, Var::tanh, sizes);
    }

    /**
     * A network with ReLU hidden layers.
     *
     * @param seed the random seed
     * @param sizes the layer sizes, input first
     * @return the network
     */
    public static SimpleMlp relu(long seed, int... sizes) {
        return of(seed, Var::relu, sizes);
    }

    /**
     * A network with any hidden activation and Glorot-normal weights (scaled by 1.4).
     *
     * @param seed the random seed
     * @param activation the hidden activation
     * @param sizes the layer sizes, input first
     * @return the network
     */
    public static SimpleMlp of(long seed, UnaryOperator<Var> activation, int... sizes) {
        if (sizes.length < 2) {
            throw new IllegalArgumentException("need at least input and output sizes");
        }
        SplittableRandom r = new SplittableRandom(seed);
        int layers = sizes.length - 1;
        Var[] w = new Var[layers];
        Var[] b = new Var[layers];
        for (int l = 0; l < layers; l++) {
            float std = (float) Math.sqrt(2.0 / (sizes[l] + sizes[l + 1])) * 1.4f;
            w[l] = Var.param(Tensor.normal(Shape.of(sizes[l], sizes[l + 1]), 0, std, r));
            b[l] = Var.param(Tensor.zeros(Shape.of(1, sizes[l + 1])));
        }
        return new SimpleMlp(sizes.clone(), w, b, activation);
    }

    /** @return the layer sizes, input first */
    public int[] sizes() {
        return sizes.clone();
    }

    /** @return the number of weight layers */
    public int depth() {
        return weights.length;
    }

    /**
     * @param l the layer
     * @return its {@code [in, out]} weight
     */
    public Var weight(int l) {
        return weights[l];
    }

    /**
     * @param l the layer
     * @return its {@code [1, out]} bias
     */
    public Var bias(int l) {
        return biases[l];
    }

    /** @return the hidden activation */
    public UnaryOperator<Var> activation() {
        return activation;
    }

    /** @return every weight and bias, layer by layer */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int l = 0; l < weights.length; l++) {
            p.add(weights[l]);
            p.add(biases[l]);
        }
        return p;
    }

    /**
     * One layer's pre-activation {@code h W + b}.
     *
     * @param l the layer
     * @param h its input
     * @return the pre-activation
     */
    public Var linear(int l, Var h) {
        return h.matmul(weights[l]).add(biases[l]);
    }

    /**
     * One layer's output: activated for hidden layers, linear for the last.
     *
     * @param l the layer
     * @param h its input
     * @return the output
     */
    public Var layer(int l, Var h) {
        Var z = linear(l, h);
        return l < weights.length - 1 ? activation.apply(z) : z;
    }

    /**
     * The logits for a batch.
     *
     * @param x the inputs, {@code [batch, sizes[0]]}
     * @return the logits
     */
    public Var forward(Var x) {
        Var h = x;
        for (int l = 0; l < weights.length; l++) {
            h = layer(l, h);
        }
        return h;
    }

    /**
     * Every layer's output for a batch, input first and logits last, without gradients.
     *
     * @param x the inputs
     * @return {@code depth() + 1} tensors
     */
    public Tensor[] activations(Tensor x) {
        Tensor[] h = new Tensor[weights.length + 1];
        h[0] = x;
        Var cur = Var.of(x);
        for (int l = 0; l < weights.length; l++) {
            cur = layer(l, cur);
            h[l + 1] = cur.value();
        }
        return h;
    }

    /**
     * Logits without a gradient graph.
     *
     * @param x the inputs
     * @return the logits
     */
    public Tensor predict(Tensor x) {
        return forward(Var.of(x)).value();
    }
}
