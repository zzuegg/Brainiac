package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A dense layer regularised by DropConnect: while training, each <em>weight</em> (rather than
 * each unit, as in dropout) is zeroed with probability {@code rate} and the survivors are
 * scaled by {@code 1 / (1 - rate)}, so every example of a batch sees one random sub-network
 * of the weight matrix. At inference the full (expected) weights are used. It is a drop-in
 * for core {@code Dense}.
 *
 * <p>Paper: Wan et al., "Regularization of Neural Networks using DropConnect" (ICML 2013),
 * <a href="http://proceedings.mlr.press/v28/wan13.html">PMLR 28</a>.
 */
public final class DropConnectDense implements Layer {

    private final int inputs;
    private final int units;
    private final Activation activation;
    private final float rate;
    private final Var weight;
    private final Var bias;
    private final RandomGenerator random;

    /**
     * Creates the layer.
     *
     * @param inputs the input features
     * @param units the output features
     * @param activation the activation
     * @param rate the share of weights dropped while training, in {@code [0, 1)}
     * @param random the source of initial weights and masks
     */
    public DropConnectDense(int inputs, int units, Activation activation, float rate, RandomGenerator random) {
        if (!(rate >= 0 && rate < 1)) {
            throw new IllegalArgumentException("rate must be in [0, 1), got " + rate);
        }
        this.inputs = inputs;
        this.units = units;
        this.activation = activation;
        this.rate = rate;
        this.weight = ClassicOps.matrix(random, inputs, units);
        this.bias = ClassicOps.zeros(units);
        this.random = random;
    }

    /**
     * Describes the layer for model builders.
     *
     * @param units the output features
     * @param activation the activation
     * @param rate the drop rate
     * @return the spec
     */
    public static LayerSpec spec(int units, Activation activation, float rate) {
        return ClassicOps.spec(s -> ClassicOps.withLast(s, units),
                (s, r) -> new DropConnectDense(s.dim(s.rank() - 1), units, activation, rate, r));
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
        Var w = weight;
        if (training && rate > 0) {
            Tensor keep = ClassicOps.bernoulli(weight.shape(), 1 - rate, random, weight.value().device());
            w = weight.mul(keep.div(1 - rate));
        }
        return activation.forward(input.matmul(w).add(bias), training);
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }

    /** @return the drop rate */
    public float rate() {
        return rate;
    }

    @Override
    public String toString() {
        return "DropConnectDense(" + inputs + " -> " + units + ", rate " + rate + ")";
    }
}
