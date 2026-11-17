package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A highway layer: {@code y = T(x) * H(x) + (1 - T(x)) * x}, with a transform
 * {@code H = act(x W_H + b_H)} and a learned transform gate {@code T = sigmoid(x W_T + b_T)}.
 * The gate bias starts negative (default -2), so every layer starts close to the identity
 * ("carry") and information and gradients pass through dozens of layers; the network learns
 * where to transform. Highway networks were the first very deep (100+ layer) feed-forward
 * nets trained with plain SGD and are the gated ancestor of residual connections.
 *
 * <p>Paper: Srivastava, Greff and Schmidhuber, "Highway Networks" (2015),
 * <a href="https://arxiv.org/abs/1505.00387">arXiv:1505.00387</a>.
 */
public final class HighwayDense implements Layer {

    private final int features;
    private final Activation activation;
    private final Var transform;
    private final Var transformBias;
    private final Var gate;
    private final Var gateBias;

    /**
     * Creates the layer.
     *
     * @param features the features (input and output)
     * @param activation the transform's activation
     * @param carryBias the gate bias at the start, such as -2
     * @param random the source of initial weights
     */
    public HighwayDense(int features, Activation activation, float carryBias, RandomGenerator random) {
        this.features = features;
        this.activation = activation;
        this.transform = Var.param(Tensor.normal(Shape.of(features, features), 0,
                (float) Math.sqrt(2.0 / features), random));
        this.transformBias = ClassicOps.zeros(features);
        this.gate = ClassicOps.matrix(random, features, features);
        this.gateBias = Var.param(Tensor.full(Shape.of(features), carryBias));
    }

    /**
     * Describes the layer for model builders.
     *
     * @param activation the transform's activation
     * @param carryBias the initial gate bias
     * @return the spec
     */
    public static LayerSpec spec(Activation activation, float carryBias) {
        return TechniqueOps.sameShape((s, r) -> new HighwayDense(s.dim(s.rank() - 1), activation, carryBias, r));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().dim(input.shape().rank() - 1) != features) {
            throw new IllegalArgumentException(this + " expects [*, " + features + "], got " + input.shape());
        }
        Var h = activation.forward(input.matmul(transform).add(transformBias), training);
        Var t = input.matmul(gate).add(gateBias).sigmoid();
        return input.add(t.mul(h.sub(input)));
    }

    /**
     * The mean transform-gate value for some inputs, showing how much each layer transforms.
     *
     * @param input the inputs
     * @return the mean of {@code T(x)}
     */
    public float meanGate(Tensor input) {
        return input.matmul(gate.value()).add(gateBias.value()).sigmoid().mean().item();
    }

    @Override
    public List<Var> parameters() {
        return List.of(transform, transformBias, gate, gateBias);
    }

    @Override
    public String toString() {
        return "HighwayDense(" + features + ")";
    }
}
