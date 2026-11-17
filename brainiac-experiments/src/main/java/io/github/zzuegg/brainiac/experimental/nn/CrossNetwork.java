package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The cross network of DCN-V2 (Wang et al., 2020, <a href="https://arxiv.org/abs/2008.13535">arXiv:2008.13535</a>):
 * layers {@code x_{l+1} = x_0 * (x_l W_l + b_l) + x_l}. Each layer raises the polynomial
 * degree of feature interactions by one, explicitly and cheaply, which plain MLPs learn
 * only inefficiently; usually stacked or run in parallel with a deep MLP for ranking and
 * tabular prediction.
 *
 * <p>Input {@code [batch, features]}, output {@code [batch, features]}.
 */
public final class CrossNetwork implements Layer {

    private final int features;
    private final List<Var> weights = new ArrayList<>();
    private final List<Var> biases = new ArrayList<>();

    /**
     * Creates the network.
     *
     * @param features input (and output) features
     * @param layers cross layers
     * @param random the initialization source
     */
    public CrossNetwork(int features, int layers, SplittableRandom random) {
        this.features = features;
        for (int l = 0; l < layers; l++) {
            weights.add(Var.param(Tensor.normal(Shape.of(features, features), 0,
                    (float) Math.sqrt(1.0 / features), random)));
            biases.add(Var.param(Tensor.zeros(Shape.of(1, features))));
        }
    }

    /**
     * @param layers cross layers
     * @return a spec for {@code Model.sequential().add(...)}
     */
    public static LayerSpec spec(int layers) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new CrossNetwork(input.dim(-1), layers, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().dim(-1) != features) {
            throw new IllegalArgumentException("CrossNetwork expects " + features + " features, got " + input.shape());
        }
        Var x = input;
        for (int l = 0; l < weights.size(); l++) {
            x = input.mul(x.matmul(weights.get(l)).add(biases.get(l))).add(x);
        }
        return x;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(weights);
        p.addAll(biases);
        return p;
    }

    @Override
    public String toString() {
        return "CrossNetwork(" + features + ", " + weights.size() + " layers)";
    }
}
