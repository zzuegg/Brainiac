package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Random node features (Sato et al., 2021, <a href="https://arxiv.org/abs/2002.03155">arXiv:2002.03155</a>;
 * Abboud et al., 2021): append a fresh random vector to every node's features on every
 * forward pass. Message passing is bounded by the 1-WL test because nodes with the same
 * neighbourhood multiset get identical embeddings; random identifiers break that symmetry,
 * which makes message-passing networks universal in probability (they can tell apart, for
 * example, regular graphs of different structure). The network has to learn to be
 * invariant to the particular random values, so its predictions become stochastic; the
 * test averages over several draws.
 *
 * <p>Input {@code [n, f]}, output {@code [n, f + dims]}.
 */
public final class RandomNodeFeatures implements Layer {

    private final int dims;
    private final RandomGenerator random;

    /**
     * Creates the layer.
     *
     * @param dims random features per node
     * @param random the source of the random features (drawn anew every call)
     */
    public RandomNodeFeatures(int dims, RandomGenerator random) {
        this.dims = dims;
        this.random = random;
    }

    /**
     * @param dims random features per node
     * @return a spec for model builders (each created layer draws from its creation seed)
     */
    public static LayerSpec spec(int dims) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), input.dim(1) + dims);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new RandomNodeFeatures(dims, random.split());
            }
        };
    }

    @Override
    public Var forward(Var input) {
        int n = input.shape().dim(0);
        return Var.concat(1, input, Var.of(Tensor.uniform(Shape.of(n, dims), 0, 1, random)));
    }

    @Override
    public int outputSize(int inputSize) {
        return inputSize + dims;
    }

    @Override
    public Shape outputShape(Shape inputShape) {
        return Shape.of(inputShape.dim(0), inputShape.dim(1) + dims);
    }

    @Override
    public String toString() {
        return "RandomNodeFeatures(" + dims + ")";
    }
}
