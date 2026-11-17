package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A factorization machine (Rendle, 2010, <a href="https://doi.org/10.1109/ICDM.2010.127">ICDM 2010</a>):
 * {@code y = w0 + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j}, a linear model plus every
 * pairwise interaction with factorized weights. Each feature gets a {@code k}-dimensional
 * vector, so interactions between features that never co-occurred in training are still
 * estimated, which is why FMs (and their deep descendants, DeepFM and DCN) dominate
 * sparse recommendation and click-through data. The pairwise sum is computed in
 * {@code O(n k)} as {@code 0.5 * sum_f ((x V)_f^2 - (x^2 V^2)_f)}.
 *
 * <p>Input {@code [batch, features]} (typically one- or multi-hot), output {@code [batch, 1]}.
 */
public final class FactorizationMachine implements Layer {

    private final int features;
    private final Var bias;
    private final Var linear;
    private final Var factors;

    /**
     * Creates the layer.
     *
     * @param features input features
     * @param rank the factor dimension {@code k}
     * @param random the initialization source
     */
    public FactorizationMachine(int features, int rank, SplittableRandom random) {
        this.features = features;
        bias = Var.param(Tensor.zeros(Shape.of(1, 1)));
        linear = Var.param(Tensor.zeros(Shape.of(features, 1)));
        factors = Var.param(Tensor.normal(Shape.of(features, rank), 0, 0.1f, random));
    }

    /**
     * @param rank the factor dimension
     * @return a spec for {@code Model.sequential().input(features).add(...)}
     */
    public static LayerSpec spec(int rank) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(1);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new FactorizationMachine(input.dim(-1), rank, random);
            }
        };
    }

    /** @return the {@code [features, rank]} factor matrix (feature embeddings) */
    public Var factors() {
        return factors;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().dim(-1) != features) {
            throw new IllegalArgumentException("FactorizationMachine expects " + features + " features, got "
                    + input.shape());
        }
        Var xv = input.matmul(factors);
        Var x2v2 = input.square().matmul(factors.square());
        Var pairs = xv.square().sub(x2v2).sum(1, true).mul(0.5f);
        return input.matmul(linear).add(bias).add(pairs);
    }

    @Override
    public List<Var> parameters() {
        return List.of(bias, linear, factors);
    }

    @Override
    public int outputSize(int inputSize) {
        return 1;
    }

    @Override
    public String toString() {
        return "FactorizationMachine(" + features + ", rank " + factors.shape().dim(1) + ")";
    }
}
