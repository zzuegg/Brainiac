package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Modern (continuous) Hopfield networks (Ramsauer et al. 2020, "Hopfield Networks is All You
 * Need", <a href="https://arxiv.org/abs/2008.02217">arXiv:2008.02217</a>). The update rule
 * {@code xi_new = X^T softmax(beta X xi)} retrieves a stored pattern from a noisy one in one
 * step, with a capacity exponential in the dimension (a classical Hopfield net stores about
 * {@code 0.14 d}); it is exactly attention with the stored patterns as keys and values.
 *
 * <p>Three forms, as in the paper's {@code hflayers}:
 * <ul>
 * <li>{@link #retrieve}: associative recall from given stored patterns;</li>
 * <li>{@link #layer}: a {@code Hopfield} lookup layer whose stored patterns are learned
 *     parameters, a differentiable associative memory (input {@code [batch, features]} or
 *     {@code [batch, steps, features]});</li>
 * <li>{@link #pooling}: {@code HopfieldPooling}, which summarizes a set {@code [batch, items,
 *     features]} with learned query patterns, for multiple-instance learning.</li>
 * </ul>
 */
public final class ModernHopfield implements Layer {

    private final boolean pool;
    private final int features;
    private final int size;
    private final int count;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var patterns;
    private final float beta;

    private ModernHopfield(boolean pool, int features, int size, int count, float beta, SplittableRandom random) {
        this.pool = pool;
        this.features = features;
        this.size = size;
        this.count = count;
        this.beta = beta > 0 ? beta : (float) (1 / Math.sqrt(size));
        if (pool) {
            patterns = Var.param(Tensor.normal(Shape.of(count, size), 0, 1, random));
            wq = null;
            wk = Var.param(Init.XAVIER.create(Shape.of(features, size), features, size, random));
            wv = Var.param(Init.XAVIER.create(Shape.of(features, size), features, size, random));
        } else {
            patterns = Var.param(Tensor.normal(Shape.of(count, size), 0, 1, random));
            wq = Var.param(Init.XAVIER.create(Shape.of(features, size), features, size, random));
            wk = Var.param(Init.XAVIER.create(Shape.of(size, size), size, size, random));
            wv = Var.param(Init.XAVIER.create(Shape.of(size, size), size, size, random));
        }
    }

    /**
     * One or more Hopfield updates of state patterns against stored ones.
     *
     * @param stored the stored patterns {@code [n, d]} or {@code [batch, n, d]}
     * @param state the state (query) patterns {@code [batch, d]}
     * @param beta the inverse temperature; larger retrieves single patterns, smaller averages
     * @param steps how many updates; one usually suffices
     * @return the retrieved patterns {@code [batch, d]}
     */
    public static Var retrieve(Var stored, Var state, float beta, int steps) {
        Var xi = state;
        boolean batched = stored.shape().rank() == 3;
        for (int s = 0; s < steps; s++) {
            if (batched) {
                int b = xi.shape().dim(0);
                int d = xi.shape().dim(1);
                Var w = Activation.SOFTMAX.forward(xi.reshape(b, 1, d).matmul(stored.transpose()).mul(beta));
                xi = w.matmul(stored).reshape(b, d);
            } else {
                xi = Activation.SOFTMAX.forward(xi.matmul(stored.transpose()).mul(beta)).matmul(stored);
            }
        }
        return xi;
    }

    /**
     * A lookup layer over learned stored patterns:
     * {@code softmax(beta (x W_q)(Y W_k)^T) (Y W_v)}.
     *
     * @param features the input features
     * @param patterns the stored patterns
     * @param size the pattern size, also the output size
     * @param beta the inverse temperature, or 0 for {@code 1 / sqrt(size)}
     * @param random the random source for the weights
     * @return the layer
     */
    public static ModernHopfield layer(int features, int patterns, int size, float beta, SplittableRandom random) {
        return new ModernHopfield(false, features, size, patterns, beta, random);
    }

    /**
     * A pooling layer over sets: {@code softmax(beta Q (X W_k)^T) (X W_v)} with learned
     * queries {@code Q}, flattened to {@code [batch, queries * size]}.
     *
     * @param features the features per item
     * @param queries the learned query patterns
     * @param size the pattern size
     * @param beta the inverse temperature, or 0 for {@code 1 / sqrt(size)}
     * @param random the random source for the weights
     * @return the layer
     */
    public static ModernHopfield pooling(int features, int queries, int size, float beta, SplittableRandom random) {
        return new ModernHopfield(true, features, size, queries, beta, random);
    }

    /**
     * Describes a lookup layer for model builders.
     *
     * @param patterns the stored patterns
     * @param size the pattern size
     * @return the spec
     */
    public static LayerSpec layerSpec(int patterns, int size) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = size;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return layer(input.dim(-1), patterns, size, 0, random);
            }
        };
    }

    /**
     * Describes a pooling layer for model builders, on inputs {@code [items, features]}.
     *
     * @param queries the learned query patterns
     * @param size the pattern size
     * @return the spec
     */
    public static LayerSpec poolingSpec(int queries, int size) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(queries * size);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return pooling(input.dim(-1), queries, size, 0, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        if (pool) {
            if (input.shape().rank() != 3 || input.shape().dim(2) != features) {
                throw new IllegalArgumentException(this + " expects [batch, items, " + features + "], got "
                        + input.shape());
            }
            int b = input.shape().dim(0);
            Var keys = input.matmul(wk);
            Var values = input.matmul(wv);
            Var w = Activation.SOFTMAX.forward(patterns.matmul(keys.transpose()).mul(beta));
            return w.matmul(values).reshape(b, count * size);
        }
        Var keys = patterns.matmul(wk);
        Var values = patterns.matmul(wv);
        Var w = Activation.SOFTMAX.forward(input.matmul(wq).matmul(keys.transpose()).mul(beta));
        return w.matmul(values);
    }

    @Override
    public List<Var> parameters() {
        return pool ? List.of(patterns, wk, wv) : List.of(patterns, wq, wk, wv);
    }

    @Override
    public int outputSize(int inputSize) {
        return pool ? count * size : size;
    }

    @Override
    public Shape outputShape(Shape input) {
        if (pool) {
            return Shape.of(count * size);
        }
        return Layer.super.outputShape(input);
    }

    @Override
    public String toString() {
        return pool ? "HopfieldPooling(" + count + " queries)" : "HopfieldLayer(" + count + " patterns)";
    }
}
