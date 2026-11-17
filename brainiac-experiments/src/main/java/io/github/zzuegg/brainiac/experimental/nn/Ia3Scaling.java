package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * (IA)^3, "infused adapter by inhibiting and amplifying inner activations" (Liu et al. 2022,
 * <a href="https://arxiv.org/abs/2205.05638">arXiv:2205.05638</a>): a learned vector
 * {@code l} (start 1) that multiplies a frozen layer's output element-wise. Applied to the
 * attention keys and values and to the feed-forward hidden activation, it adapts a
 * transformer with about {@code 2 d + d_ff} values per block, fewer than any other method,
 * and folds into the weights after training.
 */
public final class Ia3Scaling implements Layer {

    private final Layer base;
    private final Var scale;

    /**
     * @param base the frozen layer (its parameters are not trained)
     * @param outputs the width of its output
     */
    public Ia3Scaling(Layer base, int outputs) {
        this.base = base;
        this.scale = Var.param(Tensor.ones(Shape.of(outputs)));
    }

    /** @return the scaling vector */
    public Var scale() {
        return scale;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        return base.forward(input, training).mul(scale);
    }

    @Override
    public List<Var> parameters() {
        return List.of(scale);
    }

    @Override
    public int outputSize(int inputSize) {
        return base.outputSize(inputSize);
    }

    @Override
    public String toString() {
        return "Ia3Scaling(" + base + ")";
    }
}
