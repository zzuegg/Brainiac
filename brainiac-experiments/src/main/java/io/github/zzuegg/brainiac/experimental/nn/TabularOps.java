package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/** Shared helpers of the tabular deep learning blocks. */
final class TabularOps {

    private TabularOps() {
    }

    /** A normal tensor parameter. */
    static Var normal(RandomGenerator random, float std, int... dims) {
        return Var.param(Tensor.normal(Shape.of(dims), 0, std, random));
    }

    /** A zero tensor parameter. */
    static Var zeros(int... dims) {
        return Var.param(Tensor.zeros(Shape.of(dims)));
    }

    /** A constant tensor parameter. */
    static Var full(float value, int... dims) {
        return Var.param(Tensor.full(Shape.of(dims), value));
    }

    /**
     * One linear map per feature: {@code [b, f, k] x [f, k, d] -> [b, f, d]}.
     */
    static Var perFeature(Var x, Var weight) {
        return x.permute(1, 0, 2).matmul(weight).permute(1, 0, 2);
    }

    /** Requires {@code [batch, columns]} input. */
    static void requireRows(Var x, int columns, Object layer) {
        if (x.shape().rank() != 2 || x.shape().dim(1) != columns) {
            throw new IllegalArgumentException(layer + " expects [batch, " + columns + "], got " + x.shape());
        }
    }
}
