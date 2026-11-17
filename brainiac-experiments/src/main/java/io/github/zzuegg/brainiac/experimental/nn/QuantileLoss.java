package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * The pinball (quantile) loss (Koenker and Bassett, 1978): for quantile {@code q},
 * {@code max(q (y - f), (q - 1)(y - f))}. Its minimizer is the {@code q}-quantile of the
 * target distribution, so a model with one output per quantile learns prediction
 * intervals directly, without assuming a distribution (as MQ-RNN, TFT and most
 * probabilistic forecasting competitions do). A distribution-free alternative to
 * {@link GaussianHead}.
 */
public final class QuantileLoss {

    private QuantileLoss() {
    }

    /**
     * @param predictions {@code [batch, quantiles]} (or {@code [..., quantiles]})
     * @param target {@code [batch, 1]}, broadcast over the quantiles
     * @param quantiles the quantile of every output column, each in {@code (0, 1)}
     * @return the mean pinball loss
     */
    public static Var pinball(Var predictions, Tensor target, float... quantiles) {
        int rank = predictions.shape().rank();
        if (predictions.shape().dim(rank - 1) != quantiles.length) {
            throw new IllegalArgumentException("one output per quantile expected, got " + predictions.shape());
        }
        int[] dims = new int[rank];
        java.util.Arrays.fill(dims, 1);
        dims[rank - 1] = quantiles.length;
        Tensor q = Tensor.of(Shape.of(dims), quantiles).to(target.device());
        Var error = Var.of(target).sub(predictions);
        return error.mul(q).maximum(error.mul(q.sub(1f))).mean();
    }
}
