package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * A dense layer factored into two thin matrices, {@code y = act((x U) V + b)} with
 * {@code U [in, r]} and {@code V [r, out]}: {@code r (in + out)} values instead of
 * {@code in * out}, and as many multiply-adds. Built from a trained {@link Dense} by
 * truncated SVD ({@link #factorize}), the classic low-rank compression of weight
 * matrices, or by activation-aware SVD ({@link #activationAware}, after ASVD, Yuan et al.
 * 2023, <a href="https://arxiv.org/abs/2312.05821">arXiv:2312.05821</a>): the weight is
 * scaled by the input features' root-mean-square before the SVD and unscaled after, so the
 * truncation spends its rank on directions that carry large activations. Both factors stay
 * trainable for a short recovery fine-tune.
 */
public final class LowRankDense implements Layer {

    private final Var u;
    private final Var v;
    private final Var bias;
    private final Activation activation;

    /**
     * @param u {@code [in, r]}
     * @param v {@code [r, out]}
     * @param bias {@code [out]}
     * @param activation the activation
     */
    public LowRankDense(Tensor u, Tensor v, Tensor bias, Activation activation) {
        this.u = Var.param(u);
        this.v = Var.param(v);
        this.bias = Var.param(bias);
        this.activation = activation;
    }

    /**
     * @param base a trained layer
     * @param rank the rank to keep
     * @return the best rank-{@code r} approximation (Eckart-Young), singular values split evenly
     */
    public static LowRankDense factorize(Dense base, int rank) {
        ThinSvd svd = ThinSvd.of(base.weight().value());
        return new LowRankDense(svd.left(rank, 0.5), svd.right(rank, 0.5), base.bias().value(), base.activation());
    }

    /**
     * @param base a trained layer
     * @param inputs calibration inputs {@code [samples, in]}
     * @param rank the rank to keep
     * @return the activation-aware factorization
     */
    public static LowRankDense activationAware(Dense base, Tensor inputs, int rank) {
        double[][] w = ThinSvd.toMatrix(base.weight().value());
        double[][] x = ThinSvd.toMatrix(inputs);
        int in = w.length;
        double[] s = new double[in];
        for (double[] row : x) {
            for (int i = 0; i < in; i++) {
                s[i] += row[i] * row[i];
            }
        }
        for (int i = 0; i < in; i++) {
            s[i] = Math.sqrt(s[i] / x.length) + 1e-6;
            for (int j = 0; j < w[0].length; j++) {
                w[i][j] *= s[i];
            }
        }
        ThinSvd svd = ThinSvd.of(w);
        Tensor left = svd.left(rank, 0.5);
        float[] l = left.toArray();
        for (int i = 0; i < in; i++) {
            for (int k = 0; k < rank; k++) {
                l[i * rank + k] /= (float) s[i];
            }
        }
        return new LowRankDense(Tensor.of(Shape.of(in, rank), l), svd.right(rank, 0.5), base.bias().value(),
                base.activation());
    }

    /** @return the rank */
    public int rank() {
        return u.shape().dim(1);
    }

    /** @return the product {@code U V}, the weight it stands for */
    public Tensor weight() {
        return u.value().matmul(v.value());
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        return activation.forward(input.matmul(u).matmul(v).add(bias), training);
    }

    @Override
    public List<Var> parameters() {
        return List.of(u, v, bias);
    }

    @Override
    public int outputSize(int inputSize) {
        return v.shape().dim(1);
    }

    @Override
    public String toString() {
        return "LowRankDense(" + u.shape().dim(0) + " -> " + rank() + " -> " + v.shape().dim(1) + ")";
    }
}
