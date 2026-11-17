package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;

/**
 * Sparsemax (Martins and Astudillo, 2016, <a href="https://arxiv.org/abs/1602.02068">arXiv:1602.02068</a>):
 * the Euclidean projection of a score vector onto the probability simplex,
 * {@code p = max(z - tau, 0)} with {@code tau} chosen so {@code p} sums to 1. Like softmax
 * it gives a distribution, but low scores get exactly zero, so attention or feature
 * selection is truly sparse and interpretable; TabNet's feature masks use it. The
 * gradient is {@code s * (g - mean_s(g))} over the support {@code s}, written by hand
 * here as an {@link Op}.
 */
public final class Sparsemax implements Op {

    /** Sparsemax over the last axis as an {@link Activation}. */
    public static final Activation ACTIVATION = Activation.named("sparsemax", Sparsemax::apply);

    private static final Sparsemax INSTANCE = new Sparsemax();

    private Sparsemax() {
    }

    /**
     * @param x scores; sparsemax is taken over the last axis
     * @return the sparse probabilities
     */
    public static Var apply(Var x) {
        return Var.apply(INSTANCE, x);
    }

    @Override
    public Tensor forward(Tensor... inputs) {
        Tensor x = inputs[0];
        int n = x.shape().dim(x.shape().rank() - 1);
        float[] z = x.to(DType.FLOAT32).toArray();
        float[] p = new float[z.length];
        float[] row = new float[n];
        for (int r = 0; r < z.length / n; r++) {
            System.arraycopy(z, r * n, row, 0, n);
            Arrays.sort(row);
            double cumulative = 0;
            double tau = 0;
            for (int k = 1; k <= n; k++) {
                float zk = row[n - k];
                cumulative += zk;
                if (1 + k * zk > cumulative) {
                    tau = (cumulative - 1) / k;
                }
            }
            for (int i = 0; i < n; i++) {
                p[r * n + i] = (float) Math.max(0, z[r * n + i] - tau);
            }
        }
        return Tensor.of(x.shape(), p).to(x.device());
    }

    @Override
    public Tensor[] backward(Tensor grad, Tensor output, Tensor... inputs) {
        int axis = output.shape().rank() - 1;
        Tensor support = output.greater(Tensor.full(io.github.zzuegg.brainiac.tensor.Shape.of(1), 0f)
                .to(output.device()));
        Tensor mean = grad.mul(support).sum(axis, true).div(support.sum(axis, true));
        return new Tensor[] {grad.sub(mean).mul(support)};
    }

    @Override
    public String name() {
        return "sparsemax";
    }
}
