package io.github.zzuegg.brainiac.experimental.metric;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/** Shared helpers of the metric losses. */
final class MetricMath {

    private MetricMath() {
    }

    /** Class ids of a label tensor. */
    static int[] labels(Tensor target) {
        float[] t = target.toArray();
        int[] y = new int[t.length];
        for (int i = 0; i < t.length; i++) {
            y[i] = Math.round(t[i]);
        }
        return y;
    }

    /** Cosine similarities of normalised rows {@code [n, n]}. */
    static Var cosine(Var a, Var b) {
        Var an = a.div(a.square().sum(-1, true).add(1e-8f).sqrt());
        Var bn = b.div(b.square().sum(-1, true).add(1e-8f).sqrt());
        return an.matmul(bn.transpose());
    }

    /** Euclidean distances of all row pairs {@code [n, n]}, differentiable (with a floor). */
    static Var distances(Var z) {
        Var sq = z.square().sum(-1, true);
        Var d2 = sq.add(sq.transpose()).sub(z.matmul(z.transpose()).mul(2f));
        return d2.maximum(0f).add(1e-8f).sqrt();
    }

    /** Row-major values of a {@code [n, n]} matrix. */
    static float[][] matrix(Var m) {
        int n = m.shape().dim(0);
        int k = m.shape().dim(1);
        float[] v = m.value().toArray();
        float[][] out = new float[n][k];
        for (int i = 0; i < n; i++) {
            System.arraycopy(v, i * k, out[i], 0, k);
        }
        return out;
    }
}
