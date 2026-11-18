package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/** Small helpers shared by the recsys blocks. */
final class RecsysOps {

    private RecsysOps() {
    }

    static Tensor ids(int[] ids) {
        float[] f = new float[ids.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = ids[i];
        }
        return Tensor.of(Shape.of(ids.length), f);
    }

    static Tensor ids(int[][] ids) {
        int rows = ids.length;
        int cols = ids[0].length;
        float[] f = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                f[i * cols + j] = ids[i][j];
            }
        }
        return Tensor.of(Shape.of(rows, cols), f);
    }

    /** Xavier-scaled weight matrix. */
    static Var matrix(RandomGenerator random, int rows, int cols) {
        return Var.param(Tensor.normal(Shape.of(rows, cols), 0, (float) Math.sqrt(2.0 / (rows + cols)), random));
    }

    static Var zeros(int... dims) {
        return Var.param(Tensor.zeros(Shape.of(dims)));
    }

    static Var normal(RandomGenerator random, float std, int... dims) {
        return Var.param(Tensor.normal(Shape.of(dims), 0, std, random));
    }

    /** Row-wise L2 normalization of {@code [rows, dim]}. */
    static Var normalize(Var x) {
        return x.div(x.square().sum(-1, true).add(1e-8f).sqrt());
    }

    static float[][] rows(Tensor t) {
        int n = t.shape().dim(0);
        int d = (int) (t.shape().size() / n);
        float[] a = t.toArray();
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a, i * d, out[i], 0, d);
        }
        return out;
    }

    static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
