package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/** Helpers shared by the generative building blocks in this package. */
final class GenBlocks {
    private GenBlocks() {
    }

    static Dense linear(int in, int out, RandomGenerator random) {
        return new Dense(in, out, Activation.IDENTITY, Init.LECUN, random);
    }

    static Dense zeroLinear(int in, int out) {
        return new Dense(in, out, Activation.IDENTITY, Tensor.zeros(Shape.of(in, out)), Tensor.zeros(Shape.of(out)));
    }

    static Var silu(Var x) {
        return x.mul(x.sigmoid());
    }

    /**
     * Sinusoidal features of continuous times in [0, 1], as in transformers' position
     * encodings, so a network can tell nearby noise levels apart.
     */
    static Tensor timeEmbedding(float[] t, int dim) {
        int half = dim / 2;
        float[] out = new float[t.length * dim];
        for (int i = 0; i < t.length; i++) {
            for (int k = 0; k < half; k++) {
                double freq = Math.exp(-Math.log(100.0) * k / Math.max(1, half - 1));
                double a = t[i] * 100.0 * freq;
                out[i * dim + k] = (float) Math.sin(a);
                out[i * dim + half + k] = (float) Math.cos(a);
            }
        }
        return Tensor.of(Shape.of(t.length, dim), out);
    }

    /** One-hot rows; a label of -1 selects the extra "no label" slot at index {@code classes}. */
    static Tensor oneHot(int[] labels, int classes) {
        float[] out = new float[labels.length * (classes + 1)];
        for (int i = 0; i < labels.length; i++) {
            int c = labels[i] < 0 ? classes : labels[i];
            out[i * (classes + 1) + c] = 1;
        }
        return Tensor.of(Shape.of(labels.length, classes + 1), out);
    }

    static Tensor column(float[] values) {
        return Tensor.of(Shape.of(values.length, 1), values.clone());
    }

    static Tensor matrix(float[][] rows) {
        int cols = rows[0].length;
        float[] flat = new float[rows.length * cols];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(rows[i], 0, flat, i * cols, cols);
        }
        return Tensor.of(Shape.of(rows.length, cols), flat);
    }

    static float[][] rows(Tensor t) {
        int r = t.shape().dim(0);
        int c = Math.toIntExact(t.size() / r);
        float[] flat = t.toArray();
        float[][] out = new float[r][c];
        for (int i = 0; i < r; i++) {
            System.arraycopy(flat, i * c, out[i], 0, c);
        }
        return out;
    }

    static Tensor gaussian(int rows, int cols, RandomGenerator random) {
        return Tensor.normal(Shape.of(rows, cols), 0, 1, random);
    }

    static Tensor times(float[] t) {
        return Tensor.of(Shape.of(t.length), t.clone());
    }

    static void clip(java.util.List<Var> params, float maxNorm) {
        double total = 0;
        for (Var p : params) {
            if (p.grad() != null) {
                total += p.grad().square().sum().item();
            }
        }
        double norm = Math.sqrt(total);
        if (norm > maxNorm) {
            float f = (float) (maxNorm / norm);
            for (Var p : params) {
                if (p.grad() != null) {
                    p.setGrad(p.grad().mul(f));
                }
            }
        }
    }
}
