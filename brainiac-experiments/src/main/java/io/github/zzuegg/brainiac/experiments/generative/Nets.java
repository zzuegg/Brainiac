package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/** Small building blocks shared by the generative experiments. */
final class Nets {
    private Nets() {
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

    /** Scales all gradients down so their global norm is at most {@code maxNorm}. */
    static void clip(List<Var> params, float maxNorm) {
        double total = 0;
        for (Var p : params) {
            if (p.grad() != null) {
                total += p.grad().square().sum().item();
            }
        }
        double norm = Math.sqrt(total);
        if (norm > maxNorm) {
            float s = (float) (maxNorm / norm);
            for (Var p : params) {
                if (p.grad() != null) {
                    p.setGrad(p.grad().mul(s));
                }
            }
        }
    }

    /** A column {@code [n, 1]} of per-row scalars, for broadcasting over features. */
    static Tensor column(float[] values) {
        return Tensor.of(Shape.of(values.length, 1), values.clone());
    }

    /** Squared Euclidean distance between two vectors. */
    static double distance2(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }
}
