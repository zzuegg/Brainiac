package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/** Small helpers shared by the wave-6 generative blocks in this package. */
final class GenerativeOps {
    private GenerativeOps() {
    }

    /** A residual-free MLP with SiLU between layers; the last layer may start at zero. */
    static final class Mlp {
        final List<Var> weights = new ArrayList<>();
        final List<Var> biases = new ArrayList<>();

        Mlp(RandomGenerator random, boolean zeroLast, int... sizes) {
            for (int i = 0; i + 1 < sizes.length; i++) {
                boolean last = i + 2 == sizes.length;
                weights.add(last && zeroLast ? TechniqueOps.filled(0f, sizes[i], sizes[i + 1])
                        : TechniqueOps.weight(random, sizes[i], sizes[i + 1]));
                biases.add(TechniqueOps.filled(0f, sizes[i + 1]));
            }
        }

        Var apply(Var x) {
            Var h = x;
            for (int i = 0; i < weights.size(); i++) {
                h = h.matmul(weights.get(i)).add(biases.get(i));
                if (i + 1 < weights.size()) {
                    h = silu(h);
                }
            }
            return h;
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            for (int i = 0; i < weights.size(); i++) {
                p.add(weights.get(i));
                p.add(biases.get(i));
            }
            return p;
        }
    }

    static Var silu(Var x) {
        return x.mul(x.sigmoid());
    }

    /** A column {@code [n, 1]} of per-row values. */
    static Tensor column(float[] values) {
        return Tensor.of(Shape.of(values.length, 1), values.clone());
    }

    /** A column {@code [n, 1]} filled with one value. */
    static Tensor column(int n, float value) {
        return Tensor.full(Shape.of(n, 1), value);
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

    /** Sinusoidal features {@code [n, 2 * half]} of per-row times, as a constant. */
    static Tensor timeFeatures(float[] t, int half, double maxFrequency) {
        float[] out = new float[t.length * 2 * half];
        for (int i = 0; i < t.length; i++) {
            for (int k = 0; k < half; k++) {
                double f = maxFrequency * Math.exp(-Math.log(maxFrequency) * k / Math.max(1, half - 1));
                out[i * 2 * half + k] = (float) Math.sin(t[i] * f);
                out[i * 2 * half + half + k] = (float) Math.cos(t[i] * f);
            }
        }
        return Tensor.of(Shape.of(t.length, 2 * half), out);
    }

    /** Sinusoidal features of a differentiable time column {@code [n, 1]}. */
    static Var timeFeatures(Var t, int half, double maxFrequency) {
        float[] f = new float[half];
        for (int k = 0; k < half; k++) {
            f[k] = (float) (maxFrequency * Math.exp(-Math.log(maxFrequency) * k / Math.max(1, half - 1)));
        }
        Var a = t.matmul(Var.of(Tensor.of(Shape.of(1, half), f)));
        return Var.concat(1, a.sin(), a.cos());
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

    /** Zeroes the gradients of all parameters. */
    static void zeroGrad(List<Var> params) {
        for (Var p : params) {
            p.zeroGrad();
        }
    }
}
