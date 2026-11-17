package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/** Small building blocks shared by the generative experiments. */
final class Nets {
    private Nets() {
    }

    /** A fully connected layer {@code x W + b}. */
    static final class Linear {
        final Var weight;
        final Var bias;

        Linear(int in, int out, RandomGenerator random) {
            this(in, out, (float) Math.sqrt(1.0 / in), random);
        }

        Linear(int in, int out, float std, RandomGenerator random) {
            weight = Var.param(std == 0 ? Tensor.zeros(Shape.of(in, out))
                    : Tensor.normal(Shape.of(in, out), 0, std, random));
            bias = Var.param(Tensor.zeros(Shape.of(out)));
        }

        Var forward(Var x) {
            return x.matmul(weight).add(bias);
        }

        List<Var> parameters() {
            return List.of(weight, bias);
        }
    }

    /** A multi-layer perceptron with SiLU between layers and a linear last layer. */
    static final class Mlp {
        final List<Linear> layers = new ArrayList<>();

        Mlp(RandomGenerator random, int... sizes) {
            for (int i = 0; i + 1 < sizes.length; i++) {
                layers.add(new Linear(sizes[i], sizes[i + 1], random));
            }
        }

        Var forward(Var x) {
            for (int i = 0; i < layers.size(); i++) {
                x = layers.get(i).forward(x);
                if (i + 1 < layers.size()) {
                    x = silu(x);
                }
            }
            return x;
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            for (Linear l : layers) {
                p.addAll(l.parameters());
            }
            return p;
        }
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
