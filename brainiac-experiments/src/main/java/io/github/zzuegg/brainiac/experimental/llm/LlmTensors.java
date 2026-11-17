package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.List;
import java.util.SplittableRandom;

/** Small tensor and sampling helpers shared by the experimental language-model blocks. */
final class LlmTensors {

    private LlmTensors() {
    }

    static Tensor rows(float[][] rows) {
        int width = rows[0].length;
        float[] flat = new float[rows.length * width];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(rows[i], 0, flat, i * width, width);
        }
        return Tensor.of(Shape.of(rows.length, width), flat);
    }

    static Tensor column(float[] values) {
        return Tensor.of(Shape.of(values.length, 1), values);
    }

    static Tensor oneHot(int[] ids, int n) {
        float[] h = new float[ids.length * n];
        for (int i = 0; i < ids.length; i++) {
            h[i * n + ids[i]] = 1;
        }
        return Tensor.of(Shape.of(ids.length, n), h);
    }

    static float[] values(Var v) {
        return v.value().to(DType.FLOAT32).toArray();
    }

    static float[] predict(Model model, float[] x) {
        return values(model.forward(Var.of(Tensor.of(Shape.of(1, x.length), x)), false));
    }

    /** Row-wise log-softmax. */
    static Var logSoftmax(Var logits) {
        return Activation.LOG_SOFTMAX.forward(logits);
    }

    /** Picks {@code matrix[i, ids[i]]} for every row: {@code [n, 1]}. */
    static Var pick(Var matrix, int[] ids) {
        return matrix.mul(oneHot(ids, matrix.shape().dim(1))).sum(1, true);
    }

    static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    static double[] softmax(float[] logits, double temperature) {
        double max = Double.NEGATIVE_INFINITY;
        for (float l : logits) {
            max = Math.max(max, l);
        }
        double[] p = new double[logits.length];
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            p[i] = Math.exp((logits[i] - max) / temperature);
            sum += p[i];
        }
        for (int i = 0; i < p.length; i++) {
            p[i] /= sum;
        }
        return p;
    }

    static int sample(double[] p, SplittableRandom random) {
        double u = random.nextDouble();
        double c = 0;
        for (int i = 0; i < p.length; i++) {
            c += p[i];
            if (u < c) {
                return i;
            }
        }
        return p.length - 1;
    }

    static int sample(float[] logits, double temperature, SplittableRandom random) {
        if (temperature <= 0) {
            return argmax(logits);
        }
        return sample(softmax(logits, temperature), random);
    }

    static double mean(float[] v) {
        double s = 0;
        for (float x : v) {
            s += x;
        }
        return v.length == 0 ? 0 : s / v.length;
    }

    static void copy(List<Var> from, List<Var> to) {
        for (int i = 0; i < from.size(); i++) {
            to.get(i).assign(from.get(i).value());
        }
    }

    static Var param(SplittableRandom random, float scale, int... dims) {
        Shape shape = Shape.of(dims);
        float[] v = new float[Math.toIntExact(shape.size())];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (random.nextGaussian() * scale);
        }
        return Var.param(Tensor.of(shape, v));
    }

    static Var zeros(int... dims) {
        return Var.param(Tensor.zeros(Shape.of(dims)));
    }
}
