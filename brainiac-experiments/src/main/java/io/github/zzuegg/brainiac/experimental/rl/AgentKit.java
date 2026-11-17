package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.List;
import java.util.SplittableRandom;

/** Small helpers shared by the experimental RL blocks. */
final class AgentKit {

    private AgentKit() {
    }

    static Model mlp(int in, int out, long seed, int... hidden) {
        Model.Builder b = Model.sequential().input(in);
        for (int h : hidden) {
            b.dense(h, Activation.TANH);
        }
        return b.dense(out).seed(seed).build();
    }

    static Tensor row(float[] x) {
        return Tensor.of(Shape.of(1, x.length), x);
    }

    static Tensor rows(List<float[]> rows) {
        int w = rows.get(0).length;
        float[] flat = new float[rows.size() * w];
        for (int i = 0; i < rows.size(); i++) {
            System.arraycopy(rows.get(i), 0, flat, i * w, w);
        }
        return Tensor.of(Shape.of(rows.size(), w), flat);
    }

    static Tensor column(float[] v) {
        return Tensor.of(Shape.of(v.length, 1), v);
    }

    static float[] values(Var v) {
        return v.value().to(DType.FLOAT32).toArray();
    }

    static float[] predict(Model m, float[] x) {
        return values(m.forward(Var.of(row(x)), false));
    }

    static float[] oneHot(int i, int n) {
        float[] h = new float[n];
        h[i] = 1;
        return h;
    }

    static float[] concat(float[] a, float[] b) {
        float[] out = java.util.Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
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

    static int sample(float[] logits, SplittableRandom random) {
        double max = Double.NEGATIVE_INFINITY;
        for (float l : logits) {
            max = Math.max(max, l);
        }
        double[] p = new double[logits.length];
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            p[i] = Math.exp(logits[i] - max);
            sum += p[i];
        }
        double u = random.nextDouble() * sum;
        for (int i = 0; i < p.length; i++) {
            u -= p[i];
            if (u <= 0) {
                return i;
            }
        }
        return p.length - 1;
    }

    /** Running mean and variance (Welford), for normalizing rewards and observations. */
    static final class Running {
        private double mean;
        private double m2;
        private long n;

        void add(double x) {
            n++;
            double d = x - mean;
            mean += d / n;
            m2 += d * (x - mean);
        }

        double mean() {
            return mean;
        }

        double std() {
            return n < 2 ? 1 : Math.sqrt(m2 / (n - 1)) + 1e-8;
        }

        long count() {
            return n;
        }
    }
}
