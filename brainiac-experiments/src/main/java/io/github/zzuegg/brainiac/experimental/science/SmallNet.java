package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/** A small fully connected network (tanh hidden layers, linear output) on autograd {@link Var}s. */
final class SmallNet {

    private final List<Var> weights = new ArrayList<>();
    private final List<Var> biases = new ArrayList<>();

    SmallNet(int[] sizes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        for (int l = 0; l + 1 < sizes.length; l++) {
            int in = sizes[l];
            int out = sizes[l + 1];
            float[] w = new float[in * out];
            double sd = Math.sqrt(1.0 / in);
            for (int i = 0; i < w.length; i++) {
                w[i] = (float) (sd * r.nextGaussian());
            }
            weights.add(Var.param(Tensor.of(Shape.of(in, out), w)));
            biases.add(Var.param(Tensor.zeros(Shape.of(out))));
        }
    }

    List<Var> parameters() {
        List<Var> p = new ArrayList<>(weights);
        p.addAll(biases);
        return p;
    }

    Var forward(Var x) {
        Var h = x;
        for (int l = 0; l < weights.size(); l++) {
            h = h.matmul(weights.get(l)).add(biases.get(l));
            if (l + 1 < weights.size()) {
                h = h.tanh();
            }
        }
        return h;
    }

    static Tensor tensor(double[][] rows) {
        int n = rows.length;
        int d = rows[0].length;
        float[] v = new float[n * d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) rows[i][j];
            }
        }
        return Tensor.of(Shape.of(n, d), v);
    }

    double[][] predict(double[][] x) {
        float[] out = Var.noGrad(() -> forward(Var.of(tensor(x)))).value().toArray();
        int d = out.length / x.length;
        double[][] y = new double[x.length][d];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < d; j++) {
                y[i][j] = out[i * d + j];
            }
        }
        return y;
    }
}
