package io.github.zzuegg.brainiac.experiments.statistics;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/** A one-hidden-layer ReLU trunk shared by the statistics experiments' neural heads. */
final class TinyMlp {

    private final Var w1;
    private final Var b1;

    TinyMlp(int in, int hidden, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float bound = (float) Math.sqrt(6.0 / (in + hidden));
        w1 = Var.param(Tensor.uniform(Shape.of(in, hidden), -bound, bound, r));
        b1 = Var.param(Tensor.zeros(Shape.of(1, hidden)));
    }

    Var hidden(Var x) {
        return x.matmul(w1).add(b1).relu();
    }

    List<Var> parameters() {
        return new ArrayList<>(List.of(w1, b1));
    }

    static Var linear(int in, int out, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float bound = (float) Math.sqrt(6.0 / (in + out));
        return Var.param(Tensor.uniform(Shape.of(in, out), -bound, bound, r));
    }

    static Tensor matrix(double[][] x) {
        int n = x.length;
        int p = x[0].length;
        float[] v = new float[n * p];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                v[i * p + j] = (float) x[i][j];
            }
        }
        return Tensor.of(Shape.of(n, p), v);
    }

    static double[][] rows(Tensor t) {
        int n = t.shape().dim(0);
        int p = t.shape().dim(1);
        float[] v = t.toArray();
        double[][] out = new double[n][p];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                out[i][j] = v[i * p + j];
            }
        }
        return out;
    }
}
