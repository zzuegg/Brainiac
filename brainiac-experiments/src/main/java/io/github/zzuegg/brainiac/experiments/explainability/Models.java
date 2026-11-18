package io.github.zzuegg.brainiac.experiments.explainability;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;
import java.util.function.Function;

/** Shared model fitting and conversion helpers of the explainability experiments. */
final class Models {

    private Models() {
    }

    static Tensor tensor(double[][] x) {
        int d = x[0].length;
        float[] v = new float[x.length * d];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) x[i][j];
            }
        }
        return Tensor.of(Shape.of(x.length, d), v);
    }

    static Tensor labels(double[] y) {
        float[] v = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            v[i] = (float) y[i];
        }
        return Tensor.of(Shape.of(y.length), v);
    }

    /** An MLP regressor trained by Adam on mean squared error. */
    static Model mlpRegressor(double[][] x, double[] y, int hidden, int steps, long seed) {
        Model m = Model.sequential().input(x[0].length).dense(hidden, Activation.TANH).dense(hidden, Activation.TANH)
                .dense(1, Activation.IDENTITY).seed(seed).build();
        Tensor tx = tensor(x);
        Tensor ty = labels(y).reshape(y.length, 1);
        Optimizer opt = Optimizer.adam(5e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int batch = Math.min(64, x.length);
        for (int s = 0; s < steps; s++) {
            int from = r.nextInt(x.length - batch + 1);
            Var out = m.forward(Var.of(tx.slice(0, from, from + batch)), true);
            out.sub(ty.slice(0, from, from + batch)).square().mean().backward();
            opt.step(m.parameters());
        }
        return m;
    }

    /** An MLP classifier with {@code classes} logits trained by cross-entropy. */
    static Model mlpClassifier(double[][] x, double[] y, int classes, int hidden, int steps, long seed) {
        Model m = Model.sequential().input(x[0].length).dense(hidden, Activation.RELU).dense(hidden, Activation.RELU)
                .dense(classes, Activation.IDENTITY).seed(seed).build();
        Tensor tx = tensor(x);
        Tensor ty = labels(y);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int batch = Math.min(64, x.length);
        for (int s = 0; s < steps; s++) {
            int from = r.nextInt(x.length - batch + 1);
            Functional.crossEntropy(m.forward(Var.of(tx.slice(0, from, from + batch)), true),
                    ty.slice(0, from, from + batch)).backward();
            opt.step(m.parameters());
        }
        return m;
    }

    /** The first output of a model as a batch function. */
    static Function<double[][], double[]> output(Model m) {
        return rows -> {
            double[] v = m.forward(Var.of(tensor(rows))).value().toDoubleArray();
            int k = v.length / rows.length;
            double[] out = new double[rows.length];
            for (int i = 0; i < rows.length; i++) {
                out[i] = v[i * k];
            }
            return out;
        };
    }

    /** The class-1 probability of a two-logit classifier as a batch function. */
    static Function<double[][], double[]> probability(Model m) {
        return rows -> {
            double[] v = m.forward(Var.of(tensor(rows))).value().toDoubleArray();
            double[] out = new double[rows.length];
            for (int i = 0; i < rows.length; i++) {
                out[i] = 1 / (1 + Math.exp(v[2 * i] - v[2 * i + 1]));
            }
            return out;
        };
    }

    static double accuracy(Function<double[][], double[]> probability, double[][] x, double[] y) {
        double[] p = probability.apply(x);
        int ok = 0;
        for (int i = 0; i < y.length; i++) {
            if ((p[i] > 0.5 ? 1 : 0) == (int) y[i]) {
                ok++;
            }
        }
        return ok / (double) y.length;
    }

    /** AUROC of scores against binary labels. */
    static double auroc(double[] score, double[] y) {
        double pos = 0;
        double neg = 0;
        double wins = 0;
        for (int i = 0; i < y.length; i++) {
            if (y[i] != 1) {
                continue;
            }
            pos++;
            for (int j = 0; j < y.length; j++) {
                if (y[j] == 1) {
                    continue;
                }
                wins += score[i] > score[j] ? 1 : score[i] == score[j] ? 0.5 : 0;
            }
        }
        for (double v : y) {
            if (v != 1) {
                neg++;
            }
        }
        return wins / (pos * neg);
    }

    /** Spearman rank correlation. */
    static double spearman(double[] a, double[] b) {
        return pearson(ranks(a), ranks(b));
    }

    static double[] ranks(double[] v) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        java.util.Arrays.sort(idx, (p, q) -> Double.compare(v[p], v[q]));
        double[] r = new double[v.length];
        for (int k = 0; k < idx.length; k++) {
            r[idx[k]] = k;
        }
        return r;
    }

    static double pearson(double[] a, double[] b) {
        double ma = java.util.Arrays.stream(a).average().orElse(0);
        double mb = java.util.Arrays.stream(b).average().orElse(0);
        double sab = 0;
        double saa = 0;
        double sbb = 0;
        for (int i = 0; i < a.length; i++) {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }
        return saa == 0 || sbb == 0 ? 0 : sab / Math.sqrt(saa * sbb);
    }
}
