package io.github.zzuegg.brainiac.experimental.causal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.RegressionForest;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Any weighted regression method, the plug-in nuisance or effect model of the meta-learners,
 * double machine learning, causal forests and uplift models. Causal estimators only ever call
 * {@link #fit}, so swapping ridge for a forest, boosted trees or an MLP is one argument.
 *
 * <p>Factories: {@link #ridge} (closed form), {@link #forest} ({@link RegressionForest}, weighted),
 * {@link #boosted} (core {@link GradientBoostedTrees}; it has no sample weights, so weighted fits
 * resample rows in proportion to the weights) and {@link #mlp} (a core {@link Model} trained by
 * Adam on weighted squared error).
 */
@FunctionalInterface
public interface Regressor {

    /** A fitted regression function. */
    @FunctionalInterface
    interface Fitted {
        /**
         * @param x a row
         * @return the prediction
         */
        double predict(double[] x);

        /**
         * @param x rows
         * @return one prediction per row
         */
        default double[] predict(double[][] x) {
            double[] out = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                out[i] = predict(x[i]);
            }
            return out;
        }
    }

    /**
     * Fits the model.
     *
     * @param x rows {@code [n][d]}
     * @param y targets
     * @param weights sample weights, or null for equal weights
     * @return the fitted function
     */
    Fitted fit(double[][] x, double[] y, double[] weights);

    /**
     * @param x rows
     * @param y targets
     * @return the unweighted fit
     */
    default Fitted fit(double[][] x, double[] y) {
        return fit(x, y, null);
    }

    /**
     * Ridge regression with an intercept, solved in closed form with the core {@link Linalg#solve}.
     *
     * @param l2 the ridge on the weights (not the intercept)
     * @return the regressor
     */
    static Regressor ridge(double l2) {
        return (x, y, w) -> {
            int d = x[0].length + 1;
            double[][] a = new double[d][d];
            double[] b = new double[d];
            double[] r = new double[d];
            for (int i = 0; i < x.length; i++) {
                double wi = w == null ? 1 : w[i];
                r[0] = 1;
                System.arraycopy(x[i], 0, r, 1, d - 1);
                for (int p = 0; p < d; p++) {
                    b[p] += wi * r[p] * y[i];
                    for (int q = 0; q < d; q++) {
                        a[p][q] += wi * r[p] * r[q];
                    }
                }
            }
            for (int p = 0; p < d; p++) {
                a[p][p] += p == 0 ? 1e-9 : l2;
            }
            double[] beta = solve(a, b);
            return row -> {
                double s = beta[0];
                for (int j = 1; j < d; j++) {
                    s += beta[j] * row[j - 1];
                }
                return s;
            };
        };
    }

    /**
     * A weighted random forest.
     *
     * @param settings the forest settings
     * @return the regressor
     */
    static Regressor forest(RegressionForest.Settings settings) {
        return (x, y, w) -> RegressionForest.fit(x, y, w, settings)::predict;
    }

    /**
     * Core gradient-boosted trees for squared error; weights are honoured by drawing a weighted
     * resample of the rows.
     *
     * @param settings the boosting settings
     * @return the regressor
     */
    static Regressor boosted(GradientBoostedTrees.Settings settings) {
        return (x, y, w) -> {
            double[][] xs = x;
            double[] ys = y;
            if (w != null) {
                int n = x.length;
                double[] cdf = new double[n];
                double acc = 0;
                for (int i = 0; i < n; i++) {
                    acc += w[i];
                    cdf[i] = acc;
                }
                SplittableRandom r = new SplittableRandom(settings.seed() + 17);
                xs = new double[n][];
                ys = new double[n];
                for (int i = 0; i < n; i++) {
                    int k = java.util.Arrays.binarySearch(cdf, r.nextDouble() * acc);
                    k = Math.min(n - 1, k < 0 ? -k - 1 : k);
                    xs[i] = x[k];
                    ys[i] = y[k];
                }
            }
            GradientBoostedTrees g = GradientBoostedTrees.fit(xs, ys, GradientBoostedTrees.Objective.REGRESSION,
                    settings);
            return g::predict;
        };
    }

    /**
     * A two-hidden-layer ReLU MLP trained by Adam on weighted squared error over standardised
     * inputs and targets.
     *
     * @param hidden the hidden width
     * @param steps Adam steps of batch 128
     * @param lr the learning rate
     * @param seed the seed
     * @return the regressor
     */
    static Regressor mlp(int hidden, int steps, float lr, long seed) {
        return (x, y, w) -> {
            int n = x.length;
            int d = x[0].length;
            double[] mean = new double[d];
            double[] sd = new double[d];
            for (double[] row : x) {
                for (int j = 0; j < d; j++) {
                    mean[j] += row[j] / n;
                }
            }
            for (double[] row : x) {
                for (int j = 0; j < d; j++) {
                    sd[j] += (row[j] - mean[j]) * (row[j] - mean[j]) / n;
                }
            }
            for (int j = 0; j < d; j++) {
                sd[j] = Math.sqrt(sd[j]) + 1e-8;
            }
            double ym = 0;
            double ys = 0;
            for (double v : y) {
                ym += v / n;
            }
            for (double v : y) {
                ys += (v - ym) * (v - ym) / n;
            }
            double yScale = Math.sqrt(ys) + 1e-8;
            double yMean = ym;
            Model m = Model.sequential().input(d).dense(hidden, Activation.RELU).dense(hidden, Activation.RELU)
                    .dense(1).seed(seed).build();
            Optimizer opt = Optimizer.adam(lr);
            SplittableRandom r = new SplittableRandom(seed);
            int batch = Math.min(128, n);
            for (int s = 0; s < steps; s++) {
                float[] bx = new float[batch * d];
                float[] by = new float[batch];
                float[] bw = new float[batch];
                for (int i = 0; i < batch; i++) {
                    int k = r.nextInt(n);
                    for (int j = 0; j < d; j++) {
                        bx[i * d + j] = (float) ((x[k][j] - mean[j]) / sd[j]);
                    }
                    by[i] = (float) ((y[k] - yMean) / yScale);
                    bw[i] = w == null ? 1 : (float) w[k];
                }
                m.forward(Var.of(Tensor.of(Shape.of(batch, d), bx)), true).sub(Tensor.of(Shape.of(batch, 1), by))
                        .square().mul(Tensor.of(Shape.of(batch, 1), bw)).mean().backward();
                opt.step(m.parameters());
            }
            return new Fitted() {
                @Override
                public double predict(double[] row) {
                    return predict(new double[][] {row})[0];
                }

                @Override
                public double[] predict(double[][] rows) {
                    float[] f = new float[rows.length * d];
                    for (int i = 0; i < rows.length; i++) {
                        for (int j = 0; j < d; j++) {
                            f[i * d + j] = (float) ((rows[i][j] - mean[j]) / sd[j]);
                        }
                    }
                    float[] out = Var.noGrad(() -> m.forward(Var.of(Tensor.of(Shape.of(rows.length, d), f)), false))
                            .value().toArray();
                    double[] p = new double[rows.length];
                    for (int i = 0; i < p.length; i++) {
                        p[i] = out[i] * yScale + yMean;
                    }
                    return p;
                }
            };
        };
    }

    /**
     * Solves {@code a x = b} in float64 with the core {@link Linalg#solve}.
     *
     * @param a the square matrix
     * @param b the right-hand side
     * @return the solution
     */
    static double[] solve(double[][] a, double[] b) {
        int n = b.length;
        double[] flat = new double[n * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, flat, i * n, n);
        }
        Tensor ta = Tensor.of(Shape.of(n, n), flat, DType.FLOAT64, Device.cpu());
        Tensor tb = Tensor.of(Shape.of(n, 1), b, DType.FLOAT64, Device.cpu());
        return Linalg.solve(ta, tb).toDoubleArray();
    }
}
