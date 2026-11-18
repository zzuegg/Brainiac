package io.github.zzuegg.brainiac.experimental.tsc;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Lbfgs;

import java.util.List;

/**
 * Soft-DTW (Cuturi and Blondel, ICML 2017, <a href="https://arxiv.org/abs/1703.01541">arXiv:1703.01541</a>):
 * DTW's recursion {@code R(i, j) = D(i, j) + min(R(i-1, j-1), R(i-1, j), R(i, j-1))} with the
 * minimum replaced by the soft minimum {@code -gamma log sum exp(-a / gamma)}. The result is
 * differentiable everywhere; its gradient with respect to the cost matrix is the expected
 * alignment matrix {@code E}, computed by a backward recursion of the same cost as the forward one
 * (the paper's Algorithm 2), and with squared Euclidean costs the gradient with respect to a series
 * is {@code dL / dx_i = sum_j E_ij 2 (x_i - y_j)}. As {@code gamma -> 0} it tends to DTW.
 *
 * <p>{@link #loss} wraps it as an autograd loss between predicted and target series (a
 * shift-tolerant replacement for the mean squared error when training forecasters), and
 * {@link #barycenter} averages series under soft-DTW by L-BFGS ({@link Lbfgs}).
 */
public final class SoftDtw {

    private SoftDtw() {
    }

    /**
     * The soft-DTW value and alignment of two univariate series.
     *
     * @param value the soft-DTW discrepancy
     * @param alignment {@code E}: the gradient of the value with respect to the cost matrix
     */
    public record Result(double value, double[][] alignment) {
    }

    /**
     * Soft-DTW between univariate series with squared costs, value and expected alignment.
     *
     * @param x one series, length n
     * @param y another, length m
     * @param gamma the smoothing, positive
     * @return the value and the alignment matrix {@code [n][m]}
     */
    public static Result compute(double[] x, double[] y, double gamma) {
        int n = x.length;
        int m = y.length;
        double[][] d = new double[n + 2][m + 2];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                double v = x[i - 1] - y[j - 1];
                d[i][j] = v * v;
            }
        }
        double inf = Double.POSITIVE_INFINITY;
        double[][] r = new double[n + 2][m + 2];
        for (double[] row : r) {
            java.util.Arrays.fill(row, inf);
        }
        r[0][0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                r[i][j] = d[i][j] + softmin(r[i - 1][j - 1], r[i - 1][j], r[i][j - 1], gamma);
            }
        }
        double value = r[n][m];
        double[][] e = new double[n + 2][m + 2];
        for (int i = 1; i <= n; i++) {
            r[i][m + 1] = Double.NEGATIVE_INFINITY;
        }
        for (int j = 1; j <= m; j++) {
            r[n + 1][j] = Double.NEGATIVE_INFINITY;
        }
        r[n + 1][m + 1] = r[n][m];
        e[n + 1][m + 1] = 1;
        for (int j = m; j >= 1; j--) {
            for (int i = n; i >= 1; i--) {
                double a = Math.exp((r[i + 1][j] - r[i][j] - d[i + 1][j]) / gamma);
                double b = Math.exp((r[i][j + 1] - r[i][j] - d[i][j + 1]) / gamma);
                double c = Math.exp((r[i + 1][j + 1] - r[i][j] - d[i + 1][j + 1]) / gamma);
                e[i][j] = e[i + 1][j] * a + e[i][j + 1] * b + e[i + 1][j + 1] * c;
            }
        }
        double[][] out = new double[n][m];
        for (int i = 0; i < n; i++) {
            System.arraycopy(e[i + 1], 1, out[i], 0, m);
        }
        return new Result(value, out);
    }

    private static double softmin(double a, double b, double c, double gamma) {
        double min = Math.min(a, Math.min(b, c));
        if (min == Double.POSITIVE_INFINITY) {
            return min;
        }
        double s = Math.exp(-(a - min) / gamma) + Math.exp(-(b - min) / gamma) + Math.exp(-(c - min) / gamma);
        return min - gamma * Math.log(s);
    }

    /**
     * The gradient of soft-DTW with respect to the first series.
     *
     * @param x one series
     * @param y another
     * @param gamma the smoothing
     * @return {@code dL / dx}
     */
    public static double[] gradient(double[] x, double[] y, double gamma) {
        double[][] e = compute(x, y, gamma).alignment();
        double[] g = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < y.length; j++) {
                g[i] += e[i][j] * 2 * (x[i] - y[j]);
            }
        }
        return g;
    }

    /**
     * The mean soft-DTW between predicted and target rows, as an autograd loss.
     *
     * @param prediction predicted series {@code [batch, n]}
     * @param target target series {@code [batch, m]}
     * @param gamma the smoothing, such as 0.1
     * @return the loss, shape {@code [1]}, differentiable in {@code prediction}
     */
    public static Var loss(Var prediction, Tensor target, double gamma) {
        int b = prediction.shape().dim(0);
        int n = prediction.shape().dim(1);
        int m = target.shape().dim(1);
        float[] tv = target.toArray();
        double[][] grads = new double[b][];
        return Var.custom(in -> {
            float[] pv = in[0].toArray();
            double total = 0;
            for (int r = 0; r < b; r++) {
                double[] x = new double[n];
                double[] y = new double[m];
                for (int t = 0; t < n; t++) {
                    x[t] = pv[r * n + t];
                }
                for (int t = 0; t < m; t++) {
                    y[t] = tv[r * m + t];
                }
                Result res = compute(x, y, gamma);
                total += res.value();
                double[] g = new double[n];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        g[i] += res.alignment()[i][j] * 2 * (x[i] - y[j]);
                    }
                }
                grads[r] = g;
            }
            return Tensor.of(Shape.of(1), (float) (total / b));
        }, (in, out, grad) -> {
            float scale = grad.toArray()[0] / b;
            float[] g = new float[b * n];
            for (int r = 0; r < b; r++) {
                for (int i = 0; i < n; i++) {
                    g[r * n + i] = (float) (grads[r][i] * scale);
                }
            }
            return new Tensor[] {Tensor.of(Shape.of(b, n), g)};
        }, prediction);
    }

    /**
     * The soft-DTW barycentre: {@code argmin_z sum_i softDTW(z, x_i)}, by L-BFGS from an initial
     * guess (such as the Euclidean mean or a medoid).
     *
     * @param series the series to average
     * @param init the starting point
     * @param gamma the smoothing
     * @param iterations L-BFGS iterations
     * @return the barycentre
     */
    public static double[] barycenter(double[][] series, double[] init, double gamma, int iterations) {
        int n = init.length;
        float[] z0 = new float[n];
        for (int t = 0; t < n; t++) {
            z0[t] = (float) init[t];
        }
        Var z = Var.param(Tensor.of(Shape.of(1, n), z0));
        int m = series[0].length;
        float[] all = new float[series.length * m];
        for (int i = 0; i < series.length; i++) {
            for (int t = 0; t < m; t++) {
                all[i * m + t] = (float) series[i][t];
            }
        }
        Tensor target = Tensor.of(Shape.of(series.length, m), all);
        new Lbfgs(10, 1e-6).minimize(List.of(z), () -> loss(z.expand(Shape.of(series.length, n)), target, gamma),
                iterations);
        float[] v = z.value().toArray();
        double[] out = new double[n];
        for (int t = 0; t < n; t++) {
            out[t] = v[t];
        }
        return out;
    }
}
