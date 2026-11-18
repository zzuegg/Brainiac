package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * DAGMA: structure learning of a linear structural equation model {@code X = X W + noise}
 * with the log-determinant acyclicity function
 * {@code h(W) = -log det(sI - W o W) + d log s}, which is zero exactly for DAGs inside the
 * set {@code M^s} of matrices whose {@code W o W} has spectral radius below {@code s}.
 * Unlike NOTEARS' trace of a matrix exponential (or power), its gradient
 * {@code 2 W o (sI - W o W)^-T} does not vanish for large cycles, and instead of an
 * augmented Lagrangian it follows a central path: for {@code mu = 1, 0.1, 0.01, ...} it
 * minimizes {@code mu (score(W) + lambda |W|_1) + h(W)}, each problem warm-started from
 * the last, so {@code W} approaches a DAG as {@code mu -> 0}. A step that leaves
 * {@code M^s} (the determinant turns non-positive) is undone and the learning rate halved.
 *
 * <p>Drop-in beside {@link Notears}: same {@code fit(x)} contract and the same threshold.
 *
 * <p>Paper: Bello, Aragam, Ravikumar, "DAGMA: Learning DAGs via M-matrices and a
 * Log-Determinant Acyclicity Characterization" (NeurIPS 2022),
 * <a href="https://arxiv.org/abs/2209.08037">arXiv:2209.08037</a>.
 */
public final class Dagma {

    private final double lambda;
    private final double threshold;
    private final int innerSteps;
    private final int pathSteps;

    /**
     * Creates the learner.
     *
     * @param lambda the L1 weight, such as 0.03
     * @param threshold the absolute weight below which an edge is dropped, such as 0.3
     * @param innerSteps Adam steps per central-path point
     * @param pathSteps how many central-path points ({@code mu} shrinks tenfold each)
     */
    public Dagma(double lambda, double threshold, int innerSteps, int pathSteps) {
        this.lambda = lambda;
        this.threshold = threshold;
        this.innerSteps = innerSteps;
        this.pathSteps = pathSteps;
    }

    /**
     * Learns the weighted adjacency.
     *
     * @param x samples {@code [n][d]}; centered here
     * @return {@code W[i][j]} the weight of edge {@code i -> j}, 0 where there is none
     */
    public double[][] fit(double[][] x) {
        int d = x[0].length;
        double[][] cov = covariance(x);
        double[] w = new double[d * d];
        double mu = 1;
        for (int t = 0; t < pathSteps; t++) {
            double s = t == pathSteps - 1 ? 1.0 : Math.max(0.6, 1.0 - 0.1 * t);   // the paper's s schedule
            double lr = 3e-3;
            while (!adam(cov, w, mu, s, lr, t == pathSteps - 1 ? 2 * innerSteps : innerSteps)) {
                lr /= 2;   // left M^s: retry from the warm start with a smaller step
                if (lr < 1e-7) {
                    break;
                }
            }
            mu *= 0.1;
        }
        double[][] out = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                double v = w[i * d + j];
                out[i][j] = i != j && Math.abs(v) >= threshold ? v : 0;
            }
        }
        return out;
    }

    static double[][] covariance(double[][] x) {
        int n = x.length;
        int d = x[0].length;
        double[] mean = new double[d];
        for (double[] r : x) {
            for (int j = 0; j < d; j++) {
                mean[j] += r[j] / n;
            }
        }
        double[][] cov = new double[d][d];
        for (double[] r : x) {
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    cov[i][j] += (r[i] - mean[i]) * (r[j] - mean[j]) / n;
                }
            }
        }
        return cov;
    }

    /** Minimizes one central-path objective in place; false (and w untouched) if it left M^s. */
    private boolean adam(double[][] cov, double[] w0, double mu, double s, double lr, int steps) {
        int d = cov.length;
        double[] w = w0.clone();
        double[] m = new double[d * d];
        double[] v = new double[d * d];
        for (int step = 1; step <= steps; step++) {
            double[] inv = inverseInDomain(w, s, d);
            if (inv == null) {
                return false;
            }
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    int k = i * d + j;
                    if (i == j) {
                        continue;
                    }
                    double g = -cov[i][j];   // d/dW of tr((I - W)^T S (I - W)) / 2
                    for (int a = 0; a < d; a++) {
                        g += cov[i][a] * w[a * d + j];
                    }
                    g = mu * (g + lambda * Math.signum(w[k])) + 2 * w[k] * inv[j * d + i];
                    m[k] = 0.99 * m[k] + 0.01 * g;
                    v[k] = 0.999 * v[k] + 0.001 * g * g;
                    double mh = m[k] / (1 - Math.pow(0.99, step));
                    double vh = v[k] / (1 - Math.pow(0.999, step));
                    w[k] -= lr * mh / (Math.sqrt(vh) + 1e-8);
                }
            }
        }
        if (inverseInDomain(w, s, d) == null) {
            return false;
        }
        System.arraycopy(w, 0, w0, 0, w.length);
        return true;
    }

    /** {@code (sI - W o W)^-1} flattened, or null when {@code W} is outside {@code M^s}. */
    private static double[] inverseInDomain(double[] w, double s, int d) {
        double[] a = new double[d * d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                a[i * d + j] = (i == j ? s : 0) - w[i * d + j] * w[i * d + j];
            }
        }
        if (!(logDet(a, d) > Double.NEGATIVE_INFINITY)) {
            return null;
        }
        return Linalg.inverse(Tensor.of(Shape.of(d, d), a, DType.FLOAT64, Device.cpu())).toDoubleArray();
    }

    /**
     * The acyclicity value {@code h(W) = -log det(sI - W o W) + d log s}.
     *
     * @param w the adjacency {@code [d][d]}
     * @param s the domain parameter, such as 1
     * @return {@code h}, zero for a DAG, {@code +infinity} outside {@code M^s}
     */
    public static double acyclicity(double[][] w, double s) {
        int d = w.length;
        double[] a = new double[d * d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                a[i * d + j] = (i == j ? s : 0) - w[i][j] * w[i][j];
            }
        }
        double ld = logDet(a, d);
        return ld == Double.NEGATIVE_INFINITY ? Double.POSITIVE_INFINITY : -ld + d * Math.log(s);
    }

    /**
     * {@code log det} of an M-matrix candidate by LU with partial pivoting; negative infinity
     * when the determinant is not positive.
     */
    private static double logDet(double[] a0, int d) {
        double[] a = a0.clone();
        double logDet = 0;
        int sign = 1;
        for (int c = 0; c < d; c++) {
            int p = c;
            for (int r = c + 1; r < d; r++) {
                if (Math.abs(a[r * d + c]) > Math.abs(a[p * d + c])) {
                    p = r;
                }
            }
            if (a[p * d + c] == 0) {
                return Double.NEGATIVE_INFINITY;
            }
            if (p != c) {
                for (int k = 0; k < d; k++) {
                    double tmp = a[p * d + k];
                    a[p * d + k] = a[c * d + k];
                    a[c * d + k] = tmp;
                }
                sign = -sign;
            }
            double piv = a[c * d + c];
            if (piv < 0) {
                sign = -sign;
            }
            logDet += Math.log(Math.abs(piv));
            for (int r = c + 1; r < d; r++) {
                double f = a[r * d + c] / piv;
                for (int k = c; k < d; k++) {
                    a[r * d + k] -= f * a[c * d + k];
                }
            }
        }
        return sign > 0 ? logDet : Double.NEGATIVE_INFINITY;
    }
}
