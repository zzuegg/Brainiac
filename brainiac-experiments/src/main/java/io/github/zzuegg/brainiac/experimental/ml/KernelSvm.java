package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A soft-margin kernel support vector machine (Cortes and Vapnik 1995,
 * <a href="https://link.springer.com/article/10.1007/BF00994018">Support-vector networks</a>)
 * trained by sequential minimal optimisation with the second-order working-set selection of
 * LIBSVM (Fan, Chen and Lin 2005). The dual
 * {@code min 1/2 a^T Q a - sum a, 0 <= a <= C, y^T a = 0} with {@code Q_ij = y_i y_j k(x_i, x_j)}
 * is solved two coordinates at a time in closed form; the decision function
 * {@code f(x) = sum_i a_i y_i k(x_i, x) - rho} only touches the support vectors
 * ({@code a_i > 0}). More than two classes are handled one-versus-one with voting.
 *
 * <p>Why it matters: a convex problem with a unique optimum, a sparse model, and with an RBF
 * kernel a strong non-linear baseline on small tabular data. Its cost is the kernel matrix,
 * quadratic in the data; {@link RandomFourierFeatures} with a {@link LinearSvm} trades that for
 * an explicit approximate feature map.
 */
public final class KernelSvm {

    /** One binary machine: support vectors, their signed weights and the offset. */
    private record Binary(int positive, int negative, double[][] vectors, double[] weights, double rho) {
    }

    private final KernelFunction kernel;
    private final int classes;
    private final List<Binary> machines;

    private KernelSvm(KernelFunction kernel, int classes, List<Binary> machines) {
        this.kernel = kernel;
        this.classes = classes;
        this.machines = machines;
    }

    /**
     * Trains a machine, one-versus-one when there are more than two classes.
     *
     * @param x the training points {@code [n][d]}
     * @param labels their classes in {@code [0, classes)}
     * @param classes the number of classes, at least 2
     * @param kernel the kernel
     * @param c the box constraint (inverse regularisation), positive
     * @param tolerance the KKT violation at which SMO stops, for example {@code 1e-3}
     * @return the trained machine
     */
    public static KernelSvm fit(double[][] x, int[] labels, int classes, KernelFunction kernel, double c,
            double tolerance) {
        Objects.requireNonNull(kernel, "kernel");
        List<Binary> machines = new ArrayList<>();
        for (int a = 0; a < classes; a++) {
            for (int b = a + 1; b < classes; b++) {
                List<double[]> px = new ArrayList<>();
                List<Double> py = new ArrayList<>();
                for (int i = 0; i < x.length; i++) {
                    if (labels[i] == a || labels[i] == b) {
                        px.add(x[i]);
                        py.add(labels[i] == a ? 1.0 : -1.0);
                    }
                }
                double[][] sx = px.toArray(new double[0][]);
                double[] sy = new double[py.size()];
                for (int i = 0; i < sy.length; i++) {
                    sy[i] = py.get(i);
                }
                machines.add(binary(a, b, sx, sy, kernel, c, tolerance));
                if (classes == 2) {
                    return new KernelSvm(kernel, classes, machines);
                }
            }
        }
        return new KernelSvm(kernel, classes, machines);
    }

    /** SMO with second-order working-set selection on one binary problem. */
    private static Binary binary(int positive, int negative, double[][] x, double[] y, KernelFunction kernel,
            double c, double tolerance) {
        int n = x.length;
        double[][] k = kernel.gram(x);
        double[] alpha = new double[n];
        double[] grad = new double[n];
        java.util.Arrays.fill(grad, -1);
        int maxIter = Math.max(10_000_000, 100 * n);
        for (int iter = 0; iter < maxIter; iter++) {
            double gmax = Double.NEGATIVE_INFINITY;
            int i = -1;
            for (int t = 0; t < n; t++) {
                if (y[t] > 0 ? alpha[t] < c : alpha[t] > 0) {
                    double v = -y[t] * grad[t];
                    if (v >= gmax) {
                        gmax = v;
                        i = t;
                    }
                }
            }
            double gmax2 = Double.NEGATIVE_INFINITY;
            int j = -1;
            double objMin = Double.POSITIVE_INFINITY;
            for (int t = 0; t < n; t++) {
                if (y[t] > 0 ? alpha[t] > 0 : alpha[t] < c) {
                    double v = y[t] * grad[t];
                    if (v >= gmax2) {
                        gmax2 = v;
                    }
                    double diff = gmax + v;
                    if (i >= 0 && diff > 0) {
                        double quad = k[i][i] + k[t][t] - 2 * k[i][t];
                        double obj = -diff * diff / (quad > 0 ? quad : 1e-12);
                        if (obj <= objMin) {
                            objMin = obj;
                            j = t;
                        }
                    }
                }
            }
            if (i < 0 || j < 0 || gmax + gmax2 < tolerance) {
                break;
            }
            double oldI = alpha[i];
            double oldJ = alpha[j];
            double quad = k[i][i] + k[j][j] - 2 * k[i][j];
            if (quad <= 0) {
                quad = 1e-12;
            }
            if (y[i] != y[j]) {
                double delta = (-grad[i] - grad[j]) / quad;
                double diff = alpha[i] - alpha[j];
                alpha[i] += delta;
                alpha[j] += delta;
                if (diff > 0) {
                    if (alpha[j] < 0) {
                        alpha[j] = 0;
                        alpha[i] = diff;
                    }
                } else if (alpha[i] < 0) {
                    alpha[i] = 0;
                    alpha[j] = -diff;
                }
                if (diff > 0) {
                    if (alpha[i] > c) {
                        alpha[i] = c;
                        alpha[j] = c - diff;
                    }
                } else if (alpha[j] > c) {
                    alpha[j] = c;
                    alpha[i] = c + diff;
                }
            } else {
                double delta = (grad[i] - grad[j]) / quad;
                double sum = alpha[i] + alpha[j];
                alpha[i] -= delta;
                alpha[j] += delta;
                if (sum > c) {
                    if (alpha[i] > c) {
                        alpha[i] = c;
                        alpha[j] = sum - c;
                    }
                } else if (alpha[j] < 0) {
                    alpha[j] = 0;
                    alpha[i] = sum;
                }
                if (sum > c) {
                    if (alpha[j] > c) {
                        alpha[j] = c;
                        alpha[i] = sum - c;
                    }
                } else if (alpha[i] < 0) {
                    alpha[i] = 0;
                    alpha[j] = sum;
                }
            }
            double dI = alpha[i] - oldI;
            double dJ = alpha[j] - oldJ;
            for (int t = 0; t < n; t++) {
                grad[t] += y[t] * (y[i] * k[t][i] * dI + y[j] * k[t][j] * dJ);
            }
        }
        // the offset from the free support vectors, or the middle of the feasible interval
        double ub = Double.POSITIVE_INFINITY;
        double lb = Double.NEGATIVE_INFINITY;
        double sum = 0;
        int free = 0;
        for (int t = 0; t < n; t++) {
            double yg = y[t] * grad[t];
            if (alpha[t] >= c) {
                if (y[t] < 0) {
                    ub = Math.min(ub, yg);
                } else {
                    lb = Math.max(lb, yg);
                }
            } else if (alpha[t] <= 0) {
                if (y[t] > 0) {
                    ub = Math.min(ub, yg);
                } else {
                    lb = Math.max(lb, yg);
                }
            } else {
                sum += yg;
                free++;
            }
        }
        double rho = free > 0 ? sum / free : (ub + lb) / 2;
        List<double[]> sv = new ArrayList<>();
        List<Double> w = new ArrayList<>();
        for (int t = 0; t < n; t++) {
            if (alpha[t] > 0) {
                sv.add(x[t]);
                w.add(alpha[t] * y[t]);
            }
        }
        double[] weights = new double[w.size()];
        for (int t = 0; t < weights.length; t++) {
            weights[t] = w.get(t);
        }
        return new Binary(positive, negative, sv.toArray(new double[0][]), weights, rho);
    }

    /**
     * The binary decision value {@code f(x)}: positive for class 0 against class 1. For more
     * classes, the value of the first one-versus-one machine (class 0 against class 1).
     *
     * @param x a point
     * @return the signed distance-like score
     */
    public double decision(double[] x) {
        return decision(machines.getFirst(), x);
    }

    private double decision(Binary m, double[] x) {
        double s = -m.rho();
        for (int i = 0; i < m.weights().length; i++) {
            s += m.weights()[i] * kernel.value(m.vectors()[i], x);
        }
        return s;
    }

    /**
     * Predicts a class by one-versus-one voting (ties go to the smaller decision margin sum).
     *
     * @param x a point
     * @return the class
     */
    public int predict(double[] x) {
        double[] votes = new double[classes];
        for (Binary m : machines) {
            double f = decision(m, x);
            votes[f > 0 ? m.positive() : m.negative()] += 1 + 1e-6 * Math.tanh(Math.abs(f));
        }
        int best = 0;
        for (int c = 1; c < classes; c++) {
            if (votes[c] > votes[best]) {
                best = c;
            }
        }
        return best;
    }

    /**
     * Predicts many points.
     *
     * @param x points
     * @return their classes
     */
    public int[] predict(double[][] x) {
        int[] out = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = predict(x[i]);
        }
        return out;
    }

    /**
     * The number of distinct support vectors summed over the binary machines.
     *
     * @return the support-vector count
     */
    public int supportVectors() {
        int s = 0;
        for (Binary m : machines) {
            s += m.weights().length;
        }
        return s;
    }

    /** @return the number of classes */
    public int classes() {
        return classes;
    }
}
