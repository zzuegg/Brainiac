package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * SINDy, sparse identification of nonlinear dynamics (Brunton, Proctor and Kutz, 2016): build
 * a library {@code Theta(X)} of candidate terms (polynomials, sines and cosines) evaluated on
 * measured states, estimate the derivatives {@code X'}, and solve {@code X' = Theta(X) Xi} for
 * a sparse {@code Xi} by sequentially thresholded least squares (STLSQ), so the governing
 * equations come out in closed form.
 *
 * <p>{@link #derivatives} differentiates noisy samples with a Savitzky-Golay filter (a local
 * polynomial fit per window), which is far more robust than raw finite differences; window 3
 * with order 2 is the plain central difference. Least squares use the core {@link Linalg}
 * in double precision.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/1509.03580">arXiv:1509.03580</a>.
 */
public final class Sindy {

    /**
     * A library of candidate terms.
     *
     * @param names the term names
     * @param features evaluates every term at a state
     */
    public record Library(String[] names, Function<double[], double[]> features) {

        /**
         * @param x states {@code [n][d]}
         * @return {@code Theta(X)}, {@code [n][terms]}
         */
        public double[][] evaluate(double[][] x) {
            double[][] t = new double[x.length][];
            for (int i = 0; i < x.length; i++) {
                t[i] = features.apply(x[i]);
            }
            return t;
        }
    }

    private final Library library;
    private final double[][] xi;

    private Sindy(Library library, double[][] xi) {
        this.library = library;
        this.xi = xi;
    }

    /**
     * Monomials up to a degree, optionally with {@code sin} and {@code cos} of each variable.
     *
     * @param variables the variable names
     * @param degree the largest total degree
     * @param trigonometric whether to add {@code sin(x_i)} and {@code cos(x_i)}
     * @return the library
     */
    public static Library polynomial(String[] variables, int degree, boolean trigonometric) {
        int d = variables.length;
        List<int[]> powers = new ArrayList<>();
        enumerate(new int[d], 0, degree, powers);
        powers.sort((a, b) -> Integer.compare(sum(a), sum(b)));
        List<String> names = new ArrayList<>();
        for (int[] p : powers) {
            StringBuilder s = new StringBuilder();
            for (int i = 0; i < d; i++) {
                for (int k = 0; k < p[i]; k++) {
                    s.append(s.isEmpty() ? "" : "*").append(variables[i]);
                }
            }
            names.add(s.isEmpty() ? "1" : s.toString());
        }
        if (trigonometric) {
            for (String v : variables) {
                names.add("sin(" + v + ")");
                names.add("cos(" + v + ")");
            }
        }
        return new Library(names.toArray(new String[0]), x -> {
            double[] f = new double[names.size()];
            int k = 0;
            for (int[] p : powers) {
                double v = 1;
                for (int i = 0; i < d; i++) {
                    for (int e = 0; e < p[i]; e++) {
                        v *= x[i];
                    }
                }
                f[k++] = v;
            }
            if (trigonometric) {
                for (int i = 0; i < d; i++) {
                    f[k++] = Math.sin(x[i]);
                    f[k++] = Math.cos(x[i]);
                }
            }
            return f;
        });
    }

    private static int sum(int[] a) {
        int s = 0;
        for (int v : a) {
            s += v;
        }
        return s;
    }

    private static void enumerate(int[] cur, int i, int left, List<int[]> out) {
        if (i == cur.length) {
            out.add(cur.clone());
            return;
        }
        for (int e = 0; e <= left; e++) {
            cur[i] = e;
            enumerate(cur, i + 1, left - e, out);
        }
        cur[i] = 0;
    }

    /**
     * Savitzky-Golay derivatives of uniformly sampled states: a least-squares polynomial of
     * the given order is fit in each window and differentiated at its centre (windows are
     * shifted inwards at the ends).
     *
     * @param x states {@code [n][d]}
     * @param dt the sample interval
     * @param window the odd window length, at least {@code order + 1}
     * @param order the polynomial order
     * @return derivatives {@code [n][d]}
     */
    public static double[][] derivatives(double[][] x, double dt, int window, int order) {
        int n = x.length;
        int d = x[0].length;
        int half = window / 2;
        double[][] out = new double[n][d];
        for (int c = 0; c < n; c++) {
            int start = Math.clamp(c - half, 0, n - window);
            double[][] a = new double[window][order + 1];
            double[][] y = new double[window][d];
            for (int r = 0; r < window; r++) {
                double s = (start + r - c) * dt;
                double p = 1;
                for (int o = 0; o <= order; o++) {
                    a[r][o] = p;
                    p *= s;
                }
                y[r] = x[start + r];
            }
            double[][] coef = lstsq(a, y, 0);
            out[c] = coef[1];
        }
        return out;
    }

    static Tensor f64(double[][] a) {
        int n = a.length;
        int m = a[0].length;
        double[] f = new double[n * m];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, f, i * m, m);
        }
        return Tensor.of(Shape.of(n, m), f, DType.FLOAT64, Device.cpu());
    }

    static double[][] matrix(Tensor t) {
        int n = t.shape().dim(0);
        int m = t.shape().dim(1);
        double[] f = t.toDoubleArray();
        double[][] a = new double[n][m];
        for (int i = 0; i < n; i++) {
            System.arraycopy(f, i * m, a[i], 0, m);
        }
        return a;
    }

    /**
     * Least squares {@code min |a x - y|^2 + ridge |x|^2} in double precision.
     *
     * @param a {@code [n][k]}
     * @param y {@code [n][d]}
     * @param ridge the regularisation
     * @return {@code x}, {@code [k][d]}
     */
    public static double[][] lstsq(double[][] a, double[][] y, double ridge) {
        return matrix(Linalg.lstsq(f64(a), f64(y), ridge));
    }

    /**
     * Sequentially thresholded least squares.
     *
     * @param theta the library matrix {@code [n][k]}
     * @param dx the derivatives {@code [n][d]}
     * @param threshold coefficients below it (in absolute value) are zeroed
     * @param ridge a small ridge for conditioning, for example 1e-8
     * @param iterations threshold-and-refit rounds, for example 10
     * @return sparse coefficients {@code [k][d]}
     */
    public static double[][] stlsq(double[][] theta, double[][] dx, double threshold, double ridge, int iterations) {
        int k = theta[0].length;
        int d = dx[0].length;
        double[][] xi = lstsq(theta, dx, ridge);
        for (int it = 0; it < iterations; it++) {
            boolean changed = false;
            for (int j = 0; j < d; j++) {
                List<Integer> keep = new ArrayList<>();
                for (int i = 0; i < k; i++) {
                    if (Math.abs(xi[i][j]) >= threshold) {
                        keep.add(i);
                    } else if (xi[i][j] != 0) {
                        xi[i][j] = 0;
                        changed = true;
                    }
                }
                if (keep.isEmpty()) {
                    continue;
                }
                double[][] sub = new double[theta.length][keep.size()];
                double[][] target = new double[theta.length][1];
                for (int r = 0; r < theta.length; r++) {
                    for (int c = 0; c < keep.size(); c++) {
                        sub[r][c] = theta[r][keep.get(c)];
                    }
                    target[r][0] = dx[r][j];
                }
                double[][] fit = lstsq(sub, target, ridge);
                for (int c = 0; c < keep.size(); c++) {
                    xi[keep.get(c)][j] = fit[c][0];
                }
            }
            if (!changed && it > 0) {
                break;
            }
        }
        return xi;
    }

    /**
     * Fits a SINDy model.
     *
     * @param x states {@code [n][d]}
     * @param dx their derivatives
     * @param library the candidate terms
     * @param threshold the STLSQ threshold
     * @return the model
     */
    public static Sindy fit(double[][] x, double[][] dx, Library library, double threshold) {
        return new Sindy(library, stlsq(library.evaluate(x), dx, threshold, 1e-10, 10));
    }

    /** @return the coefficients {@code [terms][d]} */
    public double[][] coefficients() {
        double[][] c = new double[xi.length][];
        for (int i = 0; i < xi.length; i++) {
            c[i] = xi[i].clone();
        }
        return c;
    }

    /** @return the library */
    public Library library() {
        return library;
    }

    /**
     * @param x a state
     * @return the model's derivative {@code Theta(x) Xi}
     */
    public double[] derivative(double[] x) {
        double[] f = library.features().apply(x);
        double[] out = new double[xi[0].length];
        for (int i = 0; i < f.length; i++) {
            for (int j = 0; j < out.length; j++) {
                out[j] += f[i] * xi[i][j];
            }
        }
        return out;
    }

    /**
     * The discovered equations as text.
     *
     * @param variables the variable names
     * @return one line per variable, such as {@code x' = -10.000 x + 10.000 y}
     */
    public String equations(String[] variables) {
        StringBuilder s = new StringBuilder();
        for (int j = 0; j < xi[0].length; j++) {
            s.append(variables[j]).append("' =");
            boolean any = false;
            for (int i = 0; i < xi.length; i++) {
                if (xi[i][j] != 0) {
                    s.append(String.format(" %+.3f %s", xi[i][j], library.names()[i]));
                    any = true;
                }
            }
            s.append(any ? "" : " 0").append('\n');
        }
        return s.toString();
    }

    /**
     * Scores recovered structure against true coefficients.
     *
     * @param truth the true coefficients {@code [terms][d]}
     * @return {@code {share of correctly classified terms (zero vs non-zero), relative
     *     coefficient error on the true support}}
     */
    public double[] score(double[][] truth) {
        int correct = 0;
        int total = 0;
        double num = 0;
        double den = 0;
        for (int i = 0; i < xi.length; i++) {
            for (int j = 0; j < xi[0].length; j++) {
                total++;
                if ((truth[i][j] != 0) == (xi[i][j] != 0)) {
                    correct++;
                }
                if (truth[i][j] != 0) {
                    num += (xi[i][j] - truth[i][j]) * (xi[i][j] - truth[i][j]);
                    den += truth[i][j] * truth[i][j];
                }
            }
        }
        return new double[] {(double) correct / total, Math.sqrt(num / den)};
    }
}
