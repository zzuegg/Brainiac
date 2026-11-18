package io.github.zzuegg.brainiac.tensor;

import java.util.Arrays;
import java.util.function.ToDoubleFunction;

/**
 * Small host optimizers without gradients: Nelder–Mead for smooth or noisy functions of a few
 * variables, and the simplex method for linear programs, as SciPy's {@code minimize(method=
 * "Nelder-Mead")} and {@code linprog}.
 */
public final class Optimize {

    private Optimize() {
    }

    /**
     * Where a minimization ended.
     *
     * @param x the best point
     * @param value the function there
     * @param iterations the iterations taken
     * @param converged whether the tolerance was met
     */
    public record Minimum(double[] x, double value, int iterations, boolean converged) {
    }

    /**
     * Minimizes by the Nelder–Mead simplex method with the adaptive coefficients of Gao and
     * Han (2012), which keep it working in higher dimensions.
     *
     * @param f the function
     * @param x0 the starting point
     * @param step the initial simplex's size along each axis, such as 0.05 of each value's scale
     * @param maxIterations the most iterations
     * @param tolerance stop when the simplex's values and points spread less than this
     * @return the minimum found
     */
    public static Minimum nelderMead(ToDoubleFunction<double[]> f, double[] x0, double step, int maxIterations,
            double tolerance) {
        int n = x0.length;
        double alpha = 1;
        double beta = 1 + 2.0 / n;
        double gamma = 0.75 - 1.0 / (2 * n);
        double delta = 1 - 1.0 / n;
        double[][] s = new double[n + 1][];
        double[] fs = new double[n + 1];
        s[0] = x0.clone();
        for (int i = 0; i < n; i++) {
            s[i + 1] = x0.clone();
            s[i + 1][i] += x0[i] != 0 ? step * Math.max(1, Math.abs(x0[i])) : step;
        }
        for (int i = 0; i <= n; i++) {
            fs[i] = f.applyAsDouble(s[i]);
        }
        int it = 0;
        boolean converged = false;
        Integer[] order = new Integer[n + 1];
        while (it < maxIterations) {
            for (int i = 0; i <= n; i++) {
                order[i] = i;
            }
            double[] fv = fs;
            Arrays.sort(order, (a, b) -> Double.compare(fv[a], fv[b]));
            double[][] sorted = new double[n + 1][];
            double[] fsorted = new double[n + 1];
            for (int i = 0; i <= n; i++) {
                sorted[i] = s[order[i]];
                fsorted[i] = fs[order[i]];
            }
            s = sorted;
            fs = fsorted;
            double spreadF = fs[n] - fs[0];
            double spreadX = 0;
            for (int i = 1; i <= n; i++) {
                for (int j = 0; j < n; j++) {
                    spreadX = Math.max(spreadX, Math.abs(s[i][j] - s[0][j]));
                }
            }
            if (spreadF <= tolerance && spreadX <= tolerance) {
                converged = true;
                break;
            }
            it++;
            double[] centroid = new double[n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    centroid[j] += s[i][j] / n;
                }
            }
            double[] xr = along(centroid, s[n], -alpha);
            double fr = f.applyAsDouble(xr);
            if (fr < fs[0]) {
                double[] xe = along(centroid, s[n], -alpha * beta);
                double fe = f.applyAsDouble(xe);
                if (fe < fr) {
                    s[n] = xe;
                    fs[n] = fe;
                } else {
                    s[n] = xr;
                    fs[n] = fr;
                }
            } else if (fr < fs[n - 1]) {
                s[n] = xr;
                fs[n] = fr;
            } else {
                boolean outside = fr < fs[n];
                double[] xc = outside ? along(centroid, s[n], -alpha * gamma) : along(centroid, s[n], gamma);
                double fc = f.applyAsDouble(xc);
                if (fc <= (outside ? fr : fs[n])) {
                    s[n] = xc;
                    fs[n] = fc;
                } else {
                    for (int i = 1; i <= n; i++) {
                        for (int j = 0; j < n; j++) {
                            s[i][j] = s[0][j] + delta * (s[i][j] - s[0][j]);
                        }
                        fs[i] = f.applyAsDouble(s[i]);
                    }
                }
            }
        }
        int best = 0;
        for (int i = 1; i <= n; i++) {
            if (fs[i] < fs[best]) {
                best = i;
            }
        }
        return new Minimum(s[best].clone(), fs[best], it, converged);
    }

    // centroid + t (point - centroid)
    private static double[] along(double[] centroid, double[] point, double t) {
        double[] out = new double[centroid.length];
        for (int j = 0; j < out.length; j++) {
            out[j] = centroid[j] + t * (point[j] - centroid[j]);
        }
        return out;
    }

    // ------------------------------------------------------------------ linear programs

    /** How a linear program ended. */
    public enum Status {
        /** an optimum was found */
        OPTIMAL,
        /** no point meets the constraints */
        INFEASIBLE,
        /** the objective decreases without bound */
        UNBOUNDED
    }

    /**
     * The solution of a linear program.
     *
     * @param status how it ended
     * @param x the optimal point, or null
     * @param value the optimal objective, or NaN
     */
    public record LinearSolution(Status status, double[] x, double value) {
    }

    /**
     * Minimizes {@code c x} subject to {@code A_ub x <= b_ub}, {@code A_eq x = b_eq} and
     * {@code x >= 0}, by the two-phase simplex method with Bland's rule (so it cannot cycle).
     * Right-hand sides may have any sign. To maximize, negate {@code c}; for a free variable,
     * write it as the difference of two non-negative ones.
     *
     * @param c the costs, one per variable
     * @param aUb the inequality rows, or null
     * @param bUb their bounds, or null
     * @param aEq the equality rows, or null
     * @param bEq their values, or null
     * @return the solution
     */
    public static LinearSolution linearProgram(double[] c, double[][] aUb, double[] bUb, double[][] aEq, double[] bEq) {
        int n = c.length;
        int mu = aUb == null ? 0 : aUb.length;
        int me = aEq == null ? 0 : aEq.length;
        int m = mu + me;
        // columns: n variables, mu slacks, m artificials, then the right-hand side
        int cols = n + mu + m + 1;
        double[][] t = new double[m + 1][cols];
        int[] basis = new int[m];
        for (int i = 0; i < m; i++) {
            double[] row = i < mu ? aUb[i] : aEq[i - mu];
            double rhs = i < mu ? bUb[i] : bEq[i - mu];
            if (row.length != n) {
                throw new IllegalArgumentException("constraint " + i + " has " + row.length + " coefficients, not " + n);
            }
            double sign = rhs < 0 ? -1 : 1;
            for (int j = 0; j < n; j++) {
                t[i][j] = sign * row[j];
            }
            if (i < mu) {
                t[i][n + i] = sign;
            }
            t[i][n + mu + i] = 1;
            t[i][cols - 1] = sign * rhs;
            basis[i] = n + mu + i;
        }
        // phase 1: minimize the sum of the artificials
        double[] phase1 = new double[cols];
        for (int i = 0; i < m; i++) {
            phase1[n + mu + i] = 1;
        }
        setObjective(t, basis, phase1);
        if (!simplex(t, basis, n + mu + m)) {
            throw new IllegalStateException("phase 1 cannot be unbounded");
        }
        if (-t[m][cols - 1] > 1e-9 * Math.max(1, maxAbs(bUb, bEq))) {
            return new LinearSolution(Status.INFEASIBLE, null, Double.NaN);
        }
        // drive artificials out of the basis where a real column can replace them
        for (int i = 0; i < m; i++) {
            if (basis[i] >= n + mu) {
                for (int j = 0; j < n + mu; j++) {
                    if (Math.abs(t[i][j]) > 1e-12) {
                        pivot(t, basis, i, j);
                        break;
                    }
                }
            }
        }
        // phase 2 on the real columns only
        double[] cost = new double[cols];
        System.arraycopy(c, 0, cost, 0, n);
        setObjective(t, basis, cost);
        if (!simplex(t, basis, n + mu)) {
            return new LinearSolution(Status.UNBOUNDED, null, Double.NEGATIVE_INFINITY);
        }
        double[] x = new double[n];
        for (int i = 0; i < m; i++) {
            if (basis[i] < n) {
                x[basis[i]] = t[i][cols - 1];
            }
        }
        double value = 0;
        for (int j = 0; j < n; j++) {
            value += c[j] * x[j];
        }
        return new LinearSolution(Status.OPTIMAL, x, value);
    }

    // the objective row: reduced costs of cost given the basis, and minus the objective value
    private static void setObjective(double[][] t, int[] basis, double[] cost) {
        int m = basis.length;
        double[] obj = t[m];
        System.arraycopy(cost, 0, obj, 0, obj.length);
        for (int i = 0; i < m; i++) {
            double cb = cost[basis[i]];
            if (cb != 0) {
                for (int j = 0; j < obj.length; j++) {
                    obj[j] -= cb * t[i][j];
                }
            }
        }
    }

    // minimizes over the first `usable` columns; false if unbounded
    private static boolean simplex(double[][] t, int[] basis, int usable) {
        int m = basis.length;
        int rhs = t[0].length - 1;
        for (int iter = 0; iter < 100_000; iter++) {
            int enter = -1;
            for (int j = 0; j < usable; j++) {
                if (t[m][j] < -1e-12) {
                    enter = j;   // Bland: the lowest index
                    break;
                }
            }
            if (enter < 0) {
                return true;
            }
            int leave = -1;
            double best = Double.POSITIVE_INFINITY;
            for (int i = 0; i < m; i++) {
                if (t[i][enter] > 1e-12) {
                    double ratio = t[i][rhs] / t[i][enter];
                    if (ratio < best - 1e-12 || (Math.abs(ratio - best) <= 1e-12 && basis[i] < basis[leave])) {
                        best = ratio;
                        leave = i;
                    }
                }
            }
            if (leave < 0) {
                return false;
            }
            pivot(t, basis, leave, enter);
        }
        throw new ArithmeticException("the simplex method did not finish");
    }

    private static void pivot(double[][] t, int[] basis, int row, int col) {
        double p = t[row][col];
        for (int j = 0; j < t[row].length; j++) {
            t[row][j] /= p;
        }
        for (int i = 0; i < t.length; i++) {
            if (i != row && t[i][col] != 0) {
                double f = t[i][col];
                for (int j = 0; j < t[i].length; j++) {
                    t[i][j] -= f * t[row][j];
                }
            }
        }
        basis[row] = col;
    }

    private static double maxAbs(double[] a, double[] b) {
        double m = 0;
        for (double[] v : new double[][] {a, b}) {
            if (v != null) {
                for (double x : v) {
                    m = Math.max(m, Math.abs(x));
                }
            }
        }
        return m;
    }
}
