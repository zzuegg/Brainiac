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
        // equilibrate: scale each variable by its column's largest coefficient, then each row by
        // its largest, so that badly scaled problems (cutting planes) keep their pivots readable
        double[] colScale = new double[n];
        for (int i = 0; i < m; i++) {
            double[] row = i < mu ? aUb[i] : aEq[i - mu];
            if (row.length != n) {
                throw new IllegalArgumentException("constraint " + i + " has " + row.length + " coefficients, not " + n);
            }
            for (int j = 0; j < n; j++) {
                colScale[j] = Math.max(colScale[j], Math.abs(row[j]));
            }
        }
        for (int j = 0; j < n; j++) {
            colScale[j] = colScale[j] > 0 ? colScale[j] : 1;
        }
        // columns: n variables (as x_j * colScale_j), mu slacks, m artificials, then the right-hand side
        int cols = n + mu + m + 1;
        double[][] t = new double[m + 1][cols];
        int[] basis = new int[m];
        for (int i = 0; i < m; i++) {
            double[] row = i < mu ? aUb[i] : aEq[i - mu];
            double rhs = i < mu ? bUb[i] : bEq[i - mu];
            double rowMax = 0;
            for (int j = 0; j < n; j++) {
                rowMax = Math.max(rowMax, Math.abs(row[j] / colScale[j]));
            }
            double scale = (rhs < 0 ? -1 : 1) / (rowMax > 0 ? rowMax : 1);
            for (int j = 0; j < n; j++) {
                t[i][j] = scale * row[j] / colScale[j];
            }
            if (i < mu) {
                t[i][n + i] = Math.signum(scale);   // the slack's own scale is free
            }
            t[i][n + mu + i] = 1;
            t[i][cols - 1] = scale * rhs;
            basis[i] = n + mu + i;
        }
        double rhsMax = 0;
        for (int i = 0; i < m; i++) {
            rhsMax = Math.max(rhsMax, Math.abs(t[i][cols - 1]));
        }
        // phase 1: minimize the sum of the artificials (never unbounded: columns that only look
        // unbounded through rounding are set aside)
        double[] phase1 = new double[cols];
        for (int i = 0; i < m; i++) {
            phase1[n + mu + i] = 1;
        }
        setObjective(t, basis, phase1);
        simplex(t, basis, n + mu + m, true);
        if (-t[m][cols - 1] > 1e-9 * Math.max(1, rhsMax)) {
            return new LinearSolution(Status.INFEASIBLE, null, Double.NaN);
        }
        // drive artificials out of the basis where a real column can replace them
        for (int i = 0; i < m; i++) {
            if (basis[i] >= n + mu) {
                int best = -1;
                for (int j = 0; j < n + mu; j++) {
                    if (Math.abs(t[i][j]) > 1e-14 && (best < 0 || Math.abs(t[i][j]) > Math.abs(t[i][best]))) {
                        best = j;
                    }
                }
                if (best >= 0) {
                    pivot(t, basis, i, best);
                } else {
                    // a redundant row: keep it from moving in phase 2
                    java.util.Arrays.fill(t[i], 0, n + mu, 0);
                }
            }
        }
        // phase 2 on the real columns only
        double[] cost = new double[cols];
        for (int j = 0; j < n; j++) {
            cost[j] = c[j] / colScale[j];
        }
        setObjective(t, basis, cost);
        if (!simplex(t, basis, n + mu, false)) {
            return new LinearSolution(Status.UNBOUNDED, null, Double.NEGATIVE_INFINITY);
        }
        double[] x = new double[n];
        for (int i = 0; i < m; i++) {
            if (basis[i] < n) {
                x[basis[i]] = Math.max(0, t[i][cols - 1]) / colScale[basis[i]];
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

    // minimizes over the first `usable` columns; false if unbounded. With `bounded` (phase 1, whose
    // objective cannot fall below 0) a column with no pivot is rounding noise and is set aside.
    private static boolean simplex(double[][] t, int[] basis, int usable, boolean bounded) {
        int m = basis.length;
        int rhs = t[0].length - 1;
        boolean[] aside = new boolean[usable];
        double costScale = 0;
        for (int j = 0; j < usable; j++) {
            costScale = Math.max(costScale, Math.abs(t[m][j]));
        }
        double enterTol = 1e-11 * Math.max(1, costScale);
        for (int iter = 0; iter < 200_000; iter++) {
            int enter = -1;
            for (int j = 0; j < usable; j++) {
                if (!aside[j] && t[m][j] < -enterTol) {
                    enter = j;   // Bland: the lowest index
                    break;
                }
            }
            if (enter < 0) {
                return true;
            }
            double colMax = 0;
            for (int i = 0; i < m; i++) {
                colMax = Math.max(colMax, Math.abs(t[i][enter]));
            }
            double pivotTol = 1e-11 * Math.max(1, colMax);
            int leave = -1;
            double best = Double.POSITIVE_INFINITY;
            for (int i = 0; i < m; i++) {
                if (t[i][enter] > pivotTol) {
                    double ratio = Math.max(0, t[i][rhs]) / t[i][enter];
                    double tie = leave < 0 ? 0 : 1e-12 * Math.max(1, best);
                    if (leave < 0 || ratio < best - tie
                            || (Math.abs(ratio - best) <= tie && basis[i] < basis[leave])) {
                        best = ratio;
                        leave = i;
                    }
                }
            }
            if (leave < 0) {
                if (!bounded) {
                    return false;
                }
                aside[enter] = true;
                continue;
            }
            pivot(t, basis, leave, enter);
        }
        throw new ArithmeticException("the simplex method did not finish");
    }

    private static void pivot(double[][] t, int[] basis, int row, int col) {
        double p = t[row][col];
        double[] r = t[row];
        for (int j = 0; j < r.length; j++) {
            r[j] /= p;
        }
        r[col] = 1;
        for (int i = 0; i < t.length; i++) {
            if (i != row && t[i][col] != 0) {
                double f = t[i][col];
                double[] ti = t[i];
                for (int j = 0; j < ti.length; j++) {
                    ti[j] -= f * r[j];
                }
                ti[col] = 0;
            }
        }
        basis[row] = col;
    }

    /**
     * Minimizes by CMA-ES (Hansen's covariance matrix adaptation evolution strategy, with the
     * default weights and learning rates), which handles rugged, badly scaled and non-separable
     * functions without gradients. Each generation samples {@code 4 + 3 ln n} points.
     *
     * @param f the function
     * @param x0 the starting mean
     * @param sigma the starting step size, about a third of the search range
     * @param maxGenerations the most generations
     * @param tolerance stop when the step size times the largest axis falls below this, or the
     *     generation's values spread less than this
     * @param seed the random seed
     * @return the best point seen
     */
    public static Minimum cmaEs(ToDoubleFunction<double[]> f, double[] x0, double sigma, int maxGenerations,
            double tolerance, long seed) {
        int n = x0.length;
        if (n == 0 || !(sigma > 0)) {
            throw new IllegalArgumentException("give a starting point and a positive step size");
        }
        java.util.SplittableRandom rng = new java.util.SplittableRandom(seed);
        int lambda = 4 + (int) Math.floor(3 * Math.log(n));
        int mu = lambda / 2;
        double[] w = new double[mu];
        double wsum = 0;
        for (int i = 0; i < mu; i++) {
            w[i] = Math.log(mu + 0.5) - Math.log(i + 1);
            wsum += w[i];
        }
        double w2 = 0;
        for (int i = 0; i < mu; i++) {
            w[i] /= wsum;
            w2 += w[i] * w[i];
        }
        double mueff = 1 / w2;
        double cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n);
        double cs = (mueff + 2) / (n + mueff + 5);
        double c1 = 2 / ((n + 1.3) * (n + 1.3) + mueff);
        double cmu = Math.min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) * (n + 2) + mueff));
        double damps = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (n + 1)) - 1) + cs;
        double chiN = Math.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21.0 * n * n));

        double[] mean = x0.clone();
        double[] pc = new double[n];
        double[] ps = new double[n];
        double[][] c = Matrices.identity(n);
        double[][] b = Matrices.identity(n);
        double[] d = new double[n];
        Arrays.fill(d, 1);
        double[] best = x0.clone();
        double bestValue = f.applyAsDouble(x0);
        int eigenEvery = Math.max(1, (int) (lambda / ((c1 + cmu) * n * 10)));
        double[][] z = new double[lambda][n];
        double[][] y = new double[lambda][n];
        double[][] x = new double[lambda][n];
        double[] values = new double[lambda];
        Integer[] order = new Integer[lambda];
        int g = 0;
        boolean converged = false;
        while (g < maxGenerations) {
            g++;
            for (int k = 0; k < lambda; k++) {
                for (int i = 0; i < n; i++) {
                    z[k][i] = gaussian(rng);
                }
                for (int i = 0; i < n; i++) {
                    double s = 0;
                    for (int j = 0; j < n; j++) {
                        s += b[i][j] * d[j] * z[k][j];
                    }
                    y[k][i] = s;
                    x[k][i] = mean[i] + sigma * s;
                }
                values[k] = f.applyAsDouble(x[k]);
                order[k] = k;
                if (values[k] < bestValue) {
                    bestValue = values[k];
                    best = x[k].clone();
                }
            }
            Arrays.sort(order, (p, q) -> Double.compare(values[p], values[q]));
            double[] yw = new double[n];
            for (int r = 0; r < mu; r++) {
                for (int i = 0; i < n; i++) {
                    yw[i] += w[r] * y[order[r]][i];
                }
            }
            for (int i = 0; i < n; i++) {
                mean[i] += sigma * yw[i];
            }
            // C^(-1/2) yw = B D^-1 B^T yw
            double[] t = new double[n];
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int i = 0; i < n; i++) {
                    s += b[i][j] * yw[i];
                }
                t[j] = s / d[j];
            }
            double psNorm = 0;
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int j = 0; j < n; j++) {
                    s += b[i][j] * t[j];
                }
                ps[i] = (1 - cs) * ps[i] + Math.sqrt(cs * (2 - cs) * mueff) * s;
                psNorm += ps[i] * ps[i];
            }
            psNorm = Math.sqrt(psNorm);
            boolean hsig = psNorm / Math.sqrt(1 - Math.pow(1 - cs, 2.0 * g)) / chiN < 1.4 + 2.0 / (n + 1);
            for (int i = 0; i < n; i++) {
                pc[i] = (1 - cc) * pc[i] + (hsig ? Math.sqrt(cc * (2 - cc) * mueff) * yw[i] : 0);
            }
            double delta = hsig ? 0 : cc * (2 - cc);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j <= i; j++) {
                    double rankMu = 0;
                    for (int r = 0; r < mu; r++) {
                        rankMu += w[r] * y[order[r]][i] * y[order[r]][j];
                    }
                    double v = (1 - c1 - cmu) * c[i][j] + c1 * (pc[i] * pc[j] + delta * c[i][j]) + cmu * rankMu;
                    c[i][j] = v;
                    c[j][i] = v;
                }
            }
            sigma *= Math.exp(Math.min(1, (cs / damps) * (psNorm / chiN - 1)));
            if (g % eigenEvery == 0) {
                Matrices.SymmetricEigen e = Matrices.symmetricEigen(c);
                for (int i = 0; i < n; i++) {
                    d[i] = Math.sqrt(Math.max(e.values()[i], 1e-300));
                    for (int j = 0; j < n; j++) {
                        b[i][j] = e.vectors()[i][j];
                    }
                }
            }
            double maxD = Arrays.stream(d).max().orElse(1);
            double spread = values[order[lambda - 1]] - values[order[0]];
            if (sigma * maxD < tolerance || (g > 1 && spread < tolerance && sigma * maxD < Math.sqrt(tolerance))) {
                converged = true;
                break;
            }
        }
        return new Minimum(best, bestValue, g, converged);
    }

    /**
     * The answer to a quadratic program.
     *
     * @param x the minimizer
     * @param value {@code x P x / 2 + q x} there
     * @param dual the multipliers of the constraint rows
     * @param iterations the iterations taken
     * @param converged whether the primal and dual residuals met the tolerance
     */
    public record QuadraticSolution(double[] x, double value, double[] dual, int iterations, boolean converged) {
    }

    /**
     * Minimizes {@code x P x / 2 + q x} subject to {@code lower <= A x <= upper}, by the ADMM
     * splitting of OSQP (Stellato et al. 2020) with adaptive step size. {@code P} must be
     * symmetric positive semi-definite. Equalities have equal bounds; bounds on x are rows of the
     * identity; a missing side is an infinity. Suits problems of up to a few hundred variables
     * (one dense factorization per step-size change).
     *
     * @param p the quadratic term, {@code n x n}
     * @param q the linear term
     * @param a the constraint rows, {@code m x n}, or null for none
     * @param lower the rows' lower bounds (may be {@code -Infinity})
     * @param upper the rows' upper bounds (may be {@code +Infinity})
     * @param tolerance the residuals to reach, such as 1e-8
     * @param maxIterations the most iterations, such as 10000
     * @return the solution
     */
    public static QuadraticSolution quadraticProgram(double[][] p, double[] q, double[][] a, double[] lower,
            double[] upper, double tolerance, int maxIterations) {
        int n = q.length;
        int m = a == null ? 0 : a.length;
        double[][] am = a == null ? new double[0][n] : a;
        double sigma = 1e-6;
        double alpha = 1.6;
        double rho = 0.1;
        double[] x = new double[n];
        double[] z = new double[m];
        double[] y = new double[m];
        double[] rhoRow = rowRho(lower, upper, rho, m);
        double[][] l = qpFactor(p, am, rhoRow, sigma);
        double[] xt = new double[n];
        double[] zt = new double[m];
        double[] rhs = new double[n];
        double[] ax = new double[m];
        double[] px = new double[n];
        double[] aty = new double[n];
        int it = 0;
        boolean converged = false;
        while (it < maxIterations) {
            it++;
            for (int j = 0; j < n; j++) {
                rhs[j] = sigma * x[j] - q[j];
            }
            for (int i = 0; i < m; i++) {
                double w = rhoRow[i] * z[i] - y[i];
                for (int j = 0; j < n; j++) {
                    rhs[j] += am[i][j] * w;
                }
            }
            choleskySolve(l, rhs, xt);
            for (int i = 0; i < m; i++) {
                double s = 0;
                for (int j = 0; j < n; j++) {
                    s += am[i][j] * xt[j];
                }
                zt[i] = s;
            }
            for (int j = 0; j < n; j++) {
                x[j] = alpha * xt[j] + (1 - alpha) * x[j];
            }
            for (int i = 0; i < m; i++) {
                double relaxed = alpha * zt[i] + (1 - alpha) * z[i];
                double zn = Math.min(upper[i], Math.max(lower[i], relaxed + y[i] / rhoRow[i]));
                y[i] += rhoRow[i] * (relaxed - zn);
                z[i] = zn;
            }
            if (it % 10 != 0 && it != maxIterations) {
                continue;
            }
            // residuals
            double prim = 0;
            double axMax = 0;
            double zMax = 0;
            for (int i = 0; i < m; i++) {
                double s = 0;
                for (int j = 0; j < n; j++) {
                    s += am[i][j] * x[j];
                }
                ax[i] = s;
                prim = Math.max(prim, Math.abs(s - z[i]));
                axMax = Math.max(axMax, Math.abs(s));
                zMax = Math.max(zMax, Math.abs(z[i]));
            }
            double dual = 0;
            double pxMax = 0;
            double atyMax = 0;
            double qMax = 0;
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int k = 0; k < n; k++) {
                    s += p[j][k] * x[k];
                }
                px[j] = s;
                double t = 0;
                for (int i = 0; i < m; i++) {
                    t += am[i][j] * y[i];
                }
                aty[j] = t;
                dual = Math.max(dual, Math.abs(s + q[j] + t));
                pxMax = Math.max(pxMax, Math.abs(s));
                atyMax = Math.max(atyMax, Math.abs(t));
                qMax = Math.max(qMax, Math.abs(q[j]));
            }
            double primScale = Math.max(axMax, zMax);
            double dualScale = Math.max(pxMax, Math.max(atyMax, qMax));
            if (prim <= tolerance * Math.max(1, primScale) && dual <= tolerance * Math.max(1, dualScale)) {
                converged = true;
                break;
            }
            if (it % 50 == 0 && m > 0) {
                double ratio = Math.sqrt((prim / Math.max(1e-30, primScale)) / Math.max(1e-30, dual / Math.max(1e-30,
                        dualScale)));
                double next = Math.min(1e6, Math.max(1e-6, rho * ratio));
                if (next > 5 * rho || next < rho / 5) {
                    for (int i = 0; i < m; i++) {
                        rhoRow[i] *= next / rho;
                    }
                    rho = next;
                    l = qpFactor(p, am, rhoRow, sigma);
                }
            }
        }
        double value = 0;
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int k = 0; k < n; k++) {
                s += p[j][k] * x[k];
            }
            value += x[j] * (0.5 * s + q[j]);
        }
        return new QuadraticSolution(x, value, y, it, converged);
    }

    private static double[] rowRho(double[] lower, double[] upper, double rho, int m) {
        if ((m > 0) && (lower == null || upper == null || lower.length != m || upper.length != m)) {
            throw new IllegalArgumentException("give lower and upper bounds for each of the " + m + " rows");
        }
        double[] r = new double[m];
        for (int i = 0; i < m; i++) {
            if (lower[i] > upper[i]) {
                throw new IllegalArgumentException("row " + i + " has lower bound above upper");
            }
            boolean free = lower[i] == Double.NEGATIVE_INFINITY && upper[i] == Double.POSITIVE_INFINITY;
            r[i] = free ? 1e-6 : lower[i] == upper[i] ? 1e3 * rho : rho;
        }
        return r;
    }

    // Cholesky of P + sigma I + A^T diag(rho) A
    private static double[][] qpFactor(double[][] p, double[][] a, double[] rho, double sigma) {
        int n = p.length;
        double[][] k = new double[n][n];
        for (int i = 0; i < n; i++) {
            if (p[i].length != n) {
                throw new IllegalArgumentException("P must be square");
            }
            for (int j = 0; j < n; j++) {
                k[i][j] = 0.5 * (p[i][j] + p[j][i]);
            }
            k[i][i] += sigma;
        }
        for (int r = 0; r < a.length; r++) {
            double[] row = a[r];
            for (int i = 0; i < n; i++) {
                if (row[i] != 0) {
                    double f = rho[r] * row[i];
                    for (int j = 0; j < n; j++) {
                        k[i][j] += f * row[j];
                    }
                }
            }
        }
        return Matrices.cholesky(k);
    }

    private static void choleskySolve(double[][] l, double[] b, double[] x) {
        int n = b.length;
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double s = b[i];
            for (int k = 0; k < i; k++) {
                s -= l[i][k] * y[k];
            }
            y[i] = s / l[i][i];
        }
        for (int i = n - 1; i >= 0; i--) {
            double s = y[i];
            for (int k = i + 1; k < n; k++) {
                s -= l[k][i] * x[k];
            }
            x[i] = s / l[i][i];
        }
    }

    private static double gaussian(java.util.SplittableRandom rng) {
        double u = 1 - rng.nextDouble();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng.nextDouble());
    }
}
