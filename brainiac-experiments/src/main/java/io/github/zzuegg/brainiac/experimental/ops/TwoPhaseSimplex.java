package io.github.zzuegg.brainiac.experimental.ops;

import java.util.Arrays;

/**
 * A dense two-phase simplex solver for general linear programs
 * {@code maximise c.x subject to A_ub x <= b_ub, A_eq x = b_eq, x >= 0}, with right-hand sides of
 * any sign. The wave 6 {@code experimental.ml.LinearProgram} handles only {@code b >= 0}
 * inequality programs (the origin feasible, no phase one); occupation-measure LPs (restless
 * bandits), Stackelberg commitment (a probability simplex), correlated equilibria and security
 * games need equalities or negative bounds, so this solver adds phase one with artificial
 * variables. Bland's rule picks the pivots, which rules out cycling; sizes up to a few hundred
 * variables solve in milliseconds.
 *
 * <p>Dantzig 1951; the two-phase method as in Chvátal, "Linear Programming" (1983), chapter 8.
 */
public final class TwoPhaseSimplex {

    private static final double EPS = 1e-9;

    private TwoPhaseSimplex() {
    }

    /** How a solve ended. */
    public enum Status {
        /** an optimal solution was found */
        OPTIMAL,
        /** no point satisfies the constraints */
        INFEASIBLE,
        /** the objective grows without bound */
        UNBOUNDED
    }

    /**
     * A solved program.
     *
     * @param status how the solve ended
     * @param x the optimal point ({@code null} unless optimal)
     * @param objective the optimal value ({@code NaN} unless optimal)
     */
    public record Result(Status status, double[] x, double objective) {
        /** @return whether an optimum was found */
        public boolean optimal() {
            return status == Status.OPTIMAL;
        }
    }

    /**
     * Solves a program with inequality constraints only.
     *
     * @param c the objective
     * @param aUb inequality rows
     * @param bUb their bounds (any sign)
     * @return the result
     */
    public static Result maximize(double[] c, double[][] aUb, double[] bUb) {
        return maximize(c, aUb, bUb, new double[0][], new double[0]);
    }

    /**
     * Solves a general program.
     *
     * @param c the objective, one entry per variable
     * @param aUb inequality rows ({@code A_ub x <= b_ub}); may be empty
     * @param bUb their bounds
     * @param aEq equality rows ({@code A_eq x = b_eq}); may be empty
     * @param bEq their values
     * @return the result
     */
    public static Result maximize(double[] c, double[][] aUb, double[] bUb, double[][] aEq, double[] bEq) {
        int n = c.length;
        int mu = aUb.length;
        int me = aEq.length;
        int m = mu + me;
        boolean[] needsArt = new boolean[m];
        int arts = 0;
        for (int i = 0; i < m; i++) {
            needsArt[i] = i >= mu || bUb[i] < 0;
            arts += needsArt[i] ? 1 : 0;
        }
        int cols = n + mu + arts;
        double[][] t = new double[m][cols + 1];
        int[] basis = new int[m];
        int art = n + mu;
        for (int i = 0; i < m; i++) {
            double[] row = i < mu ? aUb[i] : aEq[i - mu];
            double rhs = i < mu ? bUb[i] : bEq[i - mu];
            double sign = rhs < 0 ? -1 : 1;
            for (int j = 0; j < n; j++) {
                t[i][j] = sign * row[j];
            }
            if (i < mu) {
                t[i][n + i] = sign;
            }
            t[i][cols] = sign * rhs;
            if (needsArt[i]) {
                t[i][art] = 1;
                basis[i] = art++;
            } else {
                basis[i] = n + i;
            }
        }
        boolean[] barred = new boolean[cols];
        if (arts > 0) {
            double[] c1 = new double[cols];
            for (int j = n + mu; j < cols; j++) {
                c1[j] = -1;
            }
            if (!run(t, basis, c1, barred)) {
                return new Result(Status.UNBOUNDED, null, Double.NaN);
            }
            if (value(t, basis, c1) < -1e-7) {
                return new Result(Status.INFEASIBLE, null, Double.NaN);
            }
            for (int i = 0; i < m; i++) {
                if (basis[i] >= n + mu) {
                    for (int j = 0; j < n + mu; j++) {
                        if (Math.abs(t[i][j]) > EPS) {
                            pivot(t, basis, i, j);
                            break;
                        }
                    }
                }
            }
            for (int j = n + mu; j < cols; j++) {
                barred[j] = true;
            }
        }
        double[] c2 = Arrays.copyOf(c, cols);
        if (!run(t, basis, c2, barred)) {
            return new Result(Status.UNBOUNDED, null, Double.NaN);
        }
        double[] x = new double[n];
        for (int i = 0; i < m; i++) {
            if (basis[i] < n) {
                x[basis[i]] = t[i][cols];
            }
        }
        double obj = 0;
        for (int j = 0; j < n; j++) {
            obj += c[j] * x[j];
        }
        return new Result(Status.OPTIMAL, x, obj);
    }

    private static double value(double[][] t, int[] basis, double[] c) {
        int rhs = t[0].length - 1;
        double v = 0;
        for (int i = 0; i < t.length; i++) {
            v += c[basis[i]] * t[i][rhs];
        }
        return v;
    }

    /** Runs simplex iterations; returns false when unbounded. */
    private static boolean run(double[][] t, int[] basis, double[] c, boolean[] barred) {
        int m = t.length;
        if (m == 0) {
            for (int j = 0; j < c.length; j++) {
                if (!barred[j] && c[j] > EPS) {
                    return false;
                }
            }
            return true;
        }
        int cols = t[0].length - 1;
        for (int iter = 0; iter < 50_000; iter++) {
            int enter = -1;
            for (int j = 0; j < cols && enter < 0; j++) {
                if (barred[j]) {
                    continue;
                }
                double r = c[j];
                for (int i = 0; i < m; i++) {
                    r -= c[basis[i]] * t[i][j];
                }
                if (r > EPS) {
                    enter = j;
                }
            }
            if (enter < 0) {
                return true;
            }
            int leave = -1;
            double best = Double.POSITIVE_INFINITY;
            for (int i = 0; i < m; i++) {
                if (t[i][enter] > EPS) {
                    double ratio = t[i][cols] / t[i][enter];
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
        return true;
    }

    private static void pivot(double[][] t, int[] basis, int row, int col) {
        double[] p = t[row];
        double d = p[col];
        for (int j = 0; j < p.length; j++) {
            p[j] /= d;
        }
        for (int i = 0; i < t.length; i++) {
            if (i != row && t[i][col] != 0) {
                double f = t[i][col];
                double[] r = t[i];
                for (int j = 0; j < r.length; j++) {
                    r[j] -= f * p[j];
                }
            }
        }
        basis[row] = col;
    }
}
