package io.github.zzuegg.brainiac.experimental.ml;

/**
 * A small dense simplex solver for linear programs in the standard form
 * {@code maximise c.x subject to A x <= b, x >= 0} with {@code b >= 0} (so the origin is
 * feasible and no phase one is needed), plus the classic reduction of a two-player zero-sum
 * matrix game to such a program ({@link #solveZeroSum}). Bland's rule picks the pivots, which
 * rules out cycling; the problems it is meant for (meta-games of a few dozen policies) solve in
 * microseconds.
 *
 * <p>The dual solution is read off the final tableau (the reduced costs of the slack
 * variables), which is what gives the row player's equilibrium strategy for free.
 *
 * <p>Dantzig 1951; the game reduction is von Neumann's minimax theorem as an LP
 * (e.g. Shoham and Leyton-Brown, "Multiagent Systems", 2009, section 4.1).
 */
public final class LinearProgram {

    /**
     * A solution.
     *
     * @param x the primal optimum
     * @param dual the dual optimum, one value per constraint
     * @param objective the optimal value of {@code c.x}
     */
    public record Solution(double[] x, double[] dual, double objective) {
    }

    /**
     * An equilibrium of a zero-sum matrix game.
     *
     * @param row the maximising row player's mixed strategy
     * @param column the minimising column player's mixed strategy
     * @param value the game value to the row player
     */
    public record Equilibrium(double[] row, double[] column, double value) {
    }

    private static final double EPS = 1e-12;

    private LinearProgram() {
    }

    /**
     * Maximises {@code c.x} subject to {@code A x <= b} and {@code x >= 0}.
     *
     * @param c the objective, length {@code n}
     * @param a the constraint matrix {@code [m][n]}
     * @param b the bounds, length {@code m}, all non-negative
     * @return the optimum
     * @throws IllegalArgumentException if a bound is negative
     * @throws ArithmeticException if the program is unbounded
     */
    public static Solution maximize(double[] c, double[][] a, double[] b) {
        int m = a.length;
        int n = c.length;
        double[][] t = new double[m + 1][n + m + 1];
        int[] basis = new int[m];
        for (int i = 0; i < m; i++) {
            if (b[i] < 0) {
                throw new IllegalArgumentException("bounds must be non-negative");
            }
            System.arraycopy(a[i], 0, t[i], 0, n);
            t[i][n + i] = 1;
            t[i][n + m] = b[i];
            basis[i] = n + i;
        }
        for (int j = 0; j < n; j++) {
            t[m][j] = -c[j];
        }
        while (true) {
            int enter = -1;
            for (int j = 0; j < n + m; j++) {
                if (t[m][j] < -EPS) {
                    enter = j;
                    break;
                }
            }
            if (enter < 0) {
                break;
            }
            int leave = -1;
            double ratio = Double.POSITIVE_INFINITY;
            for (int i = 0; i < m; i++) {
                if (t[i][enter] > EPS) {
                    double r = t[i][n + m] / t[i][enter];
                    if (r < ratio - EPS || Math.abs(r - ratio) <= EPS && basis[i] < basis[leave]) {
                        ratio = r;
                        leave = i;
                    }
                }
            }
            if (leave < 0) {
                throw new ArithmeticException("the linear program is unbounded");
            }
            pivot(t, leave, enter);
            basis[leave] = enter;
        }
        double[] x = new double[n];
        for (int i = 0; i < m; i++) {
            if (basis[i] < n) {
                x[basis[i]] = t[i][n + m];
            }
        }
        double[] dual = new double[m];
        for (int i = 0; i < m; i++) {
            dual[i] = t[m][n + i];
        }
        return new Solution(x, dual, t[m][n + m]);
    }

    private static void pivot(double[][] t, int row, int col) {
        double p = t[row][col];
        double[] r = t[row];
        for (int j = 0; j < r.length; j++) {
            r[j] /= p;
        }
        for (int i = 0; i < t.length; i++) {
            if (i == row) {
                continue;
            }
            double f = t[i][col];
            if (f != 0) {
                double[] ti = t[i];
                for (int j = 0; j < ti.length; j++) {
                    ti[j] -= f * r[j];
                }
            }
        }
    }

    /**
     * Solves a zero-sum matrix game exactly. The payoffs are shifted to be positive, the
     * column player's problem {@code max sum(y) s.t. A y <= 1} is solved, and both strategies
     * are rescaled from the primal and dual solutions.
     *
     * @param payoff the payoff to the row player, {@code [rows][columns]}
     * @return a Nash equilibrium and the game value
     */
    public static Equilibrium solveZeroSum(double[][] payoff) {
        int m = payoff.length;
        int n = payoff[0].length;
        double min = Double.POSITIVE_INFINITY;
        for (double[] row : payoff) {
            for (double v : row) {
                min = Math.min(min, v);
            }
        }
        double shift = 1 - min;
        double[][] a = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = payoff[i][j] + shift;
            }
        }
        double[] ones = new double[n];
        java.util.Arrays.fill(ones, 1);
        double[] bounds = new double[m];
        java.util.Arrays.fill(bounds, 1);
        Solution s = maximize(ones, a, bounds);
        double v = 1 / s.objective();
        double[] column = new double[n];
        for (int j = 0; j < n; j++) {
            column[j] = s.x()[j] * v;
        }
        double[] row = new double[m];
        for (int i = 0; i < m; i++) {
            row[i] = s.dual()[i] * v;
        }
        return new Equilibrium(row, column, v - shift);
    }
}
