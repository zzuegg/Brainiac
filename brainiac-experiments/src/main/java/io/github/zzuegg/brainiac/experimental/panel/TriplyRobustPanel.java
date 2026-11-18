package io.github.zzuegg.brainiac.experimental.panel;

import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;

import java.util.SplittableRandom;

/**
 * The triply robust panel estimator TROP (Athey, Imbens, Qu and Viviano 2025,
 * <a href="https://arxiv.org/abs/2508.21536">arXiv:2508.21536</a>): a two-way fixed-effects model
 * plus a nuclear-norm-penalised low-rank term, fitted on control cells with unit and time
 * weights,
 * <pre>
 *   min over a, b, L of sum_{j,s} theta_s omega_j (1 - W_js) (Y_js - a_j - b_s - L_js)^2
 *                       + lambda_nn ||L||_*,
 * </pre>
 * with {@code theta_s = exp(-lambda_time dist(s, t))} and
 * {@code omega_j = exp(-lambda_unit dist(j, i))}, the unit distance being the RMS outcome
 * difference over periods where both units are untreated. The effect of a treated cell is
 * {@code Y_it - a_i - b_t - L_it}; the bias is a product of unit imbalance, time imbalance and
 * outcome-model error, so it vanishes when any one does. {@code lambda_nn = infinity} drops the
 * factor term, {@code lambda_unit = lambda_time = 0} gives matrix completion with fixed effects
 * (Athey et al. 2021), which {@link #matrixCompletion} exposes as a baseline.
 *
 * <p>The fit alternates weighted fixed-effect sweeps with proximal-gradient steps on {@code L}
 * (singular-value soft-thresholding by {@link ThinSvd}). To keep it cheap, one fit is shared by
 * all treated cells of a unit, with time distances measured to that unit's nearest treated
 * period (the paper refits per cell). {@link #tune} picks the three levels by the paper's
 * leave-one-out criterion, the sum of squared pseudo-effects on control cells treated as if
 * treated, over a grid and a subsample of control cells.
 */
public final class TriplyRobustPanel {

    private final double lambdaUnit;
    private final double lambdaTime;
    private final double lambdaNn;
    private final int iterations;

    /**
     * Creates the estimator.
     *
     * @param lambdaUnit the unit-weight decay
     * @param lambdaTime the time-weight decay
     * @param lambdaNn the nuclear-norm level ({@code Double.POSITIVE_INFINITY} for no factor term)
     * @param iterations solver iterations
     */
    public TriplyRobustPanel(double lambdaUnit, double lambdaTime, double lambdaNn, int iterations) {
        this.lambdaUnit = lambdaUnit;
        this.lambdaTime = lambdaTime;
        this.lambdaNn = lambdaNn;
        this.iterations = iterations;
    }

    /**
     * Matrix completion with fixed effects (uniform weights), the TROP special case.
     *
     * @param lambdaNn the nuclear-norm level
     * @param iterations solver iterations
     * @return the estimator
     */
    public static TriplyRobustPanel matrixCompletion(double lambdaNn, int iterations) {
        return new TriplyRobustPanel(0, 0, lambdaNn, iterations);
    }

    /** @return {@code {lambdaUnit, lambdaTime, lambdaNn}} */
    public double[] levels() {
        return new double[] {lambdaUnit, lambdaTime, lambdaNn};
    }

    /**
     * Estimates the average effect over treated cells.
     *
     * @param y outcomes {@code [units][periods]}
     * @param treated treatment indicators
     * @return the ATT
     */
    public double fit(double[][] y, boolean[][] treated) {
        double[][] tau = cellEffects(y, treated);
        double s = 0;
        int c = 0;
        for (int i = 0; i < y.length; i++) {
            for (int t = 0; t < y[0].length; t++) {
                if (treated[i][t]) {
                    s += tau[i][t];
                    c++;
                }
            }
        }
        return c == 0 ? 0 : s / c;
    }

    /**
     * Estimates every treated cell's effect.
     *
     * @param y outcomes
     * @param treated treatment indicators
     * @return effects (0 on control cells)
     */
    public double[][] cellEffects(double[][] y, boolean[][] treated) {
        int n = y.length;
        int periods = y[0].length;
        double[][] tau = new double[n][periods];
        for (int i = 0; i < n; i++) {
            boolean any = false;
            for (int t = 0; t < periods; t++) {
                any |= treated[i][t];
            }
            if (!any) {
                continue;
            }
            double[][] w = weights(y, treated, i);
            double[][] fitted = solve(y, w);
            for (int t = 0; t < periods; t++) {
                if (treated[i][t]) {
                    tau[i][t] = y[i][t] - fitted[i][t];
                }
            }
        }
        return tau;
    }

    private double[][] weights(double[][] y, boolean[][] treated, int i) {
        int n = y.length;
        int periods = y[0].length;
        double[] theta = new double[periods];
        for (int s = 0; s < periods; s++) {
            int dist = Integer.MAX_VALUE;
            for (int t = 0; t < periods; t++) {
                if (treated[i][t]) {
                    dist = Math.min(dist, Math.abs(t - s));
                }
            }
            theta[s] = Math.exp(-lambdaTime * dist);
        }
        double[][] w = new double[n][periods];
        for (int j = 0; j < n; j++) {
            double num = 0;
            int cnt = 0;
            for (int s = 0; s < periods; s++) {
                if (!treated[i][s] && !treated[j][s]) {
                    num += (y[i][s] - y[j][s]) * (y[i][s] - y[j][s]);
                    cnt++;
                }
            }
            double dist = cnt == 0 ? 1e9 : Math.sqrt(num / cnt);
            double omega = Math.exp(-lambdaUnit * dist);
            for (int s = 0; s < periods; s++) {
                w[j][s] = treated[j][s] ? 0 : theta[s] * omega;
            }
        }
        return w;
    }

    /**
     * Fits {@code a_j + b_s + L_js} to the outcomes with cell weights.
     *
     * @param y outcomes
     * @param w cell weights (0 excludes a cell)
     * @return the fitted values on every cell
     */
    public double[][] solve(double[][] y, double[][] w) {
        int n = y.length;
        int periods = y[0].length;
        double[] a = new double[n];
        double[] b = new double[periods];
        double[][] l = new double[n][periods];
        double maxW = 0;
        for (double[] row : w) {
            for (double v : row) {
                maxW = Math.max(maxW, v);
            }
        }
        boolean lowRank = Double.isFinite(lambdaNn);
        int its = lowRank ? iterations : 30;
        for (int it = 0; it < its; it++) {
            for (int sweep = 0; sweep < 2; sweep++) {
                for (int j = 0; j < n; j++) {
                    double num = 0;
                    double den = 0;
                    for (int s = 0; s < periods; s++) {
                        num += w[j][s] * (y[j][s] - b[s] - l[j][s]);
                        den += w[j][s];
                    }
                    a[j] = den > 0 ? num / den : a[j];
                }
                for (int s = 0; s < periods; s++) {
                    double num = 0;
                    double den = 0;
                    for (int j = 0; j < n; j++) {
                        num += w[j][s] * (y[j][s] - a[j] - l[j][s]);
                        den += w[j][s];
                    }
                    b[s] = den > 0 ? num / den : b[s];
                }
            }
            if (!lowRank) {
                continue;
            }
            double[][] step = new double[n][periods];
            for (int j = 0; j < n; j++) {
                for (int s = 0; s < periods; s++) {
                    double r = y[j][s] - a[j] - b[s] - l[j][s];
                    step[j][s] = l[j][s] + w[j][s] * r / maxW;
                }
            }
            l = shrink(step, lambdaNn / (2 * maxW));
        }
        double[][] out = new double[n][periods];
        for (int j = 0; j < n; j++) {
            for (int s = 0; s < periods; s++) {
                out[j][s] = a[j] + b[s] + l[j][s];
            }
        }
        return out;
    }

    /**
     * Singular-value soft-thresholding.
     *
     * @param m a matrix
     * @param threshold the shrinkage
     * @return {@code U max(S - threshold, 0) V^T}
     */
    public static double[][] shrink(double[][] m, double threshold) {
        ThinSvd svd = ThinSvd.of(m);
        int rows = m.length;
        int cols = m[0].length;
        double[][] out = new double[rows][cols];
        for (int k = 0; k < svd.s().length; k++) {
            double s = svd.s()[k] - threshold;
            if (s <= 0) {
                continue;
            }
            for (int r = 0; r < rows; r++) {
                double ur = svd.u()[r][k] * s;
                for (int c = 0; c < cols; c++) {
                    out[r][c] += ur * svd.v()[c][k];
                }
            }
        }
        return out;
    }

    /**
     * Leave-one-out tuning over a grid: for a subsample of control units, their last
     * pre-adoption cell (or last cell) is treated as the only treated cell of that unit (real
     * treated cells stay excluded), and the levels with the smallest sum of squared
     * pseudo-effects win.
     *
     * @param y outcomes
     * @param treated treatment indicators
     * @param unitGrid {@code lambda_unit} values
     * @param timeGrid {@code lambda_time} values
     * @param nnGrid {@code lambda_nn} values (infinity allowed)
     * @param pseudoCells control units sampled
     * @param iterations solver iterations
     * @param seed the sampling seed
     * @return the tuned estimator
     */
    public static TriplyRobustPanel tune(double[][] y, boolean[][] treated, double[] unitGrid, double[] timeGrid,
            double[] nnGrid, int pseudoCells, int iterations, long seed) {
        int n = y.length;
        int periods = y[0].length;
        int[] controls = java.util.stream.IntStream.range(0, n).filter(i -> {
            for (boolean b : treated[i]) {
                if (b) {
                    return false;
                }
            }
            return true;
        }).toArray();
        SplittableRandom r = new SplittableRandom(seed);
        int[] picks = new int[Math.min(pseudoCells, controls.length)];
        for (int k = 0; k < picks.length; k++) {
            picks[k] = controls[r.nextInt(controls.length)];
        }
        int firstTreated = periods;
        for (boolean[] row : treated) {
            for (int t = 0; t < periods; t++) {
                if (row[t]) {
                    firstTreated = Math.min(firstTreated, t);
                }
            }
        }
        int cell = Math.max(1, firstTreated - 1);
        TriplyRobustPanel best = null;
        double bestQ = Double.POSITIVE_INFINITY;
        for (double lu : unitGrid) {
            for (double lt : timeGrid) {
                for (double ln : nnGrid) {
                    TriplyRobustPanel m = new TriplyRobustPanel(lu, lt, ln, iterations);
                    double q = 0;
                    for (int j : picks) {
                        boolean[][] fake = new boolean[n][];
                        for (int i = 0; i < n; i++) {
                            fake[i] = treated[i].clone();
                        }
                        fake[j][cell] = true;
                        double[][] w = m.weights(y, fake, j);
                        double fittedCell = m.solve(y, w)[j][cell];
                        q += (y[j][cell] - fittedCell) * (y[j][cell] - fittedCell);
                    }
                    if (q < bestQ) {
                        bestQ = q;
                        best = m;
                    }
                }
            }
        }
        return best;
    }
}
