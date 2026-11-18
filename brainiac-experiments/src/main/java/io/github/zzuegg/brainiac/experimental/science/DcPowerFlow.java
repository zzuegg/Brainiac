package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.tensor.Matrices;
import io.github.zzuegg.brainiac.tensor.Optimize;

/**
 * DC power flow and DC optimal power flow on a {@link ScienceSims.PowerGrid}: the power
 * transfer distribution factors {@code PTDF = diag(1/x) A B_r^-1} (bus 0 the slack), the
 * economic dispatch LP {@code min c g} subject to power balance, line limits
 * {@code |PTDF (C g - d)| <= f_max} and generator bounds (core {@link Optimize#linearProgram}),
 * load shedding by LP when no dispatch is feasible, a violation report for any proposed
 * dispatch, and a weighted-least-squares state estimate of bus injections from noisy flow and
 * injection measurements.
 */
public final class DcPowerFlow {

    /**
     * An LP result.
     *
     * @param feasible whether a dispatch meets every constraint
     * @param dispatch generator outputs (MW), or null
     * @param cost the dispatch cost, or NaN
     */
    public record Dispatch(boolean feasible, double[] dispatch, double cost) {
    }

    /**
     * A load-shedding result.
     *
     * @param dispatch generator outputs
     * @param shed the load shed per bus (MW)
     * @param totalShed the sum of {@code shed}
     */
    public record Shedding(double[] dispatch, double[] shed, double totalShed) {
    }

    /**
     * How far a dispatch is from feasible.
     *
     * @param balance |sum g - sum d| (MW)
     * @param lineExcess the summed line-flow excess over limits (MW)
     * @param boundExcess the summed generator-bound excess (MW)
     * @param overloadedLines lines over their limit by more than 1e-6 MW
     */
    public record Violation(double balance, double lineExcess, double boundExcess, int overloadedLines) {

        /**
         * @param tol a tolerance in MW
         * @return whether all violations are within {@code tol}
         */
        public boolean ok(double tol) {
            return balance <= tol && lineExcess <= tol && boundExcess <= tol;
        }

        /** @return the summed violation */
        public double total() {
            return balance + lineExcess + boundExcess;
        }
    }

    private final ScienceSims.PowerGrid grid;
    private final double[][] ptdf;
    private final double[][] genPtdf;

    /**
     * Builds the PTDF of a grid.
     *
     * @param grid the grid
     */
    public DcPowerFlow(ScienceSims.PowerGrid grid) {
        this.grid = grid;
        int n = grid.buses();
        int m = grid.lines();
        double[][] b = new double[n - 1][n - 1];
        for (int l = 0; l < m; l++) {
            double y = 1 / grid.reactance()[l];
            int f = grid.from()[l] - 1;
            int t = grid.to()[l] - 1;
            if (f >= 0) {
                b[f][f] += y;
            }
            if (t >= 0) {
                b[t][t] += y;
            }
            if (f >= 0 && t >= 0) {
                b[f][t] -= y;
                b[t][f] -= y;
            }
        }
        double[][] x = Matrices.inverse(b);
        double[][] p = new double[m][n];
        for (int l = 0; l < m; l++) {
            int f = grid.from()[l];
            int t = grid.to()[l];
            for (int k = 1; k < n; k++) {
                double xf = f == 0 ? 0 : x[f - 1][k - 1];
                double xt = t == 0 ? 0 : x[t - 1][k - 1];
                p[l][k] = (xf - xt) / grid.reactance()[l];
            }
        }
        this.ptdf = p;
        double[][] gp = new double[m][grid.generators()];
        for (int l = 0; l < m; l++) {
            for (int g = 0; g < grid.generators(); g++) {
                gp[l][g] = p[l][grid.genBus()[g]];
            }
        }
        this.genPtdf = gp;
    }

    /** @return the grid */
    public ScienceSims.PowerGrid grid() {
        return grid;
    }

    /** @return {@code [line][bus]} distribution factors (the slack column is zero) */
    public double[][] ptdf() {
        return ptdf;
    }

    /**
     * Line flows.
     *
     * @param dispatch generator outputs
     * @param load bus loads
     * @return the flow on each line (MW, positive from {@code from} to {@code to})
     */
    public double[] flows(double[] dispatch, double[] load) {
        int m = grid.lines();
        double[] f = new double[m];
        for (int l = 0; l < m; l++) {
            double s = 0;
            for (int g = 0; g < dispatch.length; g++) {
                s += genPtdf[l][g] * dispatch[g];
            }
            for (int b = 0; b < load.length; b++) {
                s -= ptdf[l][b] * load[b];
            }
            f[l] = s;
        }
        return f;
    }

    /**
     * The flow sensitivity to each generator.
     *
     * @return {@code [line][generator]}
     */
    public double[][] generatorPtdf() {
        return genPtdf;
    }

    /**
     * Solves the DC-OPF.
     *
     * @param load bus loads
     * @return the optimal dispatch, or infeasible
     */
    public Dispatch solve(double[] load) {
        int gens = grid.generators();
        int m = grid.lines();
        double[] base = flows(new double[gens], load);
        double[][] aUb = new double[2 * m + gens][gens];
        double[] bUb = new double[2 * m + gens];
        for (int l = 0; l < m; l++) {
            for (int g = 0; g < gens; g++) {
                aUb[2 * l][g] = genPtdf[l][g];
                aUb[2 * l + 1][g] = -genPtdf[l][g];
            }
            bUb[2 * l] = grid.limit()[l] - base[l];
            bUb[2 * l + 1] = grid.limit()[l] + base[l];
        }
        for (int g = 0; g < gens; g++) {
            aUb[2 * m + g][g] = 1;
            bUb[2 * m + g] = grid.genMax()[g];
        }
        double[][] aEq = new double[1][gens];
        java.util.Arrays.fill(aEq[0], 1);
        Optimize.LinearSolution s = Optimize.linearProgram(grid.genCost(), aUb, bUb, aEq, new double[] {sum(load)});
        if (s.status() != Optimize.Status.OPTIMAL) {
            return new Dispatch(false, null, Double.NaN);
        }
        return new Dispatch(true, s.x(), s.value());
    }

    /**
     * Restores feasibility by shedding load at a high value of lost load, in one LP.
     *
     * @param load bus loads
     * @param valueOfLostLoad the cost per MW shed (well above every generator cost)
     * @return the dispatch and the shed load
     */
    public Shedding shed(double[] load, double valueOfLostLoad) {
        int gens = grid.generators();
        int n = grid.buses();
        int m = grid.lines();
        int v = gens + n;
        double[] base = flows(new double[gens], load);
        double[][] aUb = new double[2 * m + gens + n][v];
        double[] bUb = new double[2 * m + gens + n];
        for (int l = 0; l < m; l++) {
            for (int g = 0; g < gens; g++) {
                aUb[2 * l][g] = genPtdf[l][g];
                aUb[2 * l + 1][g] = -genPtdf[l][g];
            }
            for (int b = 0; b < n; b++) {
                // shedding s_b raises the flow by ptdf[l][b] s_b
                aUb[2 * l][gens + b] = ptdf[l][b];
                aUb[2 * l + 1][gens + b] = -ptdf[l][b];
            }
            bUb[2 * l] = grid.limit()[l] - base[l];
            bUb[2 * l + 1] = grid.limit()[l] + base[l];
        }
        for (int g = 0; g < gens; g++) {
            aUb[2 * m + g][g] = 1;
            bUb[2 * m + g] = grid.genMax()[g];
        }
        for (int b = 0; b < n; b++) {
            aUb[2 * m + gens + b][gens + b] = 1;
            bUb[2 * m + gens + b] = load[b];
        }
        double[] c = new double[v];
        double[][] aEq = new double[1][v];
        for (int g = 0; g < gens; g++) {
            c[g] = grid.genCost()[g];
            aEq[0][g] = 1;
        }
        for (int b = 0; b < n; b++) {
            c[gens + b] = valueOfLostLoad;
            aEq[0][gens + b] = 1;
        }
        Optimize.LinearSolution s = Optimize.linearProgram(c, aUb, bUb, aEq, new double[] {sum(load)});
        double[] d = java.util.Arrays.copyOf(s.x(), gens);
        double[] shed = java.util.Arrays.copyOfRange(s.x(), gens, v);
        return new Shedding(d, shed, sum(shed));
    }

    /**
     * Measures the constraint violations of a proposed dispatch.
     *
     * @param dispatch generator outputs
     * @param load bus loads
     * @return the violations
     */
    public Violation violation(double[] dispatch, double[] load) {
        double[] f = flows(dispatch, load);
        double line = 0;
        int over = 0;
        for (int l = 0; l < f.length; l++) {
            double e = Math.abs(f[l]) - grid.limit()[l];
            if (e > 1e-6) {
                line += e;
                over++;
            }
        }
        double bound = 0;
        for (int g = 0; g < dispatch.length; g++) {
            bound += Math.max(0, dispatch[g] - grid.genMax()[g]) + Math.max(0, -dispatch[g]);
        }
        return new Violation(Math.abs(sum(dispatch) - sum(load)), line, bound, over);
    }

    /**
     * The cost of a dispatch.
     *
     * @param dispatch generator outputs
     * @return {@code c g}
     */
    public double cost(double[] dispatch) {
        double s = 0;
        for (int g = 0; g < dispatch.length; g++) {
            s += grid.genCost()[g] * dispatch[g];
        }
        return s;
    }

    /**
     * Weighted-least-squares state estimation: the bus injections (non-slack buses; the slack
     * takes the balance) that best fit measured line flows and measured injections, solved by
     * the normal equations with core {@link Matrices#inverse}.
     *
     * @param flowMeasurements measured line flows
     * @param flowSd their noise standard deviation
     * @param injectionMeasurements measured net injections per bus
     * @param injectionSd their noise standard deviation
     * @return the estimated net injection per bus
     */
    public double[] estimateInjections(double[] flowMeasurements, double flowSd, double[] injectionMeasurements,
            double injectionSd) {
        int n = grid.buses();
        int m = grid.lines();
        int k = n - 1;
        double[][] hth = new double[k][k];
        double[] htz = new double[k];
        double wf = 1 / (flowSd * flowSd);
        double wi = 1 / (injectionSd * injectionSd);
        for (int l = 0; l < m; l++) {
            for (int a = 0; a < k; a++) {
                htz[a] += wf * ptdf[l][a + 1] * flowMeasurements[l];
                for (int b = 0; b < k; b++) {
                    hth[a][b] += wf * ptdf[l][a + 1] * ptdf[l][b + 1];
                }
            }
        }
        for (int a = 0; a < k; a++) {
            hth[a][a] += wi;
            htz[a] += wi * injectionMeasurements[a + 1];
        }
        double[][] inv = Matrices.inverse(hth);
        double[] p = new double[n];
        double total = 0;
        for (int a = 0; a < k; a++) {
            for (int b = 0; b < k; b++) {
                p[a + 1] += inv[a][b] * htz[b];
            }
            total += p[a + 1];
        }
        p[0] = -total;
        return p;
    }

    static double sum(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        return s;
    }
}
