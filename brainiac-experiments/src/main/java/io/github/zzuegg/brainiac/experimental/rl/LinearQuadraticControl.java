package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

import java.util.Objects;

/**
 * Classical optimal control on a known model ({@link ControlSystem}).
 * <ul>
 *   <li>{@link #lqr}: the linear-quadratic regulator, the infinite-horizon solution of the
 *       discrete Riccati equation for {@code x' = A x + B u} and cost
 *       {@code x'Qx + u'Ru}, giving {@code u = -K x} (Kalman 1960).</li>
 *   <li>{@link #ilqr}: iterative LQR linearises the dynamics and quadratises the cost around
 *       the current trajectory, solves the resulting time-varying LQR problem backwards for
 *       feed-forward and feedback gains, and rolls out forwards with a backtracking line
 *       search; Levenberg-Marquardt regularisation of {@code Q_uu} keeps the steps safe
 *       (Li and Todorov 2004; Tassa et al. 2012).</li>
 *   <li>{@link #ddp}: differential dynamic programming adds the second-order dynamics terms
 *       {@code sum_i V_x[i] f_i''} to the Q-function Hessians (Mayne 1966).</li>
 * </ul>
 * Dynamics Jacobians come from {@link ControlSystem#jacobians} (finite differences, or exact
 * autograd for a differentiable simulator); cost derivatives from finite differences. Actions
 * are clipped to the system's bounds in the forward pass. {@link #mpc} turns the optimiser
 * into a receding-horizon controller that warm-starts from its previous plan.
 *
 * <p>Tassa, Erez and Todorov 2012, "Synthesis and Stabilization of Complex Behaviors through
 * Online Trajectory Optimization", IROS; Li and Todorov 2004, "Iterative Linear Quadratic
 * Regulator Design for Nonlinear Biological Movement Systems", ICINCO.
 */
public final class LinearQuadraticControl {

    /**
     * An optimised open-loop trajectory.
     *
     * @param states {@code horizon + 1} states
     * @param controls {@code horizon} actions
     * @param cost the total cost
     * @param iterations iterations run
     */
    public record Trajectory(double[][] states, double[][] controls, double cost, int iterations) {
    }

    private final ControlSystem system;
    private final int horizon;
    private final boolean secondOrder;
    private final int n;
    private final int m;
    private double mu = 1e-3;

    private LinearQuadraticControl(ControlSystem system, int horizon, boolean secondOrder) {
        this.system = Objects.requireNonNull(system);
        this.horizon = horizon;
        this.secondOrder = secondOrder;
        this.n = system.stateSize();
        this.m = system.actionSize();
    }

    /**
     * @param system the model
     * @param horizon the planning horizon in steps
     * @return an iLQR optimiser
     */
    public static LinearQuadraticControl ilqr(ControlSystem system, int horizon) {
        return new LinearQuadraticControl(system, horizon, false);
    }

    /**
     * @param system the model
     * @param horizon the planning horizon in steps
     * @return a DDP optimiser
     */
    public static LinearQuadraticControl ddp(ControlSystem system, int horizon) {
        return new LinearQuadraticControl(system, horizon, true);
    }

    /**
     * The infinite-horizon discrete LQR gain, by iterating the Riccati recursion
     * {@code P <- Q + A'PA - A'PB (R + B'PB)^-1 B'PA} to convergence.
     *
     * @param a the dynamics matrix {@code [n][n]}
     * @param b the input matrix {@code [n][m]}
     * @param q the state cost {@code [n][n]}
     * @param r the action cost {@code [m][m]}
     * @param iterations the most Riccati iterations
     * @return the gain {@code K [m][n]}, with {@code u = -K x}
     */
    public static double[][] lqr(double[][] a, double[][] b, double[][] q, double[][] r, int iterations) {
        double[][] p = copy(q);
        double[][] k = new double[b[0].length][a.length];
        double[][] at = MlMatrices.transpose(a);
        double[][] bt = MlMatrices.transpose(b);
        for (int it = 0; it < iterations; it++) {
            double[][] pb = MlMatrices.multiply(p, b);
            double[][] s = add(r, MlMatrices.multiply(bt, pb));
            double[][] pa = MlMatrices.multiply(p, a);
            double[][] rhs = MlMatrices.multiply(bt, pa);
            double[][] l = cholesky(s);
            if (l == null) {
                throw new IllegalArgumentException("R + B'PB is not positive definite");
            }
            for (int j = 0; j < rhs[0].length; j++) {
                double[] col = new double[rhs.length];
                for (int i = 0; i < rhs.length; i++) {
                    col[i] = rhs[i][j];
                }
                double[] x = solve(l, col);
                for (int i = 0; i < rhs.length; i++) {
                    k[i][j] = x[i];
                }
            }
            double[][] next = add(q, MlMatrices.multiply(at, pa));
            double[][] correction = MlMatrices.multiply(MlMatrices.multiply(at, pb), k);
            double change = 0;
            for (int i = 0; i < next.length; i++) {
                for (int j = 0; j < next.length; j++) {
                    next[i][j] -= correction[i][j];
                    change = Math.max(change, Math.abs(next[i][j] - p[i][j]));
                }
            }
            p = next;
            if (change < 1e-10) {
                break;
            }
        }
        return k;
    }

    /** The lower Cholesky factor in double precision, or null if the matrix is not positive definite. */
    private static double[][] cholesky(double[][] a) {
        int d = a.length;
        double[][] l = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j <= i; j++) {
                double s = a[i][j];
                for (int k = 0; k < j; k++) {
                    s -= l[i][k] * l[j][k];
                }
                if (i == j) {
                    if (!(s > 1e-12)) {
                        return null;
                    }
                    l[i][i] = Math.sqrt(s);
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        return l;
    }

    private static double[] solve(double[][] l, double[] b) {
        int d = b.length;
        double[] y = new double[d];
        for (int i = 0; i < d; i++) {
            double s = b[i];
            for (int k = 0; k < i; k++) {
                s -= l[i][k] * y[k];
            }
            y[i] = s / l[i][i];
        }
        double[] x = new double[d];
        for (int i = d - 1; i >= 0; i--) {
            double s = y[i];
            for (int k = i + 1; k < d; k++) {
                s -= l[k][i] * x[k];
            }
            x[i] = s / l[i][i];
        }
        return x;
    }

    private static double[][] copy(double[][] a) {
        double[][] out = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i].clone();
        }
        return out;
    }

    private static double[][] add(double[][] a, double[][] b) {
        double[][] out = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                out[i][j] = a[i][j] + b[i][j];
            }
        }
        return out;
    }

    private double[] clip(double[] u) {
        double[] lo = system.actionLow();
        double[] hi = system.actionHigh();
        double[] out = new double[u.length];
        for (int i = 0; i < u.length; i++) {
            out[i] = Math.max(lo[i], Math.min(hi[i], u[i]));
        }
        return out;
    }

    private double rollout(double[] x0, double[][] us, double[][] xs) {
        xs[0] = x0.clone();
        double total = 0;
        for (int t = 0; t < horizon; t++) {
            us[t] = clip(us[t]);
            total += system.cost(xs[t], us[t]);
            xs[t + 1] = system.step(xs[t], us[t]);
        }
        return total + system.finalCost(xs[horizon]);
    }

    /**
     * Optimises an open-loop trajectory from a start state.
     *
     * @param x0 the start state
     * @param initial the initial controls {@code [horizon][m]}, or null for zeros
     * @param iterations the most iterations
     * @return the optimised trajectory
     */
    public Trajectory optimize(double[] x0, double[][] initial, int iterations) {
        double[][] us = new double[horizon][];
        for (int t = 0; t < horizon; t++) {
            us[t] = initial == null ? new double[m] : initial[t].clone();
        }
        double[][] xs = new double[horizon + 1][];
        double cost = rollout(x0, us, xs);
        int it = 0;
        for (; it < iterations; it++) {
            double[][] kff = new double[horizon][];
            double[][][] kfb = new double[horizon][][];
            if (!backward(xs, us, kff, kfb)) {
                mu = Math.min(mu * 10, 1e6);
                continue;
            }
            boolean improved = false;
            for (double alpha = 1; alpha > 1e-3; alpha /= 2) {
                double[][] nu = new double[horizon][];
                double[][] nx = new double[horizon + 1][];
                nx[0] = x0.clone();
                double total = 0;
                for (int t = 0; t < horizon; t++) {
                    double[] u = new double[m];
                    for (int i = 0; i < m; i++) {
                        u[i] = us[t][i] + alpha * kff[t][i];
                        for (int j = 0; j < n; j++) {
                            u[i] += kfb[t][i][j] * (nx[t][j] - xs[t][j]);
                        }
                    }
                    nu[t] = clip(u);
                    total += system.cost(nx[t], nu[t]);
                    nx[t + 1] = system.step(nx[t], nu[t]);
                }
                total += system.finalCost(nx[horizon]);
                if (total < cost) {
                    improved = true;
                    double gain = cost - total;
                    cost = total;
                    us = nu;
                    xs = nx;
                    mu = Math.max(mu / 3, 1e-6);
                    if (gain < 1e-6 * Math.abs(cost)) {
                        it = iterations;
                    }
                    break;
                }
            }
            if (!improved) {
                mu = Math.min(mu * 10, 1e6);
            }
        }
        return new Trajectory(xs, us, cost, Math.min(it, iterations));
    }

    private boolean backward(double[][] xs, double[][] us, double[][] kff, double[][][] kfb) {
        double[] xT = xs[horizon];
        double[][] fd = ControlSystem.secondOrder(z -> system.finalCost(z), xT);
        double[] vx = fd[0];
        double[][] vxx = new double[n][];
        for (int i = 0; i < n; i++) {
            vxx[i] = fd[1 + i];
        }
        for (int t = horizon - 1; t >= 0; t--) {
            double[] x = xs[t];
            double[] u = us[t];
            double[][][] ab = system.jacobians(x, u);
            double[][] a = ab[0];
            double[][] b = ab[1];
            double[] z = new double[n + m];
            System.arraycopy(x, 0, z, 0, n);
            System.arraycopy(u, 0, z, n, m);
            double[][] ld = ControlSystem.secondOrder(v -> system.cost(java.util.Arrays.copyOfRange(v, 0, n),
                    java.util.Arrays.copyOfRange(v, n, n + m)), z);
            double[][] fz = new double[n][n + m];
            for (int i = 0; i < n; i++) {
                System.arraycopy(a[i], 0, fz[i], 0, n);
                System.arraycopy(b[i], 0, fz[i], n, m);
            }
            // Q expansions over z = (x, u)
            double[] qz = new double[n + m];
            double[][] qzz = new double[n + m][n + m];
            double[][] qzzReg = new double[n + m][n + m];
            for (int p = 0; p < n + m; p++) {
                qz[p] = ld[0][p];
                for (int i = 0; i < n; i++) {
                    qz[p] += fz[i][p] * vx[i];
                }
            }
            for (int p = 0; p < n + m; p++) {
                for (int r = 0; r < n + m; r++) {
                    double s = 0;
                    double sReg = 0;
                    for (int i = 0; i < n; i++) {
                        double row = 0;
                        for (int j = 0; j < n; j++) {
                            row += vxx[i][j] * fz[j][r];
                        }
                        s += fz[i][p] * row;
                        sReg += fz[i][p] * row;
                    }
                    qzz[p][r] = ld[1 + p][r] + s;
                    qzzReg[p][r] = ld[1 + p][r] + (p >= n ? sReg : s);
                }
            }
            if (secondOrder) {
                double[][] curvature = dynamicsCurvature(z, vx);
                for (int p = 0; p < n + m; p++) {
                    for (int r = 0; r < n + m; r++) {
                        qzz[p][r] += curvature[p][r];
                        qzzReg[p][r] += curvature[p][r];
                    }
                }
            }
            double[][] quu = new double[m][m];
            double[][] qux = new double[m][n];
            double[] qu = new double[m];
            for (int i = 0; i < m; i++) {
                qu[i] = qz[n + i];
                for (int j = 0; j < m; j++) {
                    quu[i][j] = qzzReg[n + i][n + j] + (i == j ? mu : 0);
                }
                for (int j = 0; j < n; j++) {
                    qux[i][j] = qzzReg[n + i][j];
                }
            }
            double[][] chol = cholesky(quu);
            if (chol == null) {
                return false;
            }
            double[] k = solve(chol, qu);
            double[][] kk = new double[m][n];
            for (int j = 0; j < n; j++) {
                double[] col = new double[m];
                for (int i = 0; i < m; i++) {
                    col[i] = qux[i][j];
                }
                double[] solved = solve(chol, col);
                for (int i = 0; i < m; i++) {
                    kk[i][j] = solved[i];
                }
            }
            kff[t] = new double[m];
            kfb[t] = new double[m][n];
            for (int i = 0; i < m; i++) {
                kff[t][i] = -k[i];
                for (int j = 0; j < n; j++) {
                    kfb[t][i][j] = -kk[i][j];
                }
            }
            // value function update with the unregularised Q
            double[][] quuRaw = new double[m][m];
            double[][] quxRaw = new double[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    quuRaw[i][j] = qzz[n + i][n + j];
                }
                for (int j = 0; j < n; j++) {
                    quxRaw[i][j] = qzz[n + i][j];
                }
            }
            double[] nvx = new double[n];
            double[][] nvxx = new double[n][n];
            for (int i = 0; i < n; i++) {
                nvx[i] = qz[i];
                for (int a1 = 0; a1 < m; a1++) {
                    double kq = 0;
                    for (int a2 = 0; a2 < m; a2++) {
                        kq += quuRaw[a1][a2] * kff[t][a2];
                    }
                    nvx[i] += kfb[t][a1][i] * kq + kfb[t][a1][i] * qu[a1] + quxRaw[a1][i] * kff[t][a1];
                }
                for (int j = 0; j < n; j++) {
                    double v = qzz[i][j];
                    for (int a1 = 0; a1 < m; a1++) {
                        double kq = 0;
                        for (int a2 = 0; a2 < m; a2++) {
                            kq += quuRaw[a1][a2] * kfb[t][a2][j];
                        }
                        v += kfb[t][a1][i] * kq + kfb[t][a1][i] * quxRaw[a1][j] + quxRaw[a1][i] * kfb[t][a1][j];
                    }
                    nvxx[i][j] = v;
                }
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < i; j++) {
                    double s = (nvxx[i][j] + nvxx[j][i]) / 2;
                    nvxx[i][j] = s;
                    nvxx[j][i] = s;
                }
            }
            vx = nvx;
            vxx = nvxx;
        }
        return true;
    }

    /** {@code sum_i V_x[i] d^2 f_i / dz^2} by central differences of the Jacobians. */
    private double[][] dynamicsCurvature(double[] z, double[] vx) {
        int d = n + m;
        double[][] out = new double[d][d];
        double h = 1e-4;
        for (int p = 0; p < d; p++) {
            double[] zp = z.clone();
            double[] zm = z.clone();
            zp[p] += h;
            zm[p] -= h;
            double[][][] jp = system.jacobians(java.util.Arrays.copyOfRange(zp, 0, n),
                    java.util.Arrays.copyOfRange(zp, n, d));
            double[][][] jm = system.jacobians(java.util.Arrays.copyOfRange(zm, 0, n),
                    java.util.Arrays.copyOfRange(zm, n, d));
            for (int r = 0; r < d; r++) {
                double s = 0;
                for (int i = 0; i < n; i++) {
                    double up = r < n ? jp[0][i][r] : jp[1][i][r - n];
                    double down = r < n ? jm[0][i][r] : jm[1][i][r - n];
                    s += vx[i] * (up - down) / (2 * h);
                }
                out[p][r] = s;
            }
        }
        for (int p = 0; p < d; p++) {
            for (int r = 0; r < p; r++) {
                double s = (out[p][r] + out[r][p]) / 2;
                out[p][r] = s;
                out[r][p] = s;
            }
        }
        return out;
    }

    /**
     * A receding-horizon controller: re-optimise from every state, warm-started with the
     * previous plan shifted by one step, and apply the first action.
     *
     * @param iterationsPerStep optimiser iterations per control step (the first step gets
     *     ten times as many)
     * @return the planner
     */
    public ControlSystem.Planner mpc(int iterationsPerStep) {
        return new ControlSystem.Planner() {
            private double[][] plan;

            @Override
            public double[] act(double[] state) {
                double[][] init = null;
                if (plan != null) {
                    init = new double[horizon][];
                    for (int t = 0; t < horizon; t++) {
                        init[t] = plan[Math.min(t + 1, horizon - 1)].clone();
                    }
                }
                Trajectory tr = optimize(state, init, plan == null ? 10 * iterationsPerStep : iterationsPerStep);
                plan = tr.controls();
                return plan[0].clone();
            }

            @Override
            public void reset() {
                plan = null;
                mu = 1e-3;
            }
        };
    }
}
