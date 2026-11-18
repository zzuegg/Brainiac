package io.github.zzuegg.brainiac.experimental.geometry;

import java.util.ArrayList;
import java.util.List;

/**
 * Bundle adjustment (Triggs et al., "Bundle adjustment - a modern synthesis", 2000): joint
 * Levenberg-Marquardt refinement of camera poses and 3-D points on the reprojection error, with
 * analytic Jacobians (a left-multiplied rotation update and the translation per camera, the
 * position per point). The normal equations are block-sparse: a 6x6 block per camera, a 3x3
 * block per point and a 6x3 coupling block per observation. {@link Solver#SCHUR} eliminates the
 * points with the Schur complement {@code S = U - W V^-1 W^T}, solves the small camera system
 * and back-substitutes; {@link Solver#DENSE} solves the full system, the reference it must match.
 * A Huber loss down-weights outliers by iteratively reweighted least squares. The gauge is
 * fixed by holding the first camera constant (the scale stays free and is held by damping).
 */
public final class BundleAdjustment {

    /** How the normal equations are solved. */
    public enum Solver {
        /** Schur complement on the cameras. */
        SCHUR,
        /** The full dense system. */
        DENSE
    }

    /**
     * One observation: point {@code point} seen by camera {@code camera} at pixel {@code (u, v)}.
     *
     * @param camera the camera
     * @param point the point
     * @param u the pixel x
     * @param v the pixel y
     */
    public record Observation(int camera, int point, double u, double v) {
    }

    /**
     * The result of a run.
     *
     * @param rotations the refined rotations
     * @param translations the refined translations
     * @param points the refined points
     * @param rmse per iteration the reprojection RMSE in pixels, initial first
     * @param millisPerIteration the mean wall time of one iteration
     */
    public record Result(double[][][] rotations, double[][] translations, double[][] points, double[] rmse,
            double millisPerIteration) {
    }

    private final double focal;
    private final double cx;
    private final double cy;
    private final double huber;

    /**
     * Creates the solver for cameras sharing known intrinsics.
     *
     * @param focal the focal length
     * @param cx the principal point x
     * @param cy the principal point y
     * @param huber the Huber threshold in pixels, or {@code Double.POSITIVE_INFINITY} for least
     *     squares
     */
    public BundleAdjustment(double focal, double cx, double cy, double huber) {
        this.focal = focal;
        this.cx = cx;
        this.cy = cy;
        this.huber = huber;
    }

    /**
     * The reprojection RMSE.
     *
     * @param rotations camera rotations
     * @param translations camera translations
     * @param points the points
     * @param observations the observations
     * @return the root mean squared pixel error
     */
    public double rmse(double[][][] rotations, double[][] translations, double[][] points,
            List<Observation> observations) {
        double s = 0;
        for (Observation o : observations) {
            double[] r = residual(rotations[o.camera()], translations[o.camera()], points[o.point()], o);
            s += r[0] * r[0] + r[1] * r[1];
        }
        return Math.sqrt(s / Math.max(1, observations.size()));
    }

    private double[] residual(double[][] r, double[] t, double[] x, Observation o) {
        double[] c = GeometryLinalg.transform(r, t, x);
        return new double[] {focal * c[0] / c[2] + cx - o.u(), focal * c[1] / c[2] + cy - o.v()};
    }

    private double cost(double[][][] rs, double[][] ts, double[][] ps, List<Observation> obs) {
        double s = 0;
        for (Observation o : obs) {
            double[] r = residual(rs[o.camera()], ts[o.camera()], ps[o.point()], o);
            double e = Math.sqrt(r[0] * r[0] + r[1] * r[1]);
            s += e <= huber ? e * e : 2 * huber * e - huber * huber;
        }
        return s;
    }

    /**
     * Runs Levenberg-Marquardt.
     *
     * @param rotations initial rotations (copied)
     * @param translations initial translations (copied)
     * @param points initial points (copied)
     * @param observations the observations
     * @param iterations the iterations
     * @param solver how to solve the normal equations
     * @return the result
     */
    public Result run(double[][][] rotations, double[][] translations, double[][] points,
            List<Observation> observations, int iterations, Solver solver) {
        int m = rotations.length;
        int n = points.length;
        double[][][] rs = new double[m][][];
        double[][] ts = new double[m][];
        double[][] ps = new double[n][];
        for (int j = 0; j < m; j++) {
            rs[j] = new double[3][];
            for (int a = 0; a < 3; a++) {
                rs[j][a] = rotations[j][a].clone();
            }
            ts[j] = translations[j].clone();
        }
        for (int i = 0; i < n; i++) {
            ps[i] = points[i].clone();
        }
        List<Double> history = new ArrayList<>();
        history.add(rmse(rs, ts, ps, observations));
        double lambda = 1e-3;
        double current = cost(rs, ts, ps, observations);
        long time = 0;
        for (int it = 0; it < iterations; it++) {
            long t0 = System.nanoTime();
            int mc = m - 1;
            double[][][] u = new double[mc][6][6];
            double[][] gc = new double[mc][6];
            double[][][] v = new double[n][3][3];
            double[][] gp = new double[n][3];
            double[][][] w = new double[observations.size()][][];
            for (int k = 0; k < observations.size(); k++) {
                Observation o = observations.get(k);
                int j = o.camera();
                double[][] r = rs[j];
                double[] rx = GeometryLinalg.multiply(r, ps[o.point()]);
                double[] c = {rx[0] + ts[j][0], rx[1] + ts[j][1], rx[2] + ts[j][2]};
                double z = c[2];
                double[] res = {focal * c[0] / z + cx - o.u(), focal * c[1] / z + cy - o.v()};
                double e = Math.hypot(res[0], res[1]);
                double wt = e <= huber ? 1 : huber / e;
                double[][] dpi = {{focal / z, 0, -focal * c[0] / (z * z)}, {0, focal / z, -focal * c[1] / (z * z)}};
                double[][] jp = GeometryLinalg.multiply(dpi, r);
                for (int a = 0; a < 3; a++) {
                    gp[o.point()][a] += wt * (jp[0][a] * res[0] + jp[1][a] * res[1]);
                    for (int b = 0; b < 3; b++) {
                        v[o.point()][a][b] += wt * (jp[0][a] * jp[0][b] + jp[1][a] * jp[1][b]);
                    }
                }
                if (j == 0) {
                    continue;
                }
                double[][] sk = GeometryLinalg.skew(rx);
                double[][] dc = new double[3][6];
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        dc[a][b] = -sk[a][b];
                    }
                    dc[a][3 + a] = 1;
                }
                double[][] jc = GeometryLinalg.multiply(dpi, dc);
                double[][] wk = new double[6][3];
                for (int a = 0; a < 6; a++) {
                    gc[j - 1][a] += wt * (jc[0][a] * res[0] + jc[1][a] * res[1]);
                    for (int b = 0; b < 6; b++) {
                        u[j - 1][a][b] += wt * (jc[0][a] * jc[0][b] + jc[1][a] * jc[1][b]);
                    }
                    for (int b = 0; b < 3; b++) {
                        wk[a][b] = wt * (jc[0][a] * jp[0][b] + jc[1][a] * jp[1][b]);
                    }
                }
                w[k] = wk;
            }
            boolean accepted = false;
            for (int attempt = 0; attempt < 10 && !accepted; attempt++) {
                double[][] dcams;
                double[][] dpts;
                if (solver == Solver.SCHUR) {
                    double[][][] sol = schur(u, gc, v, gp, w, observations, mc, n, lambda);
                    dcams = sol[0];
                    dpts = sol[1];
                } else {
                    double[][][] sol = dense(u, gc, v, gp, w, observations, mc, n, lambda);
                    dcams = sol[0];
                    dpts = sol[1];
                }
                double[][][] nr = rs.clone();
                double[][] nt = ts.clone();
                double[][] np = new double[n][];
                for (int j = 1; j < m; j++) {
                    double[] d = dcams[j - 1];
                    nr[j] = GeometryLinalg.multiply(GeometryLinalg.exp(new double[] {-d[0], -d[1], -d[2]}), rs[j]);
                    nt[j] = new double[] {ts[j][0] - d[3], ts[j][1] - d[4], ts[j][2] - d[5]};
                }
                for (int i = 0; i < n; i++) {
                    np[i] = new double[] {ps[i][0] - dpts[i][0], ps[i][1] - dpts[i][1], ps[i][2] - dpts[i][2]};
                }
                double c = cost(nr, nt, np, observations);
                if (c < current) {
                    current = c;
                    rs = nr;
                    ts = nt;
                    ps = np;
                    lambda = Math.max(lambda / 10, 1e-12);
                    accepted = true;
                } else {
                    lambda *= 10;
                }
            }
            time += System.nanoTime() - t0;
            history.add(rmse(rs, ts, ps, observations));
            if (!accepted) {
                break;
            }
        }
        double[] h = new double[history.size()];
        for (int i = 0; i < h.length; i++) {
            h[i] = history.get(i);
        }
        return new Result(rs, ts, ps, h, time / 1e6 / Math.max(1, history.size() - 1));
    }

    private static double[][] damped(double[][] a, double lambda) {
        double[][] o = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            o[i] = a[i].clone();
            o[i][i] += lambda * (a[i][i] + 1e-9);
        }
        return o;
    }

    private static double[][][] schur(double[][][] u, double[][] gc, double[][][] v, double[][] gp, double[][][] w,
            List<Observation> obs, int mc, int n, double lambda) {
        double[][][] vinv = new double[n][][];
        for (int i = 0; i < n; i++) {
            vinv[i] = GeometryLinalg.inverse(damped(v[i], lambda));
            if (vinv[i] == null) {
                vinv[i] = new double[3][3];
            }
        }
        List<List<Integer>> byPoint = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            byPoint.add(new ArrayList<>());
        }
        for (int k = 0; k < obs.size(); k++) {
            if (w[k] != null) {
                byPoint.get(obs.get(k).point()).add(k);
            }
        }
        int dim = 6 * mc;
        double[][] s = new double[dim][dim];
        double[] rhs = new double[dim];
        for (int j = 0; j < mc; j++) {
            double[][] du = damped(u[j], lambda);
            for (int a = 0; a < 6; a++) {
                System.arraycopy(du[a], 0, s[6 * j + a], 6 * j, 6);
                rhs[6 * j + a] = gc[j][a];
            }
        }
        double[][][] wv = new double[obs.size()][][];
        for (int i = 0; i < n; i++) {
            for (int k : byPoint.get(i)) {
                wv[k] = GeometryLinalg.multiply(w[k], vinv[i]);
                int j = obs.get(k).camera() - 1;
                double[] wvg = GeometryLinalg.multiply(wv[k], gp[i]);
                for (int a = 0; a < 6; a++) {
                    rhs[6 * j + a] -= wvg[a];
                }
                for (int k2 : byPoint.get(i)) {
                    int j2 = obs.get(k2).camera() - 1;
                    double[][] blk = GeometryLinalg.multiply(wv[k], GeometryLinalg.transpose(w[k2]));
                    for (int a = 0; a < 6; a++) {
                        for (int b = 0; b < 6; b++) {
                            s[6 * j + a][6 * j2 + b] -= blk[a][b];
                        }
                    }
                }
            }
        }
        double[] dc = GeometryLinalg.solveSpd(s, rhs);
        double[][] dcams = new double[mc][6];
        for (int j = 0; j < mc; j++) {
            System.arraycopy(dc, 6 * j, dcams[j], 0, 6);
        }
        double[][] dpts = new double[n][];
        for (int i = 0; i < n; i++) {
            double[] r = gp[i].clone();
            for (int k : byPoint.get(i)) {
                int j = obs.get(k).camera() - 1;
                double[] wtd = GeometryLinalg.multiply(GeometryLinalg.transpose(w[k]), dcams[j]);
                for (int a = 0; a < 3; a++) {
                    r[a] -= wtd[a];
                }
            }
            dpts[i] = GeometryLinalg.multiply(vinv[i], r);
        }
        return new double[][][] {dcams, dpts};
    }

    private static double[][][] dense(double[][][] u, double[][] gc, double[][][] v, double[][] gp, double[][][] w,
            List<Observation> obs, int mc, int n, double lambda) {
        int dim = 6 * mc + 3 * n;
        double[][] h = new double[dim][dim];
        double[] g = new double[dim];
        for (int j = 0; j < mc; j++) {
            double[][] du = damped(u[j], lambda);
            for (int a = 0; a < 6; a++) {
                System.arraycopy(du[a], 0, h[6 * j + a], 6 * j, 6);
                g[6 * j + a] = gc[j][a];
            }
        }
        int off = 6 * mc;
        for (int i = 0; i < n; i++) {
            double[][] dv = damped(v[i], lambda);
            for (int a = 0; a < 3; a++) {
                System.arraycopy(dv[a], 0, h[off + 3 * i + a], off + 3 * i, 3);
                g[off + 3 * i + a] = gp[i][a];
            }
        }
        for (int k = 0; k < obs.size(); k++) {
            if (w[k] == null) {
                continue;
            }
            int j = obs.get(k).camera() - 1;
            int i = obs.get(k).point();
            for (int a = 0; a < 6; a++) {
                for (int b = 0; b < 3; b++) {
                    h[6 * j + a][off + 3 * i + b] += w[k][a][b];
                    h[off + 3 * i + b][6 * j + a] += w[k][a][b];
                }
            }
        }
        double[] d = GeometryLinalg.solveSpd(h, g);
        double[][] dcams = new double[mc][6];
        for (int j = 0; j < mc; j++) {
            System.arraycopy(d, 6 * j, dcams[j], 0, 6);
        }
        double[][] dpts = new double[n][3];
        for (int i = 0; i < n; i++) {
            System.arraycopy(d, off + 3 * i, dpts[i], 0, 3);
        }
        return new double[][][] {dcams, dpts};
    }
}
