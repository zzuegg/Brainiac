package io.github.zzuegg.brainiac.experimental.geometry;

/**
 * Camera pose from {@code n} 3-D to 2-D correspondences with known intrinsics (the
 * perspective-n-point problem), on normalised image coordinates {@code K^-1 x}:
 *
 * <ul>
 *   <li>{@link #dlt}: the direct linear transform, the null vector of a {@code 2n x 12} system
 *       for {@code [R | t]} up to scale, projected onto a rotation; needs {@code n >= 6};</li>
 *   <li>{@link #epnp}: EPnP (Lepetit, Moreno-Noguer &amp; Fua, IJCV 2009), which writes every
 *       point as a weighted sum of four control points, finds the camera-frame control points
 *       in the span of the four smallest eigenvectors of {@code M^T M}, recovers the span
 *       coefficients from the preserved control-point distances (three linearisations, each
 *       refined by Gauss-Newton), and aligns the frames by Kabsch; O(n), {@code n >= 4};</li>
 *   <li>{@link #refine}: Gauss-Newton on the reprojection error over a left-multiplied
 *       rotation update and the translation.</li>
 * </ul>
 */
public final class PoseEstimation {

    private PoseEstimation() {
    }

    /**
     * A camera pose {@code x_cam = R X + t}.
     *
     * @param rotation the rotation
     * @param translation the translation
     */
    public record Pose(double[][] rotation, double[] translation) {
    }

    /**
     * Mean reprojection error in normalised coordinates.
     *
     * @param pose a pose
     * @param world 3-D points
     * @param image normalised image points
     * @return the mean distance
     */
    public static double reprojection(Pose pose, double[][] world, double[][] image) {
        double s = 0;
        for (int i = 0; i < world.length; i++) {
            double[] c = GeometryLinalg.transform(pose.rotation(), pose.translation(), world[i]);
            s += Math.hypot(c[0] / c[2] - image[i][0], c[1] / c[2] - image[i][1]);
        }
        return s / world.length;
    }

    /**
     * The direct linear transform.
     *
     * @param world 3-D points {@code [n][3]}, {@code n >= 6}
     * @param image normalised image points {@code [n][2]}
     * @return the pose
     */
    public static Pose dlt(double[][] world, double[][] image) {
        int n = world.length;
        double[][] a = new double[2 * n][12];
        for (int i = 0; i < n; i++) {
            double[] x = world[i];
            double u = image[i][0];
            double v = image[i][1];
            double[] h = {x[0], x[1], x[2], 1};
            for (int j = 0; j < 4; j++) {
                a[2 * i][j] = h[j];
                a[2 * i][8 + j] = -u * h[j];
                a[2 * i + 1][4 + j] = h[j];
                a[2 * i + 1][8 + j] = -v * h[j];
            }
        }
        double[] p = GeometryLinalg.svd(a).nullVector();
        double[][] m = {{p[0], p[1], p[2]}, {p[4], p[5], p[6]}, {p[8], p[9], p[10]}};
        double[] t = {p[3], p[7], p[11]};
        if (GeometryLinalg.det3(m) < 0) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    m[i][j] = -m[i][j];
                }
                t[i] = -t[i];
            }
        }
        GeometryLinalg.Svd d = GeometryLinalg.svd(m);
        double scale = (d.s()[0] + d.s()[1] + d.s()[2]) / 3;
        double[][] r = GeometryLinalg.nearestRotation(m);
        return new Pose(r, new double[] {t[0] / scale, t[1] / scale, t[2] / scale});
    }

    /**
     * EPnP.
     *
     * @param world 3-D points {@code [n][3]}, {@code n >= 4}
     * @param image normalised image points
     * @return the pose with the smallest reprojection error over the three linearisations
     */
    public static Pose epnp(double[][] world, double[][] image) {
        int n = world.length;
        double[][] cw = new double[4][3];
        for (double[] x : world) {
            for (int k = 0; k < 3; k++) {
                cw[0][k] += x[k] / n;
            }
        }
        double[][] cov = new double[3][3];
        for (double[] x : world) {
            for (int p = 0; p < 3; p++) {
                for (int q = 0; q < 3; q++) {
                    cov[p][q] += (x[p] - cw[0][p]) * (x[q] - cw[0][q]);
                }
            }
        }
        double[][][] eig = GeometryLinalg.eigh(cov);
        for (int j = 0; j < 3; j++) {
            double s = Math.sqrt(Math.max(eig[0][0][j], 1e-12) / n);
            for (int k = 0; k < 3; k++) {
                cw[j + 1][k] = cw[0][k] + s * eig[1][k][j];
            }
        }
        double[][] basis = new double[3][3];
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                basis[k][j] = cw[j + 1][k] - cw[0][k];
            }
        }
        double[][] binv = GeometryLinalg.inverse(basis);
        double[][] alpha = new double[n][4];
        for (int i = 0; i < n; i++) {
            double[] b = GeometryLinalg.multiply(binv, GeometryLinalg.subtract(world[i], cw[0]));
            alpha[i] = new double[] {1 - b[0] - b[1] - b[2], b[0], b[1], b[2]};
        }
        double[][] mtm = new double[12][12];
        for (int i = 0; i < n; i++) {
            double[] r1 = new double[12];
            double[] r2 = new double[12];
            for (int j = 0; j < 4; j++) {
                r1[3 * j] = alpha[i][j];
                r1[3 * j + 2] = -alpha[i][j] * image[i][0];
                r2[3 * j + 1] = alpha[i][j];
                r2[3 * j + 2] = -alpha[i][j] * image[i][1];
            }
            for (int p = 0; p < 12; p++) {
                for (int q = 0; q < 12; q++) {
                    mtm[p][q] += r1[p] * r1[q] + r2[p] * r2[q];
                }
            }
        }
        double[][][] me = GeometryLinalg.eigh(mtm);
        double[][] v = new double[4][12];
        for (int k = 0; k < 4; k++) {
            for (int p = 0; p < 12; p++) {
                v[k][p] = me[1][p][11 - k];
            }
        }
        int[][] pairs = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
        double[][] l = new double[6][10];
        double[] rho = new double[6];
        for (int pi = 0; pi < 6; pi++) {
            int a = pairs[pi][0];
            int b = pairs[pi][1];
            double[][] dv = new double[4][3];
            for (int k = 0; k < 4; k++) {
                for (int c = 0; c < 3; c++) {
                    dv[k][c] = v[k][3 * a + c] - v[k][3 * b + c];
                }
            }
            l[pi] = new double[] {GeometryLinalg.dot(dv[0], dv[0]), 2 * GeometryLinalg.dot(dv[0], dv[1]),
                GeometryLinalg.dot(dv[1], dv[1]), 2 * GeometryLinalg.dot(dv[0], dv[2]),
                2 * GeometryLinalg.dot(dv[1], dv[2]), GeometryLinalg.dot(dv[2], dv[2]),
                2 * GeometryLinalg.dot(dv[0], dv[3]), 2 * GeometryLinalg.dot(dv[1], dv[3]),
                2 * GeometryLinalg.dot(dv[2], dv[3]), GeometryLinalg.dot(dv[3], dv[3])};
            double[] d = GeometryLinalg.subtract(cw[a], cw[b]);
            rho[pi] = GeometryLinalg.dot(d, d);
        }
        Pose best = null;
        double bestErr = Double.POSITIVE_INFINITY;
        for (int approx = 1; approx <= 3; approx++) {
            double[] beta = initialBetas(l, rho, approx);
            gaussNewtonBetas(l, rho, beta);
            Pose p = poseFromBetas(beta, v, alpha, world);
            double err = reprojection(p, world, image);
            if (err < bestErr) {
                bestErr = err;
                best = p;
            }
        }
        return best;
    }

    private static double[] lstsq(double[][] a, double[] b) {
        double[][] pinv = GeometryLinalg.pseudoInverse(a, 1e-12);
        return GeometryLinalg.multiply(pinv, b);
    }

    private static double[] initialBetas(double[][] l, double[] rho, int approx) {
        double[] beta = new double[4];
        if (approx == 1) {
            int[] cols = {0, 1, 3, 6};
            double[] b = lstsq(columns(l, cols), rho);
            double s = b[0] < 0 ? -1 : 1;
            beta[0] = Math.max(Math.sqrt(Math.abs(b[0])), 1e-12);
            for (int k = 1; k < 4; k++) {
                beta[k] = s * b[k] / beta[0];
            }
        } else if (approx == 2) {
            double[] b = lstsq(columns(l, new int[] {0, 1, 2}), rho);
            if (b[0] < 0) {
                beta[0] = Math.sqrt(-b[0]);
                beta[1] = b[2] < 0 ? Math.sqrt(-b[2]) : 0;
            } else {
                beta[0] = Math.sqrt(b[0]);
                beta[1] = b[2] > 0 ? Math.sqrt(b[2]) : 0;
            }
            if (b[1] < 0) {
                beta[0] = -beta[0];
            }
        } else {
            double[] b = lstsq(columns(l, new int[] {0, 1, 2, 3, 4}), rho);
            if (b[0] < 0) {
                beta[0] = Math.sqrt(-b[0]);
                beta[1] = b[2] < 0 ? Math.sqrt(-b[2]) : 0;
            } else {
                beta[0] = Math.sqrt(b[0]);
                beta[1] = b[2] > 0 ? Math.sqrt(b[2]) : 0;
            }
            if (b[1] < 0) {
                beta[0] = -beta[0];
            }
            beta[2] = b[3] / (Math.abs(beta[0]) < 1e-12 ? 1e-12 : beta[0]);
        }
        return beta;
    }

    private static double[][] columns(double[][] a, int[] cols) {
        double[][] o = new double[a.length][cols.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < cols.length; j++) {
                o[i][j] = a[i][cols[j]];
            }
        }
        return o;
    }

    private static double[] products(double[] b) {
        return new double[] {b[0] * b[0], b[0] * b[1], b[1] * b[1], b[0] * b[2], b[1] * b[2], b[2] * b[2],
            b[0] * b[3], b[1] * b[3], b[2] * b[3], b[3] * b[3]};
    }

    private static void gaussNewtonBetas(double[][] l, double[] rho, double[] beta) {
        for (int it = 0; it < 8; it++) {
            double[] prod = products(beta);
            double[] r = new double[6];
            double[][] j = new double[6][4];
            double[] b = beta;
            double[][] dp = {
                {2 * b[0], b[1], 0, b[2], 0, 0, b[3], 0, 0, 0},
                {0, b[0], 2 * b[1], 0, b[2], 0, 0, b[3], 0, 0},
                {0, 0, 0, b[0], b[1], 2 * b[2], 0, 0, b[3], 0},
                {0, 0, 0, 0, 0, 0, b[0], b[1], b[2], 2 * b[3]}};
            for (int i = 0; i < 6; i++) {
                r[i] = GeometryLinalg.dot(l[i], prod) - rho[i];
                for (int k = 0; k < 4; k++) {
                    j[i][k] = GeometryLinalg.dot(l[i], dp[k]);
                }
            }
            double[][] jtj = GeometryLinalg.multiply(GeometryLinalg.transpose(j), j);
            double[] jtr = GeometryLinalg.multiply(GeometryLinalg.transpose(j), r);
            for (int k = 0; k < 4; k++) {
                jtj[k][k] += 1e-12;
            }
            double[] step = GeometryLinalg.solve(jtj, jtr);
            if (step == null) {
                return;
            }
            for (int k = 0; k < 4; k++) {
                beta[k] -= step[k];
            }
        }
    }

    private static Pose poseFromBetas(double[] beta, double[][] v, double[][] alpha, double[][] world) {
        double[] cc = new double[12];
        for (int k = 0; k < 4; k++) {
            for (int p = 0; p < 12; p++) {
                cc[p] += beta[k] * v[k][p];
            }
        }
        int n = world.length;
        double[][] pc = new double[n][3];
        int negative = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 4; j++) {
                for (int c = 0; c < 3; c++) {
                    pc[i][c] += alpha[i][j] * cc[3 * j + c];
                }
            }
            negative += pc[i][2] < 0 ? 1 : 0;
        }
        if (negative > n / 2) {
            for (double[] p : pc) {
                for (int c = 0; c < 3; c++) {
                    p[c] = -p[c];
                }
            }
        }
        double[][] k = GeometryLinalg.kabsch(world, pc, null);
        return new Pose(new double[][] {k[0], k[1], k[2]}, k[3]);
    }

    /**
     * Gauss-Newton refinement of the reprojection error.
     *
     * @param pose the initial pose
     * @param world 3-D points
     * @param image normalised image points
     * @param iterations the iterations
     * @return the refined pose
     */
    public static Pose refine(Pose pose, double[][] world, double[][] image, int iterations) {
        double[][] r = pose.rotation();
        double[] t = pose.translation().clone();
        int n = world.length;
        for (int it = 0; it < iterations; it++) {
            double[][] jtj = new double[6][6];
            double[] jtr = new double[6];
            for (int i = 0; i < n; i++) {
                double[] rx = GeometryLinalg.multiply(r, world[i]);
                double[] c = {rx[0] + t[0], rx[1] + t[1], rx[2] + t[2]};
                double z = c[2];
                double[] res = {c[0] / z - image[i][0], c[1] / z - image[i][1]};
                double[][] dpi = {{1 / z, 0, -c[0] / (z * z)}, {0, 1 / z, -c[1] / (z * z)}};
                double[][] dcd = new double[3][6];
                double[][] sk = GeometryLinalg.skew(rx);
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        dcd[a][b] = -sk[a][b];
                    }
                    dcd[a][3 + a] = 1;
                }
                double[][] j = GeometryLinalg.multiply(dpi, dcd);
                for (int a = 0; a < 6; a++) {
                    jtr[a] += j[0][a] * res[0] + j[1][a] * res[1];
                    for (int b = 0; b < 6; b++) {
                        jtj[a][b] += j[0][a] * j[0][b] + j[1][a] * j[1][b];
                    }
                }
            }
            for (int a = 0; a < 6; a++) {
                jtj[a][a] *= 1 + 1e-9;
                jtj[a][a] += 1e-12;
            }
            double[] d = GeometryLinalg.solve(jtj, jtr);
            if (d == null) {
                break;
            }
            r = GeometryLinalg.multiply(GeometryLinalg.exp(new double[] {-d[0], -d[1], -d[2]}), r);
            for (int k = 0; k < 3; k++) {
                t[k] -= d[3 + k];
            }
        }
        return new Pose(r, t);
    }
}
