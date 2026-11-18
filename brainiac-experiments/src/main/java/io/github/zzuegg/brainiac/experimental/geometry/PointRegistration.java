package io.github.zzuegg.brainiac.experimental.geometry;

import java.util.Arrays;

/**
 * Rigid point-cloud registration by the iterative closest point algorithm:
 *
 * <ul>
 *   <li>{@link Method#ICP_POINT}: alternate nearest-neighbour correspondences with the
 *       closed-form Kabsch alignment (Besl &amp; McKay, TPAMI 1992);</li>
 *   <li>{@link Method#ICP_PLANE}: minimise the distance to the target's tangent planes, with
 *       normals from a PCA of each target point's neighbourhood and a linearised rotation
 *       (Chen &amp; Medioni 1992), which converges in fewer iterations on smooth surfaces;</li>
 *   <li>{@link Method#TRIMMED}: point-to-point ICP on the best {@code trim} share of
 *       correspondences, for partial overlap (Chetverikov et al. 2002).</li>
 * </ul>
 *
 * All variants need a reasonable initial pose; nearest neighbours are found by brute force.
 */
public final class PointRegistration {

    private PointRegistration() {
    }

    /** The ICP variant. */
    public enum Method {
        /** Point-to-point. */
        ICP_POINT,
        /** Point-to-plane. */
        ICP_PLANE,
        /** Trimmed point-to-point. */
        TRIMMED
    }

    /**
     * A registration.
     *
     * @param rotation the rotation taking source to target
     * @param translation the translation
     * @param iterations the iterations run
     * @param rmse the final correspondence RMSE
     */
    public record Result(double[][] rotation, double[] translation, int iterations, double rmse) {
    }

    /**
     * The index of the nearest point.
     *
     * @param cloud the cloud
     * @param p a query
     * @return the nearest index
     */
    public static int nearest(double[][] cloud, double[] p) {
        int best = 0;
        double bd = Double.POSITIVE_INFINITY;
        for (int i = 0; i < cloud.length; i++) {
            double dx = cloud[i][0] - p[0];
            double dy = cloud[i][1] - p[1];
            double dz = cloud[i][2] - p[2];
            double d = dx * dx + dy * dy + dz * dz;
            if (d < bd) {
                bd = d;
                best = i;
            }
        }
        return best;
    }

    /**
     * Surface normals by PCA of the {@code k} nearest neighbours.
     *
     * @param cloud the cloud
     * @param k the neighbourhood size
     * @return unit normals {@code [n][3]}
     */
    public static double[][] normals(double[][] cloud, int k) {
        int n = cloud.length;
        double[][] out = new double[n][];
        for (int i = 0; i < n; i++) {
            double[] d = new double[n];
            Integer[] idx = new Integer[n];
            for (int j = 0; j < n; j++) {
                double dx = cloud[j][0] - cloud[i][0];
                double dy = cloud[j][1] - cloud[i][1];
                double dz = cloud[j][2] - cloud[i][2];
                d[j] = dx * dx + dy * dy + dz * dz;
                idx[j] = j;
            }
            Arrays.sort(idx, (a, b) -> Double.compare(d[a], d[b]));
            int kk = Math.min(k, n);
            double[] mean = new double[3];
            for (int q = 0; q < kk; q++) {
                for (int c = 0; c < 3; c++) {
                    mean[c] += cloud[idx[q]][c] / kk;
                }
            }
            double[][] cov = new double[3][3];
            for (int q = 0; q < kk; q++) {
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        cov[a][b] += (cloud[idx[q]][a] - mean[a]) * (cloud[idx[q]][b] - mean[b]);
                    }
                }
            }
            double[][] vec = GeometryLinalg.eigh(cov)[1];
            out[i] = new double[] {vec[0][2], vec[1][2], vec[2][2]};
        }
        return out;
    }

    /**
     * Registers a source cloud onto a target.
     *
     * @param source the source
     * @param target the target
     * @param method the variant
     * @param rotation the initial rotation
     * @param translation the initial translation
     * @param iterations the most iterations
     * @param trim the share of correspondences kept by {@link Method#TRIMMED}
     * @return the registration
     */
    public static Result register(double[][] source, double[][] target, Method method, double[][] rotation,
            double[] translation, int iterations, double trim) {
        double[][] r = rotation;
        double[] t = translation.clone();
        double[][] tn = method == Method.ICP_PLANE ? normals(target, 10) : null;
        int n = source.length;
        double rmse = Double.POSITIVE_INFINITY;
        int it = 0;
        for (; it < iterations; it++) {
            double[][] moved = new double[n][];
            int[] match = new int[n];
            double[] dist = new double[n];
            for (int i = 0; i < n; i++) {
                moved[i] = GeometryLinalg.transform(r, t, source[i]);
                match[i] = nearest(target, moved[i]);
                double[] d = GeometryLinalg.subtract(moved[i], target[match[i]]);
                dist[i] = GeometryLinalg.dot(d, d);
            }
            boolean[] keep = new boolean[n];
            Arrays.fill(keep, true);
            if (method == Method.TRIMMED) {
                double[] sorted = dist.clone();
                Arrays.sort(sorted);
                double cut = sorted[Math.max(0, Math.min(n - 1, (int) Math.ceil(trim * n) - 1))];
                for (int i = 0; i < n; i++) {
                    keep[i] = dist[i] <= cut;
                }
            }
            double s = 0;
            int kept = 0;
            for (int i = 0; i < n; i++) {
                if (keep[i]) {
                    s += dist[i];
                    kept++;
                }
            }
            double newRmse = Math.sqrt(s / Math.max(1, kept));
            if (Math.abs(rmse - newRmse) < 1e-10) {
                rmse = newRmse;
                break;
            }
            rmse = newRmse;
            if (method == Method.ICP_PLANE) {
                double[][] h = new double[6][6];
                double[] g = new double[6];
                for (int i = 0; i < n; i++) {
                    double[] q = target[match[i]];
                    double[] nn = tn[match[i]];
                    double[] p = moved[i];
                    double res = GeometryLinalg.dot(GeometryLinalg.subtract(p, q), nn);
                    double[] cr = GeometryLinalg.cross(p, nn);
                    double[] j = {cr[0], cr[1], cr[2], nn[0], nn[1], nn[2]};
                    for (int a = 0; a < 6; a++) {
                        g[a] += j[a] * res;
                        for (int b = 0; b < 6; b++) {
                            h[a][b] += j[a] * j[b];
                        }
                    }
                }
                for (int a = 0; a < 6; a++) {
                    h[a][a] += 1e-9;
                }
                double[] d = GeometryLinalg.solve(h, g);
                if (d == null) {
                    break;
                }
                double[][] dr = GeometryLinalg.exp(new double[] {-d[0], -d[1], -d[2]});
                r = GeometryLinalg.multiply(dr, r);
                double[] tt = GeometryLinalg.multiply(dr, t);
                t = new double[] {tt[0] - d[3], tt[1] - d[4], tt[2] - d[5]};
            } else {
                double[][] a = new double[kept][];
                double[][] b = new double[kept][];
                int c = 0;
                for (int i = 0; i < n; i++) {
                    if (keep[i]) {
                        a[c] = source[i];
                        b[c] = target[match[i]];
                        c++;
                    }
                }
                double[][] k = GeometryLinalg.kabsch(a, b, null);
                r = new double[][] {k[0], k[1], k[2]};
                t = k[3];
            }
        }
        return new Result(r, t, it, rmse);
    }
}
