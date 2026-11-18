package io.github.zzuegg.brainiac.experimental.geometry;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Two-view epipolar geometry. The fundamental matrix {@code F} satisfies
 * {@code x2^T F x1 = 0} for corresponding pixels; the eight-point algorithm (Longuet-Higgins,
 * Nature 1981) solves it linearly from the null vector of an {@code n x 9} system and
 * enforces rank two, but only after Hartley's normalisation (centre the points and scale their
 * mean distance to sqrt 2; Hartley, TPAMI 1997) is it accurate. With known intrinsics the
 * essential matrix {@code E = K2^T F K1 = [t]x R} gives four {@code (R, t)} candidates, and the
 * cheirality check (points in front of both cameras) keeps one; the DLT triangulates points
 * (Hartley &amp; Sturm, CVIU 1997). The Sampson distance is the first-order geometric error, and
 * {@link #ransac} wraps the eight-point solver for outliers (Fischler &amp; Bolles 1981).
 */
public final class TwoViewGeometry {

    private TwoViewGeometry() {
    }

    /**
     * A relative pose.
     *
     * @param rotation the rotation of camera 2 relative to camera 1
     * @param translation the unit translation direction
     * @param points the triangulated points in camera-1 coordinates (up to scale)
     */
    public record Pose(double[][] rotation, double[] translation, double[][] points) {
    }

    /**
     * A RANSAC fit.
     *
     * @param fundamental the fundamental matrix refitted on the inliers
     * @param inliers per correspondence whether it is an inlier
     * @param iterations the iterations run
     */
    public record Fit(double[][] fundamental, boolean[] inliers, int iterations) {
    }

    /**
     * Hartley normalisation: the similarity that centres points and scales their mean
     * distance to sqrt 2.
     *
     * @param points pixels {@code [n][2]}
     * @return the 3x3 transform
     */
    public static double[][] normalisation(double[][] points) {
        double mx = 0;
        double my = 0;
        for (double[] p : points) {
            mx += p[0];
            my += p[1];
        }
        mx /= points.length;
        my /= points.length;
        double d = 0;
        for (double[] p : points) {
            d += Math.hypot(p[0] - mx, p[1] - my);
        }
        d /= points.length;
        double s = Math.sqrt(2) / Math.max(d, 1e-12);
        return new double[][] {{s, 0, -s * mx}, {0, s, -s * my}, {0, 0, 1}};
    }

    private static double[] apply(double[][] t, double[] p) {
        return new double[] {t[0][0] * p[0] + t[0][1] * p[1] + t[0][2], t[1][0] * p[0] + t[1][1] * p[1] + t[1][2]};
    }

    /**
     * The eight-point algorithm.
     *
     * @param x1 pixels in image 1 {@code [n][2]}, {@code n >= 8}
     * @param x2 matching pixels in image 2
     * @param normalise apply Hartley's normalisation
     * @return {@code F} with rank two
     */
    public static double[][] eightPoint(double[][] x1, double[][] x2, boolean normalise) {
        int n = x1.length;
        double[][] t1 = normalise ? normalisation(x1) : GeometryLinalg.identity(3);
        double[][] t2 = normalise ? normalisation(x2) : GeometryLinalg.identity(3);
        double[][] a = new double[n][9];
        for (int i = 0; i < n; i++) {
            double[] p = apply(t1, x1[i]);
            double[] q = apply(t2, x2[i]);
            a[i] = new double[] {q[0] * p[0], q[0] * p[1], q[0], q[1] * p[0], q[1] * p[1], q[1], p[0], p[1], 1};
        }
        double[] f = GeometryLinalg.svd(a).nullVector();
        double[][] fm = {{f[0], f[1], f[2]}, {f[3], f[4], f[5]}, {f[6], f[7], f[8]}};
        GeometryLinalg.Svd d = GeometryLinalg.svd(fm);
        double[][] s = {{d.s()[0], 0, 0}, {0, d.s()[1], 0}, {0, 0, 0}};
        fm = GeometryLinalg.multiply(GeometryLinalg.multiply(d.u(), s), GeometryLinalg.transpose(d.v()));
        fm = GeometryLinalg.multiply(GeometryLinalg.multiply(GeometryLinalg.transpose(t2), fm), t1);
        double norm = 0;
        for (double[] row : fm) {
            for (double v : row) {
                norm += v * v;
            }
        }
        norm = Math.sqrt(norm);
        for (double[] row : fm) {
            for (int j = 0; j < 3; j++) {
                row[j] /= norm;
            }
        }
        return fm;
    }

    /**
     * The Sampson distance of one correspondence (squared, in pixels squared).
     *
     * @param f the fundamental matrix
     * @param x1 a pixel in image 1
     * @param x2 its match in image 2
     * @return the squared Sampson error
     */
    public static double sampson(double[][] f, double[] x1, double[] x2) {
        double[] p = {x1[0], x1[1], 1};
        double[] q = {x2[0], x2[1], 1};
        double[] fp = GeometryLinalg.multiply(f, p);
        double[] ftq = GeometryLinalg.multiply(GeometryLinalg.transpose(f), q);
        double e = GeometryLinalg.dot(q, fp);
        double den = fp[0] * fp[0] + fp[1] * fp[1] + ftq[0] * ftq[0] + ftq[1] * ftq[1];
        return e * e / Math.max(den, 1e-300);
    }

    /**
     * The essential matrix from {@code F} and the intrinsics, projected to singular values
     * {@code (1, 1, 0)}.
     *
     * @param f the fundamental matrix
     * @param k1 intrinsics of camera 1
     * @param k2 intrinsics of camera 2
     * @return {@code E}
     */
    public static double[][] essential(double[][] f, double[][] k1, double[][] k2) {
        double[][] e = GeometryLinalg.multiply(GeometryLinalg.multiply(GeometryLinalg.transpose(k2), f), k1);
        GeometryLinalg.Svd d = GeometryLinalg.svd(e);
        double[][] s = {{1, 0, 0}, {0, 1, 0}, {0, 0, 0}};
        return GeometryLinalg.multiply(GeometryLinalg.multiply(d.u(), s), GeometryLinalg.transpose(d.v()));
    }

    /**
     * DLT triangulation of one point from two 3x4 projection matrices.
     *
     * @param p1 the first projection
     * @param p2 the second projection
     * @param x1 the pixel (or normalised point) in view 1
     * @param x2 in view 2
     * @return the 3-D point
     */
    public static double[] triangulate(double[][] p1, double[][] p2, double[] x1, double[] x2) {
        double[][] a = new double[4][4];
        for (int j = 0; j < 4; j++) {
            a[0][j] = x1[0] * p1[2][j] - p1[0][j];
            a[1][j] = x1[1] * p1[2][j] - p1[1][j];
            a[2][j] = x2[0] * p2[2][j] - p2[0][j];
            a[3][j] = x2[1] * p2[2][j] - p2[1][j];
        }
        double[] x = GeometryLinalg.svd(a).nullVector();
        double w = Math.abs(x[3]) < 1e-300 ? 1e-300 : x[3];
        return new double[] {x[0] / w, x[1] / w, x[2] / w};
    }

    /**
     * Builds {@code [R | t]}.
     *
     * @param r a rotation
     * @param t a translation
     * @return the 3x4 matrix
     */
    public static double[][] projection(double[][] r, double[] t) {
        return new double[][] {{r[0][0], r[0][1], r[0][2], t[0]}, {r[1][0], r[1][1], r[1][2], t[1]},
            {r[2][0], r[2][1], r[2][2], t[2]}};
    }

    /**
     * Normalised camera coordinates {@code K^-1 x}.
     *
     * @param k intrinsics
     * @param x pixels
     * @return {@code [n][2]}
     */
    public static double[][] normalised(double[][] k, double[][] x) {
        double[][] ki = GeometryLinalg.inverse(k);
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = apply(ki, x[i]);
        }
        return out;
    }

    /**
     * Decomposes an essential matrix and keeps the candidate with the most points in front of
     * both cameras.
     *
     * @param e the essential matrix
     * @param n1 normalised points in view 1
     * @param n2 normalised points in view 2
     * @return the pose with triangulated points
     */
    public static Pose decompose(double[][] e, double[][] n1, double[][] n2) {
        GeometryLinalg.Svd d = GeometryLinalg.svd(e);
        double[][] u = d.u();
        double[][] v = d.v();
        if (GeometryLinalg.det3(u) < 0) {
            u = negateColumn(u, 2);
        }
        if (GeometryLinalg.det3(v) < 0) {
            v = negateColumn(v, 2);
        }
        double[][] w = {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}};
        double[][] vt = GeometryLinalg.transpose(v);
        double[][] ra = GeometryLinalg.multiply(GeometryLinalg.multiply(u, w), vt);
        double[][] rb = GeometryLinalg.multiply(GeometryLinalg.multiply(u, GeometryLinalg.transpose(w)), vt);
        double[] t = {u[0][2], u[1][2], u[2][2]};
        double[] tn = {-t[0], -t[1], -t[2]};
        double[][] p1 = projection(GeometryLinalg.identity(3), new double[3]);
        Pose best = null;
        int bestCount = -1;
        for (double[][] r : new double[][][] {ra, rb}) {
            for (double[] tt : new double[][] {t, tn}) {
                double[][] p2 = projection(r, tt);
                int count = 0;
                double[][] pts = new double[n1.length][];
                for (int i = 0; i < n1.length; i++) {
                    double[] x = triangulate(p1, p2, n1[i], n2[i]);
                    pts[i] = x;
                    double z2 = GeometryLinalg.dot(r[2], x) + tt[2];
                    if (x[2] > 0 && z2 > 0) {
                        count++;
                    }
                }
                if (count > bestCount) {
                    bestCount = count;
                    best = new Pose(r, tt.clone(), pts);
                }
            }
        }
        return best;
    }

    private static double[][] negateColumn(double[][] m, int c) {
        double[][] o = new double[m.length][];
        for (int i = 0; i < m.length; i++) {
            o[i] = m[i].clone();
            o[i][c] = -o[i][c];
        }
        return o;
    }

    /**
     * RANSAC over the normalised eight-point algorithm with the Sampson distance.
     *
     * @param x1 pixels in image 1
     * @param x2 matches in image 2
     * @param threshold the inlier Sampson distance in pixels
     * @param maxIterations the most iterations
     * @param confidence the stopping confidence
     * @param seed the seed
     * @return the fit
     */
    public static Fit ransac(double[][] x1, double[][] x2, double threshold, int maxIterations, double confidence,
            long seed) {
        int n = x1.length;
        SplittableRandom r = new SplittableRandom(seed);
        boolean[] best = new boolean[n];
        int bestCount = -1;
        int needed = maxIterations;
        int it = 0;
        double t2 = threshold * threshold;
        for (; it < Math.min(needed, maxIterations); it++) {
            int[] pick = new int[8];
            for (int k = 0; k < 8; k++) {
                boolean dup;
                do {
                    pick[k] = r.nextInt(n);
                    dup = false;
                    for (int q = 0; q < k; q++) {
                        dup |= pick[q] == pick[k];
                    }
                } while (dup);
            }
            double[][] a = new double[8][];
            double[][] b = new double[8][];
            for (int k = 0; k < 8; k++) {
                a[k] = x1[pick[k]];
                b[k] = x2[pick[k]];
            }
            double[][] f = eightPoint(a, b, true);
            boolean[] in = new boolean[n];
            int count = 0;
            for (int i = 0; i < n; i++) {
                in[i] = sampson(f, x1[i], x2[i]) < t2;
                count += in[i] ? 1 : 0;
            }
            if (count > bestCount) {
                bestCount = count;
                best = in;
                double w = count / (double) n;
                double p8 = Math.pow(w, 8);
                needed = p8 >= 1 ? 1 : (int) Math.ceil(Math.log(1 - confidence) / Math.log(1 - Math.max(p8, 1e-12)));
            }
        }
        double[][] f = null;
        for (int round = 0; round < 4; round++) {
            List<double[]> a = new ArrayList<>();
            List<double[]> b = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (best[i]) {
                    a.add(x1[i]);
                    b.add(x2[i]);
                }
            }
            if (a.size() < 8) {
                break;
            }
            f = eightPoint(a.toArray(new double[0][]), b.toArray(new double[0][]), true);
            boolean[] in = new boolean[n];
            int count = 0;
            for (int i = 0; i < n; i++) {
                in[i] = sampson(f, x1[i], x2[i]) < t2;
                count += in[i] ? 1 : 0;
            }
            if (count < a.size()) {
                break;
            }
            best = in;
        }
        return new Fit(f, best, it);
    }

    /**
     * Linear relative pose from pixel correspondences with known intrinsics: normalised
     * eight-point, essential projection and cheirality.
     *
     * @param x1 pixels in image 1
     * @param x2 pixels in image 2
     * @param k1 intrinsics 1
     * @param k2 intrinsics 2
     * @param normalise use Hartley normalisation
     * @return the pose
     */
    public static Pose relativePose(double[][] x1, double[][] x2, double[][] k1, double[][] k2, boolean normalise) {
        double[][] f = eightPoint(x1, x2, normalise);
        double[][] e = essential(f, k1, k2);
        return decompose(e, normalised(k1, x1), normalised(k2, x2));
    }
}
