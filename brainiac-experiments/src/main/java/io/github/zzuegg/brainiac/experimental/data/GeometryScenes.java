package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.experimental.geometry.GeometryLinalg;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Synthetic geometry scenes with exact ground truth for multi-view geometry, registration
 * and SLAM toys:
 *
 * <ul>
 *   <li>{@link #twoView}: a general or near-planar 3-D point set seen by two pinhole cameras,
 *       with pixel noise and a share of outlier correspondences;</li>
 *   <li>{@link #multiView}: 2 to 20 cameras on an arc around a point cloud, with every
 *       point's noisy projection in every camera;</li>
 *   <li>{@link #cloudPair}: a partial, noisy point cloud of an asymmetric shape and a rigidly
 *       moved copy with a known rotation (up to a given angle) and translation;</li>
 *   <li>{@link #robotWorld}: a 2-D world of landmarks, a robot driving a loop with noisy
 *       odometry, and range-bearing measurements of the landmarks within sensor range.</li>
 * </ul>
 *
 * Cameras follow the convention {@code x_cam = R X + t}, pixel {@code K x_cam / z}.
 */
public final class GeometryScenes {

    private GeometryScenes() {
    }

    /**
     * A pinhole camera.
     *
     * @param focal the focal length in pixels
     * @param cx the principal point x
     * @param cy the principal point y
     * @param rotation world-to-camera rotation
     * @param translation world-to-camera translation
     */
    public record Camera(double focal, double cx, double cy, double[][] rotation, double[] translation) {

        /** @return the intrinsic matrix K */
        public double[][] intrinsics() {
            return new double[][] {{focal, 0, cx}, {0, focal, cy}, {0, 0, 1}};
        }

        /**
         * @param x a world point
         * @return the point in camera coordinates
         */
        public double[] toCamera(double[] x) {
            return GeometryLinalg.transform(rotation, translation, x);
        }

        /**
         * @param x a world point
         * @return its pixel, or {@code null} if it is behind the camera
         */
        public double[] project(double[] x) {
            double[] c = toCamera(x);
            if (c[2] <= 1e-9) {
                return null;
            }
            return new double[] {focal * c[0] / c[2] + cx, focal * c[1] / c[2] + cy};
        }

        /** @return the camera centre in world coordinates */
        public double[] centre() {
            double[] c = GeometryLinalg.multiply(GeometryLinalg.transpose(rotation), translation);
            return new double[] {-c[0], -c[1], -c[2]};
        }
    }

    /**
     * Two views of one scene.
     *
     * @param points the 3-D points {@code [n][3]}
     * @param first pixels in the first camera {@code [n][2]}
     * @param second pixels in the second camera
     * @param inlier whether each correspondence is true (false for injected outliers)
     * @param camera1 the first camera, at the origin
     * @param camera2 the second camera
     */
    public record TwoView(double[][] points, double[][] first, double[][] second, boolean[] inlier, Camera camera1,
            Camera camera2) {
    }

    /**
     * Many views of one scene.
     *
     * @param points the 3-D points {@code [n][3]}
     * @param cameras the cameras
     * @param observations pixels {@code [camera][point][2]}, {@code null} where not visible
     */
    public record MultiView(double[][] points, Camera[] cameras, double[][][] observations) {
    }

    /**
     * A pair of point clouds related by a rigid motion.
     *
     * @param source the source cloud {@code [n][3]}
     * @param target the target cloud {@code [m][3]}: a moved, re-sampled, partial copy
     * @param rotation the rotation taking source to target
     * @param translation the translation
     * @param correspondence for each source point its index in {@code target}, or -1 if
     *     cropped away (used by the known-correspondence oracle)
     */
    public record CloudPair(double[][] source, double[][] target, double[][] rotation, double[] translation,
            int[] correspondence) {
    }

    /**
     * A 2-D robot world.
     *
     * @param landmarks landmark positions {@code [k][2]}
     * @param poses the true poses {@code [T][3]} as {@code (x, y, theta)}
     * @param odometry noisy relative motions {@code [T-1][3]}: forward distance, sideways
     *     distance (zero in truth) and heading change, in the robot frame
     * @param measurements per step the measurements {@code (landmark, range, bearing)}
     * @param odometryNoise the odometry standard deviations {@code (forward, sideways, turn)}
     * @param rangeNoise the range standard deviation
     * @param bearingNoise the bearing standard deviation in radians
     * @param maxRange the sensor range
     */
    public record RobotWorld(double[][] landmarks, double[][] poses, double[][] odometry,
            List<List<double[]>> measurements, double[] odometryNoise, double rangeNoise, double bearingNoise,
            double maxRange) {

        /** @return the number of time steps */
        public int steps() {
            return poses.length;
        }
    }

    /**
     * A camera looking from a centre towards a point.
     *
     * @param focal the focal length
     * @param side the image side (the principal point is its centre)
     * @param centre the camera centre
     * @param target the point it looks at
     * @return the camera
     */
    public static Camera lookAt(double focal, double side, double[] centre, double[] target) {
        double[] z = GeometryLinalg.subtract(target, centre);
        double zn = GeometryLinalg.norm(z);
        for (int i = 0; i < 3; i++) {
            z[i] /= zn;
        }
        double[] up = Math.abs(z[1]) > 0.9 ? new double[] {1, 0, 0} : new double[] {0, 1, 0};
        double[] x = GeometryLinalg.cross(up, z);
        double xn = GeometryLinalg.norm(x);
        for (int i = 0; i < 3; i++) {
            x[i] /= xn;
        }
        double[] y = GeometryLinalg.cross(z, x);
        double[][] r = {x, y, z};
        double[] rc = GeometryLinalg.multiply(r, centre);
        return new Camera(focal, side / 2, side / 2, r, new double[] {-rc[0], -rc[1], -rc[2]});
    }

    /**
     * Two views of random points.
     *
     * @param n the number of correspondences
     * @param pixelNoise pixel noise standard deviation
     * @param outliers the share of correspondences replaced by random pixels
     * @param planarity 0 for a box-shaped volume, 1 for a flat plane (between: squashed depth)
     * @param random the random source
     * @return the pair
     */
    public static TwoView twoView(int n, double pixelNoise, double outliers, double planarity,
            SplittableRandom random) {
        double side = 640;
        double focal = 500;
        Camera c1 = new Camera(focal, side / 2, side / 2, GeometryLinalg.identity(3), new double[3]);
        double[] centre2 = {-1.0 - random.nextDouble(), 0.3 * random.nextGaussian(), 0.3 * random.nextGaussian()};
        Camera c2 = lookAt(focal, side, centre2,
            new double[] {0.3 * random.nextGaussian(), 0.3 * random.nextGaussian(), 6 + 0.5 * random.nextGaussian()});
        double[][] pts = new double[n][];
        double[][] a = new double[n][];
        double[][] b = new double[n][];
        boolean[] in = new boolean[n];
        int i = 0;
        double depthSpread = 2.0 * (1 - planarity) + 0.01;
        while (i < n) {
            double[] x = {3 * (random.nextDouble() - 0.5), 3 * (random.nextDouble() - 0.5),
                6 + depthSpread * (random.nextDouble() - 0.5)};
            double[] p1 = c1.project(x);
            double[] p2 = c2.project(x);
            if (p1 == null || p2 == null || !inside(p1, side) || !inside(p2, side)) {
                continue;
            }
            pts[i] = x;
            in[i] = random.nextDouble() >= outliers;
            a[i] = new double[] {p1[0] + pixelNoise * random.nextGaussian(), p1[1] + pixelNoise * random.nextGaussian()};
            b[i] = in[i]
                ? new double[] {p2[0] + pixelNoise * random.nextGaussian(), p2[1] + pixelNoise * random.nextGaussian()}
                : new double[] {side * random.nextDouble(), side * random.nextDouble()};
            i++;
        }
        return new TwoView(pts, a, b, in, c1, c2);
    }

    private static boolean inside(double[] p, double side) {
        return p[0] >= 0 && p[0] < side && p[1] >= 0 && p[1] < side;
    }

    /**
     * Cameras on an arc around a point cloud.
     *
     * @param cameras the number of cameras, 2 to 20
     * @param points the number of points
     * @param pixelNoise the pixel noise
     * @param random the random source
     * @return the scene
     */
    public static MultiView multiView(int cameras, int points, double pixelNoise, SplittableRandom random) {
        double side = 640;
        double[][] pts = new double[points][];
        for (int i = 0; i < points; i++) {
            pts[i] = new double[] {2 * random.nextGaussian() * 0.5, 2 * random.nextGaussian() * 0.5,
                2 * random.nextGaussian() * 0.5};
        }
        Camera[] cams = new Camera[cameras];
        double arc = Math.toRadians(100);
        for (int c = 0; c < cameras; c++) {
            double phi = -arc / 2 + arc * c / Math.max(1, cameras - 1) + 0.05 * random.nextGaussian();
            double radius = 8 + 0.5 * random.nextGaussian();
            double[] centre = {radius * Math.sin(phi), 0.5 * random.nextGaussian(), -radius * Math.cos(phi)};
            double[] target = {0.2 * random.nextGaussian(), 0.2 * random.nextGaussian(), 0.2 * random.nextGaussian()};
            cams[c] = lookAt(500, side, centre, target);
        }
        double[][][] obs = new double[cameras][points][];
        for (int c = 0; c < cameras; c++) {
            for (int i = 0; i < points; i++) {
                double[] p = cams[c].project(pts[i]);
                if (p != null && inside(p, side)) {
                    obs[c][i] = new double[] {p[0] + pixelNoise * random.nextGaussian(),
                        p[1] + pixelNoise * random.nextGaussian()};
                }
            }
        }
        return new MultiView(pts, cams, obs);
    }

    /**
     * Points on the surface of an asymmetric shape: an ellipsoid with a box and a cylinder
     * attached, so every rotation is identifiable.
     *
     * @param n the number of points
     * @param random the random source
     * @return {@code [n][3]}, roughly unit size
     */
    public static double[][] shape(int n, SplittableRandom random) {
        double[][] out = new double[n][];
        for (int i = 0; i < n; i++) {
            double u = random.nextDouble();
            if (u < 0.5) {
                double[] d = {random.nextGaussian(), random.nextGaussian(), random.nextGaussian()};
                double dn = GeometryLinalg.norm(d);
                out[i] = new double[] {0.8 * d[0] / dn, 0.45 * d[1] / dn, 0.3 * d[2] / dn};
            } else if (u < 0.8) {
                int face = random.nextInt(6);
                double a = random.nextDouble() - 0.5;
                double b = random.nextDouble() - 0.5;
                double c = face % 2 == 0 ? -0.5 : 0.5;
                double[] p = face / 2 == 0 ? new double[] {c, a, b} : face / 2 == 1 ? new double[] {a, c, b}
                    : new double[] {a, b, c};
                out[i] = new double[] {0.6 + 0.35 * p[0], 0.3 + 0.35 * p[1], 0.2 + 0.35 * p[2]};
            } else {
                double t = 2 * Math.PI * random.nextDouble();
                double h = random.nextDouble();
                out[i] = new double[] {-0.4 + 0.12 * Math.cos(t), 0.2 + 0.7 * h, 0.12 * Math.sin(t)};
            }
        }
        return out;
    }

    /**
     * A rotation about a random axis by a given angle.
     *
     * @param degrees the angle
     * @param random the random source
     * @return the rotation matrix
     */
    public static double[][] randomRotation(double degrees, SplittableRandom random) {
        double[] axis = {random.nextGaussian(), random.nextGaussian(), random.nextGaussian()};
        double an = GeometryLinalg.norm(axis);
        double a = Math.toRadians(degrees);
        return GeometryLinalg.exp(new double[] {axis[0] / an * a, axis[1] / an * a, axis[2] / an * a});
    }

    /**
     * A source cloud and a moved, noisy, partial copy.
     *
     * @param n the number of source points
     * @param degrees the rotation angle (about a random axis)
     * @param noise the point noise standard deviation
     * @param overlap the share of target points kept (a half-space crop), 1 for full overlap
     * @param random the random source
     * @return the pair
     */
    public static CloudPair cloudPair(int n, double degrees, double noise, double overlap, SplittableRandom random) {
        double[][] src = shape(n, random);
        double[][] r = randomRotation(degrees, random);
        double[] t = {0.3 * random.nextGaussian(), 0.3 * random.nextGaussian(), 0.3 * random.nextGaussian()};
        double[] dir = {random.nextGaussian(), random.nextGaussian(), random.nextGaussian()};
        double[] proj = new double[n];
        for (int i = 0; i < n; i++) {
            proj[i] = GeometryLinalg.dot(src[i], dir);
        }
        double[] sorted = proj.clone();
        java.util.Arrays.sort(sorted);
        double cut = overlap >= 1 ? Double.POSITIVE_INFINITY : sorted[(int) Math.min(n - 1, Math.floor(overlap * n))];
        List<double[]> tgt = new ArrayList<>();
        int[] corr = new int[n];
        for (int i = 0; i < n; i++) {
            if (proj[i] <= cut) {
                double[] q = GeometryLinalg.transform(r, t, src[i]);
                for (int k = 0; k < 3; k++) {
                    q[k] += noise * random.nextGaussian();
                }
                corr[i] = tgt.size();
                tgt.add(q);
            } else {
                corr[i] = -1;
            }
        }
        return new CloudPair(src, tgt.toArray(new double[0][]), r, t, corr);
    }

    /**
     * A robot driving a noisy loop through a field of landmarks.
     *
     * @param landmarks the number of landmarks
     * @param steps the number of time steps
     * @param laps how many times the loop is driven (more than one closes loops)
     * @param odometryNoise relative odometry noise (forward, sideways and turn standard
     *     deviations per step)
     * @param rangeNoise range noise
     * @param bearingNoise bearing noise in radians
     * @param maxRange the sensor range
     * @param random the random source
     * @return the world
     */
    public static RobotWorld robotWorld(int landmarks, int steps, double laps, double[] odometryNoise,
            double rangeNoise, double bearingNoise, double maxRange, SplittableRandom random) {
        double radius = 10;
        double[][] lm = new double[landmarks][];
        for (int i = 0; i < landmarks; i++) {
            double a = 2 * Math.PI * random.nextDouble();
            double rr = radius + 6 * (random.nextDouble() - 0.5) * 2;
            lm[i] = new double[] {rr * Math.cos(a), rr * Math.sin(a)};
        }
        double[][] poses = new double[steps][];
        double omega = 2 * Math.PI * laps / (steps - 1);
        double v = radius * omega;
        poses[0] = new double[] {radius, 0, Math.PI / 2};
        double[][] odo = new double[steps - 1][];
        for (int k = 1; k < steps; k++) {
            double wobble = 0.3 * Math.sin(5 * omega * k);
            double dth = omega * (1 + wobble);
            double dist = v * (1 + 0.5 * wobble);
            double[] p = poses[k - 1];
            double th = p[2] + dth / 2;
            poses[k] = new double[] {p[0] + dist * Math.cos(th), p[1] + dist * Math.sin(th),
                GeometryLinalg.wrap(p[2] + dth)};
            double[] rel = relative(p, poses[k]);
            odo[k - 1] = new double[] {rel[0] + odometryNoise[0] * random.nextGaussian(),
                rel[1] + odometryNoise[1] * random.nextGaussian(), rel[2] + odometryNoise[2] * random.nextGaussian()};
        }
        List<List<double[]>> meas = new ArrayList<>();
        for (int k = 0; k < steps; k++) {
            List<double[]> z = new ArrayList<>();
            for (int i = 0; i < landmarks; i++) {
                double dx = lm[i][0] - poses[k][0];
                double dy = lm[i][1] - poses[k][1];
                double range = Math.hypot(dx, dy);
                if (range > maxRange) {
                    continue;
                }
                double bearing = GeometryLinalg.wrap(Math.atan2(dy, dx) - poses[k][2]);
                z.add(new double[] {i, range + rangeNoise * random.nextGaussian(),
                    GeometryLinalg.wrap(bearing + bearingNoise * random.nextGaussian())});
            }
            meas.add(z);
        }
        return new RobotWorld(lm, poses, odo, meas, odometryNoise.clone(), rangeNoise, bearingNoise, maxRange);
    }

    /**
     * The pose of {@code b} in the frame of {@code a} (SE(2) {@code a^-1 b}).
     *
     * @param a a pose {@code (x, y, theta)}
     * @param b a pose
     * @return the relative pose
     */
    public static double[] relative(double[] a, double[] b) {
        double c = Math.cos(a[2]);
        double s = Math.sin(a[2]);
        double dx = b[0] - a[0];
        double dy = b[1] - a[1];
        return new double[] {c * dx + s * dy, -s * dx + c * dy, GeometryLinalg.wrap(b[2] - a[2])};
    }

    /**
     * Composes a pose with a relative motion (SE(2) {@code a d}).
     *
     * @param a a pose
     * @param d a relative motion in the frame of {@code a}
     * @return the new pose
     */
    public static double[] compose(double[] a, double[] d) {
        double c = Math.cos(a[2]);
        double s = Math.sin(a[2]);
        return new double[] {a[0] + c * d[0] - s * d[1], a[1] + s * d[0] + c * d[1], GeometryLinalg.wrap(a[2] + d[2])};
    }
}
