package io.github.zzuegg.brainiac.experiments.robotics;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.GeometryScenes;
import io.github.zzuegg.brainiac.experimental.geometry.BundleAdjustment;
import io.github.zzuegg.brainiac.experimental.geometry.DeepClosestPoint;
import io.github.zzuegg.brainiac.experimental.geometry.GeometryLinalg;
import io.github.zzuegg.brainiac.experimental.geometry.PointRegistration;
import io.github.zzuegg.brainiac.experimental.geometry.PoseEstimation;
import io.github.zzuegg.brainiac.experimental.geometry.TwoViewGeometry;
import io.github.zzuegg.brainiac.experimental.vision.FeatureMatching;
import io.github.zzuegg.brainiac.experimental.vision.PointNet;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 12 items 12.3.6 to 12.3.8 on {@link GeometryScenes}:
 *
 * <ul>
 *   <li>{@link #twoView}: relative pose by the raw and Hartley-normalised eight-point
 *       algorithm, with and without RANSAC, against a homography fitted by wave 11's
 *       {@link FeatureMatching#ransac} (the wrong model off the plane), by rotation and
 *       translation-direction error and inlier recall;</li>
 *   <li>{@link #pnp} and {@link #bundle}: DLT against EPnP (and EPnP with Gauss-Newton) by
 *       rotation error; bundle adjustment from perturbed cameras and points with the Schur
 *       complement against the dense solve, by reprojection RMSE and time per iteration;</li>
 *   <li>{@link #registration}: point-to-point, point-to-plane and trimmed ICP, Deep Closest
 *       Point, DCP followed by ICP, and a PointNet rotation regressor against the Kabsch oracle
 *       with known correspondences, by success rate (rotation error below 5 degrees) per initial
 *       angle.</li>
 * </ul>
 */
public final class MultiViewGeometry {

    private MultiViewGeometry() {
    }

    /**
     * A two-view row.
     *
     * @param scene the scene setting
     * @param method the method
     * @param rotationError mean rotation error in degrees
     * @param translationError mean translation-direction error in degrees
     * @param inlierRecall the share of true correspondences kept as inliers (NaN without
     *     RANSAC)
     */
    public record TwoViewRow(String scene, String method, double rotationError, double translationError,
            double inlierRecall) {
    }

    /**
     * Two-view geometry under noise, outliers and planarity.
     *
     * @param trials scenes per setting
     * @param seed the seed
     * @return rows
     */
    public static List<TwoViewRow> twoView(int trials, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        String[] scenes = {"general 1px", "general 1px 30% outliers", "near-planar 1px"};
        double[][] settings = {{1, 0, 0}, {1, 0.3, 0}, {1, 0, 0.97}};
        List<TwoViewRow> rows = new ArrayList<>();
        for (int s = 0; s < scenes.length; s++) {
            double[][] acc = new double[5][3];
            for (int t = 0; t < trials; t++) {
                GeometryScenes.TwoView tv = GeometryScenes.twoView(120, settings[s][0], settings[s][1],
                    settings[s][2], r);
                double[][] k = tv.camera1().intrinsics();
                double[][] rTrue = tv.camera2().rotation();
                double[] tTrue = tv.camera2().translation();
                TwoViewGeometry.Pose raw = TwoViewGeometry.relativePose(tv.first(), tv.second(), k, k, false);
                TwoViewGeometry.Pose norm = TwoViewGeometry.relativePose(tv.first(), tv.second(), k, k, true);
                add(acc[0], raw, rTrue, tTrue, Double.NaN);
                add(acc[1], norm, rTrue, tTrue, Double.NaN);
                TwoViewGeometry.Fit fit = TwoViewGeometry.ransac(tv.first(), tv.second(), 3.0, 500, 0.999, seed + t);
                double[][] in1 = select(tv.first(), fit.inliers());
                double[][] in2 = select(tv.second(), fit.inliers());
                TwoViewGeometry.Pose rp = TwoViewGeometry.decompose(TwoViewGeometry.essential(fit.fundamental(), k, k),
                    TwoViewGeometry.normalised(k, in1), TwoViewGeometry.normalised(k, in2));
                add(acc[2], rp, rTrue, tTrue, recall(fit.inliers(), tv.inlier()));
                TwoViewGeometry.Pose rawRansac = TwoViewGeometry.relativePose(in1, in2, k, k, false);
                add(acc[3], rawRansac, rTrue, tTrue, recall(fit.inliers(), tv.inlier()));
                FeatureMatching.Fit h = FeatureMatching.ransac(Arrays.asList(tv.first()), Arrays.asList(tv.second()),
                    3.0, 500, 0.999, seed + t);
                acc[4][2] += recall(h.inliers(), tv.inlier());
            }
            String[] names = {"8-point raw", "8-point normalised", "RANSAC + normalised 8-point",
                "RANSAC inliers + raw 8-point"};
            for (int m = 0; m < 4; m++) {
                rows.add(new TwoViewRow(scenes[s], names[m], acc[m][0] / trials, acc[m][1] / trials,
                    m < 2 ? Double.NaN : acc[m][2] / trials));
            }
            rows.add(new TwoViewRow(scenes[s], "homography RANSAC (wave 11)", Double.NaN, Double.NaN,
                acc[4][2] / trials));
        }
        return rows;
    }

    private static void add(double[] acc, TwoViewGeometry.Pose p, double[][] r, double[] t, double recall) {
        acc[0] += GeometryLinalg.angleDegrees(p.rotation(), r);
        acc[1] += GeometryLinalg.directionDegrees(p.translation(), t, false);
        if (!Double.isNaN(recall)) {
            acc[2] += recall;
        }
    }

    private static double[][] select(double[][] x, boolean[] keep) {
        List<double[]> out = new ArrayList<>();
        for (int i = 0; i < x.length; i++) {
            if (keep[i]) {
                out.add(x[i]);
            }
        }
        return out.toArray(new double[0][]);
    }

    private static double recall(boolean[] found, boolean[] truth) {
        int hit = 0;
        int total = 0;
        for (int i = 0; i < truth.length; i++) {
            if (truth[i]) {
                total++;
                hit += found[i] ? 1 : 0;
            }
        }
        return hit / (double) Math.max(1, total);
    }

    /**
     * A PnP row.
     *
     * @param points the number of correspondences
     * @param method the method
     * @param rotationError mean rotation error in degrees
     * @param translationError mean relative translation error
     */
    public record PnpRow(int points, String method, double rotationError, double translationError) {
    }

    /**
     * DLT against EPnP at several point counts, 1 px noise.
     *
     * @param counts the point counts
     * @param trials trials per count
     * @param seed the seed
     * @return rows
     */
    public static List<PnpRow> pnp(int[] counts, int trials, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<PnpRow> rows = new ArrayList<>();
        for (int n : counts) {
            double[][] acc = new double[3][2];
            for (int t = 0; t < trials; t++) {
                GeometryScenes.MultiView mv = GeometryScenes.multiView(2, 4 * n, 1.0, r);
                GeometryScenes.Camera cam = mv.cameras()[1];
                List<double[]> w = new ArrayList<>();
                List<double[]> im = new ArrayList<>();
                for (int i = 0; i < mv.points().length && w.size() < n; i++) {
                    if (mv.observations()[1][i] != null) {
                        w.add(mv.points()[i]);
                        im.add(mv.observations()[1][i]);
                    }
                }
                double[][] world = w.toArray(new double[0][]);
                double[][] image = TwoViewGeometry.normalised(cam.intrinsics(), im.toArray(new double[0][]));
                PoseEstimation.Pose[] poses = {PoseEstimation.dlt(world, image), PoseEstimation.epnp(world, image),
                    null};
                poses[2] = PoseEstimation.refine(poses[1], world, image, 5);
                for (int m = 0; m < 3; m++) {
                    acc[m][0] += GeometryLinalg.angleDegrees(poses[m].rotation(), cam.rotation());
                    acc[m][1] += GeometryLinalg.norm(GeometryLinalg.subtract(poses[m].translation(),
                        cam.translation())) / GeometryLinalg.norm(cam.translation());
                }
            }
            String[] names = {"DLT", "EPnP", "EPnP + Gauss-Newton"};
            for (int m = 0; m < 3; m++) {
                rows.add(new PnpRow(n, names[m], acc[m][0] / trials, acc[m][1] / trials));
            }
        }
        return rows;
    }

    /**
     * A bundle adjustment row.
     *
     * @param method the solver
     * @param initialRmse the reprojection RMSE before refinement
     * @param finalRmse after
     * @param iterations iterations run
     * @param millisPerIteration mean time per iteration
     * @param pointError mean 3-D point error after refinement
     */
    public record BundleRow(String method, double initialRmse, double finalRmse, int iterations,
            double millisPerIteration, double pointError) {
    }

    /**
     * Bundle adjustment from perturbed cameras and points.
     *
     * @param cameras the number of cameras
     * @param points the number of points
     * @param iterations LM iterations
     * @param seed the seed
     * @return rows: Schur, dense, and no refinement
     */
    public static List<BundleRow> bundle(int cameras, int points, int iterations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        GeometryScenes.MultiView mv = GeometryScenes.multiView(cameras, points, 1.0, r);
        List<BundleAdjustment.Observation> obs = new ArrayList<>();
        for (int c = 0; c < cameras; c++) {
            for (int i = 0; i < points; i++) {
                double[] p = mv.observations()[c][i];
                if (p != null) {
                    obs.add(new BundleAdjustment.Observation(c, i, p[0], p[1]));
                }
            }
        }
        double[][][] rs = new double[cameras][][];
        double[][] ts = new double[cameras][];
        for (int c = 0; c < cameras; c++) {
            GeometryScenes.Camera cam = mv.cameras()[c];
            if (c == 0) {
                rs[c] = cam.rotation();
                ts[c] = cam.translation().clone();
            } else {
                double a = Math.toRadians(1.0);
                rs[c] = GeometryLinalg.multiply(GeometryLinalg.exp(new double[] {a * r.nextGaussian(),
                    a * r.nextGaussian(), a * r.nextGaussian()}), cam.rotation());
                ts[c] = new double[] {cam.translation()[0] + 0.1 * r.nextGaussian(),
                    cam.translation()[1] + 0.1 * r.nextGaussian(), cam.translation()[2] + 0.1 * r.nextGaussian()};
            }
        }
        double[][] ps = new double[points][];
        for (int i = 0; i < points; i++) {
            ps[i] = new double[] {mv.points()[i][0] + 0.1 * r.nextGaussian(), mv.points()[i][1] + 0.1 * r.nextGaussian(),
                mv.points()[i][2] + 0.1 * r.nextGaussian()};
        }
        GeometryScenes.Camera c0 = mv.cameras()[0];
        BundleAdjustment ba = new BundleAdjustment(c0.focal(), c0.cx(), c0.cy(), 3.0);
        List<BundleRow> rows = new ArrayList<>();
        for (BundleAdjustment.Solver s : BundleAdjustment.Solver.values()) {
            BundleAdjustment.Result res = ba.run(rs, ts, ps, obs, iterations, s);
            rows.add(new BundleRow(s.name(), res.rmse()[0], res.rmse()[res.rmse().length - 1], res.rmse().length - 1,
                res.millisPerIteration(), pointError(res.points(), mv.points())));
        }
        rows.add(new BundleRow("no refinement", ba.rmse(rs, ts, ps, obs), ba.rmse(rs, ts, ps, obs), 0, 0,
            pointError(ps, mv.points())));
        return rows;
    }

    private static double pointError(double[][] a, double[][] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += GeometryLinalg.norm(GeometryLinalg.subtract(a[i], b[i]));
        }
        return s / a.length;
    }

    /**
     * A registration row.
     *
     * @param angle the initial rotation angle in degrees
     * @param method the method
     * @param success the share with rotation error below 5 degrees
     * @param rotationError the median rotation error in degrees
     */
    public record RegistrationRow(double angle, String method, double success, double rotationError) {
    }

    /**
     * Registration from increasing initial rotations.
     *
     * @param angles the initial rotation angles
     * @param trials pairs per angle
     * @param points points per cloud
     * @param trainSteps training steps for DCP and the PointNet regressor
     * @param seed the seed
     * @return rows
     */
    public static List<RegistrationRow> registration(double[] angles, int trials, int points, int trainSteps,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        SplittableRandom tr = new SplittableRandom(seed + 1);
        DeepClosestPoint dcp = new DeepClosestPoint(24, 8, seed + 2);
        dcp.fit(() -> {
            GeometryScenes.CloudPair p = GeometryScenes.cloudPair(points, 90 * tr.nextDouble(), 0.01, 1.0, tr);
            return new double[][][] {p.source(), p.target(), p.rotation(), {p.translation()}};
        }, trainSteps, 8, 3e-3f);
        PointNet pointNet = new PointNet(9, 32, false, new SplittableRandom(seed + 3));
        trainPointNet(pointNet, points, trainSteps, seed + 4);
        String[] names = {"Kabsch oracle", "ICP point-to-point", "ICP point-to-plane", "trimmed ICP (90%)", "DCP",
            "DCP + ICP", "PointNet regressor"};
        List<RegistrationRow> rows = new ArrayList<>();
        double[][] id = GeometryLinalg.identity(3);
        for (double angle : angles) {
            double[][] errs = new double[names.length][trials];
            for (int t = 0; t < trials; t++) {
                GeometryScenes.CloudPair p = GeometryScenes.cloudPair(points, angle, 0.01, 1.0, r);
                double[][] src = p.source();
                double[][] tgt = p.target();
                double[] init = centroidShift(src, tgt);
                double[][][] est = new double[names.length][][];
                est[0] = rotationOf(GeometryLinalg.kabsch(src, tgt, null));
                est[1] = PointRegistration.register(src, tgt, PointRegistration.Method.ICP_POINT, id, init, 50, 1)
                    .rotation();
                est[2] = PointRegistration.register(src, tgt, PointRegistration.Method.ICP_PLANE, id, init, 50, 1)
                    .rotation();
                est[3] = PointRegistration.register(src, tgt, PointRegistration.Method.TRIMMED, id, init, 50, 0.9)
                    .rotation();
                double[][] d = dcp.estimate(src, tgt);
                est[4] = rotationOf(d);
                est[5] = PointRegistration.register(src, tgt, PointRegistration.Method.ICP_POINT, est[4], d[3], 50, 1)
                    .rotation();
                est[6] = pointNetRotation(pointNet, tgt);
                for (int m = 0; m < names.length; m++) {
                    errs[m][t] = GeometryLinalg.angleDegrees(est[m], p.rotation());
                }
            }
            for (int m = 0; m < names.length; m++) {
                int ok = 0;
                for (double e : errs[m]) {
                    ok += e < 5 ? 1 : 0;
                }
                double[] sorted = errs[m].clone();
                Arrays.sort(sorted);
                rows.add(new RegistrationRow(angle, names[m], ok / (double) trials, sorted[trials / 2]));
            }
        }
        return rows;
    }

    private static double[] centroidShift(double[][] src, double[][] tgt) {
        double[] a = new double[3];
        for (double[] p : src) {
            for (int c = 0; c < 3; c++) {
                a[c] -= p[c] / src.length;
            }
        }
        for (double[] p : tgt) {
            for (int c = 0; c < 3; c++) {
                a[c] += p[c] / tgt.length;
            }
        }
        return a;
    }

    private static double[][] rotationOf(double[][] packed) {
        return new double[][] {packed[0], packed[1], packed[2]};
    }

    private static float[] centred(double[][] cloud) {
        double[] m = new double[3];
        for (double[] p : cloud) {
            for (int c = 0; c < 3; c++) {
                m[c] += p[c] / cloud.length;
            }
        }
        float[] out = new float[cloud.length * 3];
        for (int i = 0; i < cloud.length; i++) {
            for (int c = 0; c < 3; c++) {
                out[i * 3 + c] = (float) (cloud[i][c] - m[c]);
            }
        }
        return out;
    }

    private static void trainPointNet(PointNet net, int points, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        int b = 16;
        for (int s = 0; s < steps; s++) {
            float[] x = new float[b * points * 3];
            float[] y = new float[b * 9];
            for (int i = 0; i < b; i++) {
                GeometryScenes.CloudPair p = GeometryScenes.cloudPair(points, 90 * r.nextDouble(), 0.01, 1.0, r);
                System.arraycopy(centred(p.target()), 0, x, i * points * 3, points * 3);
                for (int a = 0; a < 3; a++) {
                    for (int c = 0; c < 3; c++) {
                        y[i * 9 + a * 3 + c] = (float) p.rotation()[a][c];
                    }
                }
            }
            Var out = net.forward(Var.of(Tensor.of(Shape.of(b, points, 3), x)));
            out.sub(Var.of(Tensor.of(Shape.of(b, 9), y))).square().mean().backward();
            opt.step(net.parameters());
        }
    }

    private static double[][] pointNetRotation(PointNet net, double[][] target) {
        int n = target.length;
        float[] o = Var.noGrad(() -> net.forward(Var.of(Tensor.of(Shape.of(1, n, 3), centred(target))))).value()
            .toArray();
        double[][] m = new double[3][3];
        for (int a = 0; a < 3; a++) {
            for (int c = 0; c < 3; c++) {
                m[a][c] = o[a * 3 + c];
            }
        }
        return GeometryLinalg.nearestRotation(m);
    }
}
