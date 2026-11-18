package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.experimental.vision.ClassicalFlow;
import io.github.zzuegg.brainiac.experimental.vision.Conv2dOps;
import io.github.zzuegg.brainiac.experimental.vision.LearnedKeypoints;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 11 item 11.5.14: homography estimation between a synthetic scene (flat-shaded rectangles
 * and triangles in a 3 x 3 layout over a smooth textured background, 48 px) and its warp by a
 * random homography with blur, gain and noise. Harris corners with patch or BRIEF descriptors,
 * and learned SuperPoint-style keypoints, all with RANSAC; least squares on all matches and
 * pyramidal Lucas-Kanade tracking (then RANSAC) as baselines. Accuracy (mean corner error below
 * 3 px), matching precision (matches within 3 px of the true mapping) and RANSAC inlier ratio.
 */
public final class FeatureMatching {

    private FeatureMatching() {
    }

    /**
     * A row.
     *
     * @param condition the warp setting
     * @param method the method
     * @param accuracy share of pairs with corner error below 3 px
     * @param precision matching precision
     * @param inliers RANSAC inlier ratio ({@code NaN} without RANSAC)
     */
    public record Row(String condition, String method, double accuracy, double precision, double inliers) {
    }

    private static final int S = 48;

    /** A scene. */
    private record Scene(float[] image, List<double[]> corners) {
    }

    private static Scene scene(SplittableRandom r) {
        float[] img = new float[S * S];
        double a = r.nextDouble() * Math.PI;
        for (int y = 0; y < S; y++) {
            for (int x = 0; x < S; x++) {
                img[y * S + x] = (float) (0.35 + 0.08 * Math.sin(0.25 * (x * Math.cos(a) + y * Math.sin(a))));
            }
        }
        List<double[]> corners = new ArrayList<>();
        List<Integer> cells = new ArrayList<>(List.of(0, 1, 2, 3, 4, 5, 6, 7, 8));
        java.util.Collections.shuffle(cells, new java.util.Random(r.nextLong()));
        int shapes = 5 + r.nextInt(3);
        for (int k = 0; k < shapes; k++) {
            int cx = (cells.get(k) % 3) * 16;
            int cy = (cells.get(k) / 3) * 16;
            float v = (float) (r.nextBoolean() ? 0.75 + 0.25 * r.nextDouble() : 0.05 * r.nextDouble());
            if (r.nextBoolean()) {
                int x0 = cx + 2 + r.nextInt(4);
                int y0 = cy + 2 + r.nextInt(4);
                int x1 = cx + 9 + r.nextInt(5);
                int y1 = cy + 9 + r.nextInt(5);
                for (int y = y0; y <= y1; y++) {
                    for (int x = x0; x <= x1; x++) {
                        img[y * S + x] = v;
                    }
                }
                corners.add(new double[] {x0, y0});
                corners.add(new double[] {x1, y0});
                corners.add(new double[] {x0, y1});
                corners.add(new double[] {x1, y1});
            } else {
                double[][] t = new double[3][];
                for (int q = 0; q < 3; q++) {
                    t[q] = new double[] {cx + 2 + r.nextInt(12), cy + 2 + r.nextInt(12)};
                }
                double area = (t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (t[1][1] - t[0][1]);
                if (Math.abs(area) < 30) {
                    k--;
                    continue;
                }
                for (int y = cy; y < cy + 16; y++) {
                    for (int x = cx; x < cx + 16; x++) {
                        if (inside(t, x, y)) {
                            img[y * S + x] = v;
                        }
                    }
                }
                for (double[] p : t) {
                    corners.add(p);
                }
            }
        }
        return new Scene(img, corners);
    }

    private static boolean inside(double[][] t, double x, double y) {
        double d1 = (x - t[1][0]) * (t[0][1] - t[1][1]) - (t[0][0] - t[1][0]) * (y - t[1][1]);
        double d2 = (x - t[2][0]) * (t[1][1] - t[2][1]) - (t[1][0] - t[2][0]) * (y - t[2][1]);
        double d3 = (x - t[0][0]) * (t[2][1] - t[0][1]) - (t[2][0] - t[0][0]) * (y - t[0][1]);
        boolean neg = d1 < 0 || d2 < 0 || d3 < 0;
        boolean pos = d1 > 0 || d2 > 0 || d3 > 0;
        return !(neg && pos);
    }

    private static float[] degrade(float[] img, double blur, double gain, SplittableRandom r) {
        float[] y = blur > 0 ? Conv2dOps.gaussianBlur(img, S, S, blur) : img.clone();
        for (int i = 0; i < y.length; i++) {
            y[i] = (float) (y[i] * gain + (1 - gain) * 0.4 + 0.03 * r.nextGaussian());
        }
        return y;
    }

    /** Per method accumulators: accuracy, precision, inlier ratio. */
    private static final class Tally {
        double ok;
        double precision;
        double inliers;
        int n;
        int fits;

        void add(double[] est, double[] truth, List<double[]> src, List<double[]> dst,
                io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.Fit fit) {
            n++;
            ok += io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.cornerError(est, truth, S, S) < 3 ? 1 : 0;
            int good = 0;
            for (int k = 0; k < src.size(); k++) {
                double[] p = io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.apply(truth, src.get(k));
                good += Math.hypot(p[0] - dst.get(k)[0], p[1] - dst.get(k)[1]) < 3 ? 1 : 0;
            }
            precision += src.isEmpty() ? 0 : good / (double) src.size();
            if (fit != null) {
                inliers += src.isEmpty() ? 0 : fit.count() / (double) src.size();
                fits++;
            }
        }

        Row row(String cond, String name) {
            return new Row(cond, name, ok / n, precision / n, fits == 0 ? Double.NaN : inliers / fits);
        }
    }

    /**
     * Runs the comparison.
     *
     * @param trainScenes scenes for the learned keypoints
     * @param pairs test pairs per condition
     * @param steps learned-keypoint training steps
     * @param seed the seed
     * @return rows
     */
    public static List<Row> run(int trainScenes, int pairs, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> imgs = new ArrayList<>();
        List<List<double[]>> corners = new ArrayList<>();
        for (int i = 0; i < trainScenes; i++) {
            Scene sc = scene(r);
            imgs.add(sc.image());
            corners.add(sc.corners());
        }
        LearnedKeypoints learned = new LearnedKeypoints(16, 16, seed);
        learned.train(imgs, corners, S, S, steps, 8, 3e-3f, seed);
        String[] conds = {"small warp", "large warp", "blur + illumination"};
        double[][] settings = {{4, 0, 1}, {12, 0, 1}, {6, 1.2, 0.6}};
        String[] names = {"Harris + patch + RANSAC", "Harris + BRIEF + RANSAC", "Harris + patch + least squares",
            "learned keypoints + RANSAC", "Lucas-Kanade + RANSAC"};
        List<Row> rows = new ArrayList<>();
        for (int c = 0; c < conds.length; c++) {
            Tally[] t = new Tally[names.length];
            for (int m = 0; m < names.length; m++) {
                t[m] = new Tally();
            }
            SplittableRandom tr = new SplittableRandom(seed + 100 * (c + 1));
            for (int p = 0; p < pairs; p++) {
                Scene sc = scene(tr);
                double[] hm = LearnedKeypoints.randomHomography(S, S, settings[c][0], tr);
                float[] a = degrade(sc.image(), 0, 1, tr);
                float[] b = degrade(io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.warp(sc.image(), S, S,
                        hm, 0.35f), settings[c][1], settings[c][2], tr);
                evaluate(a, b, hm, learned, t, p + 31L * c + seed);
            }
            for (int m = 0; m < names.length; m++) {
                rows.add(t[m].row(conds[c], names[m]));
            }
        }
        return rows;
    }

    private static void evaluate(float[] a, float[] b, double[] truth, LearnedKeypoints learned, Tally[] t,
            long seed) {
        int count = 60;
        List<double[]> ka = io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.keypoints(
                io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.harris(a, S, S, 1.0, 0.05), S, S,
                count, 2, 4);
        List<double[]> kb = io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.keypoints(
                io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.harris(b, S, S, 1.0, 0.05), S, S,
                count, 2, 4);
        float[][] pa = io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.patches(a, S, S, ka, 4);
        float[][] pb = io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.patches(b, S, S, kb, 4);
        List<int[]> mp = io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.match(pa, pb, 0.9);
        fit(t[0], ka, kb, mp, truth, true, seed);
        fit(t[2], ka, kb, mp, truth, false, seed);
        long[][] ba = io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.brief(a, S, S, ka, 6, 128, 7);
        long[][] bb = io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.brief(b, S, S, kb, 6, 128, 7);
        fit(t[1], ka, kb, io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.matchBinary(ba, bb, 0.9),
                truth, true, seed);
        LearnedKeypoints.Detection da = learned.detect(a, S, S, count, 2);
        LearnedKeypoints.Detection db = learned.detect(b, S, S, count, 2);
        fit(t[3], da.points(), db.points(),
                io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.match(da.descriptors(), db.descriptors(),
                        0.9), truth, true, seed);
        float[] flow = ClassicalFlow.lucasKanade(a, b, S, S, 3, 3, 3);
        List<double[]> src = new ArrayList<>();
        List<double[]> dst = new ArrayList<>();
        for (double[] p : ka) {
            int i = (int) p[1] * S + (int) p[0];
            src.add(p);
            dst.add(new double[] {p[0] + flow[i], p[1] + flow[S * S + i]});
        }
        io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.Fit f =
                io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.ransac(src, dst, 2.5, 500, 0.99, seed);
        t[4].add(f.homography(), truth, src, dst, f);
    }

    private static void fit(Tally t, List<double[]> ka, List<double[]> kb, List<int[]> matches, double[] truth,
            boolean ransac, long seed) {
        List<double[]> src = new ArrayList<>();
        List<double[]> dst = new ArrayList<>();
        for (int[] m : matches) {
            src.add(ka.get(m[0]));
            dst.add(kb.get(m[1]));
        }
        if (ransac) {
            io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.Fit f =
                    io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.ransac(src, dst, 2.5, 500, 0.99,
                            seed);
            t.add(f.homography(), truth, src, dst, f);
        } else {
            double[] h = src.size() >= 4 ? io.github.zzuegg.brainiac.experimental.vision.FeatureMatching.dlt(src, dst)
                    : null;
            t.add(h, truth, src, dst, null);
        }
    }
}
