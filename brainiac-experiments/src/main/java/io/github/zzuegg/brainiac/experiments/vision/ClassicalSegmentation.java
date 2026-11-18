package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.experimental.data.ShapeScenes;
import io.github.zzuegg.brainiac.experimental.vision.GraphCuts;
import io.github.zzuegg.brainiac.experimental.vision.HoughTransform;
import io.github.zzuegg.brainiac.experimental.vision.Superpixels;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * W12 12.3.3-12.3.5: classical segmentation.
 *
 * <ul>
 *   <li>{@link #superpixels}: SLIC, Felzenszwalb-Huttenlocher and the watershed against
 *       k-means on intensity (connected components of the clusters) on 64 x 64
 *       {@link ShapeScenes} with a sinusoidal background texture and pixel noise, scored by
 *       boundary recall (1 px), under-segmentation error and achievable segmentation
 *       accuracy against the glyph instances;</li>
 *   <li>{@link #mrf}: binary denoising of glyph masks (Gaussian noise plus salt and pepper)
 *       by graph cut, ICM and loopy BP against thresholding and a 3 x 3 median, and
 *       four-level piecewise-constant segmentation by alpha-expansion, ICM and BP against the
 *       unary argmin; pixel error, Potts energy and time;</li>
 *   <li>{@link #hough}: lines and circles in synthetic edge-point sets with gaps and clutter,
 *       by the Hough transform (and its probabilistic and gradient variants) against
 *       sequential RANSAC and a single least-squares fit; detection F1 and parameter
 *       error.</li>
 * </ul>
 */
public final class ClassicalSegmentation {

    /**
     * A superpixel method's averages.
     *
     * @param method the method
     * @param segments the mean segment count
     * @param boundaryRecall the boundary recall
     * @param undersegmentation the under-segmentation error
     * @param asa the achievable segmentation accuracy
     */
    public record SuperpixelRow(String method, double segments, double boundaryRecall, double undersegmentation,
            double asa) {
    }

    /**
     * An MRF method's averages.
     *
     * @param task the task
     * @param method the method
     * @param pixelError the share of wrong pixels
     * @param energy the mean Potts energy
     * @param millis the mean time per image
     */
    public record MrfRow(String task, String method, double pixelError, double energy, double millis) {
    }

    /**
     * A primitive detector's averages.
     *
     * @param primitive lines or circles
     * @param count primitives per image
     * @param clutter the clutter share of the points
     * @param method the method
     * @param f1 the detection F1
     * @param parameterError the mean offset (pixels) of matched detections
     */
    public record HoughRow(String primitive, int count, double clutter, String method, double f1,
            double parameterError) {
    }

    private ClassicalSegmentation() {
    }

    private static int[] kmeansIntensity(float[] image, int k, int iterations) {
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float v : image) {
            min = Math.min(min, v);
            max = Math.max(max, v);
        }
        double[] c = new double[k];
        for (int i = 0; i < k; i++) {
            c[i] = min + (max - min) * (i + 0.5) / k;
        }
        int[] a = new int[image.length];
        for (int it = 0; it < iterations; it++) {
            double[] sum = new double[k];
            int[] n = new int[k];
            for (int p = 0; p < image.length; p++) {
                int best = 0;
                for (int j = 1; j < k; j++) {
                    if (Math.abs(image[p] - c[j]) < Math.abs(image[p] - c[best])) {
                        best = j;
                    }
                }
                a[p] = best;
                sum[best] += image[p];
                n[best]++;
            }
            for (int j = 0; j < k; j++) {
                if (n[j] > 0) {
                    c[j] = sum[j] / n[j];
                }
            }
        }
        return a;
    }

    /**
     * The superpixel comparison.
     *
     * @param images how many scenes
     * @param segments the target segment count
     * @param seed the seed
     * @return SLIC, FH, watershed, k-means (4 clusters), k-means (8 clusters)
     */
    public static List<SuperpixelRow> superpixels(int images, int segments, long seed) {
        int side = 64;
        ShapeScenes gen = new ShapeScenes(side, 0.15f);
        ShapeScenes.Scene[] scenes = gen.scenes(images, 4, 8, ShapeScenes.Layout.SPARSE, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        String[] names = {"SLIC", "Felzenszwalb k=0.15", "watershed", "k-means k=4 (components)",
            "k-means k=8 (components)"};
        double[][] sums = new double[names.length][4];
        for (ShapeScenes.Scene s : scenes) {
            float[] img = s.image().clone();
            double fx = r.nextDouble(0.2, 0.6);
            double fy = r.nextDouble(0.2, 0.6);
            for (int p = 0; p < img.length; p++) {
                if (s.instances()[p] == 0) {
                    img[p] += (float) (0.3 * Math.sin(fx * (p % side)) * Math.sin(fy * (p / side)));
                }
            }
            int[][] labels = {
                Superpixels.segment(Superpixels.Method.SLIC, img, side, side, segments, 0.2),
                Superpixels.segment(Superpixels.Method.FELZENSZWALB, img, side, side, segments, 0.15),
                Superpixels.segment(Superpixels.Method.WATERSHED, img, side, side, segments, 0),
                Superpixels.components(kmeansIntensity(img, 4, 10), side, side),
                Superpixels.components(kmeansIntensity(img, 8, 10), side, side)};
            for (int m = 0; m < names.length; m++) {
                sums[m][0] += Superpixels.count(labels[m]);
                sums[m][1] += Superpixels.boundaryRecall(labels[m], s.instances(), side, side, 1);
                sums[m][2] += Superpixels.undersegmentationError(labels[m], s.instances());
                sums[m][3] += Superpixels.achievableAccuracy(labels[m], s.instances());
            }
        }
        List<SuperpixelRow> rows = new ArrayList<>();
        for (int m = 0; m < names.length; m++) {
            rows.add(new SuperpixelRow(names[m], sums[m][0] / images, sums[m][1] / images, sums[m][2] / images,
                sums[m][3] / images));
        }
        return rows;
    }

    private static int[] median(int[] x, int side) {
        int[] out = new int[x.length];
        for (int y = 0; y < side; y++) {
            for (int c = 0; c < side; c++) {
                int ones = 0;
                int n = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int yy = y + dy;
                        int xx = c + dx;
                        if (yy >= 0 && xx >= 0 && yy < side && xx < side) {
                            ones += x[yy * side + xx];
                            n++;
                        }
                    }
                }
                out[y * side + c] = 2 * ones > n ? 1 : 0;
            }
        }
        return out;
    }

    /**
     * The MRF comparison.
     *
     * @param images how many images per task
     * @param sigma the Gaussian noise of the binary task
     * @param seed the seed
     * @return binary rows (threshold, median, ICM, BP, graph cut) then four-level rows (argmin,
     *     ICM, BP, alpha-expansion)
     */
    public static List<MrfRow> mrf(int images, double sigma, long seed) {
        int side = 48;
        ShapeScenes gen = new ShapeScenes(side, 0f);
        ShapeScenes.Scene[] scenes = gen.scenes(images, 3, 6, ShapeScenes.Layout.SPARSE, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        String[] binary = {"threshold", "median 3x3", "ICM", "loopy BP", "graph cut"};
        String[] multi = {"unary argmin", "ICM", "loopy BP", "alpha-expansion"};
        double[][] bs = new double[binary.length][3];
        double[][] ms = new double[multi.length][3];
        double lambda = 1.0;
        int n = side * side;
        for (ShapeScenes.Scene s : scenes) {
            int[] truth = new int[n];
            for (int p = 0; p < n; p++) {
                truth[p] = s.instances()[p] > 0 ? 1 : 0;
            }
            double[][] unary = new double[2][n];
            for (int p = 0; p < n; p++) {
                double v = truth[p] + sigma * r.nextGaussian();
                if (r.nextDouble() < 0.05) {
                    v = r.nextBoolean() ? 1 : 0;
                }
                unary[0][p] = v * v / (2 * sigma * sigma);
                unary[1][p] = (v - 1) * (v - 1) / (2 * sigma * sigma);
            }
            for (int m = 0; m < binary.length; m++) {
                long t0 = System.nanoTime();
                int[] x = switch (m) {
                    case 0 -> GraphCuts.argmin(unary);
                    case 1 -> median(GraphCuts.argmin(unary), side);
                    case 2 -> GraphCuts.icm(unary, side, side, lambda, 20);
                    case 3 -> GraphCuts.beliefPropagation(unary, side, side, lambda, 30, 0.5);
                    default -> GraphCuts.binary(unary[0], unary[1], side, side, lambda);
                };
                bs[m][2] += (System.nanoTime() - t0) / 1e6;
                bs[m][0] += error(x, truth);
                bs[m][1] += GraphCuts.energy(unary, x, side, side, lambda);
            }
            int[] levels = new int[n];
            for (int p = 0; p < n; p++) {
                levels[p] = s.instances()[p] == 0 ? 0 : 1 + s.classes()[s.instances()[p] - 1] % 3;
            }
            double ms2 = 0.3;
            double[][] u4 = new double[4][n];
            for (int p = 0; p < n; p++) {
                double v = levels[p] / 3.0 + ms2 * r.nextGaussian();
                for (int l = 0; l < 4; l++) {
                    u4[l][p] = (v - l / 3.0) * (v - l / 3.0) / (2 * ms2 * ms2);
                }
            }
            for (int m = 0; m < multi.length; m++) {
                long t0 = System.nanoTime();
                int[] x = switch (m) {
                    case 0 -> GraphCuts.argmin(u4);
                    case 1 -> GraphCuts.icm(u4, side, side, lambda, 20);
                    case 2 -> GraphCuts.beliefPropagation(u4, side, side, lambda, 30, 0.5);
                    default -> GraphCuts.alphaExpansion(u4, side, side, lambda, 5);
                };
                ms[m][2] += (System.nanoTime() - t0) / 1e6;
                ms[m][0] += error(x, levels);
                ms[m][1] += GraphCuts.energy(u4, x, side, side, lambda);
            }
        }
        List<MrfRow> rows = new ArrayList<>();
        for (int m = 0; m < binary.length; m++) {
            rows.add(new MrfRow("binary", binary[m], bs[m][0] / images, bs[m][1] / images, bs[m][2] / images));
        }
        for (int m = 0; m < multi.length; m++) {
            rows.add(new MrfRow("4-level", multi[m], ms[m][0] / images, ms[m][1] / images, ms[m][2] / images));
        }
        return rows;
    }

    private static double error(int[] x, int[] truth) {
        int wrong = 0;
        for (int p = 0; p < x.length; p++) {
            wrong += x[p] != truth[p] ? 1 : 0;
        }
        return wrong / (double) x.length;
    }

    /** One synthetic edge set: points, per point a gradient angle, and the true primitives. */
    private record EdgeSet(double[][] points, double[] gradients, List<double[]> truth) {
    }

    private static EdgeSet edges(boolean circles, int count, double clutter, int side, SplittableRandom r) {
        List<double[]> pts = new ArrayList<>();
        List<Double> grads = new ArrayList<>();
        List<double[]> truth = new ArrayList<>();
        for (int k = 0; k < count; k++) {
            if (circles) {
                double cx = r.nextDouble(14, side - 14);
                double cy = r.nextDouble(14, side - 14);
                double rad = r.nextDouble(6, 13);
                truth.add(new double[] {cx, cy, rad});
                int steps = (int) Math.round(2 * Math.PI * rad);
                double gapStart = r.nextDouble(2 * Math.PI);
                for (int s = 0; s < steps; s++) {
                    double a = 2 * Math.PI * s / steps;
                    double rel = (a - gapStart) % (2 * Math.PI);
                    rel = rel < 0 ? rel + 2 * Math.PI : rel;
                    if (rel < 1.2 || r.nextDouble() < 0.1) {
                        continue;
                    }
                    pts.add(new double[] {cx + rad * Math.cos(a) + r.nextDouble(-0.4, 0.4),
                        cy + rad * Math.sin(a) + r.nextDouble(-0.4, 0.4)});
                    grads.add(a + 0.1 * r.nextGaussian());
                }
            } else {
                double th = r.nextDouble(Math.PI);
                double rho = (side / 2.0) * (Math.cos(th) + Math.sin(th)) + r.nextDouble(-20, 20);
                truth.add(new double[] {rho, th});
                double c = Math.cos(th);
                double s = Math.sin(th);
                double gap0 = r.nextDouble(-40, 40);
                for (double t = -2 * side; t <= 2 * side; t += 1) {
                    double x = rho * c - t * s;
                    double y = rho * s + t * c;
                    if (x < 0 || y < 0 || x >= side || y >= side || Math.abs(t - gap0) < 6 || r.nextDouble() < 0.15) {
                        continue;
                    }
                    double j = r.nextDouble(-0.4, 0.4);
                    pts.add(new double[] {x + j * c, y + j * s});
                    grads.add(th + 0.1 * r.nextGaussian());
                }
            }
        }
        int signal = pts.size();
        int noise = (int) Math.round(clutter / (1 - clutter) * signal);
        for (int i = 0; i < noise; i++) {
            pts.add(new double[] {r.nextDouble(side), r.nextDouble(side)});
            grads.add(r.nextDouble(2 * Math.PI));
        }
        double[] g = new double[grads.size()];
        for (int i = 0; i < g.length; i++) {
            g[i] = grads.get(i);
        }
        return new EdgeSet(pts.toArray(new double[0][]), g, truth);
    }

    private static double lineDistance(double[] a, double[] b) {
        double dth = Math.abs(a[1] - b[1]);
        double d1 = dth < Math.toRadians(3) ? Math.abs(a[0] - b[0]) : Double.POSITIVE_INFINITY;
        double dth2 = Math.abs(Math.abs(a[1] - b[1]) - Math.PI);
        double d2 = dth2 < Math.toRadians(3) ? Math.abs(a[0] + b[0]) : Double.POSITIVE_INFINITY;
        return Math.min(d1, d2);
    }

    private static double circleDistance(double[] a, double[] b) {
        double dc = Math.hypot(a[0] - b[0], a[1] - b[1]);
        return Math.abs(a[2] - b[2]) < 2 ? dc : Double.POSITIVE_INFINITY;
    }

    /** @return {tp, detections, truth, sum of matched errors} */
    private static double[] match(List<double[]> found, List<double[]> truth, boolean circles) {
        boolean[] used = new boolean[truth.size()];
        double tp = 0;
        double err = 0;
        for (double[] f : found) {
            int best = -1;
            double bestD = circles ? 2.5 : 3;
            for (int i = 0; i < truth.size(); i++) {
                double d = circles ? circleDistance(f, truth.get(i)) : lineDistance(f, truth.get(i));
                if (!used[i] && d < bestD) {
                    bestD = d;
                    best = i;
                }
            }
            if (best >= 0) {
                used[best] = true;
                tp++;
                err += bestD;
            }
        }
        return new double[] {tp, found.size(), truth.size(), err};
    }

    private static double[] fitLine(List<double[]> pts) {
        double mx = 0;
        double my = 0;
        for (double[] p : pts) {
            mx += p[0];
            my += p[1];
        }
        mx /= pts.size();
        my /= pts.size();
        double sxx = 0;
        double sxy = 0;
        double syy = 0;
        for (double[] p : pts) {
            sxx += (p[0] - mx) * (p[0] - mx);
            sxy += (p[0] - mx) * (p[1] - my);
            syy += (p[1] - my) * (p[1] - my);
        }
        double angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
        double th = angle + Math.PI / 2;
        double rho = mx * Math.cos(th) + my * Math.sin(th);
        if (th >= Math.PI) {
            th -= Math.PI;
            rho = -rho;
        }
        if (th < 0) {
            th += Math.PI;
            rho = -rho;
        }
        return new double[] {rho, th};
    }

    private static double[] fitCircle(List<double[]> pts) {
        double[][] a = new double[3][3];
        double[] b = new double[3];
        for (double[] p : pts) {
            double[] row = {p[0], p[1], 1};
            double z = p[0] * p[0] + p[1] * p[1];
            for (int i = 0; i < 3; i++) {
                b[i] += row[i] * z;
                for (int j = 0; j < 3; j++) {
                    a[i][j] += row[i] * row[j];
                }
            }
        }
        double[] s = io.github.zzuegg.brainiac.experimental.geometry.GeometryLinalg.solve(a, b);
        if (s == null) {
            return new double[] {0, 0, 0};
        }
        double cx = s[0] / 2;
        double cy = s[1] / 2;
        return new double[] {cx, cy, Math.sqrt(Math.max(0, s[2] + cx * cx + cy * cy))};
    }

    private static double[] circleFrom3(double[] p, double[] q, double[] t) {
        return fitCircle(List.of(p, q, t));
    }

    /**
     * Sequential RANSAC: fit the best-supported primitive, refit it on its inliers by least
     * squares, remove them and repeat.
     */
    private static List<double[]> sequentialRansac(double[][] points, boolean circles, int maxModels, int minInliers,
            int iterations, SplittableRandom r) {
        List<double[]> left = new ArrayList<>(Arrays.asList(points));
        List<double[]> out = new ArrayList<>();
        while (out.size() < maxModels && left.size() >= minInliers) {
            double[] best = null;
            int bestCount = 0;
            for (int it = 0; it < iterations; it++) {
                double[] model;
                if (circles) {
                    model = circleFrom3(left.get(r.nextInt(left.size())), left.get(r.nextInt(left.size())),
                        left.get(r.nextInt(left.size())));
                    if (model[2] < 5 || model[2] > 15) {
                        continue;
                    }
                } else {
                    double[] a = left.get(r.nextInt(left.size()));
                    double[] b = left.get(r.nextInt(left.size()));
                    if (a == b) {
                        continue;
                    }
                    model = fitLine(List.of(a, b));
                }
                int c = 0;
                for (double[] p : left) {
                    c += residual(model, p, circles) < 1.0 ? 1 : 0;
                }
                if (c > bestCount) {
                    bestCount = c;
                    best = model;
                }
            }
            if (best == null || bestCount < minInliers) {
                break;
            }
            List<double[]> inliers = new ArrayList<>();
            List<double[]> rest = new ArrayList<>();
            for (double[] p : left) {
                (residual(best, p, circles) < 1.0 ? inliers : rest).add(p);
            }
            out.add(circles ? fitCircle(inliers) : fitLine(inliers));
            left = rest;
        }
        return out;
    }

    private static double residual(double[] m, double[] p, boolean circles) {
        if (circles) {
            return Math.abs(Math.hypot(p[0] - m[0], p[1] - m[1]) - m[2]);
        }
        return Math.abs(p[0] * Math.cos(m[1]) + p[1] * Math.sin(m[1]) - m[0]);
    }

    /**
     * The primitive-detection comparison.
     *
     * @param trials edge sets per condition
     * @param seed the seed
     * @return per primitive and condition: Hough, probabilistic Hough (lines) or gradient Hough
     *     (circles), sequential RANSAC and one least-squares fit
     */
    public static List<HoughRow> hough(int trials, long seed) {
        int side = 64;
        int[][] conditions = {{1, 10}, {1, 50}, {4, 40}, {6, 60}};
        SplittableRandom r = new SplittableRandom(seed);
        List<HoughRow> rows = new ArrayList<>();
        for (boolean circles : new boolean[] {false, true}) {
            String[] names = circles
                ? new String[] {"Hough", "Hough + gradient", "sequential RANSAC", "least squares"}
                : new String[] {"Hough", "probabilistic Hough (30%)", "sequential RANSAC", "least squares"};
            for (int[] c : conditions) {
                double clutter = c[1] / 100.0;
                double[][] sums = new double[names.length][4];
                for (int t = 0; t < trials; t++) {
                    EdgeSet e = edges(circles, c[0], clutter, side, r);
                    for (int m = 0; m < names.length; m++) {
                        List<double[]> found;
                        if (m == 0) {
                            found = circles
                                ? HoughTransform.circles(e.points(), null, side, side, 5, 15, 3.5, 8, 3)
                                : HoughTransform.lines(e.points(), side, side, 180, 20, 8, 4);
                        } else if (m == 1) {
                            found = circles
                                ? HoughTransform.circles(e.points(), e.gradients(), side, side, 5, 15, 3.0, 8, 3)
                                : HoughTransform.probabilisticLines(e.points(), side, side, 180, 20, 8, 4, 0.3, r);
                        } else if (m == 2) {
                            found = sequentialRansac(e.points(), circles, 8, circles ? 25 : 20, 400, r);
                        } else {
                            found = List.of(circles ? fitCircle(Arrays.asList(e.points()))
                                : fitLine(Arrays.asList(e.points())));
                        }
                        double[] mt = match(found, e.truth(), circles);
                        for (int k = 0; k < 4; k++) {
                            sums[m][k] += mt[k];
                        }
                    }
                }
                for (int m = 0; m < names.length; m++) {
                    double tp = sums[m][0];
                    double f1 = tp == 0 ? 0 : 2 * tp / (sums[m][1] + sums[m][2]);
                    rows.add(new HoughRow(circles ? "circles" : "lines", c[0], clutter, names[m], f1,
                        tp == 0 ? Double.NaN : sums[m][3] / tp));
                }
            }
        }
        return rows;
    }
}
