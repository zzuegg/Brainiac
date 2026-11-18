package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.experimental.vision.EdgeDetectors;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 11 item 11.5.15: boundary detection on synthetic scenes of two or three flat-shaded
 * shapes (rectangles and discs) over a textured background (random sinusoidal stripes that
 * create false edges) with pixel noise; true boundaries come from the renderer's label map.
 * Sobel and Laplacian-of-Gaussian magnitudes and Canny (swept thresholds) against HED and a
 * single-output FCN of the same architecture, by ODS, OIS and AP with one-pixel tolerance.
 */
public final class EdgeDetection {

    private EdgeDetection() {
    }

    /**
     * A row.
     *
     * @param method the method
     * @param scores ODS, OIS and AP
     */
    public record Row(String method, EdgeDetectors.Scores scores) {
    }

    /** A rendered scene. */
    private record Scene(float[] image, boolean[] edges) {
    }

    private static Scene scene(int s, double texture, SplittableRandom r) {
        int[] label = new int[s * s];
        float[] img = new float[s * s];
        double a = r.nextDouble() * Math.PI;
        double fq = 0.6 + 0.8 * r.nextDouble();
        double a2 = r.nextDouble() * Math.PI;
        for (int y = 0; y < s; y++) {
            for (int x = 0; x < s; x++) {
                img[y * s + x] = (float) (0.35 + texture * Math.sin(fq * (x * Math.cos(a) + y * Math.sin(a)))
                        + 0.5 * texture * Math.sin(1.7 * fq * (x * Math.cos(a2) + y * Math.sin(a2))));
            }
        }
        int shapes = 2 + r.nextInt(2);
        for (int k = 1; k <= shapes; k++) {
            float v = (float) (r.nextBoolean() ? 0.85 + 0.15 * r.nextDouble() : -0.1 + 0.15 * r.nextDouble());
            boolean disc = r.nextBoolean();
            int cx = 6 + r.nextInt(s - 12);
            int cy = 6 + r.nextInt(s - 12);
            int rx = 3 + r.nextInt(5);
            int ry = 3 + r.nextInt(5);
            for (int y = 0; y < s; y++) {
                for (int x = 0; x < s; x++) {
                    double dx = (x - cx) / (double) rx;
                    double dy = (y - cy) / (double) ry;
                    boolean in = disc ? dx * dx + dy * dy <= 1 : Math.abs(dx) <= 1 && Math.abs(dy) <= 1;
                    if (in) {
                        label[y * s + x] = k;
                        img[y * s + x] = v;
                    }
                }
            }
        }
        boolean[] e = new boolean[s * s];
        for (int y = 0; y < s; y++) {
            for (int x = 0; x < s; x++) {
                int l = label[y * s + x];
                if ((x + 1 < s && label[y * s + x + 1] != l) || (y + 1 < s && label[(y + 1) * s + x] != l)) {
                    e[y * s + x] = true;
                }
            }
        }
        for (int i = 0; i < img.length; i++) {
            img[i] += (float) (0.05 * r.nextGaussian());
        }
        return new Scene(img, e);
    }

    /**
     * Runs the comparison.
     *
     * @param train training scenes
     * @param test test scenes
     * @param steps network training steps
     * @param texture the background texture amplitude
     * @param seed the seed
     * @return rows
     */
    public static List<Row> run(int train, int test, int steps, double texture, long seed) {
        int s = 32;
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> trX = new ArrayList<>();
        List<boolean[]> trY = new ArrayList<>();
        for (int i = 0; i < train; i++) {
            Scene sc = scene(s, texture, r);
            trX.add(sc.image());
            trY.add(sc.edges());
        }
        List<float[]> teX = new ArrayList<>();
        List<boolean[]> teY = new ArrayList<>();
        for (int i = 0; i < test; i++) {
            Scene sc = scene(s, texture, r);
            teX.add(sc.image());
            teY.add(sc.edges());
        }
        int thresholds = 20;
        int radius = 1;
        List<Row> rows = new ArrayList<>();
        List<float[]> sobel = new ArrayList<>();
        List<float[]> log = new ArrayList<>();
        for (float[] x : teX) {
            sobel.add(EdgeDetectors.sobel(x, s, s, 1.0));
            log.add(EdgeDetectors.laplacian(x, s, s, 1.0));
        }
        rows.add(new Row("Sobel", EdgeDetectors.evaluateScores(sobel, teY, s, s, thresholds, radius)));
        rows.add(new Row("Laplacian of Gaussian", EdgeDetectors.evaluateScores(log, teY, s, s, thresholds, radius)));
        List<List<boolean[]>> canny = new ArrayList<>();
        for (int k = 1; k <= thresholds; k++) {
            double high = 0.9 * k / thresholds;
            List<boolean[]> b = new ArrayList<>();
            for (float[] x : teX) {
                b.add(EdgeDetectors.canny(x, s, s, 1.0, 0.4 * high, high));
            }
            canny.add(b);
        }
        rows.add(new Row("Canny", EdgeDetectors.evaluate(canny, teY, s, s, radius)));
        for (boolean deep : new boolean[] {true, false}) {
            EdgeDetectors.Hed net = new EdgeDetectors.Hed(new int[] {8, 12, 16}, deep, seed);
            net.train(trX, trY, s, s, steps, 4, 3e-3f, seed);
            List<float[]> p = new ArrayList<>();
            for (float[] x : teX) {
                p.add(net.predict(x, s, s));
            }
            rows.add(new Row(deep ? "HED" : "FCN single output",
                    EdgeDetectors.evaluateScores(p, teY, s, s, thresholds, radius)));
        }
        return rows;
    }
}
