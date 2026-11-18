package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.experimental.data.ShapeScenes;
import io.github.zzuegg.brainiac.experimental.vision.DetectionMetrics;
import io.github.zzuegg.brainiac.experimental.vision.HaarCascade;
import io.github.zzuegg.brainiac.experimental.vision.HogDescriptor;
import io.github.zzuegg.brainiac.experimental.vision.SlidingWindowDetector;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * W12 12.3.1-12.3.2: classical single-class detection of the ring glyph among five distractor
 * glyph classes in noisy {@link ShapeScenes} (48 x 48, three glyph scales, 3-6 glyphs). All
 * methods share one {@link SlidingWindowDetector} (three window sizes, canonical 16 x 16
 * patch, greedy NMS): a linear SVM on raw (normalised) pixels, HOG + linear SVM without and
 * with hard-negative mining, a single boosted classifier of Haar-like stumps, and a
 * Viola-Jones cascade with the same number of stumps split over four stages. Reported: AP at
 * IoU 0.5, false positives per image at the operating threshold, milliseconds per image and,
 * for the Haar detectors, the features evaluated per window.
 */
public final class ClassicalDetection {

    /** The detected class. */
    public static final int TARGET = 2;

    /**
     * One detector.
     *
     * @param method the method
     * @param ap the AP at IoU 0.5
     * @param falsePositives false positives per image at the operating point
     * @param millisPerImage detection time per image
     * @param featuresPerWindow Haar features per window (0 for the SVMs)
     */
    public record Row(String method, double ap, double falsePositives, double millisPerImage,
            double featuresPerWindow) {
    }

    private ClassicalDetection() {
    }

    private static float[] normalised(float[] p) {
        double m = 0;
        for (float v : p) {
            m += v;
        }
        m /= p.length;
        double s = 0;
        for (float v : p) {
            s += (v - m) * (v - m);
        }
        s = Math.sqrt(s / p.length) + 1e-3;
        float[] out = new float[p.length];
        for (int i = 0; i < p.length; i++) {
            out[i] = (float) ((p[i] - m) / s);
        }
        return out;
    }

    private static float[] targetBoxes(ShapeScenes.Scene s) {
        List<Float> b = new ArrayList<>();
        for (int i = 0; i < s.count(); i++) {
            if (s.classes()[i] == TARGET) {
                for (float v : s.box(i)) {
                    b.add(v);
                }
            }
        }
        float[] out = new float[b.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.get(i);
        }
        return out;
    }

    /**
     * Runs the comparison.
     *
     * @param trainImages training scenes
     * @param testImages test scenes
     * @param seed the seed
     * @return raw+SVM, HOG+SVM, HOG+SVM+mining, boosted Haar, Haar cascade
     */
    public static List<Row> run(int trainImages, int testImages, long seed) {
        int side = 48;
        ShapeScenes gen = new ShapeScenes(side, 0.1f);
        ShapeScenes.Scene[] train = gen.scenes(trainImages, 3, 6, ShapeScenes.Layout.SPARSE, seed);
        ShapeScenes.Scene[] test = gen.scenes(testImages, 3, 6, ShapeScenes.Layout.SPARSE, seed + 1);
        float[][] trainX = new float[trainImages][];
        float[][] trainB = new float[trainImages][];
        for (int i = 0; i < trainImages; i++) {
            trainX[i] = train[i].image();
            trainB[i] = targetBoxes(train[i]);
        }
        double margin = 0.2;
        SlidingWindowDetector det = new SlidingWindowDetector(16, new int[] {5, 10, 15}, 0.25, margin);
        HogDescriptor hog = new HogDescriptor(16, 4, 9);
        SlidingWindowDetector.Features hogF = hog::compute;
        SlidingWindowDetector.Features rawF = p -> {
            float[] n = normalised(p);
            double[] d = new double[n.length];
            for (int i = 0; i < n.length; i++) {
                d[i] = n[i];
            }
            return d;
        };
        List<Row> rows = new ArrayList<>();
        SlidingWindowDetector.LinearSvm raw = det.train(trainX, side, trainB, rawF, 40, 1, 1500, 1e-3, 8, seed);
        rows.add(evaluate("raw pixels + SVM + mining", det, test, p -> raw.score(rawF.compute(p)), 0, null));
        SlidingWindowDetector.LinearSvm plain = det.train(trainX, side, trainB, hogF, 40, 0, 0, 1e-3, 8, seed);
        rows.add(evaluate("HOG + SVM", det, test, p -> plain.score(hog.compute(p)), 0, null));
        SlidingWindowDetector.LinearSvm mined = det.train(trainX, side, trainB, hogF, 40, 1, 1500, 1e-3, 8, seed);
        rows.add(evaluate("HOG + SVM + mining", det, test, p -> mined.score(hog.compute(p)), 0, null));

        SplittableRandom r = new SplittableRandom(seed + 2);
        List<float[]> pos = new ArrayList<>();
        List<float[]> neg = new ArrayList<>();
        for (int n = 0; n < trainImages; n++) {
            float[] gt = trainB[n];
            for (int k = 0; k < gt.length / 4; k++) {
                double bw = Math.max(gt[4 * k + 2] - gt[4 * k], gt[4 * k + 3] - gt[4 * k + 1]);
                double size = bw / (1 - 2 * margin);
                double cx = (gt[4 * k] + gt[4 * k + 2]) / 2;
                double cy = (gt[4 * k + 1] + gt[4 * k + 3]) / 2;
                for (int j = 0; j < 8; j++) {
                    double dx = j == 0 ? 0 : r.nextDouble(-1, 1);
                    double dy = j == 0 ? 0 : r.nextDouble(-1, 1);
                    double s = size * (j == 0 ? 1 : r.nextDouble(0.92, 1.08));
                    pos.add(det.crop(trainX[n], side, cx - s / 2 + dx, cy - s / 2 + dy, s));
                }
            }
            for (int[] w : det.windows(side)) {
                if (r.nextDouble() < 0.15 && SlidingWindowDetector.maxIou(det.box(w[0], w[1], w[2]), gt) < 0.3) {
                    neg.add(det.crop(trainX[n], side, w[0], w[1], w[2]));
                }
            }
        }
        HaarCascade single = new HaarCascade(16, 1500, seed);
        single.fit(pos, neg, new int[] {40}, 0.99, 1500, seed);
        rows.add(evaluate("boosted Haar (40 stumps)", det, test, single::boostedScore, 0, single));
        HaarCascade cascade = new HaarCascade(16, 1500, seed);
        cascade.fit(pos, neg, new int[] {3, 7, 12, 18}, 0.995, 1500, seed);
        rows.add(evaluate("Haar cascade (3/7/12/18)", det, test, cascade, 0, cascade));
        return rows;
    }

    private static Row evaluate(String name, SlidingWindowDetector det, ShapeScenes.Scene[] test,
            SlidingWindowDetector.Scorer scorer, double threshold, HaarCascade counter) {
        List<DetectionMetrics.Detection> all = new ArrayList<>();
        float[][] boxes = new float[test.length][];
        int[][] classes = new int[test.length][];
        int falsePositives = 0;
        if (counter != null) {
            counter.resetCounters();
        }
        long t0 = System.nanoTime();
        List<List<float[]>> found = new ArrayList<>();
        for (ShapeScenes.Scene s : test) {
            found.add(det.detect(s.image(), s.side(), scorer, -2, 0.3));
        }
        double ms = (System.nanoTime() - t0) / 1e6 / test.length;
        for (int i = 0; i < test.length; i++) {
            boxes[i] = targetBoxes(test[i]);
            classes[i] = new int[boxes[i].length / 4];
            for (float[] d : found.get(i)) {
                float[] b = {d[1], d[2], d[3], d[4]};
                all.add(new DetectionMetrics.Detection(i, 0, d[0], b));
                if (d[0] >= threshold && SlidingWindowDetector.maxIou(b, boxes[i]) < 0.5) {
                    falsePositives++;
                }
            }
        }
        double ap = DetectionMetrics.meanAp(all, boxes, classes, 1, 0.5);
        return new Row(name, ap, falsePositives / (double) test.length, ms,
            counter == null ? 0 : counter.featuresPerWindow());
    }
}
