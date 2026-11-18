package io.github.zzuegg.brainiac.experimental.vision;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A single-class sliding-window detector (Dalal &amp; Triggs, CVPR 2005): square windows of
 * several sizes (equivalent to an image pyramid) are resampled to a canonical patch and
 * scored, windows above a threshold are kept, and greedy non-maximum suppression merges
 * overlaps. {@link #train} fits a linear SVM ({@link LinearSvm}, Pegasos) on the features of
 * jittered positive windows and random negatives, then mines hard negatives (false
 * positives with margin violations on the training images) and refits.
 *
 * <p>A window of side {@code s} at {@code (x, y)} predicts the box shrunk by
 * {@code margin * s} on each side, so the object fills the middle of the canonical patch.
 */
public final class SlidingWindowDetector {

    /** Scores a canonical patch. */
    @FunctionalInterface
    public interface Scorer {

        /**
         * @param patch the canonical patch, row-major
         * @return the score, or negative infinity to reject
         */
        double score(float[] patch);
    }

    /** Computes the features of a canonical patch. */
    @FunctionalInterface
    public interface Features {

        /**
         * @param patch the canonical patch
         * @return the features
         */
        double[] compute(float[] patch);
    }

    private final int canonical;
    private final int[] windows;
    private final double strideFraction;
    private final double margin;

    /**
     * Creates the detector.
     *
     * @param canonical the canonical patch side
     * @param windows the window sides in image pixels
     * @param strideFraction the stride as a fraction of the window side
     * @param margin the context margin as a fraction of the window side
     */
    public SlidingWindowDetector(int canonical, int[] windows, double strideFraction, double margin) {
        this.canonical = canonical;
        this.windows = windows.clone();
        this.strideFraction = strideFraction;
        this.margin = margin;
    }

    /** @return the canonical patch side */
    public int canonical() {
        return canonical;
    }

    /**
     * Resamples a square window to the canonical patch (bilinear, clamped at the borders).
     *
     * @param image the image
     * @param side the image side
     * @param x the window's left edge
     * @param y the window's top edge
     * @param size the window side
     * @return the patch
     */
    public float[] crop(float[] image, int side, double x, double y, double size) {
        float[] p = new float[canonical * canonical];
        double step = size / canonical;
        for (int j = 0; j < canonical; j++) {
            double sy = y + (j + 0.5) * step - 0.5;
            for (int i = 0; i < canonical; i++) {
                double sx = x + (i + 0.5) * step - 0.5;
                int x0 = (int) Math.floor(sx);
                int y0 = (int) Math.floor(sy);
                double fx = sx - x0;
                double fy = sy - y0;
                p[j * canonical + i] = (float) ((1 - fy) * ((1 - fx) * pixel(image, side, x0, y0)
                    + fx * pixel(image, side, x0 + 1, y0))
                    + fy * ((1 - fx) * pixel(image, side, x0, y0 + 1) + fx * pixel(image, side, x0 + 1, y0 + 1)));
            }
        }
        return p;
    }

    private static float pixel(float[] image, int side, int x, int y) {
        x = Math.max(0, Math.min(side - 1, x));
        y = Math.max(0, Math.min(side - 1, y));
        return image[y * side + x];
    }

    /**
     * @param x the window's left edge
     * @param y the window's top edge
     * @param size the window side
     * @return the predicted box {@code [x1, y1, x2, y2]}
     */
    public float[] box(double x, double y, double size) {
        double m = margin * size;
        return new float[] {(float) (x + m), (float) (y + m), (float) (x + size - m), (float) (y + size - m)};
    }

    /**
     * @param side the image side
     * @return every window {@code (x, y, size)}
     */
    public List<int[]> windows(int side) {
        List<int[]> out = new ArrayList<>();
        for (int s : windows) {
            int stride = Math.max(1, (int) Math.round(s * strideFraction));
            for (int y = 0; y + s <= side; y += stride) {
                for (int x = 0; x + s <= side; x += stride) {
                    out.add(new int[] {x, y, s});
                }
            }
        }
        return out;
    }

    /**
     * Scores every window and suppresses overlaps.
     *
     * @param image the image
     * @param side the image side
     * @param scorer the window scorer
     * @param threshold the smallest kept score
     * @param nmsIou the overlap above which the weaker detection is dropped
     * @return detections {@code (score, x1, y1, x2, y2)}, strongest first
     */
    public List<float[]> detect(float[] image, int side, Scorer scorer, double threshold, double nmsIou) {
        List<float[]> raw = new ArrayList<>();
        for (int[] w : windows(side)) {
            double s = scorer.score(crop(image, side, w[0], w[1], w[2]));
            if (s >= threshold) {
                float[] b = box(w[0], w[1], w[2]);
                raw.add(new float[] {(float) s, b[0], b[1], b[2], b[3]});
            }
        }
        return nms(raw, nmsIou);
    }

    /**
     * Greedy non-maximum suppression.
     *
     * @param detections {@code (score, x1, y1, x2, y2)}
     * @param iou the overlap threshold
     * @return the kept detections, strongest first
     */
    public static List<float[]> nms(List<float[]> detections, double iou) {
        List<float[]> sorted = new ArrayList<>(detections);
        sorted.sort(Comparator.comparingDouble((float[] d) -> -d[0]));
        List<float[]> kept = new ArrayList<>();
        for (float[] d : sorted) {
            boolean keep = true;
            for (float[] k : kept) {
                if (DetectionMetrics.iou(boxOf(d), boxOf(k)) > iou) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                kept.add(d);
            }
        }
        return kept;
    }

    private static float[] boxOf(float[] d) {
        return new float[] {d[1], d[2], d[3], d[4]};
    }

    /** A linear SVM trained by Pegasos (Shalev-Shwartz et al., ICML 2007) with a bias. */
    public static final class LinearSvm {

        private final double[] w;
        private double b;

        /**
         * Creates an all-zero SVM.
         *
         * @param dims the feature dimension
         */
        public LinearSvm(int dims) {
            this.w = new double[dims];
        }

        /**
         * @param x features
         * @return the margin {@code w . x + b}
         */
        public double score(double[] x) {
            double s = b;
            for (int i = 0; i < w.length; i++) {
                s += w[i] * x[i];
            }
            return s;
        }

        /**
         * Fits by Pegasos from the current weights' zero start, balancing classes by weight.
         *
         * @param x the features
         * @param y the labels, +1 or -1
         * @param lambda the regularisation
         * @param epochs passes over the data
         * @param seed the seed
         */
        public void fit(List<double[]> x, int[] y, double lambda, int epochs, long seed) {
            java.util.Arrays.fill(w, 0);
            b = 0;
            int pos = 0;
            for (int v : y) {
                pos += v > 0 ? 1 : 0;
            }
            double wPos = x.size() / (2.0 * Math.max(1, pos));
            double wNeg = x.size() / (2.0 * Math.max(1, x.size() - pos));
            SplittableRandom r = new SplittableRandom(seed);
            long t = 0;
            for (int e = 0; e < epochs; e++) {
                for (int k = 0; k < x.size(); k++) {
                    int i = r.nextInt(x.size());
                    t++;
                    double eta = 1 / (lambda * (t + 100));
                    double m = y[i] * score(x.get(i));
                    for (int j = 0; j < w.length; j++) {
                        w[j] *= 1 - eta * lambda;
                    }
                    if (m < 1) {
                        double c = eta * y[i] * (y[i] > 0 ? wPos : wNeg);
                        double[] xi = x.get(i);
                        for (int j = 0; j < w.length; j++) {
                            w[j] += c * xi[j];
                        }
                        b += c;
                    }
                }
            }
        }
    }

    /**
     * Trains a linear SVM detector with hard-negative mining.
     *
     * @param images the training images
     * @param side the image side
     * @param boxes per image the ground-truth boxes, {@code 4 k} values
     * @param features the window features
     * @param negativesPerImage random negatives per image
     * @param miningRounds hard-negative rounds (0 for none)
     * @param hardPerRound hard negatives added per round, at most
     * @param lambda the SVM regularisation
     * @param epochs SVM passes
     * @param seed the seed
     * @return the SVM
     */
    public LinearSvm train(float[][] images, int side, float[][] boxes, Features features, int negativesPerImage,
            int miningRounds, int hardPerRound, double lambda, int epochs, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<double[]> x = new ArrayList<>();
        List<Integer> y = new ArrayList<>();
        for (int n = 0; n < images.length; n++) {
            float[] gt = boxes[n];
            for (int k = 0; k < gt.length / 4; k++) {
                double bw = Math.max(gt[4 * k + 2] - gt[4 * k], gt[4 * k + 3] - gt[4 * k + 1]);
                double size = bw / (1 - 2 * margin);
                double cx = (gt[4 * k] + gt[4 * k + 2]) / 2;
                double cy = (gt[4 * k + 1] + gt[4 * k + 3]) / 2;
                for (int j = 0; j < 5; j++) {
                    double dx = j == 0 ? 0 : r.nextDouble(-1, 1);
                    double dy = j == 0 ? 0 : r.nextDouble(-1, 1);
                    double ds = j == 0 ? 1 : r.nextDouble(0.92, 1.08);
                    double s = size * ds;
                    x.add(features.compute(crop(images[n], side, cx - s / 2 + dx, cy - s / 2 + dy, s)));
                    y.add(1);
                }
            }
            int added = 0;
            int tries = 0;
            while (added < negativesPerImage && tries++ < 50 * negativesPerImage) {
                int s = windows[r.nextInt(windows.length)];
                int wx = r.nextInt(side - s + 1);
                int wy = r.nextInt(side - s + 1);
                if (maxIou(box(wx, wy, s), gt) < 0.3) {
                    x.add(features.compute(crop(images[n], side, wx, wy, s)));
                    y.add(-1);
                    added++;
                }
            }
        }
        LinearSvm svm = new LinearSvm(x.get(0).length);
        svm.fit(x, toArray(y), lambda, epochs, seed);
        for (int round = 0; round < miningRounds; round++) {
            List<double[]> hard = new ArrayList<>();
            List<Double> hardScores = new ArrayList<>();
            for (int n = 0; n < images.length; n++) {
                for (int[] w : windows(side)) {
                    double[] f = features.compute(crop(images[n], side, w[0], w[1], w[2]));
                    double s = svm.score(f);
                    if (s > -1 && maxIou(box(w[0], w[1], w[2]), boxes[n]) < 0.3) {
                        hard.add(f);
                        hardScores.add(s);
                    }
                }
            }
            Integer[] order = new Integer[hard.size()];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            java.util.Arrays.sort(order, (a, c) -> Double.compare(hardScores.get(c), hardScores.get(a)));
            for (int i = 0; i < Math.min(hardPerRound, order.length); i++) {
                x.add(hard.get(order[i]));
                y.add(-1);
            }
            svm.fit(x, toArray(y), lambda, epochs, seed + round + 1);
        }
        return svm;
    }

    private static int[] toArray(List<Integer> y) {
        int[] out = new int[y.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = y.get(i);
        }
        return out;
    }

    /**
     * @param box a box
     * @param boxes {@code 4 k} boxes
     * @return the largest overlap with any of them
     */
    public static double maxIou(float[] box, float[] boxes) {
        double best = 0;
        for (int k = 0; k < boxes.length / 4; k++) {
            best = Math.max(best, DetectionMetrics.iou(box,
                new float[] {boxes[4 * k], boxes[4 * k + 1], boxes[4 * k + 2], boxes[4 * k + 3]}));
        }
        return best;
    }
}
