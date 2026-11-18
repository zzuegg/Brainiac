package io.github.zzuegg.brainiac.experimental.vision;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Detection and instance-segmentation metrics shared by the vision blocks: box mean average
 * precision at one IoU threshold (PASCAL VOC's mAP@0.5) and averaged over the COCO
 * thresholds 0.5 to 0.95, mask average precision, and the foreground adjusted Rand index of
 * an instance labelling.
 *
 * <p>Average precision is the area under the precision-recall curve with the precision made
 * monotone (all-point interpolation): detections are taken in order of score, each matched
 * to the unmatched ground truth it overlaps most if that overlap reaches the threshold, and
 * counted a false positive otherwise. The mean is over the classes that have ground truth.
 */
public final class DetectionMetrics {

    private DetectionMetrics() {
    }

    /**
     * One detection.
     *
     * @param image the image index
     * @param cls the class
     * @param score the confidence
     * @param box {@code [x1, y1, x2, y2]}
     */
    public record Detection(int image, int cls, float score, float[] box) {
    }

    /**
     * The overlap of two boxes.
     *
     * @param a {@code [x1, y1, x2, y2]}
     * @param b {@code [x1, y1, x2, y2]}
     * @return intersection over union, 0 if either is empty
     */
    public static double iou(float[] a, float[] b) {
        double iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
        double ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
        if (iw <= 0 || ih <= 0) {
            return 0;
        }
        double inter = iw * ih;
        double union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
        return union <= 0 ? 0 : inter / union;
    }

    /**
     * Mean average precision at one IoU threshold.
     *
     * @param detections the detections of all images
     * @param boxes per image the ground-truth boxes, {@code 4 k} values
     * @param classes per image the ground-truth classes
     * @param classCount the number of classes
     * @param threshold the IoU a match needs, such as 0.5
     * @return the mean over classes with ground truth of the average precision
     */
    public static double meanAp(List<Detection> detections, float[][] boxes, int[][] classes, int classCount,
            double threshold) {
        double sum = 0;
        int counted = 0;
        for (int c = 0; c < classCount; c++) {
            List<Detection> dets = new ArrayList<>();
            for (Detection d : detections) {
                if (d.cls() == c) {
                    dets.add(d);
                }
            }
            int positives = 0;
            List<List<float[]>> truth = new ArrayList<>();
            for (int i = 0; i < boxes.length; i++) {
                List<float[]> t = new ArrayList<>();
                for (int k = 0; k < classes[i].length; k++) {
                    if (classes[i][k] == c) {
                        t.add(new float[] {boxes[i][4 * k], boxes[i][4 * k + 1], boxes[i][4 * k + 2],
                            boxes[i][4 * k + 3]});
                    }
                }
                positives += t.size();
                truth.add(t);
            }
            if (positives == 0) {
                continue;
            }
            sum += averagePrecision(dets, truth, positives, threshold, (d, t) -> iou(d.box(), t));
            counted++;
        }
        return counted == 0 ? 0 : sum / counted;
    }

    /**
     * COCO-style mean average precision, averaged over IoU thresholds 0.50, 0.55, ..., 0.95.
     *
     * @param detections the detections
     * @param boxes per image the ground-truth boxes
     * @param classes per image the ground-truth classes
     * @param classCount the number of classes
     * @return mAP@[.5:.95]
     */
    public static double meanApCoco(List<Detection> detections, float[][] boxes, int[][] classes, int classCount) {
        double sum = 0;
        for (int t = 0; t < 10; t++) {
            sum += meanAp(detections, boxes, classes, classCount, 0.5 + 0.05 * t);
        }
        return sum / 10;
    }

    /**
     * One predicted instance mask.
     *
     * @param image the image index
     * @param score the confidence
     * @param pixels the mask, one flag per pixel
     */
    public record MaskDetection(int image, float score, boolean[] pixels) {
    }

    /**
     * Class-agnostic mask average precision at one mask-IoU threshold.
     *
     * @param masks the predicted masks
     * @param instances per image the ground-truth instance labelling (0 background,
     *     {@code 1..k} instances)
     * @param threshold the mask IoU a match needs
     * @return the average precision
     */
    public static double maskAp(List<MaskDetection> masks, int[][] instances, double threshold) {
        int positives = 0;
        List<List<boolean[]>> truth = new ArrayList<>();
        for (int[] inst : instances) {
            int k = 0;
            for (int v : inst) {
                k = Math.max(k, v);
            }
            List<boolean[]> t = new ArrayList<>();
            for (int i = 1; i <= k; i++) {
                boolean[] m = new boolean[inst.length];
                boolean any = false;
                for (int p = 0; p < inst.length; p++) {
                    m[p] = inst[p] == i;
                    any |= m[p];
                }
                if (any) {
                    t.add(m);
                }
            }
            positives += t.size();
            truth.add(t);
        }
        if (positives == 0) {
            return 0;
        }
        return averagePrecision(masks, truth, positives, threshold, (d, t) -> maskIou(d.pixels(), t));
    }

    /**
     * The overlap of two masks.
     *
     * @param a one mask
     * @param b another of the same length
     * @return intersection over union
     */
    public static double maskIou(boolean[] a, boolean[] b) {
        int inter = 0;
        int union = 0;
        for (int i = 0; i < a.length; i++) {
            inter += a[i] && b[i] ? 1 : 0;
            union += a[i] || b[i] ? 1 : 0;
        }
        return union == 0 ? 0 : (double) inter / union;
    }

    /**
     * The adjusted Rand index between a predicted and a true labelling over the true
     * foreground (pixels with a true label above 0), as object discovery reports it.
     *
     * @param predicted a cluster id per pixel
     * @param truth the true instance per pixel, 0 for background
     * @return FG-ARI, 1 for a perfect grouping, about 0 for chance
     */
    public static double foregroundAri(int[] predicted, int[] truth) {
        java.util.Map<Long, Long> joint = new java.util.HashMap<>();
        java.util.Map<Integer, Long> rows = new java.util.HashMap<>();
        java.util.Map<Integer, Long> cols = new java.util.HashMap<>();
        long n = 0;
        for (int i = 0; i < truth.length; i++) {
            if (truth[i] <= 0) {
                continue;
            }
            n++;
            joint.merge(((long) truth[i] << 32) | (predicted[i] & 0xffffffffL), 1L, Long::sum);
            rows.merge(truth[i], 1L, Long::sum);
            cols.merge(predicted[i], 1L, Long::sum);
        }
        if (n < 2) {
            return 1;
        }
        double index = 0;
        for (long v : joint.values()) {
            index += pairs(v);
        }
        double a = 0;
        for (long v : rows.values()) {
            a += pairs(v);
        }
        double b = 0;
        for (long v : cols.values()) {
            b += pairs(v);
        }
        double expected = a * b / pairs(n);
        double max = (a + b) / 2;
        return max == expected ? 1 : (index - expected) / (max - expected);
    }

    private static double pairs(long v) {
        return v * (v - 1) / 2.0;
    }

    /** Scores a detection against a ground-truth item. */
    private interface Overlap<D, T> {
        double of(D detection, T truth);
    }

    private static <D, T> double averagePrecision(List<D> detections, List<List<T>> truth, int positives,
            double threshold, Overlap<D, T> overlap) {
        List<D> sorted = new ArrayList<>(detections);
        sorted.sort(Comparator.comparingDouble((D d) -> -score(d)));
        List<boolean[]> used = new ArrayList<>();
        for (List<T> t : truth) {
            used.add(new boolean[t.size()]);
        }
        int n = sorted.size();
        double[] precision = new double[n];
        double[] recall = new double[n];
        int tp = 0;
        for (int i = 0; i < n; i++) {
            D d = sorted.get(i);
            int img = image(d);
            List<T> t = truth.get(img);
            int best = -1;
            double bestIou = threshold;
            for (int k = 0; k < t.size(); k++) {
                if (used.get(img)[k]) {
                    continue;
                }
                double o = overlap.of(d, t.get(k));
                if (o >= bestIou) {
                    bestIou = o;
                    best = k;
                }
            }
            if (best >= 0) {
                used.get(img)[best] = true;
                tp++;
            }
            precision[i] = (double) tp / (i + 1);
            recall[i] = (double) tp / positives;
        }
        for (int i = n - 2; i >= 0; i--) {
            precision[i] = Math.max(precision[i], precision[i + 1]);
        }
        double ap = 0;
        double previous = 0;
        for (int i = 0; i < n; i++) {
            ap += (recall[i] - previous) * precision[i];
            previous = recall[i];
        }
        return ap;
    }

    private static double score(Object d) {
        return d instanceof Detection x ? x.score() : ((MaskDetection) d).score();
    }

    private static int image(Object d) {
        return d instanceof Detection x ? x.image() : ((MaskDetection) d).image();
    }
}
