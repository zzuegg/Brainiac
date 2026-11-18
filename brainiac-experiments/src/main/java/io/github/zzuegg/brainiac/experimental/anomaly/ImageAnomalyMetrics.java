package io.github.zzuegg.brainiac.experimental.anomaly;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Image anomaly detection metrics: the area under the ROC curve (Mann-Whitney, ties counted
 * half) for image scores and for pixel scores, and the per-region overlap (PRO) curve area up
 * to a false-positive rate, normalised to 1 (each anomalous image's mask counts as one region).
 */
public final class ImageAnomalyMetrics {

    private ImageAnomalyMetrics() {
    }

    /**
     * @param scores scores, larger is more anomalous
     * @param positive the labels
     * @return the AUROC
     */
    public static double auroc(double[] scores, boolean[] positive) {
        Integer[] idx = new Integer[scores.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(scores[a], scores[b]));
        double rankSum = 0;
        long pos = 0;
        for (int i = 0; i < idx.length;) {
            int j = i;
            while (j + 1 < idx.length && scores[idx[j + 1]] == scores[idx[i]]) {
                j++;
            }
            double rank = (i + j) / 2.0 + 1;
            for (int k = i; k <= j; k++) {
                if (positive[idx[k]]) {
                    rankSum += rank;
                    pos++;
                }
            }
            i = j + 1;
        }
        long neg = scores.length - pos;
        if (pos == 0 || neg == 0) {
            return 0.5;
        }
        return (rankSum - pos * (pos + 1) / 2.0) / ((double) pos * neg);
    }

    /**
     * Pixel AUROC over all images.
     *
     * @param maps per image its pixel scores
     * @param masks per image its defect pixels
     * @return the AUROC
     */
    public static double pixelAuroc(List<double[]> maps, List<boolean[]> masks) {
        int n = 0;
        for (double[] m : maps) {
            n += m.length;
        }
        double[] s = new double[n];
        boolean[] y = new boolean[n];
        int k = 0;
        for (int i = 0; i < maps.size(); i++) {
            for (int p = 0; p < maps.get(i).length; p++) {
                s[k] = maps.get(i)[p];
                y[k++] = masks.get(i)[p];
            }
        }
        return auroc(s, y);
    }

    /**
     * The normalised area under the PRO curve up to {@code maxFpr}.
     *
     * @param maps per image its pixel scores
     * @param masks per image its defect pixels
     * @param maxFpr the false-positive-rate limit, such as 0.3
     * @return the area divided by {@code maxFpr}
     */
    public static double pro(List<double[]> maps, List<boolean[]> masks, double maxFpr) {
        List<Double> normal = new ArrayList<>();
        for (int i = 0; i < maps.size(); i++) {
            for (int p = 0; p < maps.get(i).length; p++) {
                if (!masks.get(i)[p]) {
                    normal.add(maps.get(i)[p]);
                }
            }
        }
        double[] neg = normal.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        int steps = 40;
        double area = 0;
        double prevFpr = 0;
        double prevPro = overlapAt(maps, masks, Double.POSITIVE_INFINITY);
        for (int s = 1; s <= steps; s++) {
            double fpr = maxFpr * s / steps;
            int at = (int) Math.min(neg.length - 1, Math.floor((1 - fpr) * neg.length));
            double thr = neg[Math.max(0, at)];
            double pro = overlapAt(maps, masks, thr);
            area += (fpr - prevFpr) * (pro + prevPro) / 2;
            prevFpr = fpr;
            prevPro = pro;
        }
        return area / maxFpr;
    }

    private static double overlapAt(List<double[]> maps, List<boolean[]> masks, double thr) {
        double sum = 0;
        int regions = 0;
        for (int i = 0; i < maps.size(); i++) {
            int size = 0;
            int hit = 0;
            for (int p = 0; p < maps.get(i).length; p++) {
                if (masks.get(i)[p]) {
                    size++;
                    hit += maps.get(i)[p] > thr ? 1 : 0;
                }
            }
            if (size > 0) {
                sum += hit / (double) size;
                regions++;
            }
        }
        return regions == 0 ? 0 : sum / regions;
    }
}
