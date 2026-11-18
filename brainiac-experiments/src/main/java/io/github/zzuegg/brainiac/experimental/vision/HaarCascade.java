package io.github.zzuegg.brainiac.experimental.vision;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A Viola-Jones detector (Viola &amp; Jones, CVPR 2001): Haar-like features (two-, three- and
 * four-rectangle) evaluated in four lookups each on an integral image of the
 * variance-normalised patch, discrete AdaBoost over decision stumps, and an attentional
 * cascade of boosted stages, each with its threshold lowered so that it keeps a target share
 * of the positives; every stage is trained on the negatives that the earlier stages still let
 * through. With one stage it is the non-cascaded boosted classifier. It implements
 * {@link SlidingWindowDetector.Scorer} and counts the features it evaluates.
 */
public final class HaarCascade implements SlidingWindowDetector.Scorer {

    private final int side;
    private final List<int[]> features;
    private final List<Stage> stages = new ArrayList<>();
    private long evaluated;
    private long windows;

    /** One boosted stage: stumps {@code (feature, threshold, polarity, alpha)}. */
    private record Stage(int[] feature, double[] threshold, int[] polarity, double[] alpha, double cut) {
    }

    /**
     * Enumerates features on a square patch, keeping a random subset.
     *
     * @param side the patch side
     * @param maxFeatures how many features to keep at most
     * @param seed the seed of the subsampling
     */
    public HaarCascade(int side, int maxFeatures, long seed) {
        this.side = side;
        List<int[]> all = new ArrayList<>();
        int[][] shapes = {{2, 1}, {1, 2}, {3, 1}, {1, 3}, {2, 2}};
        for (int type = 0; type < shapes.length; type++) {
            int sx = shapes[type][0];
            int sy = shapes[type][1];
            for (int w = sx; w <= side; w += sx) {
                for (int h = sy; h <= side; h += sy) {
                    for (int y = 0; y + h <= side; y++) {
                        for (int x = 0; x + w <= side; x++) {
                            all.add(new int[] {type, x, y, w, h});
                        }
                    }
                }
            }
        }
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = all.size() - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int[] t = all.get(i);
            all.set(i, all.get(j));
            all.set(j, t);
        }
        this.features = new ArrayList<>(all.subList(0, Math.min(maxFeatures, all.size())));
    }

    /** @return how many features the pool holds */
    public int featureCount() {
        return features.size();
    }

    /**
     * The integral image of the variance-normalised patch, {@code (side + 1)^2}.
     *
     * @param patch the patch
     * @return the integral image
     */
    public double[] integral(float[] patch) {
        double mean = 0;
        for (float v : patch) {
            mean += v;
        }
        mean /= patch.length;
        double var = 0;
        for (float v : patch) {
            var += (v - mean) * (v - mean);
        }
        double sd = Math.sqrt(var / patch.length) + 1e-3;
        int s = side + 1;
        double[] ii = new double[s * s];
        for (int y = 0; y < side; y++) {
            double row = 0;
            for (int x = 0; x < side; x++) {
                row += (patch[y * side + x] - mean) / sd;
                ii[(y + 1) * s + x + 1] = ii[y * s + x + 1] + row;
            }
        }
        return ii;
    }

    private double rect(double[] ii, int x, int y, int w, int h) {
        int s = side + 1;
        return ii[(y + h) * s + x + w] - ii[y * s + x + w] - ii[(y + h) * s + x] + ii[y * s + x];
    }

    /**
     * @param ii an integral image
     * @param f the feature index
     * @return the feature value
     */
    public double feature(double[] ii, int f) {
        int[] d = features.get(f);
        int x = d[1];
        int y = d[2];
        int w = d[3];
        int h = d[4];
        return switch (d[0]) {
            case 0 -> rect(ii, x, y, w / 2, h) - rect(ii, x + w / 2, y, w / 2, h);
            case 1 -> rect(ii, x, y, w, h / 2) - rect(ii, x, y + h / 2, w, h / 2);
            case 2 -> rect(ii, x, y, w, h) - 3 * rect(ii, x + w / 3, y, w / 3, h);
            case 3 -> rect(ii, x, y, w, h) - 3 * rect(ii, x, y + h / 3, w, h / 3);
            default -> rect(ii, x, y, w / 2, h / 2) + rect(ii, x + w / 2, y + h / 2, w / 2, h / 2)
                - rect(ii, x + w / 2, y, w / 2, h / 2) - rect(ii, x, y + h / 2, w / 2, h / 2);
        };
    }

    /**
     * Trains the cascade.
     *
     * @param positives positive patches
     * @param negatives a pool of negative patches
     * @param stageSizes the stumps per stage
     * @param minDetection the share of positives each stage keeps, such as 0.995
     * @param negativesPerStage the negatives each stage trains on, at most
     * @param seed the seed
     */
    public void fit(List<float[]> positives, List<float[]> negatives, int[] stageSizes, double minDetection,
            int negativesPerStage, long seed) {
        stages.clear();
        SplittableRandom r = new SplittableRandom(seed);
        List<double[]> pos = new ArrayList<>();
        for (float[] p : positives) {
            pos.add(integral(p));
        }
        List<double[]> pool = new ArrayList<>();
        for (float[] p : negatives) {
            pool.add(integral(p));
        }
        for (int size : stageSizes) {
            List<double[]> passing = new ArrayList<>();
            for (double[] n : pool) {
                if (passes(n)) {
                    passing.add(n);
                }
            }
            if (passing.isEmpty()) {
                break;
            }
            List<double[]> neg = new ArrayList<>();
            for (int i = 0; i < Math.min(negativesPerStage, passing.size()); i++) {
                neg.add(passing.get(r.nextInt(passing.size())));
            }
            stages.add(boost(pos, neg, size, minDetection));
        }
    }

    private boolean passes(double[] ii) {
        for (Stage s : stages) {
            if (stageScore(s, ii, false) < s.cut()) {
                return false;
            }
        }
        return true;
    }

    private double stageScore(Stage s, double[] ii, boolean count) {
        double sum = 0;
        for (int k = 0; k < s.feature().length; k++) {
            double v = feature(ii, s.feature()[k]);
            if (s.polarity()[k] * v < s.polarity()[k] * s.threshold()[k]) {
                sum += s.alpha()[k];
            }
        }
        if (count) {
            evaluated += s.feature().length;
        }
        return sum;
    }

    private Stage boost(List<double[]> pos, List<double[]> neg, int size, double minDetection) {
        int n = pos.size() + neg.size();
        int f = features.size();
        boolean[] label = new boolean[n];
        double[][] values = new double[f][n];
        for (int i = 0; i < n; i++) {
            double[] ii = i < pos.size() ? pos.get(i) : neg.get(i - pos.size());
            label[i] = i < pos.size();
            for (int k = 0; k < f; k++) {
                values[k][i] = feature(ii, k);
            }
        }
        int[][] order = new int[f][];
        for (int k = 0; k < f; k++) {
            double[] v = values[k];
            order[k] = java.util.stream.IntStream.range(0, n).boxed()
                .sorted((a, b) -> Double.compare(v[a], v[b])).mapToInt(Integer::intValue).toArray();
        }
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = label[i] ? 0.5 / pos.size() : 0.5 / neg.size();
        }
        int[] feat = new int[size];
        double[] thr = new double[size];
        int[] pol = new int[size];
        double[] alpha = new double[size];
        for (int t = 0; t < size; t++) {
            double total = 0;
            double totalPos = 0;
            for (int i = 0; i < n; i++) {
                total += w[i];
                totalPos += label[i] ? w[i] : 0;
            }
            for (int i = 0; i < n; i++) {
                w[i] /= total;
            }
            totalPos /= total;
            double totalNeg = 1 - totalPos;
            double bestErr = Double.POSITIVE_INFINITY;
            for (int k = 0; k < f; k++) {
                double belowPos = 0;
                double belowNeg = 0;
                int[] o = order[k];
                for (int j = 0; j <= n; j++) {
                    if (j > 0 && j < n && values[k][o[j]] == values[k][o[j - 1]]) {
                        int i = o[j];
                        if (label[i]) {
                            belowPos += w[i];
                        } else {
                            belowNeg += w[i];
                        }
                        continue;
                    }
                    double errBelowPositive = belowNeg + (totalPos - belowPos);
                    double errAbovePositive = belowPos + (totalNeg - belowNeg);
                    double th = j == 0 ? values[k][o[0]] - 1e-9 : j == n ? values[k][o[n - 1]] + 1e-9
                        : 0.5 * (values[k][o[j - 1]] + values[k][o[j]]);
                    if (errBelowPositive < bestErr) {
                        bestErr = errBelowPositive;
                        feat[t] = k;
                        thr[t] = th;
                        pol[t] = 1;
                    }
                    if (errAbovePositive < bestErr) {
                        bestErr = errAbovePositive;
                        feat[t] = k;
                        thr[t] = th;
                        pol[t] = -1;
                    }
                    if (j < n) {
                        int i = o[j];
                        if (label[i]) {
                            belowPos += w[i];
                        } else {
                            belowNeg += w[i];
                        }
                    }
                }
            }
            double err = Math.max(bestErr, 1e-10);
            double beta = err / (1 - err);
            alpha[t] = Math.log(1 / beta);
            for (int i = 0; i < n; i++) {
                boolean predicted = pol[t] * values[feat[t]][i] < pol[t] * thr[t];
                if (predicted == label[i]) {
                    w[i] *= beta;
                }
            }
        }
        Stage partial = new Stage(feat, thr, pol, alpha, 0);
        double[] scores = new double[pos.size()];
        for (int i = 0; i < pos.size(); i++) {
            scores[i] = stageScore(partial, pos.get(i), false);
        }
        Arrays.sort(scores);
        int idx = (int) Math.floor((1 - minDetection) * scores.length);
        double cut = scores[Math.max(0, Math.min(scores.length - 1, idx))] - 1e-9;
        return new Stage(feat, thr, pol, alpha, cut);
    }

    @Override
    public double score(float[] patch) {
        windows++;
        double[] ii = integral(patch);
        double total = 0;
        for (Stage s : stages) {
            double v = stageScore(s, ii, true);
            if (v < s.cut()) {
                return Double.NEGATIVE_INFINITY;
            }
            double sum = 0;
            for (double a : s.alpha()) {
                sum += a;
            }
            total += (v - s.cut()) / sum;
        }
        return total;
    }

    /**
     * The raw score of the last stage alone, never rejecting: the non-cascaded classifier's
     * margin when there is one stage.
     *
     * @param patch the patch
     * @return the score in units of the stage's total weight, centred at half of it
     */
    public double boostedScore(float[] patch) {
        windows++;
        Stage s = stages.get(stages.size() - 1);
        double sum = 0;
        for (double a : s.alpha()) {
            sum += a;
        }
        return stageScore(s, integral(patch), true) / sum - 0.5;
    }

    /** @return the stages trained */
    public int stageCount() {
        return stages.size();
    }

    /** @return the mean features evaluated per scored window since the last reset */
    public double featuresPerWindow() {
        return windows == 0 ? 0 : evaluated / (double) windows;
    }

    /** Resets the feature counters. */
    public void resetCounters() {
        evaluated = 0;
        windows = 0;
    }
}
