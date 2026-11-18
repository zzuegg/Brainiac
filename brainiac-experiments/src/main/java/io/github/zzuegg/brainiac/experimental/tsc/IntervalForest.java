package io.github.zzuegg.brainiac.experimental.tsc;

import io.github.zzuegg.brainiac.experimental.ml.RandomForest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Interval forests: every tree draws its own random intervals and summarises each with a few
 * statistics, so the ensemble searches the space of (interval, statistic) features at random.
 * <ul>
 *   <li>{@link Kind#TSF}: the time series forest (Deng et al., "A time series forest for
 *       classification and feature extraction", Inf. Sci. 2013): {@code sqrt(L)} random intervals
 *       of the series per tree, with mean, standard deviation and slope.</li>
 *   <li>{@link Kind#DRCIF_LITE}: a small version of the Diverse Representation Canonical Interval
 *       Forest (Middlehurst, Large and Bagnall 2020): intervals of the series, its first
 *       difference and its periodogram, nine summary statistics (mean, standard deviation, slope,
 *       median, interquartile range, minimum, maximum, mean crossings, lag-1 autocorrelation) in
 *       place of catch22, and a random subset of statistics per tree.</li>
 * </ul>
 * Each tree is one bootstrapped {@link RandomForest} tree on its own features; probabilities are
 * averaged.
 */
public final class IntervalForest {

    /** Which forest. */
    public enum Kind {
        /** Time series forest. */
        TSF,
        /** DrCIF with a small fixed feature set. */
        DRCIF_LITE
    }

    private static final int STATS = 9;

    private record Tree(int[][] intervals, int[] stats, RandomForest model) {
    }

    private final Kind kind;
    private final List<Tree> trees;
    private final int classes;

    private IntervalForest(Kind kind, List<Tree> trees, int classes) {
        this.kind = kind;
        this.trees = trees;
        this.classes = classes;
    }

    /**
     * Fits the forest.
     *
     * @param series training series (one length)
     * @param labels their classes
     * @param classes the number of classes
     * @param kind which forest
     * @param trees the number of trees, such as 100
     * @param seed the seed
     * @return the forest
     */
    public static IntervalForest fit(float[][] series, int[] labels, int classes, Kind kind, int trees, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<double[][]> reps = new ArrayList<>();
        for (float[] s : series) {
            reps.add(representations(s, kind));
        }
        List<Tree> out = new ArrayList<>();
        for (int t = 0; t < trees; t++) {
            int nRep = reps.get(0).length;
            int perRep = Math.max(1, (int) Math.sqrt(series[0].length) / (kind == Kind.TSF ? 1 : 2));
            int[][] intervals = new int[nRep * perRep][3];
            for (int k = 0; k < intervals.length; k++) {
                int rep = k / perRep;
                int len = reps.get(0)[rep].length;
                int minLen = Math.min(3, len);
                int l = minLen + r.nextInt(Math.max(1, len - minLen + 1));
                int start = r.nextInt(Math.max(1, len - l + 1));
                intervals[k] = new int[] {rep, start, l};
            }
            int[] stats;
            if (kind == Kind.TSF) {
                stats = new int[] {0, 1, 2};
            } else {
                int[] all = {0, 1, 2, 3, 4, 5, 6, 7, 8};
                for (int i = all.length - 1; i > 0; i--) {
                    int j = r.nextInt(i + 1);
                    int tmp = all[i];
                    all[i] = all[j];
                    all[j] = tmp;
                }
                stats = Arrays.copyOf(all, 5);
            }
            double[][] x = new double[series.length][];
            for (int i = 0; i < series.length; i++) {
                x[i] = features(reps.get(i), intervals, stats);
            }
            RandomForest m = RandomForest.fit(x, labels, classes, new RandomForest.Settings(
                    RandomForest.Mode.RANDOM_FOREST, 1, 30, 1, 0, r.nextLong()));
            out.add(new Tree(intervals, stats, m));
        }
        return new IntervalForest(kind, out, classes);
    }

    private static double[][] representations(float[] s, Kind kind) {
        double[] x = new double[s.length];
        for (int t = 0; t < s.length; t++) {
            x[t] = s[t];
        }
        if (kind == Kind.TSF) {
            return new double[][] {x};
        }
        List<double[]> q = Quant.representations(s);
        double[] mag = q.get(3);
        double[] power = new double[mag.length];
        for (int k = 0; k < mag.length; k++) {
            power[k] = Math.log1p(mag[k] * mag[k]);
        }
        return new double[][] {x, q.get(1), power};
    }

    private static double[] features(double[][] reps, int[][] intervals, int[] stats) {
        double[] out = new double[intervals.length * stats.length];
        int f = 0;
        for (int[] iv : intervals) {
            double[] all = statistics(reps[iv[0]], iv[1], iv[2]);
            for (int s : stats) {
                out[f++] = all[s];
            }
        }
        return out;
    }

    /** The nine interval statistics. */
    static double[] statistics(double[] r, int start, int len) {
        double[] v = Arrays.copyOfRange(r, start, start + len);
        int n = v.length;
        double mean = 0;
        for (double a : v) {
            mean += a / n;
        }
        double var = 0;
        double sxy = 0;
        double sxx = 0;
        double tm = (n - 1) / 2.0;
        for (int t = 0; t < n; t++) {
            var += (v[t] - mean) * (v[t] - mean);
            sxy += (t - tm) * (v[t] - mean);
            sxx += (t - tm) * (t - tm);
        }
        double sd = Math.sqrt(var / Math.max(1, n - 1));
        double slope = sxx > 0 ? sxy / sxx : 0;
        int crossings = 0;
        double ac = 0;
        for (int t = 1; t < n; t++) {
            if ((v[t] - mean) * (v[t - 1] - mean) < 0) {
                crossings++;
            }
            ac += (v[t] - mean) * (v[t - 1] - mean);
        }
        ac = var > 0 ? ac / var : 0;
        double[] sorted = v.clone();
        Arrays.sort(sorted);
        double median = sorted[n / 2];
        double iqr = sorted[Math.min(n - 1, 3 * n / 4)] - sorted[n / 4];
        return new double[] {mean, sd, slope, median, iqr, sorted[0], sorted[n - 1], crossings, ac};
    }

    /**
     * Class probabilities, averaged over trees.
     *
     * @param series the series
     * @return {@code [n][classes]}
     */
    public double[][] probabilities(float[][] series) {
        double[][] out = new double[series.length][classes];
        for (int i = 0; i < series.length; i++) {
            double[][] reps = representations(series[i], kind);
            for (Tree t : trees) {
                double[] p = t.model().predictProbability(features(reps, t.intervals(), t.stats()));
                for (int c = 0; c < classes; c++) {
                    out[i][c] += p[c] / trees.size();
                }
            }
        }
        return out;
    }

    /**
     * Predicts classes.
     *
     * @param series the series
     * @return the classes
     */
    public int[] predict(float[][] series) {
        return WideRidgeClassifier.argmax(probabilities(series));
    }

    /** @return the number of interval statistics available to {@link Kind#DRCIF_LITE} */
    public static int statistics() {
        return STATS;
    }
}
