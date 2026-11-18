package io.github.zzuegg.brainiac.experimental.kernels;

import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The AutoTVM cost model (Chen et al. 2018, <a href="https://arxiv.org/abs/1805.08166">
 * arXiv:1805.08166</a>): gradient-boosted trees (the core {@link GradientBoostedTrees}) that
 * regress the measured throughput of a schedule from its loop-structure features
 * ({@link ScheduleSpace#features}). Only the ranking matters, so the model is scored by
 * Spearman rank correlation ({@link #spearman}).
 */
public final class LearnedCostModel {

    private final ScheduleSpace space;
    private final List<double[]> x = new ArrayList<>();
    private final List<Double> y = new ArrayList<>();
    private GradientBoostedTrees model;

    /** @param space the schedule space whose features it learns from */
    public LearnedCostModel(ScheduleSpace space) {
        this.space = space;
    }

    /**
     * Adds a measurement (the model is refitted by {@link #fit}).
     *
     * @param s the schedule
     * @param gflops its measured throughput
     */
    public void add(ScheduleSpace.Schedule s, double gflops) {
        x.add(space.features(s));
        y.add(gflops);
    }

    /** Refits the trees on all measurements. */
    public void fit() {
        double[] target = new double[y.size()];
        for (int i = 0; i < target.length; i++) {
            target[i] = y.get(i);
        }
        GradientBoostedTrees.Settings settings = GradientBoostedTrees.Settings.defaults().withTrees(60).withDepth(3);
        model = GradientBoostedTrees.fit(x.toArray(new double[0][]), target, GradientBoostedTrees.Objective.REGRESSION,
                settings);
    }

    /**
     * @param s a schedule
     * @return the predicted GFLOP/s (0 before the first fit)
     */
    public double predict(ScheduleSpace.Schedule s) {
        return model == null ? 0 : model.predict(space.features(s));
    }

    /** @return the number of measurements */
    public int size() {
        return y.size();
    }

    /**
     * Spearman rank correlation.
     *
     * @param a one series
     * @param b the other
     * @return the correlation of their ranks
     */
    public static double spearman(double[] a, double[] b) {
        double[] ra = ranks(a);
        double[] rb = ranks(b);
        double ma = Arrays.stream(ra).average().orElse(0);
        double mb = Arrays.stream(rb).average().orElse(0);
        double num = 0;
        double da = 0;
        double db = 0;
        for (int i = 0; i < a.length; i++) {
            num += (ra[i] - ma) * (rb[i] - mb);
            da += (ra[i] - ma) * (ra[i] - ma);
            db += (rb[i] - mb) * (rb[i] - mb);
        }
        return num / Math.sqrt(da * db + 1e-300);
    }

    private static double[] ranks(double[] v) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (p, q) -> Double.compare(v[p], v[q]));
        double[] r = new double[v.length];
        int i = 0;
        while (i < idx.length) {
            int j = i;
            while (j + 1 < idx.length && v[idx[j + 1]] == v[idx[i]]) {
                j++;
            }
            for (int t = i; t <= j; t++) {
                r[idx[t]] = (i + j) / 2.0;
            }
            i = j + 1;
        }
        return r;
    }
}
