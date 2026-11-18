package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;
import io.github.zzuegg.brainiac.experimental.timeseries.ExponentialSmoothing;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Intermittent demand forecasting (wave 11, item 11.3.16): Croston, SBA, TSB, ADIDA and IMAPA
 * ({@link io.github.zzuegg.brainiac.experimental.timeseries.IntermittentDemand}, smoothing weight
 * 0.1, TSB probability weight 0.05) against simple exponential smoothing and Holt's method
 * ({@link ExponentialSmoothing}, fitted by maximum likelihood) and the zero forecast, on
 * {@link SeasonalSeries#intermittent} series: 100 training periods and a flat forecast for the
 * next 24.
 *
 * <p>Scores: scaled MAE (MAE over the in-sample mean demand), RMSSE (RMSE over the in-sample
 * root mean squared one-step naive change) and the cumulative bias
 * {@code (sum forecast - sum actual) / (h * in-sample mean)}, averaged over series. Scenarios:
 * stationary demand, a halving of the occurrence rate 20 periods before the origin, and
 * obsolescence (no demand from 30 periods before the origin).
 */
public final class IntermittentDemand {

    private IntermittentDemand() {
    }

    /** A demand scenario. */
    public enum Scenario {
        /** Stationary demand. */
        STATIONARY,
        /** The occurrence probability halves 20 periods before the origin. */
        RATE_DROP,
        /** No demand from 30 periods before the origin. */
        OBSOLESCENCE
    }

    /**
     * Runs the comparison.
     *
     * @param scenario the scenario
     * @param kind the demand-size distribution
     * @param zeroShare the share of zero periods
     * @param series simulated series
     * @param seed the seed
     * @return method to {@code {scaled MAE, RMSSE, cumulative bias}}
     */
    public static Map<String, double[]> compare(Scenario scenario, SeasonalSeries.Demand kind, double zeroShare,
            int series, long seed) {
        int train = 100;
        int horizon = 24;
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int s = 0; s < series; s++) {
            int change = scenario == Scenario.RATE_DROP ? train - 20 : -1;
            int obsolete = scenario == Scenario.OBSOLESCENCE ? train - 30 : -1;
            float[] all = SeasonalSeries.intermittent(train + horizon, kind, zeroShare, change, 0.5, obsolete,
                    seed + s);
            double[] y = new double[train];
            for (int t = 0; t < train; t++) {
                y[t] = all[t];
            }
            double mean = 0;
            double naive = 0;
            for (int t = 0; t < train; t++) {
                mean += y[t] / train;
                if (t > 0) {
                    naive += (y[t] - y[t - 1]) * (y[t] - y[t - 1]) / (train - 1);
                }
            }
            if (mean == 0) {
                continue;
            }
            Map<String, Double> f = new LinkedHashMap<>();
            f.put("zero forecast", 0.0);
            f.put("SES (ETS A,N,N)", ExponentialSmoothing.fit(y, ExponentialSmoothing.Trend.NONE, 1, 15)
                    .forecast(y, 1).mean()[0]);
            double[] holt = ExponentialSmoothing.fit(y, ExponentialSmoothing.Trend.ADDITIVE, 1, 15).forecast(y,
                    horizon).mean();
            for (io.github.zzuegg.brainiac.experimental.timeseries.IntermittentDemand.Method m
                    : io.github.zzuegg.brainiac.experimental.timeseries.IntermittentDemand.Method.values()) {
                f.put(m.name(), io.github.zzuegg.brainiac.experimental.timeseries.IntermittentDemand.forecast(y, m,
                        0.1, 0.05));
            }
            for (Map.Entry<String, Double> e : f.entrySet()) {
                score(out, e.getKey(), constant(e.getValue(), horizon), all, train, mean, naive, series);
            }
            score(out, "Holt (ETS A,A,N)", holt, all, train, mean, naive, series);
        }
        return out;
    }

    private static double[] constant(double v, int h) {
        double[] a = new double[h];
        java.util.Arrays.fill(a, v);
        return a;
    }

    private static void score(Map<String, double[]> out, String key, double[] f, float[] all, int train, double mean,
            double naive, int series) {
        double mae = 0;
        double mse = 0;
        double bias = 0;
        for (int h = 0; h < f.length; h++) {
            double e = f[h] - all[train + h];
            mae += Math.abs(e) / f.length;
            mse += e * e / f.length;
            bias += e;
        }
        double[] acc = out.computeIfAbsent(key, k -> new double[3]);
        acc[0] += mae / mean / series;
        acc[1] += Math.sqrt(mse / Math.max(naive, 1e-12)) / series;
        acc[2] += bias / (f.length * mean) / series;
    }
}
