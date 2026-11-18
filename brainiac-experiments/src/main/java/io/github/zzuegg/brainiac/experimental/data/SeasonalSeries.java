package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Synthetic univariate time series for forecasting experiments (level + trend + two
 * seasonalities + noise), sliding-window datasets cut from them, and the two classic
 * baselines every forecaster must beat: <b>naive</b> (repeat the last value) and
 * <b>seasonal naive</b> (repeat the last season).
 */
public final class SeasonalSeries {

    private SeasonalSeries() {
    }

    /**
     * Supervised windows.
     *
     * @param inputs {@code [windows, lookback]}
     * @param targets {@code [windows, horizon]}
     * @param starts each window's start index in the series
     */
    public record Windows(Tensor inputs, Tensor targets, int[] starts) {
        /** @return the number of windows */
        public int size() {
            return starts.length;
        }
    }

    /**
     * Generates a series {@code level + slope t + a1 sin(2 pi t / period) + a2 sin(2 pi t / (period / 2)) + noise}.
     *
     * @param length the number of steps
     * @param period the main season length
     * @param slope the trend per step
     * @param amplitude the main seasonal amplitude (the half-period harmonic gets 40% of it)
     * @param noise the standard deviation of Gaussian noise
     * @param seed the seed
     * @return the series
     */
    public static float[] generate(int length, int period, double slope, double amplitude, double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double level = r.nextDouble(-1, 1);
        double phase = r.nextDouble(0, 2 * Math.PI);
        float[] y = new float[length];
        for (int t = 0; t < length; t++) {
            double w = 2 * Math.PI * t / period + phase;
            y[t] = (float) (level + slope * t + amplitude * Math.sin(w) + 0.4 * amplitude * Math.sin(2 * w + 1)
                    + noise * r.nextGaussian());
        }
        return y;
    }

    /**
     * A series with two incommensurate seasonalities (no common period within the window),
     * {@code sin(2 pi t / p1) + 0.7 sin(2 pi t / p2) + noise}, the case TimesNet's multi-period
     * folding is built for.
     *
     * @param length steps
     * @param p1 the first period
     * @param p2 the second period
     * @param noise the noise standard deviation
     * @param seed the seed
     * @return the series
     */
    public static float[] twoPeriods(int length, double p1, double p2, double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double a = r.nextDouble(0, 2 * Math.PI);
        double b = r.nextDouble(0, 2 * Math.PI);
        float[] y = new float[length];
        for (int t = 0; t < length; t++) {
            y[t] = (float) (Math.sin(2 * Math.PI * t / p1 + a) + 0.7 * Math.sin(2 * Math.PI * t / p2 + b)
                    + noise * r.nextGaussian());
        }
        return y;
    }

    /**
     * Cuts sliding windows.
     *
     * @param series the series
     * @param from the first window start
     * @param to windows must end before this index
     * @param lookback input steps
     * @param horizon target steps
     * @param stride steps between window starts
     * @return the windows
     */
    public static Windows windows(float[] series, int from, int to, int lookback, int horizon, int stride) {
        int count = Math.max(0, (to - from - lookback - horizon) / stride + 1);
        float[] x = new float[count * lookback];
        float[] y = new float[count * horizon];
        int[] starts = new int[count];
        for (int w = 0; w < count; w++) {
            int s = from + w * stride;
            starts[w] = s;
            System.arraycopy(series, s, x, w * lookback, lookback);
            System.arraycopy(series, s + lookback, y, w * horizon, horizon);
        }
        return new Windows(Tensor.of(Shape.of(count, lookback), x), Tensor.of(Shape.of(count, horizon), y), starts);
    }

    /**
     * @param inputs {@code [n, lookback]}
     * @param horizon steps to forecast
     * @return the naive forecast, the last observed value repeated
     */
    public static Tensor naive(Tensor inputs, int horizon) {
        int l = inputs.shape().dim(1);
        return inputs.slice(1, l - 1, l).expand(Shape.of(inputs.shape().dim(0), horizon));
    }

    /**
     * @param inputs {@code [n, lookback]}, with {@code lookback >= period}
     * @param horizon steps to forecast
     * @param period the season length
     * @return the seasonal naive forecast, the last full season repeated
     */
    public static Tensor seasonalNaive(Tensor inputs, int horizon, int period) {
        int n = inputs.shape().dim(0);
        int l = inputs.shape().dim(1);
        float[] x = inputs.toArray();
        float[] y = new float[n * horizon];
        for (int i = 0; i < n; i++) {
            for (int h = 0; h < horizon; h++) {
                y[i * horizon + h] = x[i * l + l - period + (h % period)];
            }
        }
        return Tensor.of(Shape.of(n, horizon), y);
    }

    /**
     * Multivariate windows.
     *
     * @param inputs {@code [windows, lookback, channels]}
     * @param targets {@code [windows, horizon, channels]}
     * @param starts each window's start index in the series
     */
    public record MultiWindows(Tensor inputs, Tensor targets, int[] starts) {
        /** @return the number of windows */
        public int size() {
            return starts.length;
        }
    }

    /**
     * A multivariate seasonal series. Channel 0 is a seasonal series plus a persistent AR(1)
     * component (unpredictable from its own past beyond a few steps). When {@code coupled},
     * channel {@code j > 0} copies channel {@code j - 1} delayed by {@code lag} steps (plus a
     * little noise), so a model that reads the other channels sees each follower's future
     * {@code lag} steps early; otherwise every channel is an independent draw of the same
     * process (the control).
     *
     * @param length steps
     * @param channels channels, at least 1
     * @param period the season length
     * @param lag the cross-channel delay
     * @param coupled whether the channels are lagged copies of each other
     * @param noise observation noise
     * @param seed the seed
     * @return {@code [channels][length]}
     */
    public static float[][] multivariate(int length, int channels, int period, int lag, boolean coupled, double noise,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int burn = lag * channels;
        float[][] raw = new float[channels][];
        for (int c = 0; c < channels; c++) {
            raw[c] = ar1Seasonal(length + burn, period, r.split());
        }
        float[][] y = new float[channels][length];
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < length; t++) {
                double v = coupled ? raw[0][t + burn - c * lag] : raw[c][t + burn];
                y[c][t] = (float) (v + noise * r.nextGaussian());
            }
        }
        return y;
    }

    private static float[] ar1Seasonal(int length, int period, SplittableRandom r) {
        double phase = r.nextDouble(0, 2 * Math.PI);
        double ar = 0;
        float[] y = new float[length];
        for (int t = 0; t < length; t++) {
            ar = 0.95 * ar + 0.25 * r.nextGaussian();
            y[t] = (float) (0.6 * Math.sin(2 * Math.PI * t / period + phase) + ar);
        }
        return y;
    }

    /**
     * Cuts multivariate sliding windows.
     *
     * @param series {@code [channels][length]}
     * @param from the first window start
     * @param to windows must end before this index
     * @param lookback input steps
     * @param horizon target steps
     * @param stride steps between window starts
     * @return the windows
     */
    public static MultiWindows windows(float[][] series, int from, int to, int lookback, int horizon, int stride) {
        int c = series.length;
        int count = Math.max(0, (to - from - lookback - horizon) / stride + 1);
        float[] x = new float[count * lookback * c];
        float[] y = new float[count * horizon * c];
        int[] starts = new int[count];
        for (int w = 0; w < count; w++) {
            int s = from + w * stride;
            starts[w] = s;
            for (int t = 0; t < lookback; t++) {
                for (int k = 0; k < c; k++) {
                    x[(w * lookback + t) * c + k] = series[k][s + t];
                }
            }
            for (int t = 0; t < horizon; t++) {
                for (int k = 0; k < c; k++) {
                    y[(w * horizon + t) * c + k] = series[k][s + lookback + t];
                }
            }
        }
        return new MultiWindows(Tensor.of(Shape.of(count, lookback, c), x), Tensor.of(Shape.of(count, horizon, c), y),
                starts);
    }

    /**
     * A demand series driven by known future covariates.
     *
     * @param target the demand
     * @param price the price, a step function (known in advance)
     * @param holiday 1 on holidays (known in advance), else 0
     */
    public record Covariates(float[] target, float[] price, float[] holiday) {
        /** @return the number of steps */
        public int length() {
            return target.length;
        }
    }

    /**
     * Demand {@code = seasonality - 1.2 price + 2 holiday + noise}: the price is a step function
     * that changes every 40-80 steps, holidays are single-step spikes every 15-35 steps. Both
     * covariates are known for the forecast horizon, so a model that reads them can forecast
     * price steps and holiday spikes that no lookback window announces.
     *
     * @param length steps
     * @param period the season length
     * @param noise observation noise
     * @param seed the seed
     * @return the series and its covariates
     */
    public static Covariates covariates(int length, int period, double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] base = generate(length, period, 0.0, 0.8, 0.0, seed + 7);
        float[] price = new float[length];
        float[] holiday = new float[length];
        float[] y = new float[length];
        double p = r.nextDouble(-0.5, 0.5);
        int nextChange = 40 + r.nextInt(40);
        int nextHoliday = 15 + r.nextInt(20);
        for (int t = 0; t < length; t++) {
            if (t == nextChange) {
                p = r.nextDouble(-0.8, 0.8);
                nextChange += 40 + r.nextInt(40);
            }
            if (t == nextHoliday) {
                holiday[t] = 1;
                nextHoliday += 15 + r.nextInt(20);
            }
            price[t] = (float) p;
            y[t] = (float) (base[t] - 1.2 * p + 2.0 * holiday[t] + noise * r.nextGaussian());
        }
        return new Covariates(y, price, holiday);
    }

    /**
     * @param prediction a forecast
     * @param target the truth
     * @return the mean squared error
     */
    public static double mse(Tensor prediction, Tensor target) {
        return prediction.sub(target).square().mean().item();
    }

    /**
     * @param prediction a forecast
     * @param target the truth
     * @return the mean absolute error
     */
    public static double mae(Tensor prediction, Tensor target) {
        return prediction.sub(target).abs().mean().item();
    }

    /** The kind of intermittent demand (wave 11, 11.3.16). */
    public enum Demand {
        /** Bernoulli occurrence with gamma sizes (shape 2), rounded up to whole units. */
        POISSON_GAMMA,
        /** Bernoulli occurrence with log-normal sizes ({@code sigma = 0.6}), rounded up. */
        BERNOULLI_LOGNORMAL
    }

    /**
     * An intermittent demand series: demand occurs with probability {@code 1 - zeroShare} in each
     * period (the rate after {@code changeAt} is multiplied by {@code rateChange}), with mean
     * size 5, and stops for good from {@code obsoleteAt}.
     *
     * @param length periods
     * @param kind the size distribution
     * @param zeroShare the share of zero periods before any change, for example 0.5 to 0.95
     * @param changeAt the period where the occurrence probability changes ({@code -1} for never)
     * @param rateChange the factor on the occurrence probability from {@code changeAt}
     * @param obsoleteAt the first period of permanent zero demand ({@code -1} for never)
     * @param seed the seed
     * @return the series
     */
    public static float[] intermittent(int length, Demand kind, double zeroShare, int changeAt, double rateChange,
            int obsoleteAt, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] y = new float[length];
        for (int t = 0; t < length; t++) {
            double p = 1 - zeroShare;
            if (changeAt >= 0 && t >= changeAt) {
                p = Math.min(1, p * rateChange);
            }
            if ((obsoleteAt >= 0 && t >= obsoleteAt) || r.nextDouble() >= p) {
                continue;
            }
            double size = switch (kind) {
                case POISSON_GAMMA -> 2.5 * (-Math.log(1 - r.nextDouble()) - Math.log(1 - r.nextDouble()));
                case BERNOULLI_LOGNORMAL -> Math.exp(Math.log(5) - 0.18 + 0.6 * r.nextGaussian());
            };
            y[t] = (float) Math.max(1, Math.ceil(size));
        }
        return y;
    }
}
