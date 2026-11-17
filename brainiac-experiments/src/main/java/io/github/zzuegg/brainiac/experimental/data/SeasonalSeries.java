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
}
