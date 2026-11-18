package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;
import io.github.zzuegg.brainiac.experimental.nn.DLinear;
import io.github.zzuegg.brainiac.experimental.timeseries.AutoEts;
import io.github.zzuegg.brainiac.experimental.timeseries.ExponentialSmoothing;
import io.github.zzuegg.brainiac.experimental.timeseries.StateSpaceForecast;
import io.github.zzuegg.brainiac.experimental.timeseries.ThetaMethod;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 12 item 12.2.9: fast statistical forecasting on many short seasonal series. The existing
 * L-BFGS fit of {@link ExponentialSmoothing} is compared with its new coarse-to-fine grid fit
 * ({@link ExponentialSmoothing.Method#GRID_WARM}) for speed and accuracy, and both with
 * {@link AutoEts} (AICc selection over ETS(A/M, N/A/Ad, N/A/M)), the {@link ThetaMethod}, seasonal
 * naive and a global DLinear trained on windows of all series. Accuracy is MASE at the end of each
 * series (scaled by the in-sample seasonal-naive error); intervals are scored by empirical
 * coverage at 80% and 95%.
 */
public final class FastStatisticalForecasts {

    private FastStatisticalForecasts() {
    }

    /**
     * One forecaster on the panel.
     *
     * @param method the forecaster
     * @param series the series it was run on
     * @param mase the mean MASE over those series
     * @param coverage80 the empirical coverage of its 80% intervals ({@code NaN} if it has none)
     * @param coverage95 the empirical coverage of its 95% intervals
     * @param millisPerSeries the fit and forecast time per series
     */
    public record ForecastRow(String method, int series, double mase, double coverage80, double coverage95,
            double millisPerSeries) {
    }

    private static final int PERIOD = 12;
    private static final double Z80 = 1.2815515655446004;
    private static final double Z95 = 1.959963984540054;

    /**
     * A panel of positive seasonal series with a trend, a seasonal pattern and multiplicative noise.
     *
     * @param count series
     * @param length steps per series (history plus horizon)
     * @param seed the seed
     * @return {@code [count][length]}
     */
    public static double[][] panel(int count, int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] y = new double[count][];
        for (int i = 0; i < count; i++) {
            double slope = r.nextDouble(-0.03, 0.06);
            double amplitude = r.nextDouble(0.3, 2.0);
            double noise = r.nextDouble(0.02, 0.08);
            float[] base = SeasonalSeries.generate(length, PERIOD, slope, amplitude, 0, r.nextLong());
            y[i] = new double[length];
            for (int t = 0; t < length; t++) {
                y[i][t] = (10 + base[t]) * (1 + noise * r.nextGaussian());
            }
        }
        return y;
    }

    private static final class Score {
        private final String method;
        private double mase;
        private int hits80;
        private int hits95;
        private int points;
        private int series;
        private long nanos;
        private boolean intervals;

        Score(String method) {
            this.method = method;
        }

        void add(double[] history, double[] actual, double[] forecast, double[][] i80, double[][] i95, long time) {
            double scale = 0;
            for (int t = PERIOD; t < history.length; t++) {
                scale += Math.abs(history[t] - history[t - PERIOD]);
            }
            scale /= history.length - PERIOD;
            double err = 0;
            for (int h = 0; h < actual.length; h++) {
                err += Math.abs(actual[h] - forecast[h]);
                if (i80 != null) {
                    intervals = true;
                    hits80 += actual[h] >= i80[0][h] && actual[h] <= i80[1][h] ? 1 : 0;
                    hits95 += actual[h] >= i95[0][h] && actual[h] <= i95[1][h] ? 1 : 0;
                }
            }
            mase += err / actual.length / scale;
            points += actual.length;
            series++;
            nanos += time;
        }

        ForecastRow row() {
            return new ForecastRow(method, series, mase / series, intervals ? (double) hits80 / points : Double.NaN,
                    intervals ? (double) hits95 / points : Double.NaN, nanos / 1e6 / series);
        }
    }

    private static double[][] gaussian(double[] mean, double[] variance, double z) {
        double[][] out = new double[2][mean.length];
        for (int h = 0; h < mean.length; h++) {
            double sd = Math.sqrt(Math.max(0, variance[h]));
            out[0][h] = mean[h] - z * sd;
            out[1][h] = mean[h] + z * sd;
        }
        return out;
    }

    /**
     * Runs the comparison.
     *
     * @param count series in the panel
     * @param history steps of history per series
     * @param horizon steps forecast
     * @param lbfgsSeries how many series the slow L-BFGS fit is run on (the first ones)
     * @param seed the seed
     * @return one row per method; the L-BFGS rows and a grid-fit row on the same subset come first
     */
    public static List<ForecastRow> run(int count, int history, int horizon, int lbfgsSeries, long seed) {
        double[][] panel = panel(count, history + horizon, seed);
        Score lbfgs = new Score("ETS(A,Ad,A) L-BFGS fit (existing)");
        Score gridSubset = new Score("ETS(A,Ad,A) grid fit, same series");
        Score grid = new Score("ETS(A,Ad,A) grid fit (GRID_WARM)");
        Score auto = new Score("AutoETS (AICc over 18 models)");
        Score theta = new Score("Theta method");
        Score naive = new Score("seasonal naive");
        for (int i = 0; i < count; i++) {
            double[] h = Arrays.copyOf(panel[i], history);
            double[] a = Arrays.copyOfRange(panel[i], history, history + horizon);
            long t0 = System.nanoTime();
            ExponentialSmoothing g = ExponentialSmoothing.fit(h, ExponentialSmoothing.Trend.DAMPED, PERIOD,
                    ExponentialSmoothing.Method.GRID_WARM, 3);
            StateSpaceForecast gf = g.forecast(h, horizon);
            long t1 = System.nanoTime();
            grid.add(h, a, gf.mean(), gaussian(gf.mean(), gf.variance(), Z80), gaussian(gf.mean(), gf.variance(), Z95),
                    t1 - t0);
            if (i < lbfgsSeries) {
                gridSubset.add(h, a, gf.mean(), gaussian(gf.mean(), gf.variance(), Z80),
                        gaussian(gf.mean(), gf.variance(), Z95), t1 - t0);
                long l0 = System.nanoTime();
                ExponentialSmoothing l = ExponentialSmoothing.fit(h, ExponentialSmoothing.Trend.DAMPED, PERIOD, 100);
                StateSpaceForecast lf = l.forecast(h, horizon);
                long l1 = System.nanoTime();
                lbfgs.add(h, a, lf.mean(), gaussian(lf.mean(), lf.variance(), Z80),
                        gaussian(lf.mean(), lf.variance(), Z95), l1 - l0);
            }
            long a0 = System.nanoTime();
            AutoEts e = AutoEts.select(h, PERIOD);
            double[] ef = e.forecast(horizon);
            double[][] e80 = e.intervals(horizon, 0.8, 300, seed + i);
            double[][] e95 = e.intervals(horizon, 0.95, 300, seed + i);
            long a1 = System.nanoTime();
            auto.add(h, a, ef, e80, e95, a1 - a0);
            long th0 = System.nanoTime();
            ThetaMethod th = ThetaMethod.fit(h, PERIOD);
            double[] tf = th.forecast(horizon);
            double[][] t80 = th.intervals(horizon, Z80);
            double[][] t95 = th.intervals(horizon, Z95);
            long th1 = System.nanoTime();
            theta.add(h, a, tf, t80, t95, th1 - th0);
            double[] sn = new double[horizon];
            for (int k = 0; k < horizon; k++) {
                sn[k] = h[history - PERIOD + k % PERIOD];
            }
            naive.add(h, a, sn, null, null, 0);
        }
        List<ForecastRow> rows = new ArrayList<>();
        if (lbfgsSeries > 0) {
            rows.add(lbfgs.row());
            rows.add(gridSubset.row());
        }
        rows.add(grid.row());
        rows.add(auto.row());
        rows.add(theta.row());
        rows.add(naive.row());
        rows.add(dlinear(panel, history, horizon, seed));
        return rows;
    }

    /** A global DLinear on mean-scaled windows of every series' history. */
    private static ForecastRow dlinear(double[][] panel, int history, int horizon, long seed) {
        int lookback = 3 * PERIOD;
        long t0 = System.nanoTime();
        List<float[]> xs = new ArrayList<>();
        List<float[]> ys = new ArrayList<>();
        for (double[] s : panel) {
            for (int o = 0; o + lookback + horizon <= history; o += 2) {
                double m = 0;
                for (int k = 0; k < lookback; k++) {
                    m += s[o + k];
                }
                m /= lookback;
                float[] x = new float[lookback];
                float[] y = new float[horizon];
                for (int k = 0; k < lookback; k++) {
                    x[k] = (float) (s[o + k] / m);
                }
                for (int k = 0; k < horizon; k++) {
                    y[k] = (float) (s[o + lookback + k] / m);
                }
                xs.add(x);
                ys.add(y);
            }
        }
        DLinear model = new DLinear(lookback, horizon, PERIOD + 1, new SplittableRandom(seed));
        Optimizer opt = Optimizer.adam(0.005f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int b = 64;
        for (int step = 0; step < 600; step++) {
            float[] bx = new float[b * lookback];
            float[] by = new float[b * horizon];
            for (int i = 0; i < b; i++) {
                int j = r.nextInt(xs.size());
                System.arraycopy(xs.get(j), 0, bx, i * lookback, lookback);
                System.arraycopy(ys.get(j), 0, by, i * horizon, horizon);
            }
            Var d = model.forward(Var.of(Tensor.of(Shape.of(b, lookback), bx)), true)
                    .sub(Var.of(Tensor.of(Shape.of(b, horizon), by)));
            d.abs().mean().backward();
            opt.step(model.parameters());
        }
        Score score = new Score("DLinear (global, mean-scaled windows)");
        long train = System.nanoTime() - t0;
        for (double[] s : panel) {
            double[] h = Arrays.copyOf(s, history);
            double[] a = Arrays.copyOfRange(s, history, history + horizon);
            double m = 0;
            float[] x = new float[lookback];
            for (int k = 0; k < lookback; k++) {
                m += h[history - lookback + k];
            }
            m /= lookback;
            for (int k = 0; k < lookback; k++) {
                x[k] = (float) (h[history - lookback + k] / m);
            }
            float[] out = Var.noGrad(() -> model.forward(Var.of(Tensor.of(Shape.of(1, lookback), x)), false))
                    .value().toArray();
            double[] f = new double[horizon];
            for (int k = 0; k < horizon; k++) {
                f[k] = out[k] * m;
            }
            score.add(h, a, f, null, null, train / panel.length);
        }
        return score.row();
    }
}
