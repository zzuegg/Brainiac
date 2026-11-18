package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;
import io.github.zzuegg.brainiac.experimental.nn.DLinear;
import io.github.zzuegg.brainiac.experimental.nn.NBeats;
import io.github.zzuegg.brainiac.experimental.nn.PatchTst;
import io.github.zzuegg.brainiac.experimental.nn.RevIn;
import io.github.zzuegg.brainiac.experimental.nn.TemporalConvNet;
import io.github.zzuegg.brainiac.experimental.timeseries.Arima;
import io.github.zzuegg.brainiac.experimental.timeseries.ExponentialSmoothing;
import io.github.zzuegg.brainiac.experimental.timeseries.NHits;
import io.github.zzuegg.brainiac.experimental.timeseries.StateSpaceForecast;
import io.github.zzuegg.brainiac.experimental.timeseries.Ts2Vec;
import io.github.zzuegg.brainiac.experimental.train.RidgeRegression;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Point forecasting on a synthetic seasonal series with trend: the naive and
 * seasonal-naive baselines against a plain MLP, DLinear, N-BEATS (generic and
 * interpretable), PatchTST and a temporal convolutional network, all trained with MSE on
 * the first part of the series and scored on the last part (which, with a trend, lies
 * outside the training range: instance normalization matters there).
 *
 * <p>Papers: Zeng et al., <a href="https://arxiv.org/abs/2205.13504">arXiv:2205.13504</a> (DLinear);
 * Oreshkin et al., <a href="https://arxiv.org/abs/1905.10437">arXiv:1905.10437</a> (N-BEATS);
 * Nie et al., <a href="https://arxiv.org/abs/2211.14730">arXiv:2211.14730</a> (PatchTST);
 * Bai et al., <a href="https://arxiv.org/abs/1803.01271">arXiv:1803.01271</a> (TCN).
 */
public final class ForecastingBenchmark {

    /**
     * A model's test errors.
     *
     * @param mse mean squared error
     * @param mae mean absolute error
     */
    public record Score(double mse, double mae) {
    }

    private ForecastingBenchmark() {
    }

    private static Score score(Tensor prediction, Tensor target) {
        return new Score(SeasonalSeries.mse(prediction, target), SeasonalSeries.mae(prediction, target));
    }

    private static Score fit(UnaryOperator<Var> model, List<Var> params, SeasonalSeries.Windows train,
            SeasonalSeries.Windows test, int steps, float lr, long seed) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int l = train.inputs().shape().dim(1);
        int h = train.targets().shape().dim(1);
        float[] x = train.inputs().toArray();
        float[] y = train.targets().toArray();
        int b = 32;
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[b * l];
            float[] by = new float[b * h];
            for (int i = 0; i < b; i++) {
                int j = r.nextInt(train.size());
                System.arraycopy(x, j * l, bx, i * l, l);
                System.arraycopy(y, j * h, by, i * h, h);
            }
            model.apply(Var.of(Tensor.of(Shape.of(b, l), bx))).sub(Tensor.of(Shape.of(b, h), by)).square().mean()
                    .backward();
            opt.step(params);
        }
        return score(model.apply(Var.of(test.inputs())).value(), test.targets());
    }

    /**
     * Runs the benchmark.
     *
     * @param lookback input steps
     * @param horizon forecast steps
     * @param period the season length
     * @param slope the trend per step
     * @param noise the noise level
     * @param steps training steps per model
     * @param seed the seed
     * @return model to test errors
     */
    public static Map<String, Score> run(int lookback, int horizon, int period, double slope, double noise, int steps,
            long seed) {
        float[] series = SeasonalSeries.generate(800, period, slope, 1.0, noise, seed);
        SeasonalSeries.Windows train = SeasonalSeries.windows(series, 0, 600, lookback, horizon, 2);
        SeasonalSeries.Windows test = SeasonalSeries.windows(series, 600, 800, lookback, horizon, 4);
        Map<String, Score> out = new LinkedHashMap<>();
        out.put("naive", score(SeasonalSeries.naive(test.inputs(), horizon), test.targets()));
        out.put("seasonal naive", score(SeasonalSeries.seasonalNaive(test.inputs(), horizon, period), test.targets()));
        Model mlp = Model.sequential().input(lookback).dense(64, Activation.RELU).dense(horizon).seed(seed).build();
        out.put("MLP", fit(x -> mlp.forward(x, true), mlp.parameters(), train, test, steps, 0.003f, seed));
        Model mlpRevIn = Model.sequential().input(lookback).dense(64, Activation.RELU).dense(horizon).seed(seed)
                .build();
        out.put("MLP + RevIN", fit(x -> {
            RevIn.Stats s = RevIn.stats(x.value());
            return RevIn.denormalize(mlpRevIn.forward(RevIn.normalize(x, s), true), s);
        }, mlpRevIn.parameters(), train, test, steps, 0.003f, seed));
        DLinear dlinear = new DLinear(lookback, horizon, period + 1 - period % 2, new SplittableRandom(seed));
        out.put("DLinear", fit(x -> dlinear.forward(x, true), dlinear.parameters(), train, test, steps * 3, 0.005f,
                seed));
        NBeats generic = NBeats.generic(lookback, horizon, 2, 64, seed);
        out.put("N-BEATS", fit(x -> generic.forward(x, true), generic.parameters(), train, test, steps, 0.002f, seed));
        NBeats interpretable = NBeats.interpretable(lookback, horizon, 64, seed);
        out.put("N-BEATS interpretable", fit(x -> interpretable.forward(x, true), interpretable.parameters(), train,
                test, steps, 0.002f, seed));
        PatchTst patch = new PatchTst(lookback, horizon, period, period / 2, 16, 1, 2, true,
                new SplittableRandom(seed));
        out.put("PatchTST", fit(x -> patch.forward(x, true), patch.parameters(), train, test, steps / 2, 0.005f,
                seed));
        TemporalConvNet tcn = new TemporalConvNet(1, 16, 3, 4, new SplittableRandom(seed));
        Var head = Var.param(Tensor.normal(Shape.of(16, horizon), 0, 0.25f, new SplittableRandom(seed + 1)));
        List<Var> tcnParams = new ArrayList<>(tcn.parameters());
        tcnParams.add(head);
        out.put("TCN + RevIN", fit(x -> {
            int b = x.shape().dim(0);
            RevIn.Stats s = RevIn.stats(x.value());
            Var hidden = tcn.forward(RevIn.normalize(x, s).reshape(b, lookback, 1), true);
            return RevIn.denormalize(hidden.slice(1, lookback - 1, lookback).reshape(b, 16).matmul(head), s);
        }, tcnParams, train, test, steps / 2, 0.005f, seed));
        return out;
    }

    // ---------------------------------------------------------------- wave 8 rows

    private static void train(UnaryOperator<Var> model, List<Var> params, Tensor x, Tensor y, int steps, float lr,
            boolean absolute, long seed) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int n = x.shape().dim(0);
        for (int s = 0; s < steps; s++) {
            float[] idx = new float[32];
            for (int i = 0; i < 32; i++) {
                idx[i] = r.nextInt(n);
            }
            Tensor sel = Tensor.of(Shape.of(32), idx);
            Var d = model.apply(Var.of(x.gather(0, sel))).sub(y.gather(0, sel));
            (absolute ? d.abs() : d.square()).mean().backward();
            opt.step(params);
        }
    }

    private static Tensor stateSpaceForecasts(float[] series, int from, int lookback, int horizon, int count,
            int stride, java.util.function.BiFunction<double[], Integer, StateSpaceForecast> f) {
        float[] out = new float[count * horizon];
        for (int i = 0; i < count; i++) {
            int origin = from + i * stride + lookback;
            double[] history = new double[origin];
            for (int t = 0; t < origin; t++) {
                history[t] = series[t];
            }
            double[] m = f.apply(history, horizon).mean();
            for (int h = 0; h < horizon; h++) {
                out[i * horizon + h] = (float) m[h];
            }
        }
        return Tensor.of(Shape.of(count, horizon), out);
    }

    /**
     * The wave 8 rows on the {@link #run} task (trend + season + noise, train {@code [0, 600)},
     * test windows over {@code [600, 800)}): ETS and ARIMA with AICc selection (Kalman
     * likelihood, L-BFGS; ARIMA with one seasonal difference), N-HiTS, an N-BEATS ensemble
     * (generic N-BEATS at two lookbacks times MSE and MAE losses, median-aggregated as in the
     * paper), and TS2Vec representations with a ridge head (self-supervised on the training
     * part only), against seasonal naive, DLinear, PatchTST and a single N-BEATS.
     *
     * @param lookback input steps
     * @param horizon forecast steps
     * @param period the season length
     * @param steps training steps per learned model
     * @param iterations L-BFGS iterations per state-space fit
     * @param maxOrder the largest AR and MA orders ARIMA selection tries (seasonal MA up to 1)
     * @param seed the seed
     * @return model to test error
     */
    public static Map<String, Score> extended(int lookback, int horizon, int period, int steps, int iterations,
            int maxOrder, long seed) {
        float[] series = SeasonalSeries.generate(800, period, 0.004, 1.0, 0.3, seed);
        SeasonalSeries.Windows train = SeasonalSeries.windows(series, 0, 600, lookback, horizon, 2);
        SeasonalSeries.Windows test = SeasonalSeries.windows(series, 600, 800, lookback, horizon, 4);
        Map<String, Score> out = new LinkedHashMap<>();
        out.put("seasonal naive", score(SeasonalSeries.seasonalNaive(test.inputs(), horizon, period), test.targets()));
        DLinear dlinear = new DLinear(lookback, horizon, period + 1 - period % 2, new SplittableRandom(seed));
        out.put("DLinear", fit(x -> dlinear.forward(x, true), dlinear.parameters(), train, test, steps * 3, 0.005f,
                seed));
        PatchTst patch = new PatchTst(lookback, horizon, period, period / 2, 16, 1, 2, true,
                new SplittableRandom(seed));
        out.put("PatchTST", fit(x -> patch.forward(x, true), patch.parameters(), train, test, steps / 2, 0.005f,
                seed));
        double[] fitPart = new double[600];
        for (int t = 0; t < 600; t++) {
            fitPart[t] = series[t];
        }
        ExponentialSmoothing ets = ExponentialSmoothing.select(fitPart, period, iterations);
        out.put("ETS", score(stateSpaceForecasts(series, 600, lookback, horizon, test.size(), 4, ets::forecast),
                test.targets()));
        Arima arima = Arima.select(fitPart, maxOrder, maxOrder, period, 1, 1, iterations);
        out.put("ARIMA", score(stateSpaceForecasts(series, 600, lookback, horizon, test.size(), 4, arima::forecast),
                test.targets()));
        NHits nhits = NHits.standard(lookback, horizon, 64, seed);
        out.put("N-HiTS", fit(x -> nhits.forward(x, true), nhits.parameters(), train, test, steps, 0.002f, seed));
        NBeats single = NBeats.generic(lookback, horizon, 2, 64, seed);
        out.put("N-BEATS", fit(x -> single.forward(x, true), single.parameters(), train, test, steps, 0.002f, seed));

        List<Tensor> members = new ArrayList<>();
        int k = 0;
        for (int l : new int[] {lookback, lookback / 2}) {
            for (boolean absolute : new boolean[] {false, true}) {
                NBeats m = NBeats.generic(l, horizon, 2, 64, seed + 10 + k++);
                int from = lookback - l;
                train(x -> m.forward(x.slice(1, from, lookback), true), m.parameters(), train.inputs(),
                        train.targets(), steps, 0.002f, absolute, seed + k);
                members.add(Var.noGrad(() -> m.forward(Var.of(test.inputs().slice(1, from, lookback)), false))
                        .value());
            }
        }
        int n = test.size();
        float[][] all = new float[members.size()][];
        for (int i = 0; i < all.length; i++) {
            all[i] = members.get(i).toArray();
        }
        float[] median = new float[n * horizon];
        float[] col = new float[all.length];
        for (int j = 0; j < median.length; j++) {
            for (int i = 0; i < all.length; i++) {
                col[i] = all[i][j];
            }
            java.util.Arrays.sort(col);
            median[j] = (col[(col.length - 1) / 2] + col[col.length / 2]) / 2;
        }
        out.put("N-BEATS ensemble", score(Tensor.of(Shape.of(n, horizon), median), test.targets()));

        Ts2Vec encoder = new Ts2Vec(32, 64, 4, new SplittableRandom(seed));
        float[][] crops = new float[40][];
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < crops.length; i++) {
            int s = r.nextInt(600 - 2 * lookback);
            crops[i] = java.util.Arrays.copyOfRange(series, s, s + 2 * lookback);
        }
        encoder.fit(crops, steps, 8, seed);
        out.put("TS2Vec + ridge", score(ts2vecRidge(encoder, train, test, lookback, horizon), test.targets()));
        return out;
    }

    private static Tensor ts2vecRidge(Ts2Vec encoder, SeasonalSeries.Windows train, SeasonalSeries.Windows test,
            int lookback, int horizon) {
        double[][] tr = features(encoder, train.inputs(), lookback);
        float[] ty = train.targets().toArray();
        float[] tx = train.inputs().toArray();
        double[][] y = new double[tr.length][horizon];
        for (int i = 0; i < tr.length; i++) {
            double mean = mean(tx, i * lookback, lookback);
            for (int h = 0; h < horizon; h++) {
                y[i][h] = ty[i * horizon + h] - mean;
            }
        }
        double[][] w = RidgeRegression.fit(tr, y, 0.1);
        double[][] te = features(encoder, test.inputs(), lookback);
        float[] ex = test.inputs().toArray();
        float[] p = new float[te.length * horizon];
        for (int i = 0; i < te.length; i++) {
            double[] yi = RidgeRegression.predict(w, te[i]);
            double mean = mean(ex, i * lookback, lookback);
            for (int h = 0; h < horizon; h++) {
                p[i * horizon + h] = (float) (yi[h] + mean);
            }
        }
        return Tensor.of(Shape.of(te.length, horizon), p);
    }

    private static double mean(float[] x, int from, int n) {
        double s = 0;
        for (int i = 0; i < n; i++) {
            s += x[from + i];
        }
        return s / n;
    }

    /** Last-timestamp TS2Vec representations of de-meaned windows, plus a bias column. */
    private static double[][] features(Ts2Vec encoder, Tensor inputs, int lookback) {
        int n = inputs.shape().dim(0);
        float[] x = inputs.toArray();
        float[][] windows = new float[n][lookback];
        for (int i = 0; i < n; i++) {
            double mean = mean(x, i * lookback, lookback);
            for (int t = 0; t < lookback; t++) {
                windows[i][t] = (float) (x[i * lookback + t] - mean);
            }
        }
        double[][] rep = encoder.lastRepresentations(windows);
        double[][] out = new double[n][];
        for (int i = 0; i < n; i++) {
            out[i] = java.util.Arrays.copyOf(rep[i], rep[i].length + 1);
            out[i][rep[i].length] = 1;
        }
        return out;
    }
}
