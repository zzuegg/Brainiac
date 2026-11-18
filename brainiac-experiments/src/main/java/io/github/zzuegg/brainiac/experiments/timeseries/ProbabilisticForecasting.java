package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ChaoticSeries;
import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;
import io.github.zzuegg.brainiac.experimental.nn.DLinear;
import io.github.zzuegg.brainiac.experimental.nn.DeepAr;
import io.github.zzuegg.brainiac.experimental.nn.PatchTst;
import io.github.zzuegg.brainiac.experimental.nn.QuantileLoss;
import io.github.zzuegg.brainiac.experimental.nn.RevIn;
import io.github.zzuegg.brainiac.experimental.timeseries.Arima;
import io.github.zzuegg.brainiac.experimental.timeseries.ConformalForecaster;
import io.github.zzuegg.brainiac.experimental.timeseries.ExponentialSmoothing;
import io.github.zzuegg.brainiac.experimental.timeseries.GroupAttentionForecaster;
import io.github.zzuegg.brainiac.experimental.timeseries.KernelSynth;
import io.github.zzuegg.brainiac.experimental.timeseries.StateSpaceForecast;
import io.github.zzuegg.brainiac.experimental.timeseries.TokenizedForecaster;
import io.github.zzuegg.brainiac.experimental.timeseries.ValueTokenizer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Probabilistic forecasting: 80% prediction intervals from DeepAR (an autoregressive GRU
 * with a Gaussian head, intervals from sampled paths) and from a direct multi-quantile MLP
 * trained with the pinball loss, on a seasonal series whose noise level changes with the
 * season (so good intervals must breathe). Reported: empirical coverage (target 0.8),
 * mean interval width (narrower is better at equal coverage) and the median forecast's
 * squared error.
 *
 * <p>Papers: Salinas et al., <a href="https://arxiv.org/abs/1704.04110">arXiv:1704.04110</a>
 * (DeepAR); Wen et al., <a href="https://arxiv.org/abs/1711.11053">arXiv:1711.11053</a> (MQ-RNN,
 * direct multi-horizon quantile forecasting).
 *
 * <p>Wave 8 rows: {@link #stateSpace} (ETS and ARIMA intervals from the Kalman filter),
 * {@link #conformal} (CQR, ACI, split conformal and EnbPI around a level and variance shift)
 * and {@link #zeroShot} (a token language model and a group-attention model pretrained on
 * KernelSynth only, against per-dataset models), all scored by MSE of the median, CRPS and
 * 80% coverage.
 */
public final class ProbabilisticForecasting {

    /**
     * Interval quality.
     *
     * @param coverage the share of true values inside the 80% interval
     * @param width the mean interval width
     * @param medianMse the squared error of the median forecast
     */
    public record Intervals(double coverage, double width, double medianMse) {
    }

    private ProbabilisticForecasting() {
    }

    /**
     * A seasonal series whose noise is three times larger in the first half of each season.
     *
     * @param length steps
     * @param period season length
     * @param seed the seed
     * @return the series
     */
    public static float[] heteroscedastic(int length, int period, long seed) {
        float[] base = SeasonalSeries.generate(length, period, 0.0, 1.0, 0.0, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int t = 0; t < length; t++) {
            double sigma = (t % period) < period / 2 ? 0.3 : 0.1;
            base[t] += (float) (sigma * r.nextGaussian());
        }
        return base;
    }

    /**
     * Trains both models and scores their intervals on the last 200 steps.
     *
     * @param lookback context steps
     * @param horizon forecast steps
     * @param period season length
     * @param steps training steps per model
     * @param seed the seed
     * @return model to interval quality
     */
    public static Map<String, Intervals> run(int lookback, int horizon, int period, int steps, long seed) {
        float[] series = heteroscedastic(800, period, seed);
        int window = lookback + horizon;
        SplittableRandom r = new SplittableRandom(seed);
        DeepAr deepAr = new DeepAr(24, period, seed);
        for (int s = 0; s < steps; s++) {
            float[][] batch = new float[16][];
            int[] phases = new int[16];
            for (int i = 0; i < 16; i++) {
                int start = r.nextInt(600 - window);
                batch[i] = Arrays.copyOfRange(series, start, start + window);
                phases[i] = start % period;
            }
            deepAr.trainStep(batch, phases, lookback);
        }
        float[] qs = {0.1f, 0.5f, 0.9f};
        Model quantile = Model.sequential().input(lookback).dense(64, Activation.RELU).dense(horizon * 3).seed(seed)
                .build();
        Optimizer opt = Optimizer.adam(0.003f);
        SeasonalSeries.Windows train = SeasonalSeries.windows(series, 0, 600, lookback, horizon, 1);
        float[] x = train.inputs().toArray();
        float[] y = train.targets().toArray();
        for (int s = 0; s < steps * 3; s++) {
            float[] bx = new float[32 * lookback];
            float[] by = new float[32 * horizon];
            for (int i = 0; i < 32; i++) {
                int j = r.nextInt(train.size());
                System.arraycopy(x, j * lookback, bx, i * lookback, lookback);
                System.arraycopy(y, j * horizon, by, i * horizon, horizon);
            }
            Var in = Var.of(Tensor.of(Shape.of(32, lookback), bx));
            RevIn.Stats st = RevIn.stats(in.value());
            Var out = quantile.forward(RevIn.normalize(in, st), true).reshape(32, horizon, 3);
            Tensor target = Tensor.of(Shape.of(32, horizon), by).sub(st.mean()).div(st.std()).reshape(32, horizon, 1);
            QuantileLoss.pinball(out, target, qs).backward();
            opt.step(quantile.parameters());
        }
        double[] dar = new double[4];
        double[] mq = new double[4];
        for (int start = 600; start + window <= 800; start += 5) {
            float[] context = Arrays.copyOfRange(series, start, start + lookback);
            float[] truth = Arrays.copyOfRange(series, start + lookback, start + window);
            float[][] paths = deepAr.sample(context, start % period, horizon, 50, new SplittableRandom(start));
            accumulate(dar, DeepAr.quantile(paths, 0.1), DeepAr.quantile(paths, 0.5), DeepAr.quantile(paths, 0.9),
                    truth);
            Tensor in = Tensor.of(Shape.of(1, lookback), context);
            RevIn.Stats st = RevIn.stats(in);
            float[] q = RevIn.denormalize(quantile.forward(RevIn.normalize(Var.of(in), st), false), st).value()
                    .toArray();
            float[] lo = new float[horizon];
            float[] mid = new float[horizon];
            float[] hi = new float[horizon];
            for (int h = 0; h < horizon; h++) {
                lo[h] = q[3 * h];
                mid[h] = q[3 * h + 1];
                hi[h] = q[3 * h + 2];
            }
            accumulate(mq, lo, mid, hi, truth);
        }
        Map<String, Intervals> out = new LinkedHashMap<>();
        out.put("DeepAR", new Intervals(dar[0] / dar[3], dar[1] / dar[3], dar[2] / dar[3]));
        out.put("quantile MLP", new Intervals(mq[0] / mq[3], mq[1] / mq[3], mq[2] / mq[3]));
        return out;
    }

    private static void accumulate(double[] acc, float[] lo, float[] mid, float[] hi, float[] truth) {
        for (int h = 0; h < truth.length; h++) {
            acc[0] += truth[h] >= lo[h] && truth[h] <= hi[h] ? 1 : 0;
            acc[1] += hi[h] - lo[h];
            acc[2] += (mid[h] - truth[h]) * (mid[h] - truth[h]);
            acc[3]++;
        }
    }

    // ---------------------------------------------------------------- wave 8: shared harness

    /** The quantile levels of every quantile forecast here; 0.1 and 0.9 bound the 80% interval. */
    static final float[] LEVELS = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};

    /**
     * Quantile-forecast quality.
     *
     * @param mse the squared error of the median
     * @param crps the CRPS, approximated by twice the mean pinball loss over the nine deciles
     *     (for a point forecast it is the absolute error)
     * @param coverage the share of true values inside the 10%-90% interval ({@code NaN} for a
     *     point forecast)
     * @param width the mean width of that interval
     */
    public record Probabilistic(double mse, double crps, double coverage, double width) {
    }

    /** A forecaster that, given the index where the horizon starts, returns {@code [horizon][levels]}. */
    @FunctionalInterface
    interface QuantileForecaster {
        float[][] forecast(int start);
    }

    private static Probabilistic evaluate(float[] series, int from, int to, int horizon, int stride, boolean point,
            QuantileForecaster f) {
        double se = 0;
        double crps = 0;
        double cover = 0;
        double width = 0;
        int n = 0;
        int last = LEVELS.length - 1;
        for (int start = from; start + horizon <= to; start += stride) {
            float[][] q = f.forecast(start);
            for (int h = 0; h < horizon; h++) {
                double y = series[start + h];
                double med = q[h][LEVELS.length / 2];
                se += (y - med) * (y - med);
                double pin = 0;
                for (int k = 0; k < LEVELS.length; k++) {
                    double d = y - q[h][k];
                    pin += Math.max(LEVELS[k] * d, (LEVELS[k] - 1) * d);
                }
                crps += 2 * pin / LEVELS.length;
                cover += y >= q[h][0] && y <= q[h][last] ? 1 : 0;
                width += q[h][last] - q[h][0];
                n++;
            }
        }
        return new Probabilistic(se / n, crps / n, point ? Double.NaN : cover / n, point ? 0 : width / n);
    }

    private static float[][] fromPaths(float[][] paths) {
        int horizon = paths[0].length;
        float[][] q = new float[horizon][LEVELS.length];
        for (int k = 0; k < LEVELS.length; k++) {
            float[] col = DeepAr.quantile(paths, LEVELS[k]);
            for (int h = 0; h < horizon; h++) {
                q[h][k] = col[h];
            }
        }
        return q;
    }

    private static float[][] fromPoint(float[] point) {
        float[][] q = new float[point.length][LEVELS.length];
        for (int h = 0; h < point.length; h++) {
            Arrays.fill(q[h], point[h]);
        }
        return q;
    }

    private static float[][] fromGaussian(StateSpaceForecast f) {
        int horizon = f.mean().length;
        float[][] q = new float[horizon][LEVELS.length];
        for (int h = 0; h < horizon; h++) {
            double s = Math.sqrt(Math.max(0, f.variance()[h]));
            for (int k = 0; k < LEVELS.length; k++) {
                q[h][k] = (float) (f.mean()[h] + StateSpaceForecast.normalQuantile(LEVELS[k]) * s);
            }
        }
        return q;
    }

    private static double[] toDouble(float[] f, int to) {
        double[] d = new double[to];
        for (int i = 0; i < to; i++) {
            d[i] = f[i];
        }
        return d;
    }

    /** Seasonal naive with per-step empirical residual deciles from the training part. */
    private static QuantileForecaster seasonalNaive(float[] series, int trainTo, int horizon, int period) {
        float[][] res = new float[horizon][];
        for (int h = 0; h < horizon; h++) {
            int lag = period * ((h / period) + 1);
            int count = trainTo - lag;
            res[h] = new float[Math.max(1, count)];
            for (int t = 0; t < count; t++) {
                res[h][t] = series[t + lag] - series[t];
            }
            Arrays.sort(res[h]);
        }
        return start -> {
            float[][] q = new float[horizon][LEVELS.length];
            for (int h = 0; h < horizon; h++) {
                int lag = period * ((h / period) + 1);
                float base = series[start + h - lag];
                for (int k = 0; k < LEVELS.length; k++) {
                    q[h][k] = base + res[h][(int) Math.min(res[h].length - 1, Math.floor(LEVELS[k] * res[h].length))];
                }
            }
            return q;
        };
    }

    /** A direct multi-quantile MLP with RevIN trained with the pinball loss on windows before {@code trainTo}. */
    private static QuantileForecaster quantileMlp(float[] series, int trainTo, int lookback, int horizon, int steps,
            long seed) {
        int nq = LEVELS.length;
        Model m = Model.sequential().input(lookback).dense(64, Activation.RELU).dense(horizon * nq).seed(seed).build();
        Optimizer opt = Optimizer.adam(0.003f);
        SplittableRandom r = new SplittableRandom(seed);
        int windows = trainTo - lookback - horizon + 1;
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[32 * lookback];
            float[] by = new float[32 * horizon];
            for (int i = 0; i < 32; i++) {
                int j = r.nextInt(windows);
                System.arraycopy(series, j, bx, i * lookback, lookback);
                System.arraycopy(series, j + lookback, by, i * horizon, horizon);
            }
            Var in = Var.of(Tensor.of(Shape.of(32, lookback), bx));
            RevIn.Stats st = RevIn.stats(in.value());
            Var out = m.forward(RevIn.normalize(in, st), true).reshape(32, horizon, nq);
            Tensor target = Tensor.of(Shape.of(32, horizon), by).sub(st.mean()).div(st.std()).reshape(32, horizon, 1);
            QuantileLoss.pinball(out, target, LEVELS).backward();
            opt.step(m.parameters());
        }
        return start -> {
            Tensor in = Tensor.of(Shape.of(1, lookback), Arrays.copyOfRange(series, start - lookback, start));
            RevIn.Stats st = RevIn.stats(in);
            float[] y = Var.noGrad(() -> RevIn.denormalize(m.forward(RevIn.normalize(Var.of(in), st), false), st))
                    .value().toArray();
            float[][] q = new float[horizon][nq];
            for (int h = 0; h < horizon; h++) {
                System.arraycopy(y, h * nq, q[h], 0, nq);
                Arrays.sort(q[h]);
            }
            return q;
        };
    }

    /** DeepAR trained on windows before {@code trainTo}; deciles of 50 sampled paths. */
    private static QuantileForecaster deepAr(float[] series, int trainTo, int lookback, int horizon, int period,
            int steps, long seed) {
        DeepAr model = new DeepAr(24, period, seed);
        SplittableRandom r = new SplittableRandom(seed);
        int window = lookback + horizon;
        for (int s = 0; s < steps; s++) {
            float[][] batch = new float[16][];
            int[] phases = new int[16];
            for (int i = 0; i < 16; i++) {
                int start = r.nextInt(trainTo - window + 1);
                batch[i] = Arrays.copyOfRange(series, start, start + window);
                phases[i] = period > 0 ? start % period : 0;
            }
            model.trainStep(batch, phases, lookback);
        }
        return start -> fromPaths(model.sample(Arrays.copyOfRange(series, start - lookback, start),
                period > 0 ? (start - lookback) % period : 0, horizon, 50, new SplittableRandom(start)));
    }

    /** PatchTST (RevIN) trained with MSE before {@code trainTo}; a point forecast. */
    private static QuantileForecaster patchTst(float[] series, int trainTo, int lookback, int horizon, int patch,
            int steps, long seed) {
        PatchTst model = new PatchTst(lookback, horizon, patch, patch / 2, 16, 1, 2, true, new SplittableRandom(seed));
        Optimizer opt = Optimizer.adam(0.005f);
        SplittableRandom r = new SplittableRandom(seed);
        int windows = trainTo - lookback - horizon + 1;
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[32 * lookback];
            float[] by = new float[32 * horizon];
            for (int i = 0; i < 32; i++) {
                int j = r.nextInt(windows);
                System.arraycopy(series, j, bx, i * lookback, lookback);
                System.arraycopy(series, j + lookback, by, i * horizon, horizon);
            }
            model.forward(Var.of(Tensor.of(Shape.of(32, lookback), bx)), true)
                    .sub(Tensor.of(Shape.of(32, horizon), by)).square().mean().backward();
            opt.step(model.parameters());
        }
        return start -> fromPoint(Var.noGrad(() -> model.forward(Var.of(Tensor.of(Shape.of(1, lookback),
                Arrays.copyOfRange(series, start - lookback, start))), false)).value().toArray());
    }

    // ---------------------------------------------------------------- 8.2.1: state-space intervals

    /**
     * ETS and ARIMA (Kalman likelihood, L-BFGS, AICc selection) against seasonal naive, DeepAR,
     * the quantile MLP and DLinear on a trend + season + noise series: rolling-origin forecasts
     * over the last 100 steps.
     *
     * @param horizon steps ahead
     * @param steps training steps of the learned models
     * @param iterations L-BFGS iterations per state-space fit
     * @param seed the seed
     * @return model to quality
     */
    public static Map<String, Probabilistic> stateSpace(int horizon, int steps, int iterations, long seed) {
        int period = 12;
        int lookback = 48;
        float[] series = SeasonalSeries.generate(500, period, 0.004, 1.0, 0.3, seed);
        int trainTo = 400;
        int stride = 4;
        Map<String, Probabilistic> out = new LinkedHashMap<>();
        out.put("seasonal naive", evaluate(series, trainTo, 500, horizon, stride, false,
                seasonalNaive(series, trainTo, horizon, period)));
        ExponentialSmoothing ets = ExponentialSmoothing.select(toDouble(series, trainTo), period, iterations);
        out.put("ETS", evaluate(series, trainTo, 500, horizon, stride, false,
                start -> fromGaussian(ets.forecast(toDouble(series, start), horizon))));
        Arima arima = Arima.select(toDouble(series, trainTo), 1, 1, period, 1, 1, iterations);
        out.put("ARIMA", evaluate(series, trainTo, 500, horizon, stride, false,
                start -> fromGaussian(arima.forecast(toDouble(series, start), horizon))));
        out.put("DeepAR", evaluate(series, trainTo, 500, horizon, stride, false,
                deepAr(series, trainTo, lookback, horizon, period, steps, seed)));
        out.put("quantile MLP", evaluate(series, trainTo, 500, horizon, stride, false,
                quantileMlp(series, trainTo, lookback, horizon, steps * 2, seed)));
        DLinear dlinear = new DLinear(lookback, horizon, period + 1, new SplittableRandom(seed));
        Optimizer opt = Optimizer.adam(0.005f);
        SeasonalSeries.Windows train = SeasonalSeries.windows(series, 0, trainTo, lookback, horizon, 1);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps * 3; s++) {
            float[] idx = new float[32];
            for (int i = 0; i < 32; i++) {
                idx[i] = r.nextInt(train.size());
            }
            Tensor sel = Tensor.of(Shape.of(32), idx);
            dlinear.forward(Var.of(train.inputs().gather(0, sel)), true).sub(train.targets().gather(0, sel)).square()
                    .mean().backward();
            opt.step(dlinear.parameters());
        }
        out.put("DLinear", evaluate(series, trainTo, 500, horizon, stride, true, start -> fromPoint(Var.noGrad(
                () -> dlinear.forward(Var.of(Tensor.of(Shape.of(1, lookback),
                        Arrays.copyOfRange(series, start - lookback, start))), false)).value().toArray())));
        return out;
    }

    // ---------------------------------------------------------------- 8.2.8: conformal under drift

    /**
     * Interval quality around a drift.
     *
     * @param before coverage before the shift (target 0.8)
     * @param after coverage over the whole post-shift period
     * @param recovered coverage over the second half of the post-shift period
     * @param width the mean interval width
     */
    public record Drift(double before, double after, double recovered, double width) {
    }

    /**
     * A seasonal series (noise 0.2) whose level shifts by 1.5 and whose noise rises to about
     * 0.54 from step {@code shift}.
     *
     * @param length steps
     * @param shift the change step
     * @param seed the seed
     * @return the series
     */
    public static float[] drifting(int length, int shift, long seed) {
        float[] y = SeasonalSeries.generate(length, 12, 0.0, 1.0, 0.2, seed);
        SplittableRandom r = new SplittableRandom(seed + 3);
        for (int t = shift; t < length; t++) {
            y[t] += (float) (1.5 + 0.5 * r.nextGaussian());
        }
        return y;
    }

    /**
     * One-step 80% intervals on {@link #drifting}: train on {@code [0, 500)}, calibrate on
     * {@code [500, 700)}, test on {@code [700, 1000)} with the shift at 850. Rows: the raw
     * quantile MLP, CQR, ACI (on CQR scores, sliding window of 200), split conformal on the
     * median's residuals, EnbPI (eight bootstrap MLPs on all pre-test windows, leave-one-out
     * residuals, sliding window of 200), ETS intervals and DeepAR. The online methods see each
     * test observation after forecasting it.
     *
     * @param steps training steps of the learned models
     * @param gamma ACI's step size
     * @param seed the seed
     * @return method to coverage and width
     */
    public static Map<String, Drift> conformal(int steps, double gamma, long seed) {
        int lookback = 24;
        int trainTo = 500;
        int calTo = 700;
        int shift = 850;
        int end = 1000;
        double alpha = 0.2;
        float[] y = drifting(end, shift, seed);
        QuantileForecaster mlp = quantileMlp(y, trainTo, lookback, 1, steps, seed);
        int last = LEVELS.length - 1;
        int mid = LEVELS.length / 2;
        double[] cLo = new double[calTo - trainTo];
        double[] cMid = new double[cLo.length];
        double[] cHi = new double[cLo.length];
        double[] cY = new double[cLo.length];
        for (int t = trainTo; t < calTo; t++) {
            float[] q = mlp.forecast(t)[0];
            cLo[t - trainTo] = q[0];
            cMid[t - trainTo] = q[mid];
            cHi[t - trainTo] = q[last];
            cY[t - trainTo] = y[t];
        }
        double[] residuals = new double[cY.length];
        for (int i = 0; i < cY.length; i++) {
            residuals[i] = cY[i] - cMid[i];
        }
        ConformalForecaster cqr = ConformalForecaster.cqr(cLo, cHi, cY, alpha);
        ConformalForecaster aci = ConformalForecaster.aci(cLo, cHi, cY, alpha, gamma, 200);
        ConformalForecaster split = ConformalForecaster.split(residuals, alpha);

        int members = 8;
        int windows = calTo - lookback;
        SplittableRandom r = new SplittableRandom(seed + 11);
        List<QuantileForecaster> ens = new ArrayList<>();
        boolean[][] inBag = new boolean[members][windows];
        for (int b = 0; b < members; b++) {
            int[] boot = new int[windows];
            for (int i = 0; i < windows; i++) {
                boot[i] = r.nextInt(windows);
                inBag[b][boot[i]] = true;
            }
            ens.add(pointMlp(y, lookback, boot, steps, seed + 100 + b));
        }
        double[] loo = new double[windows];
        int kept = 0;
        for (int i = 0; i < windows; i++) {
            double s = 0;
            int c = 0;
            for (int b = 0; b < members; b++) {
                if (!inBag[b][i]) {
                    s += ens.get(b).forecast(i + lookback)[0][0];
                    c++;
                }
            }
            if (c > 0) {
                loo[kept++] = y[i + lookback] - s / c;
            }
        }
        ConformalForecaster enbpi = ConformalForecaster.enbpi(Arrays.copyOf(loo, kept), alpha, 200);

        ExponentialSmoothing ets = ExponentialSmoothing.fit(toDouble(y, calTo), ExponentialSmoothing.Trend.NONE, 12,
                15);
        QuantileForecaster deepAr = deepAr(y, trainTo, lookback, 1, 12, steps, seed);

        String[] names = {"quantile MLP", "CQR", "ACI", "split conformal", "EnbPI", "ETS", "DeepAR"};
        double[][] acc = new double[names.length][4];
        int late = (shift + end) / 2;
        for (int t = calTo; t < end; t++) {
            float[] q = mlp.forecast(t)[0];
            double lo = q[0];
            double hi = q[last];
            double med = q[mid];
            double ens0 = 0;
            for (QuantileForecaster m : ens) {
                ens0 += m.forecast(t)[0][0];
            }
            ens0 /= members;
            double[][] iv = new double[names.length][];
            iv[0] = new double[] {lo, hi};
            iv[1] = cqr.interval(lo, hi);
            iv[2] = aci.interval(lo, hi);
            iv[3] = split.interval(med, med);
            iv[4] = enbpi.interval(ens0, ens0);
            double[][] e = ets.forecast(toDouble(y, t), 1).interval(0.8);
            iv[5] = new double[] {e[0][0], e[1][0]};
            float[] d = deepAr.forecast(t)[0];
            iv[6] = new double[] {d[0], d[last]};
            for (int k = 0; k < names.length; k++) {
                double hit = y[t] >= iv[k][0] && y[t] <= iv[k][1] ? 1 : 0;
                if (t < shift) {
                    acc[k][0] += hit;
                } else {
                    acc[k][1] += hit;
                    if (t >= late) {
                        acc[k][2] += hit;
                    }
                }
                acc[k][3] += iv[k][1] - iv[k][0];
            }
            aci.update(lo, hi, y[t]);
            enbpi.update(ens0, ens0, y[t]);
        }
        Map<String, Drift> out = new LinkedHashMap<>();
        for (int k = 0; k < names.length; k++) {
            out.put(names[k], new Drift(acc[k][0] / (shift - calTo), acc[k][1] / (end - shift),
                    acc[k][2] / (end - late), acc[k][3] / (end - calTo)));
        }
        return out;
    }

    /** A one-step point MLP with RevIN trained with MSE on the given window starts. */
    private static QuantileForecaster pointMlp(float[] series, int lookback, int[] starts, int steps, long seed) {
        Model m = Model.sequential().input(lookback).dense(32, Activation.RELU).dense(1).seed(seed).build();
        Optimizer opt = Optimizer.adam(0.003f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[32 * lookback];
            float[] by = new float[32];
            for (int i = 0; i < 32; i++) {
                int j = starts[r.nextInt(starts.length)];
                System.arraycopy(series, j, bx, i * lookback, lookback);
                by[i] = series[j + lookback];
            }
            Var in = Var.of(Tensor.of(Shape.of(32, lookback), bx));
            RevIn.Stats st = RevIn.stats(in.value());
            RevIn.denormalize(m.forward(RevIn.normalize(in, st), true), st).sub(Tensor.of(Shape.of(32, 1), by))
                    .square().mean().backward();
            opt.step(m.parameters());
        }
        return start -> {
            Tensor in = Tensor.of(Shape.of(1, lookback), Arrays.copyOfRange(series, start - lookback, start));
            RevIn.Stats st = RevIn.stats(in);
            float v = Var.noGrad(() -> RevIn.denormalize(m.forward(RevIn.normalize(Var.of(in), st), false), st))
                    .value().toArray()[0];
            return new float[][] {{v}};
        };
    }

    // ---------------------------------------------------------------- 8.2.7: zero-shot forecasting

    /**
     * Chronos-style zero-shot forecasting: a token language model ({@link TokenizedForecaster})
     * and a group-attention quantile model ({@link GroupAttentionForecaster}) are pretrained on
     * KernelSynth only, then forecast three series families they never saw (a trend + season
     * series, Mackey-Glass, the demand series with price and holiday covariates), against
     * seasonal naive and DeepAR, the quantile MLP and PatchTST trained on each dataset.
     * Context 64, horizon 16; per-dataset models train on the first 400 steps, everything is
     * scored on rolling origins over the last 200.
     *
     * @param pretrainSteps pretraining steps of each zero-shot model
     * @param steps training steps of the per-dataset models
     * @param stride the step between test origins
     * @param seed the seed
     * @param datasets the datasets to score ({@code "seasonal"}, {@code "Mackey-Glass"},
     *     {@code "covariates"}); none for all three
     * @return {@code "dataset: model"} to quality
     */
    public static Map<String, Probabilistic> zeroShot(int pretrainSteps, int steps, int stride, long seed,
            String... datasets) {
        int lookback = 64;
        int horizon = 16;
        int length = lookback + horizon;
        KernelSynth synth = new KernelSynth(length, 4);
        SplittableRandom r = new SplittableRandom(seed);
        TokenizedForecaster lm = new TokenizedForecaster(new ValueTokenizer(128, -6, 6), lookback, horizon, 32, 2, 4,
                new SplittableRandom(seed));
        lm.fit(() -> synth.sample(length, r), pretrainSteps, 16, 0.002f);
        GroupAttentionForecaster group = new GroupAttentionForecaster(lookback, horizon, 8, 32, 2, 4, LEVELS,
                new SplittableRandom(seed + 1));
        int[] calls = {0};
        group.fit(() -> {
            int covariates = calls[0]++ % 2 == 0 ? 0 : 2;
            float[][][] b = new float[16][][];
            for (int i = 0; i < 16; i++) {
                b[i] = synth.sampleGroup(length, covariates, r);
            }
            return b;
        }, pretrainSteps, 0.002f);

        SeasonalSeries.Covariates cov = SeasonalSeries.covariates(600, 12, 0.1, seed + 5);
        double[] mg = ChaoticSeries.mackeyGlass(600, 17, 1.0, 300);
        float[] mgf = new float[mg.length];
        for (int i = 0; i < mg.length; i++) {
            mgf[i] = (float) mg[i];
        }
        Map<String, float[]> data = new LinkedHashMap<>();
        data.put("seasonal", SeasonalSeries.generate(600, 12, 0.003, 1.0, 0.2, seed + 9));
        data.put("Mackey-Glass", mgf);
        data.put("covariates", cov.target());
        if (datasets.length > 0) {
            data.keySet().retainAll(List.of(datasets));
        }
        int trainTo = 400;
        int end = 600;
        Map<String, Probabilistic> out = new LinkedHashMap<>();
        for (Map.Entry<String, float[]> e : data.entrySet()) {
            String name = e.getKey();
            float[] y = e.getValue();
            int period = name.equals("Mackey-Glass") ? 0 : 12;
            out.put(name + ": seasonal naive", evaluate(y, trainTo, end, horizon, stride, false,
                    seasonalNaive(y, trainTo, horizon, Math.max(1, period))));
            out.put(name + ": DeepAR", evaluate(y, trainTo, end, horizon, stride, false,
                    deepAr(y, trainTo, lookback, horizon, period, steps, seed)));
            out.put(name + ": quantile MLP", evaluate(y, trainTo, end, horizon, stride, false,
                    quantileMlp(y, trainTo, lookback, horizon, steps * 2, seed)));
            out.put(name + ": PatchTST", evaluate(y, trainTo, end, horizon, stride, true,
                    patchTst(y, trainTo, lookback, horizon, 8, steps, seed)));
            out.put(name + ": token LM zero-shot", evaluate(y, trainTo, end, horizon, stride, false,
                    start -> fromPaths(lm.sample(Arrays.copyOfRange(y, start - lookback, start), 32, 1.0,
                            new SplittableRandom(start)))));
            out.put(name + ": group attention zero-shot", evaluate(y, trainTo, end, horizon, stride, false,
                    start -> group.forecast(Arrays.copyOfRange(y, start - lookback, start))));
            if (name.equals("covariates")) {
                out.put(name + ": group attention + covariates zero-shot", evaluate(y, trainTo, end, horizon, stride,
                        false, start -> group.forecast(Arrays.copyOfRange(y, start - lookback, start),
                                Arrays.copyOfRange(cov.price(), start - lookback, start + horizon),
                                Arrays.copyOfRange(cov.holiday(), start - lookback, start + horizon))));
            }
        }
        return out;
    }
}
