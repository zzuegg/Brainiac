package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;
import io.github.zzuegg.brainiac.experimental.nn.DLinear;
import io.github.zzuegg.brainiac.experimental.nn.NBeats;
import io.github.zzuegg.brainiac.experimental.nn.PatchTst;
import io.github.zzuegg.brainiac.experimental.nn.RevIn;
import io.github.zzuegg.brainiac.experimental.nn.TemporalConvNet;
import io.github.zzuegg.brainiac.experimental.timeseries.InvertedTransformer;
import io.github.zzuegg.brainiac.experimental.timeseries.NHits;
import io.github.zzuegg.brainiac.experimental.timeseries.Tide;
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
import java.util.function.Function;

/**
 * Long-horizon, multivariate and covariate forecasting (8.2.2-8.2.6), every model trained
 * with Adam and MSE for the same number of steps on batches of 32 windows.
 * <ul>
 *   <li>{@link #horizons}: univariate {@link SeasonalSeries} at several horizons: seasonal
 *       naive, DLinear, PatchTST, a vanilla transformer, Autoformer
 *       ({@code AutoCorrelationLayer} + {@code SeriesDecomposition}), FEDformer
 *       ({@code FrequencyEnhancedBlock} + decomposition), TimesNet ({@code TimesBlock}),
 *       {@link NHits} and N-BEATS, with parameter counts. The question of Zeng et al.
 *       (<a href="https://arxiv.org/abs/2205.13504">arXiv:2205.13504</a>): do the
 *       transformer forecasters beat a linear map?</li>
 *   <li>{@link #twoPeriods}: two incommensurate seasonalities, TimesNet's case, against
 *       PatchTST, a TCN and DLinear;</li>
 *   <li>{@link #multivariate}: four channels that are lagged copies of each other
 *       (coupled) or independent draws (control): the {@link InvertedTransformer} against the
 *       channel-independent PatchTST and DLinear and the channel-mixing FEDformer;</li>
 *   <li>{@link #covariates}: demand driven by a known future price step and holidays:
 *       {@link Tide} against DLinear, PatchTST and an MLP with RevIN that ignore the
 *       covariates, and the same MLP fed the flattened future covariates; MSE overall, on
 *       holiday steps, and on steps after a price change inside the horizon.</li>
 * </ul>
 */
public final class LongHorizonForecasting {

    private LongHorizonForecasting() {
    }

    /**
     * A model's score.
     *
     * @param mse the test MSE
     * @param parameters trainable scalars
     * @param millis training and test time
     */
    public record Score(double mse, long parameters, double millis) {
        @Override
        public String toString() {
            return String.format("%.4f (%d params, %.0f ms)", mse, parameters, millis);
        }
    }

    private static Tensor univariate3d(Tensor t) {
        return t.reshape(t.shape().dim(0), t.shape().dim(1), 1);
    }

    private static Forecasters.Model patchTst(int lookback, int horizon, int period, long seed) {
        int patch = Math.max(4, Math.min(period, lookback / 4));
        PatchTst p = new PatchTst(lookback, horizon, patch, patch / 2, 16, 1, 2, true, new SplittableRandom(seed));
        return new Forecasters.Model("PatchTST", Forecasters.channelIndependent(x -> p.forward(x, true)),
                p.parameters());
    }

    private static Forecasters.Model dlinear(int lookback, int horizon, int period, long seed) {
        DLinear d = new DLinear(lookback, horizon, period + 1 - period % 2, new SplittableRandom(seed));
        return new Forecasters.Model("DLinear", Forecasters.channelIndependent(x -> d.forward(x, true)),
                d.parameters());
    }

    private static Forecasters.Model normalised(String name, Function<Var, Var> model, List<Var> parameters) {
        return new Forecasters.Model(name, Forecasters.channelIndependent(x -> {
            RevIn.Stats s = RevIn.stats(x.value());
            return RevIn.denormalize(model.apply(RevIn.normalize(x, s)), s);
        }), parameters);
    }

    private static Forecasters.Model tcn(int lookback, int horizon, long seed) {
        TemporalConvNet net = new TemporalConvNet(1, 16, 3, 4, new SplittableRandom(seed));
        Var head = Var.param(Tensor.normal(Shape.of(16, horizon), 0, 0.25f, new SplittableRandom(seed + 1)));
        List<Var> params = new ArrayList<>(net.parameters());
        params.add(head);
        return new Forecasters.Model("TCN + RevIN", Forecasters.channelIndependent(x -> {
            int b = x.shape().dim(0);
            RevIn.Stats s = RevIn.stats(x.value());
            Var hidden = net.forward(RevIn.normalize(x, s).reshape(b, lookback, 1), true);
            return RevIn.denormalize(hidden.slice(1, lookback - 1, lookback).reshape(b, 16).matmul(head), s);
        }), params);
    }

    private static void train(Map<String, Score> out, Forecasters.Model m, SeasonalSeries.MultiWindows train,
            SeasonalSeries.MultiWindows test, int steps, float lr, long seed, int seeds) {
        long t0 = System.nanoTime();
        double mse = Forecasters.fit(m, train.inputs(), train.targets(), test.inputs(), test.targets(), steps, lr,
                seed);
        double ms = (System.nanoTime() - t0) / 1e6;
        Score prev = out.get(m.name());
        out.put(m.name(), new Score((prev == null ? 0 : prev.mse()) + mse / seeds, m.size(),
                (prev == null ? 0 : prev.millis()) + ms / seeds));
    }

    private static SeasonalSeries.MultiWindows univariateWindows(float[] series, int from, int to, int lookback,
            int horizon, int stride) {
        SeasonalSeries.Windows w = SeasonalSeries.windows(series, from, to, lookback, horizon, stride);
        return new SeasonalSeries.MultiWindows(univariate3d(w.inputs()), univariate3d(w.targets()), w.starts());
    }

    /**
     * Univariate forecasting at several horizons.
     *
     * @param lookback input steps
     * @param horizons the horizons
     * @param steps training steps per model
     * @param seeds series and initialisations to average over
     * @return horizon to model to score
     */
    public static Map<Integer, Map<String, Score>> horizons(int lookback, int[] horizons, int steps, int seeds) {
        int period = 24;
        Map<Integer, Map<String, Score>> out = new LinkedHashMap<>();
        for (int horizon : horizons) {
            Map<String, Score> row = new LinkedHashMap<>();
            for (int s = 0; s < seeds; s++) {
                long seed = 11 + s;
                float[] series = SeasonalSeries.generate(1000 + 2 * lookback + 2 * horizon, period, 0.002, 1.0, 0.15,
                        seed);
                int split = 800 + lookback + horizon;
                SeasonalSeries.MultiWindows train = univariateWindows(series, 0, split, lookback, horizon, 2);
                SeasonalSeries.MultiWindows test = univariateWindows(series, split, series.length, lookback, horizon,
                        4);
                Tensor naive = univariate3d(SeasonalSeries.seasonalNaive(test.inputs().reshape(test.size(), lookback),
                        horizon, period));
                Score prev = row.get("seasonal naive");
                row.put("seasonal naive", new Score((prev == null ? 0 : prev.mse())
                        + naive.sub(test.targets()).square().mean().item() / seeds, 0, 0));
                train(row, dlinear(lookback, horizon, period, seed), train, test, steps, 0.005f, seed, seeds);
                train(row, patchTst(lookback, horizon, period, seed), train, test, steps, 0.005f, seed, seeds);
                train(row, Forecasters.transformer("vanilla transformer", Forecasters.Mixer.ATTENTION, lookback,
                        horizon, 1, 16, 1, 25, seed), train, test, steps, 0.003f, seed, seeds);
                train(row, Forecasters.transformer("Autoformer", Forecasters.Mixer.AUTO_CORRELATION, lookback,
                        horizon, 1, 16, 1, 25, seed), train, test, steps, 0.003f, seed, seeds);
                train(row, Forecasters.transformer("FEDformer", Forecasters.Mixer.FOURIER, lookback, horizon, 1, 16,
                        1, 25, seed), train, test, steps, 0.003f, seed, seeds);
                train(row, Forecasters.timesNet(lookback, horizon, 1, 8, 1, 2, seed), train, test, steps, 0.003f,
                        seed, seeds);
                NHits nhits = NHits.standard(lookback, horizon, 64, seed);
                train(row, new Forecasters.Model("N-HiTS", Forecasters.channelIndependent(x -> nhits.forward(x,
                        true)), nhits.parameters()), train, test, steps, 0.002f, seed, seeds);
                NBeats nbeats = NBeats.generic(lookback, horizon, 3, 64, seed);
                train(row, normalised("N-BEATS", v -> nbeats.forward(v, true), nbeats.parameters()), train, test,
                        steps, 0.002f, seed, seeds);
            }
            out.put(horizon, row);
        }
        return out;
    }

    /**
     * Two incommensurate periods (12 and 31.4 steps), lookback 64, horizon 32.
     *
     * @param steps training steps per model
     * @param seeds seeds to average over
     * @return model to score
     */
    public static Map<String, Score> twoPeriods(int steps, int seeds) {
        int lookback = 64;
        int horizon = 32;
        Map<String, Score> row = new LinkedHashMap<>();
        for (int s = 0; s < seeds; s++) {
            long seed = 21 + s;
            float[] series = SeasonalSeries.twoPeriods(1300, 12, 31.4, 0.1, seed);
            SeasonalSeries.MultiWindows train = univariateWindows(series, 0, 1000, lookback, horizon, 2);
            SeasonalSeries.MultiWindows test = univariateWindows(series, 1000, 1300, lookback, horizon, 4);
            train(row, dlinear(lookback, horizon, 12, seed), train, test, steps, 0.005f, seed, seeds);
            train(row, patchTst(lookback, horizon, 12, seed), train, test, steps, 0.005f, seed, seeds);
            train(row, tcn(lookback, horizon, seed), train, test, steps, 0.005f, seed, seeds);
            train(row, Forecasters.timesNet(lookback, horizon, 1, 8, 1, 3, seed), train, test, steps, 0.003f, seed,
                    seeds);
        }
        return row;
    }

    /**
     * Four channels, lagged copies of one AR + seasonal process (lag 12) or independent.
     *
     * @param coupled whether the channels are coupled
     * @param steps training steps per model
     * @param seeds seeds to average over
     * @return model to score
     */
    public static Map<String, Score> multivariate(boolean coupled, int steps, int seeds) {
        int lookback = 48;
        int horizon = 12;
        int channels = 4;
        Map<String, Score> row = new LinkedHashMap<>();
        for (int s = 0; s < seeds; s++) {
            long seed = 31 + s;
            float[][] series = SeasonalSeries.multivariate(900, channels, 24, 12, coupled, 0.05, seed);
            SeasonalSeries.MultiWindows train = SeasonalSeries.windows(series, 0, 650, lookback, horizon, 1);
            SeasonalSeries.MultiWindows test = SeasonalSeries.windows(series, 650, 900, lookback, horizon, 2);
            train(row, dlinear(lookback, horizon, 24, seed), train, test, steps, 0.005f, seed, seeds);
            train(row, patchTst(lookback, horizon, 24, seed), train, test, steps, 0.005f, seed, seeds);
            train(row, Forecasters.transformer("FEDformer", Forecasters.Mixer.FOURIER, lookback, horizon, channels,
                    16, 1, 13, seed), train, test, steps, 0.003f, seed, seeds);
            InvertedTransformer it = new InvertedTransformer(lookback, horizon, channels, 32, 1, 2,
                    new SplittableRandom(seed));
            train(row, new Forecasters.Model("iTransformer", x -> it.forward(x, true), it.parameters()), train, test,
                    steps, 0.003f, seed, seeds);
        }
        return row;
    }

    /**
     * Covariate scores.
     *
     * @param mse overall test MSE
     * @param holidayMse MSE on holiday steps
     * @param priceChangeMse MSE on steps after a price change inside the horizon
     * @param parameters trainable scalars
     */
    public record CovariateScore(double mse, double holidayMse, double priceChangeMse, long parameters) {
        @Override
        public String toString() {
            return String.format("%.4f (holiday %.3f, after price step %.3f; %d params)", mse, holidayMse,
                    priceChangeMse, parameters);
        }
    }

    /**
     * Demand with known future price and holiday covariates, lookback 48, horizon 24.
     *
     * @param steps training steps per model
     * @param seeds seeds to average over
     * @return model to scores
     */
    public static Map<String, CovariateScore> covariates(int steps, int seeds) {
        int lookback = 48;
        int horizon = 24;
        int period = 24;
        Map<String, double[]> acc = new LinkedHashMap<>();
        Map<String, Long> sizes = new LinkedHashMap<>();
        for (int s = 0; s < seeds; s++) {
            long seed = 41 + s;
            SeasonalSeries.Covariates data = SeasonalSeries.covariates(1400, period, 0.1, seed);
            int split = 1000;
            CovariateWindows train = covariateWindows(data, 0, split, lookback, horizon, 1);
            CovariateWindows test = covariateWindows(data, split, data.length(), lookback, horizon, 2);
            Map<String, CovariateModel> models = new LinkedHashMap<>();
            models.put("seasonal naive", null);
            Forecasters.Model dl = dlinear(lookback, horizon, period, seed);
            models.put("DLinear (no covariates)", new CovariateModel(in -> dl.forward().apply(in.get(0)), dl.parameters(),
                    0.005f));
            Forecasters.Model pt = patchTst(lookback, horizon, period, seed);
            models.put("PatchTST (no covariates)", new CovariateModel(in -> pt.forward().apply(in.get(0)),
                    pt.parameters(), 0.005f));
            Model mlp = Model.sequential().input(lookback)
                    .dense(64, Activation.RELU).dense(horizon).seed(seed).build();
            Forecasters.Model mlpModel = normalised("MLP + RevIN", v -> mlp.forward(v, true), mlp.parameters());
            models.put("MLP + RevIN (no covariates)", new CovariateModel(in -> mlpModel.forward().apply(in.get(0)),
                    mlp.parameters(), 0.003f));
            Model mlpCov = Model.sequential()
                    .input(lookback + 2 * horizon).dense(64, Activation.RELU).dense(horizon).seed(seed).build();
            models.put("MLP + RevIN + future covariates", new CovariateModel(in -> {
                Var y = in.get(0).reshape(in.get(0).shape().dim(0), lookback);
                int b = y.shape().dim(0);
                RevIn.Stats st = RevIn.stats(y.value());
                Var x = Var.concat(1, RevIn.normalize(y, st), in.get(2).reshape(b, 2 * horizon));
                return RevIn.denormalize(mlpCov.forward(x, true), st).reshape(b, horizon, 1);
            }, mlpCov.parameters(), 0.003f));
            Tide tide = new Tide(lookback, horizon, 2, 64, 4, 8, 1, new SplittableRandom(seed));
            models.put("TiDE", new CovariateModel(in -> {
                int b = in.get(0).shape().dim(0);
                return tide.forward(List.of(in.get(0).reshape(b, lookback), in.get(1), in.get(2)), true)
                        .reshape(b, horizon, 1);
            }, tide.parameters(), 0.003f));
            for (Map.Entry<String, CovariateModel> e : models.entrySet()) {
                Tensor pred;
                if (e.getValue() == null) {
                    pred = univariate3d(SeasonalSeries.seasonalNaive(test.y().reshape(test.y().shape().dim(0),
                            lookback), horizon, period));
                    sizes.put(e.getKey(), 0L);
                } else {
                    pred = e.getValue().fit(train, test, steps, seed);
                    long n = 0;
                    for (Var v : e.getValue().parameters()) {
                        n += v.shape().size();
                    }
                    sizes.put(e.getKey(), n);
                }
                double[] sc = covariateErrors(pred, test);
                double[] a = acc.computeIfAbsent(e.getKey(), k -> new double[3]);
                for (int i = 0; i < 3; i++) {
                    a[i] += sc[i] / seeds;
                }
            }
        }
        Map<String, CovariateScore> out = new LinkedHashMap<>();
        acc.forEach((k, a) -> out.put(k, new CovariateScore(a[0], a[1], a[2], sizes.get(k))));
        return out;
    }

    /** Windows of values and covariates; {@code y} and {@code target} are {@code [n, steps, 1]}. */
    private record CovariateWindows(Tensor y, Tensor past, Tensor future, Tensor target, boolean[] holiday,
            boolean[] afterPriceStep) {
    }

    private static CovariateWindows covariateWindows(SeasonalSeries.Covariates d, int from, int to, int lookback,
            int horizon, int stride) {
        int count = (to - from - lookback - horizon) / stride + 1;
        float[] y = new float[count * lookback];
        float[] past = new float[count * lookback * 2];
        float[] future = new float[count * horizon * 2];
        float[] target = new float[count * horizon];
        boolean[] holiday = new boolean[count * horizon];
        boolean[] after = new boolean[count * horizon];
        for (int w = 0; w < count; w++) {
            int s = from + w * stride;
            for (int t = 0; t < lookback; t++) {
                y[w * lookback + t] = d.target()[s + t];
                past[(w * lookback + t) * 2] = d.price()[s + t];
                past[(w * lookback + t) * 2 + 1] = d.holiday()[s + t];
            }
            boolean changed = false;
            for (int t = 0; t < horizon; t++) {
                int i = s + lookback + t;
                target[w * horizon + t] = d.target()[i];
                future[(w * horizon + t) * 2] = d.price()[i];
                future[(w * horizon + t) * 2 + 1] = d.holiday()[i];
                holiday[w * horizon + t] = d.holiday()[i] > 0;
                changed |= d.price()[i] != d.price()[i - 1];
                after[w * horizon + t] = changed;
            }
        }
        return new CovariateWindows(Tensor.of(Shape.of(count, lookback, 1), y),
                Tensor.of(Shape.of(count, lookback, 2), past), Tensor.of(Shape.of(count, horizon, 2), future),
                Tensor.of(Shape.of(count, horizon, 1), target), holiday, after);
    }

    private static double[] covariateErrors(Tensor pred, CovariateWindows test) {
        float[] p = pred.toArray();
        float[] t = test.target().toArray();
        double all = 0;
        double hol = 0;
        double aft = 0;
        int nh = 0;
        int na = 0;
        for (int i = 0; i < p.length; i++) {
            double e = (p[i] - t[i]) * (p[i] - t[i]);
            all += e;
            if (test.holiday()[i]) {
                hol += e;
                nh++;
            }
            if (test.afterPriceStep()[i]) {
                aft += e;
                na++;
            }
        }
        return new double[] {all / p.length, hol / Math.max(1, nh), aft / Math.max(1, na)};
    }

    /** A model over {values, past covariates, future covariates}. */
    private record CovariateModel(Function<List<Var>, Var> forward, List<Var> parameters,
            float lr) {
        Tensor fit(CovariateWindows train, CovariateWindows test, int steps, long seed) {
            Optimizer opt = Optimizer.adam(lr);
            SplittableRandom r = new SplittableRandom(seed);
            int n = train.y().shape().dim(0);
            for (int s = 0; s < steps; s++) {
                float[] idx = new float[32];
                for (int i = 0; i < 32; i++) {
                    idx[i] = r.nextInt(n);
                }
                Tensor sel = Tensor.of(Shape.of(32), idx);
                forward.apply(List.of(Var.of(train.y().gather(0, sel)), Var.of(train.past().gather(0, sel)),
                        Var.of(train.future().gather(0, sel)))).sub(train.target().gather(0, sel)).square().mean()
                        .backward();
                opt.step(parameters);
            }
            return Var.noGrad(() -> forward.apply(List.of(Var.of(test.y()), Var.of(test.past()),
                    Var.of(test.future())))).value();
        }
    }
}
