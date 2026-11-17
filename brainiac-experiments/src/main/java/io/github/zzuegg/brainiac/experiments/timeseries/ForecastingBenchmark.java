package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;
import io.github.zzuegg.brainiac.experimental.nn.DLinear;
import io.github.zzuegg.brainiac.experimental.nn.NBeats;
import io.github.zzuegg.brainiac.experimental.nn.PatchTst;
import io.github.zzuegg.brainiac.experimental.nn.RevIn;
import io.github.zzuegg.brainiac.experimental.nn.TemporalConvNet;
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
}
