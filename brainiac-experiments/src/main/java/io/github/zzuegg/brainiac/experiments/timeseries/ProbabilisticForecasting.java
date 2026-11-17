package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;
import io.github.zzuegg.brainiac.experimental.nn.DeepAr;
import io.github.zzuegg.brainiac.experimental.nn.QuantileLoss;
import io.github.zzuegg.brainiac.experimental.nn.RevIn;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.LinkedHashMap;
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
}
