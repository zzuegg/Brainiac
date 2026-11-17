package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

class ForecastingLayersTest {

    static final int LOOKBACK = 48;
    static final int HORIZON = 12;
    static final int PERIOD = 12;
    static final float[] SERIES = SeasonalSeries.generate(700, PERIOD, 0.004, 1.0, 0.1, 3);
    static final SeasonalSeries.Windows TRAIN = SeasonalSeries.windows(SERIES, 0, 500, LOOKBACK, HORIZON, 2);
    static final SeasonalSeries.Windows TEST = SeasonalSeries.windows(SERIES, 500, 700, LOOKBACK, HORIZON, 4);

    /** Trains a point forecaster with MSE and returns its test MSE. */
    static double fit(UnaryOperator<Var> model, List<Var> params, int steps, float lr) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(1);
        int n = TRAIN.size();
        for (int s = 0; s < steps; s++) {
            int[] idx = r.ints(32, 0, n).toArray();
            float[] x = new float[32 * LOOKBACK];
            float[] y = new float[32 * HORIZON];
            float[] all = TRAIN.inputs().toArray();
            float[] tgt = TRAIN.targets().toArray();
            for (int i = 0; i < 32; i++) {
                System.arraycopy(all, idx[i] * LOOKBACK, x, i * LOOKBACK, LOOKBACK);
                System.arraycopy(tgt, idx[i] * HORIZON, y, i * HORIZON, HORIZON);
            }
            Var loss = model.apply(Var.of(Tensor.of(Shape.of(32, LOOKBACK), x)))
                    .sub(Tensor.of(Shape.of(32, HORIZON), y)).square().mean();
            loss.backward();
            opt.step(params);
        }
        return SeasonalSeries.mse(model.apply(Var.of(TEST.inputs())).value(), TEST.targets());
    }

    static double naive() {
        return SeasonalSeries.mse(SeasonalSeries.naive(TEST.inputs(), HORIZON), TEST.targets());
    }

    static double seasonalNaive() {
        return SeasonalSeries.mse(SeasonalSeries.seasonalNaive(TEST.inputs(), HORIZON, PERIOD), TEST.targets());
    }

    static double score(Layer layer, int steps, float lr) {
        return fit(x -> layer.forward(x, true), layer.parameters(), steps, lr);
    }

    @Test
    void forecastersBeatTheNaiveBaselines() {
        double naive = naive();
        double seasonal = seasonalNaive();
        double dlinear = score(new DLinear(LOOKBACK, HORIZON, 13, new SplittableRandom(1)), 1500, 0.005f);
        double nbeats = score(NBeats.generic(LOOKBACK, HORIZON, 2, 64, 1), 300, 0.002f);
        double interpretable = score(NBeats.interpretable(LOOKBACK, HORIZON, 64, 1), 300, 0.002f);
        double patch = score(new PatchTst(LOOKBACK, HORIZON, 12, 6, 16, 1, 2, true, new SplittableRandom(1)), 150,
                0.005f);
        TemporalConvNet tcn = new TemporalConvNet(1, 16, 3, 4, new SplittableRandom(1));
        Var tcnHead = Var.param(Tensor.normal(Shape.of(16, HORIZON), 0, 0.25f, new SplittableRandom(2)));
        List<Var> tcnParams = new ArrayList<>(tcn.parameters());
        tcnParams.add(tcnHead);
        double tcnScore = fit(x -> {
            RevIn.Stats s = RevIn.stats(x.value());
            Var h = tcn.forward(RevIn.normalize(x, s).reshape(x.shape().dim(0), LOOKBACK, 1), true);
            return RevIn.denormalize(h.slice(1, LOOKBACK - 1, LOOKBACK).reshape(x.shape().dim(0), 16)
                    .matmul(tcnHead), s);
        }, tcnParams, 200, 0.005f);
        System.out.printf("test MSE: naive %.4f seasonal-naive %.4f | DLinear %.4f N-BEATS %.4f N-BEATS-I %.4f "
                + "PatchTST %.4f TCN %.4f%n", naive, seasonal, dlinear, nbeats, interpretable, patch, tcnScore);
        for (double m : new double[] {dlinear, nbeats, interpretable, patch, tcnScore}) {
            assertTrue(m < naive / 3, "a forecaster should beat the naive forecast clearly: " + m);
        }
        assertTrue(dlinear < seasonal && patch < seasonal, "DLinear and PatchTST should beat seasonal naive");
    }

    @Test
    void temporalConvolutionIsCausal() {
        TemporalConvNet tcn = new TemporalConvNet(1, 4, 2, 3, new SplittableRandom(1));
        assertEquals(15, tcn.receptiveField());
        float[] a = new float[20];
        float[] b = new float[20];
        for (int i = 0; i < 20; i++) {
            a[i] = (float) Math.sin(i);
            b[i] = i < 10 ? a[i] : -a[i];
        }
        float[] ya = tcn.forward(Var.of(Tensor.of(Shape.of(1, 20, 1), a))).value().toArray();
        float[] yb = tcn.forward(Var.of(Tensor.of(Shape.of(1, 20, 1), b))).value().toArray();
        for (int i = 0; i < 10 * 4; i++) {
            assertEquals(ya[i], yb[i], 1e-6, "outputs before the change must not see it");
        }
        GradCheck.Result g = GradCheck.check(xs -> TemporalConvNet.causalConv(xs.get(0), xs.get(1), xs.get(2), 2, 2)
                .tanh().sum(), List.of(Tensor.normal(Shape.of(2, 5, 2), 0, 1, new SplittableRandom(3)),
                        Tensor.normal(Shape.of(4, 3), 0, 0.5f, new SplittableRandom(4)),
                        Tensor.zeros(Shape.of(1, 3))));
        assertTrue(g.passed(), g.toString());
    }

    @Test
    void deepArGivesCalibratedIntervals() {
        DeepAr model = new DeepAr(24, PERIOD, 1);
        SplittableRandom r = new SplittableRandom(2);
        int window = LOOKBACK + HORIZON;
        for (int s = 0; s < 120; s++) {
            float[][] batch = new float[16][];
            int[] phases = new int[16];
            for (int i = 0; i < 16; i++) {
                int start = r.nextInt(500 - window);
                batch[i] = java.util.Arrays.copyOfRange(SERIES, start, start + window);
                phases[i] = start % PERIOD;
            }
            model.trainStep(batch, phases, LOOKBACK);
        }
        int inside = 0;
        int total = 0;
        double err = 0;
        double naiveErr = 0;
        for (int start = 500; start + window <= 700; start += 20) {
            float[] context = java.util.Arrays.copyOfRange(SERIES, start, start + LOOKBACK);
            float[][] paths = model.sample(context, start % PERIOD, HORIZON, 40, new SplittableRandom(start));
            float[] lo = DeepAr.quantile(paths, 0.1);
            float[] hi = DeepAr.quantile(paths, 0.9);
            float[] mid = DeepAr.quantile(paths, 0.5);
            for (int h = 0; h < HORIZON; h++) {
                float truth = SERIES[start + LOOKBACK + h];
                inside += truth >= lo[h] && truth <= hi[h] ? 1 : 0;
                total++;
                err += (mid[h] - truth) * (mid[h] - truth);
                naiveErr += (context[LOOKBACK - 1] - truth) * (context[LOOKBACK - 1] - truth);
            }
        }
        double coverage = inside / (double) total;
        System.out.printf("DeepAR: 80%% interval coverage %.2f, median MSE %.4f vs naive %.4f, nll %.3f%n", coverage,
                err / total, naiveErr / total, model.lastLoss());
        assertTrue(coverage > 0.5 && coverage <= 1.0, "coverage " + coverage);
        assertTrue(err < naiveErr / 2, "median forecast should beat naive");
    }

    @Test
    void pinballLossLearnsPredictionIntervals() {
        SplittableRandom r = new SplittableRandom(4);
        io.github.zzuegg.brainiac.nn.Model m = io.github.zzuegg.brainiac.nn.Model.sequential().input(1)
                .dense(32, io.github.zzuegg.brainiac.nn.Activation.RELU).dense(3).seed(1).build();
        Optimizer opt = Optimizer.adam(0.01f);
        float[] qs = {0.1f, 0.5f, 0.9f};
        for (int s = 0; s < 800; s++) {
            float[] x = new float[128];
            float[] y = new float[128];
            for (int i = 0; i < 128; i++) {
                x[i] = (float) r.nextDouble(-2, 2);
                y[i] = (float) (x[i] + (0.2 + Math.abs(x[i])) * r.nextGaussian());
            }
            Var loss = QuantileLoss.pinball(m.forward(Var.of(Tensor.of(Shape.of(128, 1), x)), true),
                    Tensor.of(Shape.of(128, 1), y), qs);
            loss.backward();
            opt.step(m.parameters());
        }
        int inside = 0;
        int below = 0;
        int n = 2000;
        float[] x = new float[n];
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) r.nextDouble(-2, 2);
            y[i] = (float) (x[i] + (0.2 + Math.abs(x[i])) * r.nextGaussian());
        }
        float[] p = m.forward(Var.of(Tensor.of(Shape.of(n, 1), x))).value().toArray();
        double width0 = 0;
        double width2 = 0;
        for (int i = 0; i < n; i++) {
            inside += y[i] >= p[3 * i] && y[i] <= p[3 * i + 2] ? 1 : 0;
            below += y[i] < p[3 * i + 1] ? 1 : 0;
            if (Math.abs(x[i]) < 0.3) {
                width0 += p[3 * i + 2] - p[3 * i];
            } else if (Math.abs(x[i]) > 1.7) {
                width2 += p[3 * i + 2] - p[3 * i];
            }
        }
        System.out.printf("pinball: 80%% interval coverage %.3f, median below-fraction %.3f%n", inside / (double) n,
                below / (double) n);
        assertEquals(0.8, inside / (double) n, 0.06);
        assertEquals(0.5, below / (double) n, 0.06);
        assertTrue(width2 > width0, "intervals should widen where noise grows");
    }
}
