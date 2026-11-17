package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A DeepAR-style probabilistic forecaster (Salinas et al., 2017,
 * <a href="https://arxiv.org/abs/1704.04110">arXiv:1704.04110</a>): an autoregressive recurrent
 * network (GRU) reads the series one step at a time, its input at step {@code t} being the
 * observed value at {@code t - 1} plus calendar features, and a {@link GaussianHead}
 * predicts the distribution of the value at {@code t}. Training maximizes the likelihood
 * of observed values; forecasting feeds sampled values back in to draw whole future
 * paths, from which means and quantiles (prediction intervals) follow. Each window is
 * scaled by its context's mean and deviation so one model serves series of any level.
 *
 * <p>Calendar features here are {@code sin} and {@code cos} of the position within a
 * known season length, a common DeepAR covariate.
 */
public final class DeepAr {

    private final int season;
    private final Layer rnn;
    private final GaussianHead head;
    private Optimizer optimizer = Optimizer.adam(5e-3f);
    private double lastLoss = Double.NaN;

    /**
     * Creates the model.
     *
     * @param hidden GRU units
     * @param season the season length for calendar features (0 for none)
     * @param seed the initialization seed
     */
    public DeepAr(int hidden, int season, long seed) {
        this.season = season;
        SplittableRandom r = new SplittableRandom(seed);
        this.rnn = Recurrent.gru(hidden).returnSequences(true).create(Shape.of(8, inputs()), r);
        this.head = new GaussianHead(hidden, r);
    }

    private int inputs() {
        return season > 0 ? 3 : 1;
    }

    /** @param o the optimizer @return this model */
    public DeepAr optimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    /** @return the loss of the last training batch */
    public double lastLoss() {
        return lastLoss;
    }

    /** @return every trainable parameter */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(rnn.parameters());
        p.addAll(head.parameters());
        return p;
    }

    /** Builds inputs {@code [batch, steps, inputs]} from scaled values (lagged by one step). */
    private Tensor inputs(float[][] scaled, int[] startPhase, int steps) {
        int b = scaled.length;
        int k = inputs();
        float[] x = new float[b * steps * k];
        for (int i = 0; i < b; i++) {
            for (int t = 0; t < steps; t++) {
                int o = (i * steps + t) * k;
                x[o] = t == 0 ? 0 : scaled[i][t - 1];
                if (season > 0) {
                    double phase = 2 * Math.PI * (startPhase[i] + t) / season;
                    x[o + 1] = (float) Math.sin(phase);
                    x[o + 2] = (float) Math.cos(phase);
                }
            }
        }
        return Tensor.of(Shape.of(b, steps, k), x);
    }

    private static float[] scale(float[] context) {
        double mean = Arrays.stream(toDouble(context)).average().orElse(0);
        double var = 0;
        for (float v : context) {
            var += (v - mean) * (v - mean);
        }
        return new float[] {(float) mean, (float) Math.sqrt(var / context.length + 1e-6)};
    }

    private static double[] toDouble(float[] f) {
        double[] d = new double[f.length];
        for (int i = 0; i < f.length; i++) {
            d[i] = f[i];
        }
        return d;
    }

    /**
     * One gradient step on windows of the series.
     *
     * @param windows {@code [batch][steps]} raw values; the first {@code context} steps fix the scale
     * @param phases each window's start position within the season
     * @param context steps used for scaling
     * @return the negative log-likelihood
     */
    public double trainStep(float[][] windows, int[] phases, int context) {
        int b = windows.length;
        int steps = windows[0].length;
        float[][] scaled = new float[b][steps];
        float[] target = new float[b * steps];
        for (int i = 0; i < b; i++) {
            float[] s = scale(Arrays.copyOf(windows[i], context));
            for (int t = 0; t < steps; t++) {
                scaled[i][t] = (windows[i][t] - s[0]) / s[1];
                target[i * steps + t] = scaled[i][t];
            }
        }
        Var out = head.forward(rnn.forward(Var.of(inputs(scaled, phases, steps)), true), true);
        Var loss = GaussianHead.nll(out.slice(1, 1, steps), Tensor.of(Shape.of(b, steps, 1), target)
                .slice(1, 1, steps));
        loss.backward();
        optimizer.step(parameters());
        lastLoss = loss.value().item();
        return lastLoss;
    }

    /**
     * Draws future paths.
     *
     * @param context the observed history
     * @param phase the history's start position within the season
     * @param horizon steps to forecast
     * @param samples paths to draw
     * @param random the sampling source
     * @return {@code [samples][horizon]} sampled futures in the series' units
     */
    public float[][] sample(float[] context, int phase, int horizon, int samples, SplittableRandom random) {
        float[] s = scale(context);
        int total = context.length + horizon;
        float[][] scaled = new float[samples][total];
        for (int i = 0; i < samples; i++) {
            for (int t = 0; t < context.length; t++) {
                scaled[i][t] = (context[t] - s[0]) / s[1];
            }
        }
        int[] phases = new int[samples];
        Arrays.fill(phases, phase);
        for (int h = 0; h < horizon; h++) {
            int t = context.length + h;
            int steps = t + 1;
            float[][] prefix = new float[samples][];
            for (int i = 0; i < samples; i++) {
                prefix[i] = Arrays.copyOf(scaled[i], steps);
            }
            Tensor out = head.forward(rnn.forward(Var.of(inputs(prefix, phases, steps)), false), false).value()
                    .slice(1, t, t + 1).to(DType.FLOAT32);
            float[] o = out.toArray();
            for (int i = 0; i < samples; i++) {
                scaled[i][t] = (float) (o[2 * i] + o[2 * i + 1] * random.nextGaussian());
            }
        }
        float[][] paths = new float[samples][horizon];
        for (int i = 0; i < samples; i++) {
            for (int h = 0; h < horizon; h++) {
                paths[i][h] = scaled[i][context.length + h] * s[1] + s[0];
            }
        }
        return paths;
    }

    /**
     * @param paths sampled paths {@code [samples][horizon]}
     * @param q the quantile in {@code [0, 1]}
     * @return the per-step empirical quantile
     */
    public static float[] quantile(float[][] paths, double q) {
        int horizon = paths[0].length;
        float[] out = new float[horizon];
        float[] col = new float[paths.length];
        for (int h = 0; h < horizon; h++) {
            for (int i = 0; i < paths.length; i++) {
                col[i] = paths[i][h];
            }
            Arrays.sort(col);
            out[h] = col[(int) Math.min(col.length - 1, Math.floor(q * col.length))];
        }
        return out;
    }
}
