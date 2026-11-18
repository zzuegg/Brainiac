package io.github.zzuegg.brainiac.experiments.events;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.IrregularSeries;
import io.github.zzuegg.brainiac.experimental.data.SeriesClassificationTasks;
import io.github.zzuegg.brainiac.experimental.events.IrregularBatch;
import io.github.zzuegg.brainiac.experimental.events.IrregularSequenceModels;
import io.github.zzuegg.brainiac.experimental.events.IrregularSequenceModels.SequenceModel;
import io.github.zzuegg.brainiac.experimental.events.TimeSeriesImputer;
import io.github.zzuegg.brainiac.experimental.ml.GaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.StationaryKernel;
import io.github.zzuegg.brainiac.experimental.nn.ControlPath;
import io.github.zzuegg.brainiac.experimental.nn.NeuralCde;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wave 10 items 10.5.10 and 10.5.11 on {@link IrregularSeries}: models for irregularly sampled
 * series and deep imputers, against classical fills. Because the generator keeps the complete
 * series, every unobserved value is scored against the truth (in standardised units).
 * <ul>
 *   <li>{@link #reconstruction}: interpolation (MSE on all unobserved values) and extrapolation
 *       (MSE on the last quarter of the window, observed only up to 75%) for GRU-D, ODE-RNN, the
 *       latent ODE and mTAN against a GRU on forward-filled values with mask and time-gap channels,
 *       linear interpolation, last observation carried forward, and (extrapolation) linear
 *       interpolation followed by a DLinear-style ridge forecaster.</li>
 *   <li>{@link #classification}: cylinder-bell-funnel series after random subsampling, for GRU-D,
 *       mTAN, the forward-fill GRU and the wave 8 {@link NeuralCde} on the filled path.</li>
 *   <li>{@link #imputation}: BRITS, SAITS and CSDI against the series mean, LOCF, linear
 *       interpolation, a Kalman smoother (local linear trend, noise chosen by likelihood) and a
 *       Gaussian process per channel; MAE, and CRPS for the probabilistic ones.</li>
 * </ul>
 * A downstream forecaster trained on imputed data and {@code LiquidCell} were not run.
 */
public final class IrregularSeriesShootout {

    /**
     * Sizes.
     *
     * @param train training series
     * @param test test series
     * @param steps grid length
     * @param channels channels
     * @param trainSteps optimiser steps per model
     * @param batch series per step
     * @param units hidden width
     */
    public record Budget(int train, int test, int steps, int channels, int trainSteps, int batch, int units) {
    }

    private IrregularSeriesShootout() {
    }

    // ------------------------------------------------------------ classical fills

    /** Linear interpolation per channel, holding the nearest value beyond the ends (0 if empty). */
    static float[][][] linear(float[][][] v, boolean[][][] m) {
        int n = v.length;
        int steps = v[0].length;
        int c = v[0][0].length;
        float[][][] out = new float[n][steps][c];
        for (int i = 0; i < n; i++) {
            for (int ch = 0; ch < c; ch++) {
                int prev = -1;
                for (int t = 0; t < steps; t++) {
                    if (!m[i][t][ch]) {
                        continue;
                    }
                    if (prev < 0) {
                        for (int u = 0; u <= t; u++) {
                            out[i][u][ch] = v[i][t][ch];
                        }
                    } else {
                        for (int u = prev; u <= t; u++) {
                            double a = (double) (u - prev) / (t - prev);
                            out[i][u][ch] = (float) ((1 - a) * v[i][prev][ch] + a * v[i][t][ch]);
                        }
                    }
                    prev = t;
                }
                if (prev >= 0) {
                    for (int u = prev; u < steps; u++) {
                        out[i][u][ch] = v[i][prev][ch];
                    }
                }
            }
        }
        return out;
    }

    /** Last observation carried forward (the first observation carried backward before it). */
    static float[][][] locf(float[][][] v, boolean[][][] m) {
        int n = v.length;
        int steps = v[0].length;
        int c = v[0][0].length;
        float[][][] out = new float[n][steps][c];
        for (int i = 0; i < n; i++) {
            for (int ch = 0; ch < c; ch++) {
                float first = 0;
                for (int t = 0; t < steps; t++) {
                    if (m[i][t][ch]) {
                        first = v[i][t][ch];
                        break;
                    }
                }
                float cur = first;
                for (int t = 0; t < steps; t++) {
                    if (m[i][t][ch]) {
                        cur = v[i][t][ch];
                    }
                    out[i][t][ch] = cur;
                }
            }
        }
        return out;
    }

    /** The mean of the observed values of each series and channel. */
    static float[][][] mean(float[][][] v, boolean[][][] m) {
        int n = v.length;
        int steps = v[0].length;
        int c = v[0][0].length;
        float[][][] out = new float[n][steps][c];
        for (int i = 0; i < n; i++) {
            for (int ch = 0; ch < c; ch++) {
                double s = 0;
                int k = 0;
                for (int t = 0; t < steps; t++) {
                    if (m[i][t][ch]) {
                        s += v[i][t][ch];
                        k++;
                    }
                }
                for (int t = 0; t < steps; t++) {
                    out[i][t][ch] = k == 0 ? 0 : (float) (s / k);
                }
            }
        }
        return out;
    }

    /**
     * A Kalman smoother per channel with a local linear trend and missing observations skipped;
     * the process noise is chosen from a grid by the observed log-likelihood.
     *
     * @return {@code {means, variances}} per value
     */
    static float[][][][] kalman(float[][][] v, boolean[][][] m) {
        int n = v.length;
        int steps = v[0].length;
        int c = v[0][0].length;
        float[][][] mu = new float[n][steps][c];
        float[][][] var = new float[n][steps][c];
        double obsNoise = 0.05 * 0.05;
        for (int i = 0; i < n; i++) {
            for (int ch = 0; ch < c; ch++) {
                double best = Double.NEGATIVE_INFINITY;
                double[][] bestOut = null;
                for (double q : new double[] {1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1}) {
                    double[][] o = smooth(v[i], m[i], ch, q, obsNoise);
                    if (o[2][0] > best) {
                        best = o[2][0];
                        bestOut = o;
                    }
                }
                for (int t = 0; t < steps; t++) {
                    mu[i][t][ch] = (float) bestOut[0][t];
                    var[i][t][ch] = (float) (bestOut[1][t] + obsNoise);
                }
            }
        }
        return new float[][][][] {mu, var};
    }

    /** RTS smoothing of a local linear trend: {@code {means, variances, {log-likelihood}}}. */
    private static double[][] smooth(float[][] v, boolean[][] m, int ch, double q, double r) {
        int steps = v.length;
        double[][] fm = new double[steps][2];
        double[][][] fp = new double[steps][2][2];
        double[][] pm = new double[steps][2];
        double[][][] pp = new double[steps][2][2];
        double[] x = {0, 0};
        double[][] p = {{4, 0}, {0, 1}};
        double ll = 0;
        for (int t = 0; t < steps; t++) {
            if (t > 0) {
                x = new double[] {x[0] + x[1], x[1]};
                double[][] np = {{p[0][0] + p[0][1] + p[1][0] + p[1][1] + q / 4, p[0][1] + p[1][1] + q / 2},
                    {p[1][0] + p[1][1] + q / 2, p[1][1] + q}};
                p = np;
            }
            pm[t] = x.clone();
            pp[t] = new double[][] {p[0].clone(), p[1].clone()};
            if (m[t][ch]) {
                double s = p[0][0] + r;
                double innov = v[t][ch] - x[0];
                ll += -0.5 * (Math.log(2 * Math.PI * s) + innov * innov / s);
                double k0 = p[0][0] / s;
                double k1 = p[1][0] / s;
                x = new double[] {x[0] + k0 * innov, x[1] + k1 * innov};
                p = new double[][] {{p[0][0] - k0 * p[0][0], p[0][1] - k0 * p[0][1]},
                    {p[1][0] - k1 * p[0][0], p[1][1] - k1 * p[0][1]}};
            }
            fm[t] = x.clone();
            fp[t] = new double[][] {p[0].clone(), p[1].clone()};
        }
        double[] means = new double[steps];
        double[] vars = new double[steps];
        double[] sx = fm[steps - 1].clone();
        double[][] sp = fp[steps - 1];
        means[steps - 1] = sx[0];
        vars[steps - 1] = sp[0][0];
        for (int t = steps - 2; t >= 0; t--) {
            double[][] f = {{1, 1}, {0, 1}};
            double[][] pf = mul(fp[t], transpose(f));
            double[][] inv = inverse(pp[t + 1]);
            double[][] g = mul(pf, inv);
            double[] d = {sx[0] - pm[t + 1][0], sx[1] - pm[t + 1][1]};
            double[] nx = {fm[t][0] + g[0][0] * d[0] + g[0][1] * d[1], fm[t][1] + g[1][0] * d[0] + g[1][1] * d[1]};
            double[][] dp = {{sp[0][0] - pp[t + 1][0][0], sp[0][1] - pp[t + 1][0][1]},
                {sp[1][0] - pp[t + 1][1][0], sp[1][1] - pp[t + 1][1][1]}};
            double[][] np = add(fp[t], mul(mul(g, dp), transpose(g)));
            sx = nx;
            sp = np;
            means[t] = sx[0];
            vars[t] = Math.max(1e-9, sp[0][0]);
        }
        return new double[][] {means, vars, {ll}};
    }

    private static double[][] mul(double[][] a, double[][] b) {
        double[][] o = new double[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
            }
        }
        return o;
    }

    private static double[][] add(double[][] a, double[][] b) {
        return new double[][] {{a[0][0] + b[0][0], a[0][1] + b[0][1]}, {a[1][0] + b[1][0], a[1][1] + b[1][1]}};
    }

    private static double[][] transpose(double[][] a) {
        return new double[][] {{a[0][0], a[1][0]}, {a[0][1], a[1][1]}};
    }

    private static double[][] inverse(double[][] a) {
        double det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        return new double[][] {{a[1][1] / det, -a[0][1] / det}, {-a[1][0] / det, a[0][0] / det}};
    }

    /** A Gaussian process per series and channel (RBF, hyperparameters by marginal likelihood). */
    static float[][][][] gaussianProcess(float[][][] v, boolean[][][] m, double[] times) {
        int n = v.length;
        int steps = v[0].length;
        int c = v[0][0].length;
        float[][][] mu = new float[n][steps][c];
        float[][][] var = new float[n][steps][c];
        double[][] grid = new double[steps][];
        for (int t = 0; t < steps; t++) {
            grid[t] = new double[] {times[t]};
        }
        for (int i = 0; i < n; i++) {
            for (int ch = 0; ch < c; ch++) {
                List<double[]> xs = new ArrayList<>();
                List<Double> ys = new ArrayList<>();
                for (int t = 0; t < steps; t++) {
                    if (m[i][t][ch]) {
                        xs.add(grid[t]);
                        ys.add((double) v[i][t][ch]);
                    }
                }
                if (xs.size() < 2) {
                    for (int t = 0; t < steps; t++) {
                        mu[i][t][ch] = xs.isEmpty() ? 0 : ys.get(0).floatValue();
                        var[i][t][ch] = 1;
                    }
                    continue;
                }
                double[] y = ys.stream().mapToDouble(Double::doubleValue).toArray();
                GaussianProcess gp = GaussianProcess.optimize(xs.toArray(double[][]::new), y,
                        StationaryKernel.rbf(0.1, 1), 0.01, 30);
                GaussianProcess.Prediction p = gp.predict(grid);
                for (int t = 0; t < steps; t++) {
                    mu[i][t][ch] = (float) p.mean()[t];
                    var[i][t][ch] = (float) Math.max(1e-6, p.variance()[t]);
                }
            }
        }
        return new float[][][][] {mu, var};
    }

    /** A DLinear-style forecaster: ridge regression from the interpolated history to the horizon. */
    static float[][][] dlinear(float[][][] trainV, boolean[][][] trainIn, float[][][] testV, boolean[][][] testIn,
            int cut) {
        float[][][] trainFill = linear(trainV, trainIn);
        float[][][] testFill = linear(testV, testIn);
        int steps = trainV[0].length;
        int c = trainV[0][0].length;
        int h = steps - cut;
        int p = cut + 1;
        double[][] xtx = new double[p][p];
        double[][] xty = new double[p][h];
        for (int i = 0; i < trainV.length; i++) {
            for (int ch = 0; ch < c; ch++) {
                double[] x = new double[p];
                for (int t = 0; t < cut; t++) {
                    x[t] = trainFill[i][t][ch];
                }
                x[cut] = 1;
                for (int a = 0; a < p; a++) {
                    for (int b = 0; b < p; b++) {
                        xtx[a][b] += x[a] * x[b];
                    }
                    for (int k = 0; k < h; k++) {
                        xty[a][k] += x[a] * trainV[i][cut + k][ch];
                    }
                }
            }
        }
        for (int a = 0; a < p; a++) {
            xtx[a][a] += 1.0;
        }
        double[][] w = solve(xtx, xty);
        float[][][] out = new float[testV.length][steps][c];
        for (int i = 0; i < testV.length; i++) {
            for (int ch = 0; ch < c; ch++) {
                for (int t = 0; t < cut; t++) {
                    out[i][t][ch] = testFill[i][t][ch];
                }
                for (int k = 0; k < h; k++) {
                    double s = w[cut][k];
                    for (int t = 0; t < cut; t++) {
                        s += w[t][k] * testFill[i][t][ch];
                    }
                    out[i][cut + k][ch] = (float) s;
                }
            }
        }
        return out;
    }

    private static double[][] solve(double[][] a, double[][] b) {
        int n = a.length;
        int m = b[0].length;
        double[][] aa = new double[n][];
        double[][] bb = new double[n][];
        for (int i = 0; i < n; i++) {
            aa[i] = a[i].clone();
            bb[i] = b[i].clone();
        }
        for (int col = 0; col < n; col++) {
            int piv = col;
            for (int r = col + 1; r < n; r++) {
                if (Math.abs(aa[r][col]) > Math.abs(aa[piv][col])) {
                    piv = r;
                }
            }
            double[] t = aa[col];
            aa[col] = aa[piv];
            aa[piv] = t;
            t = bb[col];
            bb[col] = bb[piv];
            bb[piv] = t;
            for (int r = 0; r < n; r++) {
                if (r == col) {
                    continue;
                }
                double f = aa[r][col] / aa[col][col];
                for (int k = col; k < n; k++) {
                    aa[r][k] -= f * aa[col][k];
                }
                for (int k = 0; k < m; k++) {
                    bb[r][k] -= f * bb[col][k];
                }
            }
        }
        double[][] x = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < m; k++) {
                x[i][k] = bb[i][k] / aa[i][i];
            }
        }
        return x;
    }

    // ------------------------------------------------------------ forward-fill GRU and CDE baselines

    /** A GRU on {@code [forward-filled values, mask, time since observation]}. */
    static SequenceModel forwardFillGru(int channels, int units, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Layer gru = Recurrent.gru(units).returnSequences(true).create(Shape.of(1, 3 * channels), r);
        float limit = (float) Math.sqrt(6.0 / (units + channels));
        Var out = Var.param(Tensor.uniform(Shape.of(units, channels), -limit, limit, r));
        List<Var> params = new ArrayList<>(gru.parameters());
        params.add(out);
        return new SequenceModel() {
            @Override
            public Var[] forward(IrregularBatch b) {
                Tensor m = b.mask();
                Tensor filled = b.values().add(Tensor.full(m.shape(), 1f).sub(m).mul(b.last()));
                Var x = Var.of(filled);
                Var h = gru.forward(Var.concat(-1, x, Var.of(m), Var.of(b.delta().mul((float) (b.steps() - 1)))));
                int u = h.shape().dim(2);
                return new Var[] {h.matmul(out), h.slice(1, b.steps() - 1, b.steps()).reshape(b.batch(), u), null};
            }

            @Override
            public int summaryWidth() {
                return units;
            }

            @Override
            public List<Var> parameters() {
                return params;
            }
        };
    }

    /** The wave 8 neural CDE on the path {@code (time, forward-filled values)}. */
    static SequenceModel neuralCde(int channels, int units, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        NeuralCde cde = new NeuralCde(channels + 1, units, units, ControlPath.Interpolation.LINEAR, 1, r);
        return new SequenceModel() {
            @Override
            public Var[] forward(IrregularBatch b) {
                Tensor m = b.mask();
                float[] filled = b.values().add(Tensor.full(m.shape(), 1f).sub(m).mul(b.last())).toArray();
                int bs = b.batch();
                int steps = b.steps();
                int c = b.channels();
                float[] path = new float[bs * steps * (c + 1)];
                for (int i = 0; i < bs; i++) {
                    for (int t = 0; t < steps; t++) {
                        path[(i * steps + t) * (c + 1)] = (float) b.times()[t] * 4;
                        for (int ch = 0; ch < c; ch++) {
                            path[(i * steps + t) * (c + 1) + 1 + ch] = filled[(i * steps + t) * c + ch];
                        }
                    }
                }
                return new Var[] {null, cde.forward(Var.of(Tensor.of(Shape.of(bs, steps, c + 1), path))), null};
            }

            @Override
            public int summaryWidth() {
                return units;
            }

            @Override
            public List<Var> parameters() {
                return cde.parameters();
            }
        };
    }

    // ------------------------------------------------------------ metrics

    private static double mse(float[][][] pred, float[][][] truth, boolean[][][] where) {
        double s = 0;
        int k = 0;
        for (int i = 0; i < pred.length; i++) {
            for (int t = 0; t < pred[i].length; t++) {
                for (int ch = 0; ch < pred[i][t].length; ch++) {
                    if (where[i][t][ch]) {
                        double d = pred[i][t][ch] - truth[i][t][ch];
                        s += d * d;
                        k++;
                    }
                }
            }
        }
        return s / Math.max(1, k);
    }

    private static double mae(float[][][] pred, float[][][] truth, boolean[][][] where) {
        double s = 0;
        int k = 0;
        for (int i = 0; i < pred.length; i++) {
            for (int t = 0; t < pred[i].length; t++) {
                for (int ch = 0; ch < pred[i][t].length; ch++) {
                    if (where[i][t][ch]) {
                        s += Math.abs(pred[i][t][ch] - truth[i][t][ch]);
                        k++;
                    }
                }
            }
        }
        return s / Math.max(1, k);
    }

    private static double gaussianCrps(float[][][] mu, float[][][] var, float[][][] truth, boolean[][][] where) {
        double s = 0;
        int k = 0;
        for (int i = 0; i < mu.length; i++) {
            for (int t = 0; t < mu[i].length; t++) {
                for (int ch = 0; ch < mu[i][t].length; ch++) {
                    if (where[i][t][ch]) {
                        double sd = Math.sqrt(var[i][t][ch]);
                        double z = (truth[i][t][ch] - mu[i][t][ch]) / sd;
                        double pdf = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
                        double cdf = 0.5 * (1 + erf(z / Math.sqrt(2)));
                        s += sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / Math.sqrt(Math.PI));
                        k++;
                    }
                }
            }
        }
        return s / Math.max(1, k);
    }

    private static double erf(double x) {
        double t = 1 / (1 + 0.3275911 * Math.abs(x));
        double y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
                * Math.exp(-x * x);
        return x >= 0 ? y : -y;
    }

    private static double sampleCrps(float[][][][] samples, float[][][] truth, boolean[][][] where) {
        int s = samples.length;
        double total = 0;
        int k = 0;
        for (int i = 0; i < truth.length; i++) {
            for (int t = 0; t < truth[i].length; t++) {
                for (int ch = 0; ch < truth[i][t].length; ch++) {
                    if (!where[i][t][ch]) {
                        continue;
                    }
                    double a = 0;
                    double b = 0;
                    for (int p = 0; p < s; p++) {
                        a += Math.abs(samples[p][i][t][ch] - truth[i][t][ch]);
                        for (int q = 0; q < s; q++) {
                            b += Math.abs(samples[p][i][t][ch] - samples[q][i][t][ch]);
                        }
                    }
                    total += a / s - b / (2.0 * s * s);
                    k++;
                }
            }
        }
        return total / Math.max(1, k);
    }

    private static boolean[][][] not(boolean[][][] m) {
        boolean[][][] o = new boolean[m.length][m[0].length][m[0][0].length];
        for (int i = 0; i < m.length; i++) {
            for (int t = 0; t < m[i].length; t++) {
                for (int ch = 0; ch < m[i][t].length; ch++) {
                    o[i][t][ch] = !m[i][t][ch];
                }
            }
        }
        return o;
    }

    // ------------------------------------------------------------ experiments

    private static Map<String, SequenceModel> models(int c, Budget b, long seed) {
        Map<String, SequenceModel> m = new LinkedHashMap<>();
        m.put("GRU-D", IrregularSequenceModels.gruD(c, b.units(), seed));
        m.put("ODE-RNN", IrregularSequenceModels.odeRnn(c, b.units(), seed));
        m.put("latent ODE", IrregularSequenceModels.latentOde(c, b.units(), 8, 1f, seed));
        m.put("mTAN", IrregularSequenceModels.mtan(c, b.units(), 16, 16, seed));
        m.put("forward-fill GRU", forwardFillGru(c, b.units(), seed));
        return m;
    }

    /**
     * Interpolation and extrapolation.
     *
     * @param kind the dynamics
     * @param missing the missing share
     * @param mechanism the missingness
     * @param budget sizes
     * @param seed the seed
     * @return per model {@code {interpolation MSE, extrapolation MSE, ms}}
     */
    public static Map<String, double[]> reconstruction(IrregularSeries.Kind kind, double missing,
            IrregularSeries.Missing mechanism, Budget budget, long seed) {
        double[] times = IrregularSeries.times(budget.steps());
        float[][][] train = IrregularSeries.generate(kind, budget.train(), budget.steps(), budget.channels(), seed);
        float[][][] test = IrregularSeries.generate(kind, budget.test(), budget.steps(), budget.channels(), seed + 1);
        boolean[][][] trainObs = IrregularSeries.mask(train, missing, mechanism, seed + 2);
        boolean[][][] testObs = IrregularSeries.mask(test, missing, mechanism, seed + 3);
        int c = train[0][0].length;
        int steps = budget.steps();
        int cut = (int) (steps * 0.75);
        boolean[][][] testHistory = new boolean[test.length][steps][c];
        boolean[][][] future = new boolean[test.length][steps][c];
        boolean[][][] trainHistory = new boolean[train.length][steps][c];
        for (int i = 0; i < test.length; i++) {
            for (int t = 0; t < steps; t++) {
                for (int ch = 0; ch < c; ch++) {
                    testHistory[i][t][ch] = t < cut && testObs[i][t][ch];
                    future[i][t][ch] = t >= cut;
                }
            }
        }
        for (int i = 0; i < train.length; i++) {
            for (int t = 0; t < cut; t++) {
                trainHistory[i][t] = trainObs[i][t].clone();
            }
        }
        boolean[][][] unobserved = not(testObs);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, SequenceModel> e : models(c, budget, seed).entrySet()) {
            long t0 = System.currentTimeMillis();
            SequenceModel m = e.getValue();
            IrregularSequenceModels.fit(m, train, trainObs, times, IrregularSequenceModels.Task.INTERPOLATION,
                    budget.trainSteps(), budget.batch(), 1e-2f, seed);
            double interp = mse(IrregularSequenceModels.predict(m, test, testObs, times), test, unobserved);
            SequenceModel f = models(c, budget, seed + 11).get(e.getKey());
            IrregularSequenceModels.fit(f, train, trainObs, times, IrregularSequenceModels.Task.EXTRAPOLATION,
                    budget.trainSteps(), budget.batch(), 1e-2f, seed);
            double extra = mse(IrregularSequenceModels.predict(f, test, testHistory, times), test, future);
            out.put(e.getKey(), new double[] {interp, extra, System.currentTimeMillis() - t0});
        }
        out.put("linear interpolation", new double[] {mse(linear(test, testObs), test, unobserved),
            mse(linear(test, testHistory), test, future), 0});
        out.put("LOCF", new double[] {mse(locf(test, testObs), test, unobserved), mse(locf(test, testHistory), test,
            future), 0});
        out.put("linear + DLinear", new double[] {Double.NaN,
            mse(dlinear(train, trainHistory, test, testHistory, cut), test, future), 0});
        return out;
    }

    /**
     * Classification of subsampled cylinder-bell-funnel series.
     *
     * @param missing the missing share
     * @param budget sizes (channels is ignored: 1)
     * @param seed the seed
     * @return per model {@code {accuracy, ms}}
     */
    public static Map<String, double[]> classification(double missing, Budget budget, long seed) {
        SeriesClassificationTasks.Task trainTask = SeriesClassificationTasks.cylinderBellFunnel(budget.train(),
                budget.steps(), seed);
        SeriesClassificationTasks.Task testTask = SeriesClassificationTasks.cylinderBellFunnel(budget.test(),
                budget.steps(), seed + 1);
        float[][][] train = lift(trainTask.series());
        float[][][] test = lift(testTask.series());
        double[] times = IrregularSeries.times(budget.steps());
        boolean[][][] trainObs = IrregularSeries.mask(train, missing, IrregularSeries.Missing.MAR, seed + 2);
        boolean[][][] testObs = IrregularSeries.mask(test, missing, IrregularSeries.Missing.MAR, seed + 3);
        Map<String, SequenceModel> models = new LinkedHashMap<>();
        models.put("GRU-D", IrregularSequenceModels.gruD(1, budget.units(), seed));
        models.put("mTAN", IrregularSequenceModels.mtan(1, budget.units(), 16, 16, seed));
        models.put("forward-fill GRU", forwardFillGru(1, budget.units(), seed));
        models.put("neural CDE", neuralCde(1, budget.units(), seed));
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, SequenceModel> e : models.entrySet()) {
            long t0 = System.currentTimeMillis();
            Var head = IrregularSequenceModels.fitClassifier(e.getValue(), train, trainObs, trainTask.labels(),
                    trainTask.classes(), times, budget.trainSteps(), budget.batch(), 1e-2f, seed);
            int[] pred = IrregularSequenceModels.classify(e.getValue(), head, test, testObs, times);
            double right = 0;
            for (int i = 0; i < pred.length; i++) {
                right += pred[i] == testTask.labels()[i] ? 1 : 0;
            }
            out.put(e.getKey(), new double[] {right / pred.length, System.currentTimeMillis() - t0});
        }
        return out;
    }

    private static float[][][] lift(float[][] series) {
        float[][][] out = new float[series.length][series[0].length][1];
        for (int i = 0; i < series.length; i++) {
            for (int t = 0; t < series[i].length; t++) {
                out[i][t][0] = series[i][t];
            }
        }
        return out;
    }

    /**
     * Imputation of every unobserved value.
     *
     * @param kind the dynamics
     * @param missing the missing share
     * @param mechanism the missingness
     * @param budget sizes
     * @param diffusionSteps CSDI diffusion steps
     * @param samples CSDI samples
     * @param seed the seed
     * @return per method {@code {MAE, CRPS (NaN for point methods), ms}}
     */
    public static Map<String, double[]> imputation(IrregularSeries.Kind kind, double missing,
            IrregularSeries.Missing mechanism, Budget budget, int diffusionSteps, int samples, long seed) {
        double[] times = IrregularSeries.times(budget.steps());
        float[][][] train = IrregularSeries.generate(kind, budget.train(), budget.steps(), budget.channels(), seed);
        float[][][] test = IrregularSeries.generate(kind, budget.test(), budget.steps(), budget.channels(), seed + 1);
        boolean[][][] trainObs = IrregularSeries.mask(train, missing, mechanism, seed + 2);
        boolean[][][] testObs = IrregularSeries.mask(test, missing, mechanism, seed + 3);
        boolean[][][] target = not(testObs);
        int c = train[0][0].length;
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("mean", new double[] {mae(mean(test, testObs), test, target), Double.NaN, 0});
        out.put("LOCF", new double[] {mae(locf(test, testObs), test, target), Double.NaN, 0});
        out.put("linear interpolation", new double[] {mae(linear(test, testObs), test, target), Double.NaN, 0});
        long t0 = System.currentTimeMillis();
        float[][][][] k = kalman(test, testObs);
        out.put("Kalman smoother", new double[] {mae(k[0], test, target), gaussianCrps(k[0], k[1], test, target),
            System.currentTimeMillis() - t0});
        t0 = System.currentTimeMillis();
        float[][][][] gp = gaussianProcess(test, testObs, times);
        out.put("Gaussian process", new double[] {mae(gp[0], test, target), gaussianCrps(gp[0], gp[1], test, target),
            System.currentTimeMillis() - t0});
        for (TimeSeriesImputer imp : List.of(TimeSeriesImputer.brits(c, budget.units(), seed),
                TimeSeriesImputer.saits(c, budget.steps(), budget.units(), seed),
                TimeSeriesImputer.csdi(c, 16, diffusionSteps, seed))) {
            t0 = System.currentTimeMillis();
            imp.fit(train, trainObs, times, budget.trainSteps(), budget.batch(), imp.kind() == TimeSeriesImputer.Kind.CSDI
                    ? 3e-3f : 5e-3f, seed);
            if (imp.kind() == TimeSeriesImputer.Kind.CSDI) {
                float[][][][] s = imp.sample(test, testObs, times, samples);
                float[][][] median = medianOf(s);
                out.put("CSDI", new double[] {mae(median, test, target), sampleCrps(s, test, target),
                    System.currentTimeMillis() - t0});
            } else {
                out.put(imp.kind().name(), new double[] {mae(imp.impute(test, testObs, times, 1), test, target),
                    Double.NaN, System.currentTimeMillis() - t0});
            }
        }
        t0 = System.currentTimeMillis();
        SequenceModel latent = IrregularSequenceModels.latentOde(c, budget.units(), 8, 1f, seed);
        IrregularSequenceModels.fit(latent, train, trainObs, times, IrregularSequenceModels.Task.INTERPOLATION,
                budget.trainSteps(), budget.batch(), 1e-2f, seed);
        out.put("latent ODE", new double[] {mae(IrregularSequenceModels.predict(latent, test, testObs, times), test,
            target), Double.NaN, System.currentTimeMillis() - t0});
        return out;
    }

    private static float[][][] medianOf(float[][][][] s) {
        int n = s[0].length;
        int steps = s[0][0].length;
        int c = s[0][0][0].length;
        float[][][] out = new float[n][steps][c];
        float[] buf = new float[s.length];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < steps; t++) {
                for (int ch = 0; ch < c; ch++) {
                    for (int k = 0; k < s.length; k++) {
                        buf[k] = s[k][i][t][ch];
                    }
                    java.util.Arrays.sort(buf);
                    out[i][t][ch] = s.length % 2 == 1 ? buf[s.length / 2] : (buf[s.length / 2 - 1] + buf[s.length / 2]) / 2;
                }
            }
        }
        return out;
    }

    /**
     * Larger budgets over all dynamics and missingness levels.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        Budget budget = new Budget(1000, 200, 48, 3, 1500, 32, 48);
        for (IrregularSeries.Kind kind : IrregularSeries.Kind.values()) {
            for (double missing : new double[] {0.3, 0.6, 0.9}) {
                for (IrregularSeries.Missing mech : new IrregularSeries.Missing[] {IrregularSeries.Missing.MAR,
                    IrregularSeries.Missing.MNAR}) {
                    print(kind + " " + missing + " " + mech + " interpolation MSE, extrapolation MSE, ms",
                            reconstruction(kind, missing, mech, budget, 1));
                }
            }
            print(kind + " block 0.5 imputation MAE, CRPS, ms",
                    imputation(kind, 0.5, IrregularSeries.Missing.BLOCK, budget, 50, 10, 1));
        }
        for (double missing : new double[] {0.3, 0.6, 0.9}) {
            print("CBF classification " + missing + " accuracy, ms", classification(missing, budget, 1));
        }
    }

    private static void print(String title, Map<String, double[]> r) {
        StringBuilder s = new StringBuilder(title).append(':');
        r.forEach((k, v) -> {
            s.append("\n  ").append(k);
            for (double x : v) {
                s.append(String.format(" %.4f", x));
            }
        });
        System.out.println(s);
    }
}
