package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;
import io.github.zzuegg.brainiac.experimental.ml.DeepGaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.GaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.SparseVariationalGp;
import io.github.zzuegg.brainiac.experimental.ml.SpectralMixtureKernel;
import io.github.zzuegg.brainiac.experimental.ml.StationaryKernel;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Deep Gaussian processes and the spectral mixture kernel against single-layer GPs.
 *
 * <ul>
 *   <li>{@link #deepGp}: a step function and a heteroscedastic series (noise growing with
 *       {@code x}), 100 training points; an exact RBF GP, an RBF SVGP with 20 inducing points,
 *       and doubly stochastic deep GPs with one, two and three layers. Scores: test NLL and
 *       CRPS on fresh samples.</li>
 *   <li>{@link #spectralForecast}: a {@link SeasonalSeries} (period 24, two harmonics, small
 *       trend) observed for five periods and forecast four periods ahead. A GP with a fitted
 *       spectral mixture kernel, a GP with a fitted RBF kernel (both on linearly detrended
 *       data) and the seasonal naive forecast. Scores: MSE over the first 1, 2, 3 and 4
 *       periods of the horizon.</li>
 * </ul>
 */
public final class GaussianProcessKernels {

    private GaussianProcessKernels() {
    }

    private static double[][] column(double[] v) {
        double[][] x = new double[v.length][1];
        for (int i = 0; i < v.length; i++) {
            x[i][0] = v[i];
        }
        return x;
    }

    private static double target(String task, double x, SplittableRandom r) {
        return task.equals("step") ? (x > 0 ? 1 : -1) + 0.05 * r.nextGaussian()
                : Math.sin(2 * x) + (0.05 + 0.4 * Math.max(0, x)) * r.nextGaussian();
    }

    /**
     * Deep GPs on a non-stationary 1-D task.
     *
     * @param task {@code "step"} (on [-1, 1]) or {@code "heteroscedastic"} (on [-2, 2])
     * @param seed the seed
     * @return model to {@code {test NLL, test CRPS}}
     */
    public static Map<String, double[]> deepGp(String task, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double range = task.equals("step") ? 1 : 2;
        int n = 100;
        double[] xs = new double[n];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = range * (2 * r.nextDouble() - 1);
            ys[i] = target(task, xs[i], r);
        }
        double[] xt = new double[200];
        double[] yt = new double[200];
        for (int i = 0; i < 200; i++) {
            xt[i] = range * (2 * r.nextDouble() - 1);
            yt[i] = target(task, xt[i], r);
        }
        double[][] x = column(xs);
        double[][] xtest = column(xt);
        Map<String, double[]> out = new LinkedHashMap<>();
        GaussianProcess gp = GaussianProcess.optimize(x, ys, StationaryKernel.rbf(0.5, 1), 0.05, 80);
        GaussianProcess.Prediction p = gp.predict(xtest);
        out.put("exact GP (RBF)", new double[] {MlMetrics.gaussianNll(p.mean(), p.variance(), yt),
            MlMetrics.gaussianCrps(p.mean(), p.variance(), yt)});
        double[][] z = new double[20][1];
        for (int i = 0; i < 20; i++) {
            z[i][0] = -range + 2 * range * i / 19.0;
        }
        SparseVariationalGp svgp = SparseVariationalGp.fit(x, ys, z, StationaryKernel.rbf(0.3 * range, 1), 0.05, 50,
                300, 0.5, seed);
        GaussianProcess.Prediction sp = svgp.predict(xtest);
        out.put("SVGP (RBF, M=20)", new double[] {MlMetrics.gaussianNll(sp.mean(), sp.variance(), yt),
            MlMetrics.gaussianCrps(sp.mean(), sp.variance(), yt)});
        double[][] xn = column(scale(xs, range));
        double[][] xtn = column(scale(xt, range));
        int[][] widths = {{1}, {1, 1}, {1, 1, 1}};
        double[][] scales = {{0.3}, {0.3, 1}, {0.3, 1, 1}};
        for (int k = 0; k < widths.length; k++) {
            DeepGaussianProcess dgp = DeepGaussianProcess.fit(xn, ys, widths[k], scales[k], 20, 800, 4, seed);
            out.put("deep GP L=" + widths[k].length, new double[] {dgp.nll(xtn, yt, 50, seed),
                dgp.crps(xtn, yt, 50, seed)});
        }
        return out;
    }

    private static double[] scale(double[] v, double range) {
        double[] o = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            o[i] = v[i] / range;
        }
        return o;
    }

    /**
     * Long-horizon extrapolation of a seasonal series.
     *
     * @param seed the seed
     * @return method to {@code {MSE over 1, 2, 3 and 4 periods}}
     */
    public static Map<String, double[]> spectralForecast(long seed) {
        int period = 24;
        int train = 5 * period;
        int horizon = 4 * period;
        float[] series = SeasonalSeries.generate(train + horizon, period, 0.005, 1, 0.1, seed);
        double[] t = new double[train];
        double[] y = new double[train];
        for (int i = 0; i < train; i++) {
            t[i] = i;
            y[i] = series[i];
        }
        // linear detrend by least squares
        double tm = (train - 1) / 2.0;
        double ym = 0;
        for (double v : y) {
            ym += v / train;
        }
        double sxy = 0;
        double sxx = 0;
        for (int i = 0; i < train; i++) {
            sxy += (t[i] - tm) * (y[i] - ym);
            sxx += (t[i] - tm) * (t[i] - tm);
        }
        double slope = sxy / sxx;
        double[] resid = new double[train];
        for (int i = 0; i < train; i++) {
            resid[i] = y[i] - ym - slope * (t[i] - tm);
        }
        double[][] tf = new double[horizon][1];
        double[] truth = new double[horizon];
        for (int h = 0; h < horizon; h++) {
            tf[h][0] = train + h;
            truth[h] = series[train + h];
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        SpectralMixtureKernel.Fit sm = SpectralMixtureKernel.fit(column(t), resid, 3, 150);
        double[] smMean = sm.gp().mean(tf);
        out.put("GP spectral mixture (Q=3)", horizonMse(smMean, truth, ym, slope, tm, train, period));
        GaussianProcess rbf = GaussianProcess.optimize(column(t), resid, StationaryKernel.rbf(5, 1), 0.05, 100);
        out.put("GP RBF", horizonMse(rbf.predict(tf).mean(), truth, ym, slope, tm, train, period));
        out.put("linear trend only", horizonMse(new double[horizon], truth, ym, slope, tm, train, period));
        double[] naive = new double[horizon];
        for (int h = 0; h < horizon; h++) {
            naive[h] = series[train - period + h % period];
        }
        out.put("seasonal naive", horizonMse(naive, truth, 0, 0, tm, train, period));
        return out;
    }

    private static double[] horizonMse(double[] residPred, double[] truth, double level, double slope, double tm,
            int train, int period) {
        double[] out = new double[4];
        double s = 0;
        for (int h = 0; h < truth.length; h++) {
            double pred = residPred[h] + level + slope * (train + h - tm);
            s += (pred - truth[h]) * (pred - truth[h]);
            if ((h + 1) % period == 0) {
                out[(h + 1) / period - 1] = s / (h + 1);
            }
        }
        return out;
    }
}
