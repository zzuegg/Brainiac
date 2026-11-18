package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.experimental.data.IrregularSeries;
import io.github.zzuegg.brainiac.experimental.events.TimeSeriesImputer;
import io.github.zzuegg.brainiac.experimental.ml.LinearDynamicalSystemEm;
import io.github.zzuegg.brainiac.experimental.ml.StateSpaceFilter;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * EM-fitted linear dynamical systems with missing data (wave 11, item 11.3.14).
 *
 * <ul>
 *   <li>{@link #imputation}: {@link IrregularSeries} series under MAR, MNAR or block missingness.
 *       {@link LinearDynamicalSystemEm} fitted per series (state dimension two per channel)
 *       against linear interpolation, a local-linear-trend smoother per channel (wave 10's
 *       baseline, rebuilt on the {@code NaN}-masked {@link StateSpaceFilter#kalman}, process
 *       noise picked from a grid by the observed likelihood) and BRITS trained on separate series.
 *       Scores: MAE on the missing entries, MAE of one-step-ahead predictions of the observed
 *       entries after step 10 (NaN for methods without a predictive filter) and milliseconds.</li>
 *   <li>{@link #recovery}: a planted system (two damped rotations, three channels) observed with
 *       missing values; the error of the fitted transition's eigenvalues (basis invariant), the
 *       relative error of the observation noise, and whether the log-likelihood ever
 *       decreased.</li>
 * </ul>
 */
public final class StateSpaceEm {

    private StateSpaceEm() {
    }

    private static double[][] asNaN(float[][] v, boolean[][] m) {
        double[][] out = new double[v.length][v[0].length];
        for (int t = 0; t < v.length; t++) {
            for (int c = 0; c < v[0].length; c++) {
                out[t][c] = m[t][c] ? v[t][c] : Double.NaN;
            }
        }
        return out;
    }

    private static double[] linear(double[] y) {
        int n = y.length;
        double[] out = new double[n];
        int prev = -1;
        for (int t = 0; t < n; t++) {
            if (!Double.isNaN(y[t])) {
                prev = t;
                out[t] = y[t];
                continue;
            }
            int next = t + 1;
            while (next < n && Double.isNaN(y[next])) {
                next++;
            }
            if (prev < 0 && next >= n) {
                out[t] = 0;
            } else if (prev < 0) {
                out[t] = y[next];
            } else if (next >= n) {
                out[t] = y[prev];
            } else {
                out[t] = y[prev] + (y[next] - y[prev]) * (t - prev) / (double) (next - prev);
            }
        }
        return out;
    }

    /**
     * The local-linear-trend smoother of one channel.
     *
     * @param y the channel, {@code NaN} where missing
     * @param obsNoise the observation noise variance
     * @return {@code {smoothed level [T], one-step-ahead level predictions [T]}}
     */
    static double[][] localTrend(double[] y, double obsNoise) {
        int steps = y.length;
        double[][] ys = new double[steps][1];
        for (int t = 0; t < steps; t++) {
            ys[t][0] = y[t];
        }
        StateSpaceFilter.LinearModel best = null;
        double bestLl = Double.NEGATIVE_INFINITY;
        for (double q : new double[] {1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1}) {
            StateSpaceFilter.LinearModel m = new StateSpaceFilter.LinearModel(new double[][] {{1, 1}, {0, 1}},
                    new double[][] {{q / 4, q / 2}, {q / 2, q}}, new double[][] {{1, 0}}, new double[][] {{obsNoise}},
                    new double[] {0, 0}, new double[][] {{4, 0}, {0, 1}});
            double ll = StateSpaceFilter.kalman(m, ys, false).logLikelihood();
            if (ll > bestLl) {
                bestLl = ll;
                best = m;
            }
        }
        StateSpaceFilter.Estimate s = StateSpaceFilter.kalman(best, ys, true);
        StateSpaceFilter.Estimate f = StateSpaceFilter.kalman(best, ys, false);
        double[][] out = new double[2][steps];
        for (int t = 0; t < steps; t++) {
            out[0][t] = s.means()[t][0];
            out[1][t] = t == 0 ? 0 : f.means()[t - 1][0] + f.means()[t - 1][1];
        }
        return out;
    }

    /**
     * Imputation and one-step prediction.
     *
     * @param kind the dynamics
     * @param mechanism the missingness
     * @param missing the missing share
     * @param series test series
     * @param steps steps per series
     * @param channels channels
     * @param iterations EM iterations
     * @param britsSteps BRITS training steps on as many training series (0 skips BRITS)
     * @param seed the seed
     * @return method to {@code {imputation MAE, one-step MAE, ms}}
     */
    public static Map<String, double[]> imputation(IrregularSeries.Kind kind, IrregularSeries.Missing mechanism,
            double missing, int series, int steps, int channels, int iterations, int britsSteps, long seed) {
        float[][][] test = IrregularSeries.generate(kind, series, steps, channels, seed);
        boolean[][][] obs = IrregularSeries.mask(test, missing, mechanism, seed + 1);
        int c = test[0][0].length;
        double[] acc = new double[9];
        int[] counts = new int[2];
        long[] ms = new long[3];
        for (int i = 0; i < series; i++) {
            double[][] ys = asNaN(test[i], obs[i]);
            long t0 = System.nanoTime();
            double[][] lin = new double[steps][c];
            for (int ch = 0; ch < c; ch++) {
                double[] col = new double[steps];
                for (int t = 0; t < steps; t++) {
                    col[t] = ys[t][ch];
                }
                double[] li = linear(col);
                for (int t = 0; t < steps; t++) {
                    lin[t][ch] = li[t];
                }
            }
            ms[0] += System.nanoTime() - t0;
            t0 = System.nanoTime();
            double[][] trend = new double[steps][c];
            double[][] trendAhead = new double[steps][c];
            for (int ch = 0; ch < c; ch++) {
                double[] col = new double[steps];
                for (int t = 0; t < steps; t++) {
                    col[t] = ys[t][ch];
                }
                double[][] o = localTrend(col, 0.05 * 0.05);
                for (int t = 0; t < steps; t++) {
                    trend[t][ch] = o[0][t];
                    trendAhead[t][ch] = o[1][t];
                }
            }
            ms[1] += System.nanoTime() - t0;
            t0 = System.nanoTime();
            LinearDynamicalSystemEm.Fit fit = LinearDynamicalSystemEm.fit(new double[][][] {ys}, 2 * c, iterations,
                    seed + i);
            double[][] lds = fit.reconstruct(ys)[0];
            double[][] ldsAhead = fit.oneStepAhead(ys);
            ms[2] += System.nanoTime() - t0;
            for (int t = 0; t < steps; t++) {
                for (int ch = 0; ch < c; ch++) {
                    double truth = test[i][t][ch];
                    if (!obs[i][t][ch]) {
                        acc[0] += Math.abs(lin[t][ch] - truth);
                        acc[1] += Math.abs(trend[t][ch] - truth);
                        acc[2] += Math.abs(lds[t][ch] - truth);
                        counts[0]++;
                    } else if (t >= 10) {
                        acc[4] += Math.abs(trendAhead[t][ch] - truth);
                        acc[5] += Math.abs(ldsAhead[t][ch] - truth);
                        counts[1]++;
                    }
                }
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("linear interpolation", new double[] {acc[0] / counts[0], Double.NaN, ms[0] / 1e6});
        out.put("local linear trend smoother", new double[] {acc[1] / counts[0], acc[4] / counts[1], ms[1] / 1e6});
        out.put("LDS EM", new double[] {acc[2] / counts[0], acc[5] / counts[1], ms[2] / 1e6});
        if (britsSteps > 0) {
            float[][][] train = IrregularSeries.generate(kind, Math.max(16, series), steps, channels, seed + 7);
            boolean[][][] trainObs = IrregularSeries.mask(train, missing, mechanism, seed + 8);
            double[] times = IrregularSeries.times(steps);
            long t0 = System.nanoTime();
            TimeSeriesImputer brits = TimeSeriesImputer.brits(c, 16, seed);
            brits.fit(train, trainObs, times, britsSteps, 8, 5e-3f, seed);
            float[][][] imp = brits.impute(test, obs, times, 1);
            double e = 0;
            for (int i = 0; i < series; i++) {
                for (int t = 0; t < steps; t++) {
                    for (int ch = 0; ch < c; ch++) {
                        if (!obs[i][t][ch]) {
                            e += Math.abs(imp[i][t][ch] - test[i][t][ch]);
                        }
                    }
                }
            }
            out.put("BRITS", new double[] {e / counts[0], Double.NaN, (System.nanoTime() - t0) / 1e6});
        }
        return out;
    }

    /**
     * Parameter recovery on a planted system.
     *
     * @param steps sequence length
     * @param sequences sequences
     * @param missing the MAR missing share
     * @param iterations EM iterations
     * @param seed the seed
     * @return {@code {mean |eigenvalue error|, mean relative R error, log-likelihood decreases,
     *     first log-likelihood per step, last log-likelihood per step}}
     */
    public static double[] recovery(int steps, int sequences, double missing, int iterations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] f = new double[4][4];
        double[] rho = {0.95, 0.85};
        double[] theta = {2 * Math.PI / 12, 2 * Math.PI / 30};
        for (int b = 0; b < 2; b++) {
            f[2 * b][2 * b] = rho[b] * Math.cos(theta[b]);
            f[2 * b][2 * b + 1] = -rho[b] * Math.sin(theta[b]);
            f[2 * b + 1][2 * b] = rho[b] * Math.sin(theta[b]);
            f[2 * b + 1][2 * b + 1] = rho[b] * Math.cos(theta[b]);
        }
        double[][] h = new double[3][4];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                h[i][j] = r.nextGaussian();
            }
        }
        double[] rTrue = {0.05, 0.1, 0.2};
        double[][][] data = new double[sequences][steps][3];
        int total = 0;
        for (int s = 0; s < sequences; s++) {
            double[] x = new double[4];
            for (int t = 0; t < steps; t++) {
                double[] nx = new double[4];
                for (int a = 0; a < 4; a++) {
                    for (int b = 0; b < 4; b++) {
                        nx[a] += f[a][b] * x[b];
                    }
                    nx[a] += Math.sqrt(0.1) * r.nextGaussian();
                }
                x = nx;
                for (int i = 0; i < 3; i++) {
                    double y = Math.sqrt(rTrue[i]) * r.nextGaussian();
                    for (int j = 0; j < 4; j++) {
                        y += h[i][j] * x[j];
                    }
                    data[s][t][i] = r.nextDouble() < missing ? Double.NaN : y;
                    total += Double.isNaN(data[s][t][i]) ? 0 : 1;
                }
            }
        }
        LinearDynamicalSystemEm.Fit fit = LinearDynamicalSystemEm.fit(data, 4, iterations, seed + 1);
        double[][] et = LinearDynamicalSystemEm.eigenvalues(f);
        double[][] ef = LinearDynamicalSystemEm.eigenvalues(fit.model().f());
        // match each true eigenvalue to its nearest fitted one (greedy)
        boolean[] used = new boolean[ef.length];
        double err = 0;
        for (double[] e : et) {
            int best = -1;
            double bd = Double.POSITIVE_INFINITY;
            for (int k = 0; k < ef.length; k++) {
                double dd = Math.hypot(e[0] - ef[k][0], Math.abs(e[1]) - Math.abs(ef[k][1]));
                if (!used[k] && dd < bd) {
                    bd = dd;
                    best = k;
                }
            }
            used[best] = true;
            err += bd / et.length;
        }
        double rErr = 0;
        for (int i = 0; i < 3; i++) {
            rErr += Math.abs(fit.model().r()[i][i] - rTrue[i]) / rTrue[i] / 3;
        }
        double[] ll = fit.logLikelihoods();
        int decreases = 0;
        for (int k = 1; k < ll.length; k++) {
            decreases += ll[k] < ll[k - 1] - 1e-6 * Math.abs(ll[k - 1]) ? 1 : 0;
        }
        return new double[] {err, rErr, decreases, ll[0] / total, ll[ll.length - 1] / total};
    }
}
