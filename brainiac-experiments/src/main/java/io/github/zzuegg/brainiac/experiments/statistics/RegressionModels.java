package io.github.zzuegg.brainiac.experiments.statistics;

import io.github.zzuegg.brainiac.experimental.boost.BoostLosses;
import io.github.zzuegg.brainiac.experimental.boost.ExplainableBoosting;
import io.github.zzuegg.brainiac.experimental.boost.QuantileForest;
import io.github.zzuegg.brainiac.experimental.data.SparseDesigns;
import io.github.zzuegg.brainiac.experimental.discovery.LeastSquares;
import io.github.zzuegg.brainiac.experimental.ml.RegressionForest;
import io.github.zzuegg.brainiac.experimental.stats.AdditiveModel;
import io.github.zzuegg.brainiac.experimental.stats.CountGlm;
import io.github.zzuegg.brainiac.experimental.stats.Mars;
import io.github.zzuegg.brainiac.experimental.stats.QuantileRegression;
import io.github.zzuegg.brainiac.experimental.stats.RobustCovariance;
import io.github.zzuegg.brainiac.experimental.stats.RobustRegression;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Regression beyond least squares (items 12.1.7-12.1.9 and 12.1.17): robust M-, LTS, Theil-Sen
 * and MM estimators against OLS under vertical and leverage contamination, MCD distances against
 * classical Mahalanobis for outlier detection and median-of-means against the mean under heavy
 * tails; linear quantile regression (LP and IRLS) against quantile forests and GBT quantile
 * loss, with monotone rearrangement of crossing quantiles; P-spline GAMs and MARS against
 * explainable boosting and polynomial ridge on Friedman #1 and additive data; and count GLMs
 * (Poisson, NB, ZIP, Tweedie) against OLS on {@code log(1 + y)} and GBT with squared loss.
 * (The wave 11 RANSAC in {@code FeatureMatching} fits point correspondences, not regressions, so
 * it is not a row here.)
 */
public final class RegressionModels {

    private RegressionModels() {
    }

    /**
     * Coefficient error under contamination.
     *
     * @param n rows
     * @param p features
     * @param share contaminated share
     * @param leverage whether outliers are leverage points
     * @param seed the seed
     * @return {@code ||b - beta||_2} per method
     */
    public static Map<String, Double> robust(int n, int p, double share, boolean leverage, long seed) {
        SparseDesigns.Regression clean = SparseDesigns.linear(n, p, p, 0, 10, SparseDesigns.Noise.GAUSSIAN,
                SparseDesigns.Coefficients.UNIFORM, seed);
        SparseDesigns.Contaminated c = SparseDesigns.contaminate(clean, share, leverage, seed + 1);
        Map<String, Double> out = new LinkedHashMap<>();
        double[][] x = c.data().x();
        double[] y = c.data().y();
        out.put("OLS", Scores.distance(RobustRegression.ols(x, y).beta(), clean.beta()));
        for (RobustRegression.Method m : RobustRegression.Method.values()) {
            out.put(m.name(), Scores.distance(RobustRegression.fit(m, x, y, seed).beta(), clean.beta()));
        }
        return out;
    }

    /**
     * Outlier detection by robust against classical distances, with a cluster of outliers that
     * breaks the clean data's positive correlation (the masking case).
     *
     * @param n rows
     * @param p dimensions
     * @param share outlier share
     * @param seed the seed
     * @return {@code {AUROC classical, AUROC MCD}}
     */
    public static double[] outlierDetection(int n, int p, double share, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][p];
        boolean[] bad = new boolean[n];
        for (int i = 0; i < n; i++) {
            bad[i] = r.nextDouble() < share;
            for (int j = 0; j < p; j++) {
                x[i][j] = bad[i] ? (j % 2 == 0 ? 2.5 : -2.5) + 0.3 * r.nextGaussian()
                        : r.nextGaussian() + (j > 0 ? 0.8 * x[i][j - 1] : 0);
            }
        }
        double[] classical = RobustCovariance.empirical(x).distances(x);
        double[] mcd = RobustCovariance.mcd(x, 100, seed).distances(x);
        return new double[] {Scores.auc(classical, bad), Scores.auc(mcd, bad)};
    }

    /**
     * Mean absolute error of the sample mean and median-of-means under t2 noise.
     *
     * @param n sample size
     * @param reps replications
     * @param seed the seed
     * @return {@code {mean error of the mean, of median-of-means (10 blocks), 99th percentile of each}}
     */
    public static double[] medianOfMeans(int n, int reps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] em = new double[reps];
        double[] emom = new double[reps];
        for (int k = 0; k < reps; k++) {
            double[] v = new double[n];
            for (int i = 0; i < n; i++) {
                v[i] = r.nextGaussian() / Math.sqrt(SparseDesigns.gamma(1, r));
            }
            em[k] = Math.abs(Scores.mean(v));
            emom[k] = Math.abs(RobustCovariance.medianOfMeans(v, 10, seed + k));
        }
        double[] sm = em.clone();
        double[] smom = emom.clone();
        java.util.Arrays.sort(sm);
        java.util.Arrays.sort(smom);
        int q = (int) (0.99 * (reps - 1));
        return new double[] {Scores.mean(em), Scores.mean(emom), sm[q], smom[q]};
    }

    /** Heteroscedastic, right-skewed linear data: {@code y = 1 + 2 x1 - x2 + (0.3 + x1) e}, e ~ Exp(1) - 1. */
    private static double[][][] heteroscedastic(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][2];
        double[][] y = new double[n][1];
        for (int i = 0; i < n; i++) {
            x[i][0] = 2 * r.nextDouble();
            x[i][1] = r.nextGaussian();
            double e = -Math.log(1 - r.nextDouble()) - 1;
            y[i][0] = 1 + 2 * x[i][0] - x[i][1] + (0.3 + x[i][0]) * e;
        }
        return new double[][][] {x, y};
    }

    /**
     * Quantile regression against nonlinear quantile learners on linear heteroscedastic data.
     *
     * @param n training rows
     * @param seed the seed
     * @return {@code {mean pinball over tau in 0.1/0.5/0.9, 80% coverage, 90% coverage}} per method
     */
    public static Map<String, double[]> quantiles(int n, long seed) {
        double[][][] tr = heteroscedastic(n, seed);
        double[][][] te = heteroscedastic(2000, seed + 1);
        double[][] x = tr[0];
        double[] y = column(tr[1]);
        double[][] xt = te[0];
        double[] yt = column(te[1]);
        double[] taus = {0.05, 0.1, 0.5, 0.9, 0.95};
        Map<String, double[][]> preds = new LinkedHashMap<>();
        for (QuantileRegression.Solver s : new QuantileRegression.Solver[] {QuantileRegression.Solver.LP,
            QuantileRegression.Solver.IRLS}) {
            double[][] q = new double[taus.length][];
            for (int k = 0; k < taus.length; k++) {
                QuantileRegression.Fit f = QuantileRegression.fit(s, x, y, taus[k], 0);
                q[k] = new double[xt.length];
                for (int i = 0; i < xt.length; i++) {
                    q[k][i] = f.predict(xt[i]);
                }
            }
            preds.put("linear QR " + s.name(), q);
        }
        QuantileForest qf = QuantileForest.fit(x, y, new RegressionForest.Settings(100, 10, 10, 0, 1, false, seed));
        double[][] qq = new double[taus.length][xt.length];
        for (int i = 0; i < xt.length; i++) {
            for (int k = 0; k < taus.length; k++) {
                qq[k][i] = qf.quantile(xt[i], taus[k]);
            }
        }
        preds.put("quantile forest", qq);
        double[][] gq = new double[taus.length][];
        for (int k = 0; k < taus.length; k++) {
            GradientBoostedTrees g = GradientBoostedTrees.fit(x, y, BoostLosses.quantile(taus[k]),
                    GradientBoostedTrees.Settings.defaults());
            gq[k] = g.predict(xt);
        }
        preds.put("GBT quantile loss", gq);
        Map<String, double[]> out = new LinkedHashMap<>();
        preds.forEach((name, q) -> {
            double pin = 0;
            for (int k : new int[] {1, 2, 3}) {
                pin += BoostLosses.pinball(yt, q[k], taus[k]) / 3;
            }
            out.put(name, new double[] {pin, coverage(yt, q[1], q[3]), coverage(yt, q[0], q[4])});
        });
        return out;
    }

    /**
     * Crossing stress: few rows, many levels, separate LP fits.
     *
     * @param n training rows
     * @param seed the seed
     * @return {@code {crossing rate before, after, pinball before, pinball after}}
     */
    public static double[] crossing(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int p = 4;
        double[][] x = new double[n][p];
        double[] y = new double[n];
        double[][] xt = new double[1000][p];
        double[] yt = new double[1000];
        for (double[][] xs : new double[][][] {x, xt}) {
            double[] ys = xs == x ? y : yt;
            for (int i = 0; i < xs.length; i++) {
                for (int j = 0; j < p; j++) {
                    xs[i][j] = 3 * r.nextGaussian();
                }
                ys[i] = xs[i][0] + (1 + 0.5 * Math.abs(xs[i][1])) * r.nextGaussian();
            }
        }
        double[] taus = new double[9];
        QuantileRegression.Fit[] fits = new QuantileRegression.Fit[9];
        for (int k = 0; k < 9; k++) {
            taus[k] = 0.1 * (k + 1);
            fits[k] = QuantileRegression.fit(QuantileRegression.Solver.LP, x, y, taus[k], 0);
        }
        double[][] q = QuantileRegression.predict(fits, xt);
        double[][] s = QuantileRegression.rearrange(q);
        return new double[] {QuantileRegression.crossingRate(q), QuantileRegression.crossingRate(s),
            pinballAll(q, yt, taus), pinballAll(s, yt, taus)};
    }

    private static double pinballAll(double[][] q, double[] y, double[] taus) {
        double s = 0;
        for (int i = 0; i < y.length; i++) {
            for (int k = 0; k < taus.length; k++) {
                s += QuantileRegression.pinball(y[i] - q[i][k], taus[k]);
            }
        }
        return s / (y.length * taus.length);
    }

    /** The smooth-regression tasks. */
    public enum SmoothTask {
        /** Friedman #1: {@code 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5}, 10 uniform features. */
        FRIEDMAN1,
        /** Additive: {@code 2 sin(2 pi x1) + 2 * 1[x2 > 0.5] + 4 (x3 - 0.5)^2 + 2 x4}, 8 uniform features. */
        ADDITIVE
    }

    private static double truth(SmoothTask task, double[] x) {
        return task == SmoothTask.FRIEDMAN1
                ? 10 * Math.sin(Math.PI * x[0] * x[1]) + 20 * (x[2] - 0.5) * (x[2] - 0.5) + 10 * x[3] + 5 * x[4]
                : 2 * Math.sin(2 * Math.PI * x[0]) + (x[1] > 0.5 ? 2 : 0) + 4 * (x[2] - 0.5) * (x[2] - 0.5) + 2 * x[3];
    }

    /**
     * Smooth regressors on a task.
     *
     * @param task the task
     * @param n training rows
     * @param seed the seed
     * @return {@code {test RMSE against the noiseless truth, fit ms}} per method
     */
    public static Map<String, double[]> smooth(SmoothTask task, int n, long seed) {
        int p = task == SmoothTask.FRIEDMAN1 ? 10 : 8;
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][p];
        double[] y = new double[n];
        double[][] xt = new double[1000][p];
        double[] ft = new double[1000];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                x[i][j] = r.nextDouble();
            }
            y[i] = truth(task, x[i]) + (task == SmoothTask.FRIEDMAN1 ? 1.0 : 0.5) * r.nextGaussian();
        }
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < p; j++) {
                xt[i][j] = r.nextDouble();
            }
            ft[i] = truth(task, xt[i]);
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        long t0 = System.nanoTime();
        AdditiveModel gam = AdditiveModel.fit(x, y, AdditiveModel.Family.GAUSSIAN, 20);
        long t1 = System.nanoTime();
        out.put("GAM (P-splines)", new double[] {rmse(gam::predict, xt, ft), (t1 - t0) / 1e6});
        t0 = System.nanoTime();
        Mars mars = Mars.fit(x, y, 21, 2, 12);
        t1 = System.nanoTime();
        out.put("MARS", new double[] {rmse(mars::predict, xt, ft), (t1 - t0) / 1e6});
        t0 = System.nanoTime();
        ExplainableBoosting ebm = ExplainableBoosting.fit(x, y, BoostLosses.squared(),
                ExplainableBoosting.Settings.defaults().withInteractions(0));
        t1 = System.nanoTime();
        out.put("EBM", new double[] {rmse(ebm::predict, xt, ft), (t1 - t0) / 1e6});
        t0 = System.nanoTime();
        double[][] px = poly(x);
        LeastSquares.Fit pr = LeastSquares.fit(px, y, true, 1.0);
        t1 = System.nanoTime();
        double[][] pxt = poly(xt);
        out.put("poly-2 ridge", new double[] {rmse(row -> {
            double s = pr.beta()[0];
            for (int j = 0; j < row.length; j++) {
                s += pr.beta()[j + 1] * row[j];
            }
            return s;
        }, pxt, ft), (t1 - t0) / 1e6});
        if (task == SmoothTask.ADDITIVE) {
            // shape error of the sine component against the truth (centred on a grid)
            double[] grid = new double[50];
            double[] g = new double[50];
            double[] e = new double[50];
            double[] tru = new double[50];
            double mg = 0;
            double me = 0;
            double mt = 0;
            for (int k = 0; k < 50; k++) {
                grid[k] = (k + 0.5) / 50;
                g[k] = gam.shape(0, grid[k]);
                e[k] = ebm.shape(0, grid[k]);
                tru[k] = 2 * Math.sin(2 * Math.PI * grid[k]);
                mg += g[k] / 50;
                me += e[k] / 50;
                mt += tru[k] / 50;
            }
            double sg = 0;
            double se = 0;
            for (int k = 0; k < 50; k++) {
                sg += Math.pow(g[k] - mg - (tru[k] - mt), 2) / 50;
                se += Math.pow(e[k] - me - (tru[k] - mt), 2) / 50;
            }
            out.put("shape RMSE x1: GAM vs EBM", new double[] {Math.sqrt(sg), Math.sqrt(se)});
        }
        return out;
    }

    private static double[][] poly(double[][] x) {
        int p = x[0].length;
        int q = p + p * (p + 1) / 2;
        double[][] out = new double[x.length][q];
        for (int i = 0; i < x.length; i++) {
            int k = 0;
            for (int a = 0; a < p; a++) {
                out[i][k++] = x[i][a];
            }
            for (int a = 0; a < p; a++) {
                for (int b = a; b < p; b++) {
                    out[i][k++] = x[i][a] * x[i][b];
                }
            }
        }
        return out;
    }

    /**
     * Count GLMs against log-OLS and GBT with squared loss.
     *
     * @param family the generating family
     * @param dispersion the overdispersion (NB) or phi (Tweedie)
     * @param zeroShare structural zeros (ZIP)
     * @param n rows
     * @param seed the seed
     * @return {@code {test deviance (Tweedie deviance for Tweedie data, Poisson otherwise),
     *     |predicted - actual zero share|, 90% interval coverage}} per method
     */
    public static Map<String, double[]> counts(SparseDesigns.CountFamily family, double dispersion, double zeroShare,
            int n, long seed) {
        SparseDesigns.Counts d = SparseDesigns.counts(family, n, 4, dispersion, zeroShare, seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        double actualZeros = 0;
        for (double v : d.yTest()) {
            actualZeros += v == 0 ? 1.0 / d.yTest().length : 0;
        }
        boolean tweedie = family == SparseDesigns.CountFamily.TWEEDIE;
        for (CountGlm.Family f : CountGlm.Family.values()) {
            if (tweedie != (f == CountGlm.Family.TWEEDIE) && f != CountGlm.Family.POISSON) {
                continue;
            }
            CountGlm m = CountGlm.fit(f, d.x(), d.exposure(), d.y(), 1e-6);
            double[] mu = new double[d.yTest().length];
            double zeros = 0;
            int covered = 0;
            SplittableRandom r = new SplittableRandom(seed);
            for (int i = 0; i < mu.length; i++) {
                mu[i] = m.mean(d.xTest()[i], d.exposureTest()[i]);
                zeros += m.zeroProbability(d.xTest()[i], d.exposureTest()[i]) / mu.length;
                double[] iv = m.interval(d.xTest()[i], d.exposureTest()[i], 0.9, 200, r);
                if (d.yTest()[i] >= iv[0] && d.yTest()[i] <= iv[1]) {
                    covered++;
                }
            }
            out.put(f.name() + " GLM", new double[] {deviance(tweedie, d.yTest(), mu), Math.abs(zeros - actualZeros),
                (double) covered / mu.length});
        }
        // OLS on log(1 + y) with log exposure as a feature; intervals from residual quantiles
        double[][] xl = withLogExposure(d.x(), d.exposure());
        double[][] xlt = withLogExposure(d.xTest(), d.exposureTest());
        double[] ly = new double[n];
        for (int i = 0; i < n; i++) {
            ly[i] = Math.log1p(d.y()[i]);
        }
        LeastSquares.Fit ols = LeastSquares.fit(xl, ly, true, 1e-9);
        double[] res = ols.residuals().clone();
        java.util.Arrays.sort(res);
        double lo = res[(int) (0.05 * (n - 1))];
        double hi = res[(int) (0.95 * (n - 1))];
        double[] mu = new double[xlt.length];
        int covered = 0;
        for (int i = 0; i < xlt.length; i++) {
            double e = ols.beta()[0];
            for (int j = 0; j < xlt[i].length; j++) {
                e += ols.beta()[j + 1] * xlt[i][j];
            }
            mu[i] = Math.max(1e-6, Math.expm1(e));
            double y = d.yTest()[i];
            if (y >= Math.max(0, Math.expm1(e + lo)) && y <= Math.expm1(e + hi)) {
                covered++;
            }
        }
        // log-OLS has no point mass at zero, so its zero-share error is the whole actual share
        out.put("OLS on log(1+y)", new double[] {deviance(tweedie, d.yTest(), mu), actualZeros,
            (double) covered / mu.length});
        GradientBoostedTrees gbt = GradientBoostedTrees.fit(xl, d.y(), BoostLosses.squared(),
                GradientBoostedTrees.Settings.defaults());
        double[] gp = gbt.predict(xlt);
        double[] gtr = gbt.predict(xl);
        double[] gres = new double[n];
        for (int i = 0; i < n; i++) {
            gres[i] = d.y()[i] - gtr[i];
        }
        java.util.Arrays.sort(gres);
        double glo = gres[(int) (0.05 * (n - 1))];
        double ghi = gres[(int) (0.95 * (n - 1))];
        covered = 0;
        for (int i = 0; i < gp.length; i++) {
            gp[i] = Math.max(1e-6, gp[i]);
            double y = d.yTest()[i];
            if (y >= gp[i] + glo && y <= gp[i] + ghi) {
                covered++;
            }
        }
        out.put("GBT squared loss", new double[] {deviance(tweedie, d.yTest(), gp), actualZeros,
            (double) covered / gp.length});
        return out;
    }

    private static double deviance(boolean tweedie, double[] y, double[] mu) {
        return tweedie ? CountGlm.tweedieDeviance(y, mu, CountGlm.TWEEDIE_POWER) : CountGlm.poissonDeviance(y, mu);
    }

    private static double[][] withLogExposure(double[][] x, double[] e) {
        double[][] o = new double[x.length][x[0].length + 1];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(x[i], 0, o[i], 0, x[i].length);
            o[i][x[i].length] = Math.log(e[i]);
        }
        return o;
    }

    private static double coverage(double[] y, double[] lo, double[] hi) {
        int c = 0;
        for (int i = 0; i < y.length; i++) {
            if (y[i] >= lo[i] && y[i] <= hi[i]) {
                c++;
            }
        }
        return (double) c / y.length;
    }

    private static double[] column(double[][] a) {
        double[] o = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            o[i] = a[i][0];
        }
        return o;
    }

    private static double rmse(java.util.function.ToDoubleFunction<double[]> f, double[][] x, double[] truth) {
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            double e = f.applyAsDouble(x[i]) - truth[i];
            s += e * e;
        }
        return Math.sqrt(s / x.length);
    }
}
