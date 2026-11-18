package io.github.zzuegg.brainiac.experiments.statistics;

import io.github.zzuegg.brainiac.experimental.data.SparseDesigns;
import io.github.zzuegg.brainiac.experimental.discovery.CiTest;
import io.github.zzuegg.brainiac.experimental.discovery.LeastSquares;
import io.github.zzuegg.brainiac.experimental.discovery.Lingam;
import io.github.zzuegg.brainiac.experimental.discovery.PcAlgorithm;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.sciml.Sindy;
import io.github.zzuegg.brainiac.experimental.stats.Lars;
import io.github.zzuegg.brainiac.experimental.stats.PenalizedRegression;
import io.github.zzuegg.brainiac.experimental.stats.PenalizedRegression.Penalty;
import io.github.zzuegg.brainiac.experimental.testing.GraphicalLasso;
import io.github.zzuegg.brainiac.experimental.timeseries.Reconciliation;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Sparse and penalised regression on {@link SparseDesigns} with a planted sparse truth (items
 * 12.1.1-12.1.3): the lasso and elastic net by coordinate descent with CV ({@code lambda_min}
 * and the 1-SE rule) against OLS, ridge, forward stepwise, {@code Lingam.lassoBic} and SINDy's
 * STLSQ; strong-rule screening against plain cyclic descent and LARS against coordinate descent;
 * SCAD, MCP and the adaptive lasso on mixed large/small coefficients; the group lasso on a
 * grouped design; and the graphical lasso and neighbourhood selection against thresholded
 * inverse covariances (sample and Reconciliation's Schafer-Strimmer shrinkage) and the PC skeleton.
 */
public final class SparseRegression {

    private SparseRegression() {
    }

    /**
     * Compares estimators on one design: test MSE and support F1.
     *
     * @param data the data set
     * @param seed the CV seed
     * @return {@code {test MSE, support F1}} per method
     */
    public static Map<String, double[]> linearMethods(SparseDesigns.Regression data, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        double[][] x = data.x();
        double[] y = data.y();
        int n = y.length;
        int p = x[0].length;
        boolean[] truth = data.support();
        if (n > p + 1) {
            LeastSquares.Fit ols = LeastSquares.fit(x, y, true, 0);
            out.put("OLS", score(data, ols.beta()[0], tail(ols.beta()), truth));
        }
        out.put("ridge (holdout)", ridge(data, truth));
        out.put("forward stepwise (BIC)", stepwise(data, truth));
        double[] lb = Lingam.lassoBic(x, y, n, false);
        out.put("Lingam.lassoBic", score(data, interceptFor(x, y, lb), lb, truth));
        out.put("STLSQ", stlsq(data, truth));
        PenalizedRegression lasso = PenalizedRegression.of(Penalty.LASSO);
        PenalizedRegression.CrossValidation cv = lasso.crossValidate(x, y, 50, 0.01, 5, seed);
        out.put("lasso CV min", score(data, cv.min(), truth));
        out.put("lasso CV 1-SE", score(data, cv.oneStandardError(), truth));
        PenalizedRegression enet = PenalizedRegression.of(Penalty.ELASTIC_NET).alpha(0.5);
        PenalizedRegression.CrossValidation ecv = enet.crossValidate(x, y, 50, 0.01, 5, seed);
        out.put("elastic net CV min", score(data, ecv.min(), truth));
        return out;
    }

    /**
     * Path cost with and without strong rules, and LARS against coordinate descent.
     *
     * @param data the data set
     * @param count grid points
     * @return {@code {updates with screening, updates without, max |b| difference between the
     *     two, max |b| difference LARS vs CD, ms with, ms without}}
     */
    public static double[] pathCost(SparseDesigns.Regression data, int count) {
        double[][] x = data.x();
        double[] y = data.y();
        PenalizedRegression on = PenalizedRegression.of(Penalty.LASSO).tolerance(1e-10);
        PenalizedRegression off = PenalizedRegression.of(Penalty.LASSO).screening(false).tolerance(1e-10);
        long t0 = System.nanoTime();
        PenalizedRegression.Path a = on.path(x, y, count, 0.01);
        long t1 = System.nanoTime();
        PenalizedRegression.Path b = off.path(x, y, count, 0.01);
        long t2 = System.nanoTime();
        double diff = 0;
        for (int k = 0; k < count; k++) {
            diff = Math.max(diff, maxAbsDiff(a.fits().get(k).beta(), b.fits().get(k).beta()));
        }
        Lars.LassoPath lars = Lars.lassoPath(x, y, a.lambdas()[count - 1] * 0.99);
        double larsDiff = 0;
        for (int k = 0; k < count; k++) {
            larsDiff = Math.max(larsDiff,
                    maxAbsDiff(lars.at(a.lambdas()[k]).beta(), a.fits().get(k).beta()));
        }
        return new double[] {a.updates(), b.updates(), diff, larsDiff, (t1 - t0) / 1e6, (t2 - t1) / 1e6};
    }

    /**
     * Folded-concave and adaptive penalties on mixed large/small coefficients, each with
     * 5-fold CV ({@code lambda_min}).
     *
     * @param data the data set
     * @param seed the CV seed
     * @return {@code {||b - beta||_2, support F1, false positives, error on the large coefficients}}
     *     per penalty
     */
    public static Map<String, double[]> nonConvex(SparseDesigns.Regression data, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        boolean[] truth = data.support();
        double[] beta = data.beta();
        for (Penalty pen : new Penalty[] {Penalty.LASSO, Penalty.ELASTIC_NET, Penalty.ADAPTIVE_LASSO, Penalty.SCAD,
                Penalty.MCP}) {
            PenalizedRegression m = PenalizedRegression.of(pen).alpha(0.5);
            PenalizedRegression.Fit f = m.crossValidate(data.x(), data.y(), 40, 0.01, 5, seed).min();
            double large = 0;
            for (int j = 0; j < beta.length; j++) {
                if (Math.abs(beta[j]) > 2) {
                    large += (f.beta()[j] - beta[j]) * (f.beta()[j] - beta[j]);
                }
            }
            out.put(pen.name(), new double[] {Scores.distance(f.beta(), beta), Scores.f1(f.support(), truth),
                Scores.falsePositives(f.support(), truth), Math.sqrt(large)});
        }
        return out;
    }

    /**
     * Group recovery on a grouped design: group lasso, sparse-group lasso and lasso (CV min).
     *
     * @param data a design from {@link SparseDesigns#grouped}
     * @param groupSize features per group
     * @param seed the CV seed
     * @return {@code {group F1, feature support F1, test MSE, features selected}} per method
     */
    public static Map<String, double[]> grouped(SparseDesigns.Regression data, int groupSize, long seed) {
        int p = data.beta().length;
        int[] groups = new int[p];
        for (int j = 0; j < p; j++) {
            groups[j] = j / groupSize;
        }
        int ng = p / groupSize;
        boolean[] truth = new boolean[ng];
        for (int j = 0; j < p; j++) {
            truth[j / groupSize] |= data.beta()[j] != 0;
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        PenalizedRegression[] models = {PenalizedRegression.of(Penalty.LASSO),
            PenalizedRegression.of(Penalty.GROUP_LASSO).groups(groups),
            PenalizedRegression.of(Penalty.SPARSE_GROUP_LASSO).groups(groups).alpha(0.2)};
        String[] names = {"lasso", "group lasso", "sparse-group lasso"};
        for (int k = 0; k < models.length; k++) {
            PenalizedRegression.Fit f = models[k].crossValidate(data.x(), data.y(), 30, 0.01, 5, seed)
                    .oneStandardError();
            boolean[] sel = new boolean[ng];
            for (int j = 0; j < p; j++) {
                sel[j / groupSize] |= f.beta()[j] != 0;
            }
            out.put(names[k], new double[] {Scores.f1(sel, truth), Scores.f1(f.support(), data.support()),
                Scores.mse(f.predict(data.xTest()), data.yTest()), f.nonZero()});
        }
        return out;
    }

    /**
     * Graph recovery for a Gaussian graphical model.
     *
     * @param g the model and samples
     * @return {@code {edge F1, Frobenius error of Theta (NaN for graph-only methods)}} per method
     */
    public static Map<String, double[]> graphs(SparseDesigns.GaussianGraph g) {
        double[][] x = g.samples();
        int n = x.length;
        int p = x[0].length;
        Map<String, double[]> out = new LinkedHashMap<>();
        for (GraphicalLasso.Method m : GraphicalLasso.Method.values()) {
            GraphicalLasso.Estimate e = GraphicalLasso.fitEbic(m, x, 12, 0.5);
            out.put(m.name(), new double[] {GraphicalLasso.edgeF1(e.edges(), g.edges()),
                e.precision() == null ? Double.NaN : frobenius(e.precision(), g.precision())});
        }
        double[][] s = GraphicalLasso.covariance(x);
        double[][] sr = s;
        if (n <= p + 1) {
            sr = new double[p][p];
            for (int i = 0; i < p; i++) {
                sr[i] = s[i].clone();
                sr[i][i] += 0.05 * s[i][i];
            }
        }
        double[][] inv = MlMatrices.choleskyInverse(MlMatrices.cholesky(sr));
        out.put("thresholded inverse", new double[] {GraphicalLasso.edgeF1(threshold(inv, n), g.edges()),
            frobenius(inv, g.precision())});
        double[][] lw = MlMatrices.choleskyInverse(MlMatrices.cholesky(Reconciliation.shrinkCovariance(x)));
        out.put("shrinkage inverse", new double[] {GraphicalLasso.edgeF1(threshold(lw, n), g.edges()),
            frobenius(lw, g.precision())});
        PcAlgorithm.Skeleton sk = PcAlgorithm.skeleton(CiTest.fisherZ(x), p, 0.01, 2);
        out.put("PC skeleton", new double[] {GraphicalLasso.edgeF1(sk.adjacent(), g.edges()), Double.NaN});
        return out;
    }

    // ---------------------------------------------------------------- baselines and helpers

    private static boolean[][] threshold(double[][] inv, int n) {
        int p = inv.length;
        boolean[][] e = new boolean[p][p];
        double cut = 2 / Math.sqrt(n);
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                double pc = -inv[i][j] / Math.sqrt(inv[i][i] * inv[j][j]);
                e[i][j] = i != j && Math.abs(pc) > cut;
            }
        }
        return e;
    }

    private static double frobenius(double[][] a, double[][] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) {
                s += (a[i][j] - b[i][j]) * (a[i][j] - b[i][j]);
            }
        }
        return Math.sqrt(s);
    }

    private static double[] score(SparseDesigns.Regression d, PenalizedRegression.Fit f, boolean[] truth) {
        return score(d, f.intercept(), f.beta(), truth);
    }

    private static double[] score(SparseDesigns.Regression d, double b0, double[] beta, boolean[] truth) {
        return new double[] {Scores.mse(Scores.predict(d.xTest(), b0, beta), d.yTest()),
            Scores.f1(Scores.nonZero(beta), truth)};
    }

    private static double[] tail(double[] b) {
        double[] t = new double[b.length - 1];
        System.arraycopy(b, 1, t, 0, t.length);
        return t;
    }

    private static double interceptFor(double[][] x, double[] y, double[] beta) {
        double s = 0;
        for (int i = 0; i < y.length; i++) {
            double f = 0;
            for (int j = 0; j < beta.length; j++) {
                f += x[i][j] * beta[j];
            }
            s += y[i] - f;
        }
        return s / y.length;
    }

    private static double[] ridge(SparseDesigns.Regression d, boolean[] truth) {
        int n = d.y().length;
        int cut = n * 3 / 4;
        double[][] xa = java.util.Arrays.copyOfRange(d.x(), 0, cut);
        double[] ya = java.util.Arrays.copyOfRange(d.y(), 0, cut);
        double[][] xb = java.util.Arrays.copyOfRange(d.x(), cut, n);
        double[] yb = java.util.Arrays.copyOfRange(d.y(), cut, n);
        double best = Double.POSITIVE_INFINITY;
        double bestRidge = 1;
        for (double r = 1e-2; r <= 1e4; r *= 3) {
            LeastSquares.Fit f = LeastSquares.fit(xa, ya, true, r);
            double e = Scores.mse(Scores.predict(xb, f.beta()[0], tail(f.beta())), yb);
            if (e < best) {
                best = e;
                bestRidge = r;
            }
        }
        LeastSquares.Fit f = LeastSquares.fit(d.x(), d.y(), true, bestRidge);
        return score(d, f.beta()[0], tail(f.beta()), truth);
    }

    private static double[] stepwise(SparseDesigns.Regression d, boolean[] truth) {
        double[][] x = d.x();
        double[] y = d.y();
        int n = y.length;
        int p = x[0].length;
        List<Integer> chosen = new ArrayList<>();
        double bestBic = n * Math.log(variance(y));
        double[] bestBeta = new double[p];
        double bestB0 = Scores.mean(y);
        double[] resid = y.clone();
        double ym = Scores.mean(y);
        for (int i = 0; i < n; i++) {
            resid[i] -= ym;
        }
        int maxK = Math.min(n / 3, 40);
        for (int k = 0; k < maxK; k++) {
            int pick = -1;
            double bestC = -1;
            for (int j = 0; j < p; j++) {
                if (chosen.contains(j)) {
                    continue;
                }
                double c = Math.abs(LeastSquares.correlation(LeastSquares.column(x, j), resid));
                if (c > bestC) {
                    bestC = c;
                    pick = j;
                }
            }
            chosen.add(pick);
            int[] cols = chosen.stream().mapToInt(Integer::intValue).toArray();
            LeastSquares.Fit f = LeastSquares.fit(LeastSquares.columns(x, cols), y, true, 1e-9);
            resid = f.residuals();
            double bic = n * Math.log(f.rss() / n) + (k + 1) * Math.log(n);
            if (bic < bestBic) {
                bestBic = bic;
                bestBeta = new double[p];
                for (int a = 0; a < cols.length; a++) {
                    bestBeta[cols[a]] = f.beta()[a + 1];
                }
                bestB0 = f.beta()[0];
            }
        }
        return score(d, bestB0, bestBeta, truth);
    }

    private static double variance(double[] y) {
        double m = Scores.mean(y);
        double s = 0;
        for (double v : y) {
            s += (v - m) * (v - m);
        }
        return s / y.length;
    }

    private static double[] stlsq(SparseDesigns.Regression d, boolean[] truth) {
        double[][] x = d.x();
        int n = x.length;
        int p = x[0].length;
        double[][] theta = new double[n][p + 1];
        double[][] dy = new double[n][1];
        for (int i = 0; i < n; i++) {
            theta[i][0] = 1;
            System.arraycopy(x[i], 0, theta[i], 1, p);
            dy[i][0] = d.y()[i];
        }
        double[][] xi = Sindy.stlsq(theta, dy, 0.5, n > p ? 1e-8 : 1.0, 10);
        double[] beta = new double[p];
        for (int j = 0; j < p; j++) {
            beta[j] = xi[j + 1][0];
        }
        return score(d, xi[0][0], beta, truth);
    }

    private static double maxAbsDiff(double[] a, double[] b) {
        double m = 0;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }
}
