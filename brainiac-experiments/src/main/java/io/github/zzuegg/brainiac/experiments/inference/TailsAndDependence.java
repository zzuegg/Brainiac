package io.github.zzuegg.brainiac.experiments.inference;

import io.github.zzuegg.brainiac.experimental.data.DependenceSamples;
import io.github.zzuegg.brainiac.experimental.ml.IsolationForest;
import io.github.zzuegg.brainiac.experimental.ml.NormalTails;
import io.github.zzuegg.brainiac.experimental.stats.ExtremeValues;
import io.github.zzuegg.brainiac.experimental.stats.FunctionalData;
import io.github.zzuegg.brainiac.experimental.stats.SpotThreshold;
import io.github.zzuegg.brainiac.experimental.stats.VineCopula;
import io.github.zzuegg.brainiac.experimental.stats.VineCopula.Family;
import io.github.zzuegg.brainiac.tensor.Matrices;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Tails, dependence and curves: extreme quantiles by peaks over threshold, Hill and their
 * baselines, and SPOT thresholds on streams (13.5.13); vine copulas against a Gaussian copula
 * (13.5.14); sparse functional PCA, functional regression and depth-based outliers (13.5.15).
 */
public final class TailsAndDependence {

    private TailsAndDependence() {
    }

    private static final String[] LAWS = {"gpd", "frechet", "student-t", "stable"};

    /**
     * Extreme-quantile table: mean relative error at the {@code 1e-3} and {@code 1e-4} upper
     * quantiles per estimator, averaged over the four heavy-tailed laws, plus the coverage of
     * the 95% profile-likelihood interval of the POT-MLE {@code 1e-3} quantile.
     *
     * @param n the sample size
     * @param replicates replicates per law
     * @param seed the seed
     * @return rows {@code {rel err @1e-3, rel err @1e-4}}; the "POT-MLE interval" row holds
     *     {@code {coverage, mean width / q}}
     */
    public static Map<String, double[]> tails(int n, int replicates, long seed) {
        String[] names = {"POT-MLE", "POT-PWM", "Hill", "empirical", "Gaussian tail"};
        double[][] err = new double[names.length][2];
        double[] p = {1e-3, 1e-4};
        double covered = 0;
        double width = 0;
        int count = 0;
        for (String law : LAWS) {
            for (int rep = 0; rep < replicates; rep++) {
                DependenceSamples.HeavyTail h = DependenceSamples.heavyTail(law, n,
                        seed + 1000L * rep + law.hashCode());
                double[] x = h.sample();
                ExtremeValues.Pot mle = ExtremeValues.pot(x, 0.95, true);
                ExtremeValues.Pot pwm = ExtremeValues.pot(x, 0.95, false);
                int k = ExtremeValues.hillK(x);
                double[] s = x.clone();
                Arrays.sort(s);
                double mean = Arrays.stream(x).average().orElse(0);
                double sd = Math.sqrt(Arrays.stream(x).map(v -> (v - mean) * (v - mean)).sum() / (n - 1));
                for (int j = 0; j < 2; j++) {
                    double truth = h.quantile(p[j]);
                    double[] est = {
                        mle.quantile(1 - p[j]),
                        pwm.quantile(1 - p[j]),
                        ExtremeValues.hillQuantile(x, k, 1 - p[j]),
                        s[Math.min(n - 1, (int) Math.floor((1 - p[j]) * n))],
                        mean + sd * NormalTails.quantile(1 - p[j])};
                    for (int m = 0; m < names.length; m++) {
                        err[m][j] += Math.abs(est[m] - truth) / truth;
                    }
                }
                double truth = h.quantile(1e-3);
                double[] ci = mle.quantileInterval(1 - 1e-3, 3.841);
                covered += ci[0] <= truth && truth <= ci[1] ? 1 : 0;
                width += (ci[1] - ci[0]) / truth;
                count++;
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int m = 0; m < names.length; m++) {
            out.put(names[m], new double[] {err[m][0] / count, err[m][1] / count});
        }
        out.put("POT-MLE interval", new double[] {covered / count, width / count});
        return out;
    }

    /**
     * Streaming thresholds at a target false-alarm probability {@code q = 1e-3}: realised
     * false-alarm rate (per normal point) and recall of planted anomalies, on a stationary
     * heavy-tailed stream and on one with a drifting level.
     *
     * @param length the stream length after calibration
     * @param seed the seed
     * @return rows {@code {FA stationary, recall stationary, FA drift, recall drift}}
     */
    public static Map<String, double[]> anomalies(int length, long seed) {
        double q = 1e-3;
        int calibration = 2000;
        Map<String, double[]> out = new LinkedHashMap<>();
        String[] names = {"SPOT", "DSPOT", "fixed top-1%", "empirical 1-q", "Gaussian 1-q"};
        double[][] res = new double[names.length][4];
        for (int drift = 0; drift < 2; drift++) {
            SplittableRandom r = new SplittableRandom(seed + drift);
            int total = calibration + length;
            double[] x = new double[total];
            boolean[] anomaly = new boolean[total];
            double level = 0;
            for (int i = 0; i < total; i++) {
                if (drift == 1) {
                    level += 8.0 / total;
                }
                double c = 0;
                for (int k = 0; k < 4; k++) {
                    double z = NormalTails.quantile(Math.min(1 - 1e-12, Math.max(1e-12, r.nextDouble())));
                    c += z * z;
                }
                double t = NormalTails.quantile(Math.min(1 - 1e-12, Math.max(1e-12, r.nextDouble())))
                        / Math.sqrt(c / 4);
                x[i] = level + Math.abs(t);
                if (i >= calibration && r.nextDouble() < 0.002) {
                    anomaly[i] = true;
                    x[i] += 12;
                }
            }
            double[] cal = Arrays.copyOf(x, calibration);
            double[] sorted = cal.clone();
            Arrays.sort(sorted);
            double mean = Arrays.stream(cal).average().orElse(0);
            double sd = Math.sqrt(Arrays.stream(cal).map(v -> (v - mean) * (v - mean)).sum() / (calibration - 1));
            double[] fixed = {
                Double.NaN, Double.NaN,
                sorted[(int) (0.99 * calibration)],
                sorted[(int) Math.floor((1 - q) * (calibration - 1))],
                mean + sd * NormalTails.quantile(1 - q)};
            SpotThreshold spot = SpotThreshold.spot(q, 0.98);
            spot.calibrate(cal);
            SpotThreshold dspot = SpotThreshold.dspot(q, 0.98, 50);
            dspot.calibrate(cal);
            for (int m = 0; m < names.length; m++) {
                double fa = 0;
                double normal = 0;
                double hit = 0;
                double planted = 0;
                for (int i = calibration; i < total; i++) {
                    boolean alarm = m == 0 ? spot.update(x[i]) : m == 1 ? dspot.update(x[i]) : x[i] > fixed[m];
                    if (anomaly[i]) {
                        planted++;
                        hit += alarm ? 1 : 0;
                    } else {
                        normal++;
                        fa += alarm ? 1 : 0;
                    }
                }
                res[m][2 * drift] = fa / normal;
                res[m][2 * drift + 1] = hit / Math.max(planted, 1);
            }
        }
        for (int m = 0; m < names.length; m++) {
            out.put(names[m], res[m]);
        }
        return out;
    }

    /** A Gaussian copula on normal scores. */
    private record GaussianCopula(double[][] precisionMinusIdentity, double logDet) {

        static GaussianCopula fit(double[][] u) {
            int d = u[0].length;
            double[][] c = correlation(u);
            double[][] inv = Matrices.inverse(c);
            for (int a = 0; a < d; a++) {
                inv[a][a] -= 1;
            }
            return new GaussianCopula(inv, Math.log(Matrices.determinant(c)));
        }

        static double[][] correlation(double[][] u) {
            int n = u.length;
            int d = u[0].length;
            double[][] z = scores(u);
            double[][] c = new double[d][d];
            for (double[] row : z) {
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        c[a][b] += row[a] * row[b] / n;
                    }
                }
            }
            double[][] r = new double[d][d];
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    r[a][b] = c[a][b] / Math.sqrt(c[a][a] * c[b][b]);
                }
            }
            return r;
        }

        static double[][] scores(double[][] u) {
            double[][] z = new double[u.length][u[0].length];
            for (int i = 0; i < u.length; i++) {
                for (int a = 0; a < u[0].length; a++) {
                    z[i][a] = NormalTails.quantile(Math.min(1 - 1e-10, Math.max(1e-10, u[i][a])));
                }
            }
            return z;
        }

        double[] logDensity(double[][] u) {
            double[][] z = scores(u);
            double[] out = new double[u.length];
            int d = z[0].length;
            for (int i = 0; i < u.length; i++) {
                double q = 0;
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        q += z[i][a] * precisionMinusIdentity[a][b] * z[i][b];
                    }
                }
                out[i] = -0.5 * logDet - 0.5 * q;
            }
            return out;
        }
    }

    private static double mean(double[] v) {
        return Arrays.stream(v).average().orElse(Double.NaN);
    }

    /**
     * Dependence table on the planted 4-D vine.
     *
     * @param n training rows (and as many held-out rows)
     * @param allStructures whether to fit the C- and D-vines too (slower)
     * @param seed the seed
     * @return rows {@code {held-out log c per row, |lambda_L(0,1) err|, |lambda_U(0,2) err|,
     *     rel err P(U0,U1 < 0.01), rel err P(U0,U2 > 0.99), first-tree recovery}}
     */
    public static Map<String, double[]> dependence(int n, boolean allStructures, long seed) {
        DependenceSamples.Vine truth = DependenceSamples.vine();
        double[][] train = VineCopula.pseudoObservations(truth.sample(n, seed));
        double[][] test = truth.sample(n, seed + 1);
        double a = 0.01;
        double lowTruth = truth.families()[0][0].cdf(a, a, truth.parameters()[0][0]);
        double highTruth = 1 - 2 * (1 - a) + truth.families()[0][1].cdf(1 - a, 1 - a, truth.parameters()[0][1]);
        double[] lamTruth = {truth.tailDependence(0)[0], truth.tailDependence(1)[1]};
        Map<String, double[]> out = new LinkedHashMap<>();
        double oracle = mean(trueLogDensity(truth, test));
        out.put("true vine (oracle)", new double[] {oracle, 0, 0, 0, 0, 1});
        Family[] all = {Family.GAUSSIAN, Family.STUDENT_T, Family.CLAYTON, Family.GUMBEL, Family.FRANK};
        for (VineCopula.Structure s : VineCopula.Structure.values()) {
            if (!allStructures && s != VineCopula.Structure.R_VINE) {
                continue;
            }
            VineCopula v = VineCopula.fit(train, s, all);
            out.put(s.name(), vineRow(v, test, lamTruth, lowTruth, highTruth, truth.firstTree()));
        }
        VineCopula gaussVine = VineCopula.fit(train, VineCopula.Structure.R_VINE, Family.GAUSSIAN);
        out.put("R_VINE Gaussian pairs", vineRow(gaussVine, test, lamTruth, lowTruth, highTruth, truth.firstTree()));
        GaussianCopula g = GaussianCopula.fit(train);
        double[][] corr = GaussianCopula.correlation(train);
        double low = Family.GAUSSIAN.cdf(a, a, corr[0][1]);
        double high = 1 - 2 * (1 - a) + Family.GAUSSIAN.cdf(1 - a, 1 - a, corr[0][2]);
        out.put("Gaussian copula", new double[] {
            mean(g.logDensity(test)), lamTruth[0], lamTruth[1],
            Math.abs(low - lowTruth) / lowTruth, Math.abs(high - highTruth) / highTruth, Double.NaN});
        return out;
    }

    private static double[] vineRow(VineCopula v, double[][] test, double[] lamTruth, double lowTruth,
            double highTruth, int[][] tree) {
        double a = 0.01;
        double ll = mean(v.logDensity(test));
        VineCopula.Pair p01 = v.firstTreePair(0, 1);
        VineCopula.Pair p02 = v.firstTreePair(0, 2);
        double lamL = p01 == null ? Double.NaN
                : Math.abs(p01.family().tailDependence(p01.parameter())[0] - lamTruth[0]);
        double lamU = p02 == null ? Double.NaN
                : Math.abs(p02.family().tailDependence(p02.parameter())[1] - lamTruth[1]);
        double low = p01 == null ? Double.NaN : p01.family().cdf(a, a, p01.parameter());
        double high = p02 == null ? Double.NaN : 1 - 2 * (1 - a) + p02.family().cdf(1 - a, 1 - a, p02.parameter());
        int[][] got = v.firstTree();
        int hit = 0;
        for (int[] e : tree) {
            for (int[] f : got) {
                hit += e[0] == f[0] && e[1] == f[1] ? 1 : 0;
            }
        }
        return new double[] {ll, lamL, lamU, Math.abs(low - lowTruth) / lowTruth,
            Math.abs(high - highTruth) / highTruth, hit / (double) tree.length};
    }

    /** The true C-vine log density (Aas et al. 2009, Algorithm 3). */
    private static double[] trueLogDensity(DependenceSamples.Vine vine, double[][] u) {
        int d = u[0].length;
        Family[][] f = vine.families();
        double[][] par = vine.parameters();
        double[] out = new double[u.length];
        for (int r = 0; r < u.length; r++) {
            double[] v = u[r].clone();
            double ll = 0;
            for (int j = 0; j < d - 1; j++) {
                double[] next = new double[d];
                for (int i = 0; i < d - 1 - j; i++) {
                    double root = v[j];
                    double other = v[j + 1 + i];
                    ll += f[j][i].logDensity(root, other, par[j][i]);
                    next[j + 1 + i] = f[j][i].h(other, root, par[j][i]);
                }
                for (int i = j + 1; i < d; i++) {
                    v[i] = next[i];
                }
            }
            out[r] = ll;
        }
        return out;
    }

    /**
     * Functional table: eigenfunction recovery, reconstruction and regression from sparse
     * curves (PACE against interpolate-then-PCA and dense FPCA), and depth-based outlier
     * detection (AUROC for magnitude and shape outliers against an isolation forest and the
     * distance to the pointwise median).
     *
     * @param n curves
     * @param seed the seed
     * @return rows; the curve rows hold {@code {angle phi1 (deg), angle phi2 (deg),
     *     reconstruction RMSE, outcome RMSE (held out), relative beta error}}, the outlier rows
     *     {@code {AUROC magnitude, AUROC shape}}
     */
    public static Map<String, double[]> functional(int n, long seed) {
        DependenceSamples.Curves c = DependenceSamples.curves(n, 41, 3, 5, 0, seed);
        double[] grid = c.grid();
        int k = 3;
        int train = n * 3 / 4;
        Map<String, double[]> out = new LinkedHashMap<>();
        FunctionalData.Basis pace = FunctionalData.pace(c.times(), c.values(), grid, k, 10, 0.08);
        out.put("PACE (sparse)", curveRow(c, pace, pace.blup(c.times(), c.values()), train));
        double[][] interp = new double[n][];
        for (int i = 0; i < n; i++) {
            interp[i] = interpolateSparse(c.times()[i], c.values()[i], grid);
        }
        FunctionalData.Basis ip = FunctionalData.fpca(grid, interp, k);
        out.put("interpolate+PCA (sparse)", curveRow(c, ip, ip.project(interp), train));
        FunctionalData.Basis dense = FunctionalData.fpca(grid, c.dense(), k);
        out.put("dense FPCA (oracle data)", curveRow(c, dense, dense.project(c.dense()), train));

        DependenceSamples.Curves o = DependenceSamples.curves(100, 41, 3, 5, 0.12, seed + 7);
        double[][] x = o.dense();
        double[] mbd = FunctionalData.modifiedBandDepth(x);
        double[] negDepth = Arrays.stream(mbd).map(v -> -v).toArray();
        double[] box = FunctionalData.boxplotOutliers(x);
        double[] gram = FunctionalData.outliergram(x);
        double[] combined = new double[x.length];
        double[] rb = ranks(box);
        double[] rg = ranks(gram);
        for (int i = 0; i < x.length; i++) {
            combined[i] = Math.max(rb[i], rg[i]);
        }
        double[] iso = IsolationForest.fit(x, 100, 64, seed).score(x);
        double[] dist = new double[x.length];
        for (int g = 0; g < grid.length; g++) {
            double[] col = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                col[i] = x[i][g];
            }
            Arrays.sort(col);
            double med = col[x.length / 2];
            for (int i = 0; i < x.length; i++) {
                dist[i] += (x[i][g] - med) * (x[i][g] - med);
            }
        }
        out.put("band depth (MBD)", aurocs(negDepth, o.outlier()));
        out.put("functional boxplot", aurocs(box, o.outlier()));
        out.put("outliergram", aurocs(gram, o.outlier()));
        out.put("boxplot+outliergram", aurocs(combined, o.outlier()));
        out.put("isolation forest", aurocs(iso, o.outlier()));
        out.put("L2 to median", aurocs(dist, o.outlier()));
        return out;
    }

    private static double[] interpolateSparse(double[] t, double[] v, double[] grid) {
        Integer[] ord = new Integer[t.length];
        for (int i = 0; i < t.length; i++) {
            ord[i] = i;
        }
        Arrays.sort(ord, (a, b) -> Double.compare(t[a], t[b]));
        double[] ts = new double[t.length];
        double[] vs = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            ts[i] = t[ord[i]];
            vs[i] = v[ord[i]];
        }
        double[] out = new double[grid.length];
        for (int g = 0; g < grid.length; g++) {
            out[g] = FunctionalData.interpolate(ts, vs, grid[g]);
        }
        return out;
    }

    private static double[] curveRow(DependenceSamples.Curves c, FunctionalData.Basis b, double[][] scores,
            int train) {
        int n = scores.length;
        int m = c.grid().length;
        double h = 1.0 / (m - 1);
        double[] angles = new double[2];
        for (int j = 0; j < 2; j++) {
            double dot = 0;
            double na = 0;
            double nb = 0;
            for (int g = 0; g < m; g++) {
                dot += b.components()[j][g] * c.eigenfunctions()[j][g];
                na += b.components()[j][g] * b.components()[j][g];
                nb += c.eigenfunctions()[j][g] * c.eigenfunctions()[j][g];
            }
            angles[j] = Math.toDegrees(Math.acos(Math.min(1, Math.abs(dot) / Math.sqrt(na * nb))));
        }
        double[][] rec = b.reconstruct(scores);
        double se = 0;
        for (int i = 0; i < n; i++) {
            for (int g = 0; g < m; g++) {
                se += (rec[i][g] - c.truth()[i][g]) * (rec[i][g] - c.truth()[i][g]) / (n * m);
            }
        }
        double[][] trS = Arrays.copyOf(scores, train);
        double[] trY = Arrays.copyOf(c.y(), train);
        double[] coef = FunctionalData.regression(trS, trY);
        double[] pred = FunctionalData.predict(coef, Arrays.copyOfRange(scores, train, n));
        double yse = 0;
        for (int i = train; i < n; i++) {
            yse += (pred[i - train] - c.y()[i]) * (pred[i - train] - c.y()[i]) / (n - train);
        }
        double num = 0;
        double den = 0;
        for (int g = 0; g < m; g++) {
            double beta = 0;
            for (int j = 0; j < b.components().length; j++) {
                beta += coef[j + 1] * b.components()[j][g];
            }
            num += (beta - c.beta()[g]) * (beta - c.beta()[g]) * h;
            den += c.beta()[g] * c.beta()[g] * h;
        }
        return new double[] {angles[0], angles[1], Math.sqrt(se), Math.sqrt(yse), Math.sqrt(num / den)};
    }

    private static double[] ranks(double[] v) {
        Integer[] ord = new Integer[v.length];
        for (int i = 0; i < v.length; i++) {
            ord[i] = i;
        }
        Arrays.sort(ord, (a, b) -> Double.compare(v[a], v[b]));
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            r[ord[i]] = i;
        }
        return r;
    }

    private static double[] aurocs(double[] score, int[] kind) {
        return new double[] {auroc(score, kind, 1), auroc(score, kind, 2)};
    }

    /** AUROC of positives of one outlier kind against the regular curves. */
    private static double auroc(double[] score, int[] kind, int positive) {
        double wins = 0;
        double pairs = 0;
        for (int i = 0; i < score.length; i++) {
            if (kind[i] != positive) {
                continue;
            }
            for (int j = 0; j < score.length; j++) {
                if (kind[j] != 0) {
                    continue;
                }
                wins += score[i] > score[j] ? 1 : score[i] == score[j] ? 0.5 : 0;
                pairs++;
            }
        }
        return wins / pairs;
    }
}
