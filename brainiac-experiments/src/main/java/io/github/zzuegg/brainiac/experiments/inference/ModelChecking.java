package io.github.zzuegg.brainiac.experiments.inference;

import io.github.zzuegg.brainiac.experimental.data.SimulatorSuite;
import io.github.zzuegg.brainiac.experimental.estimators.NestedSampling;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Prior;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Simulator;
import io.github.zzuegg.brainiac.experimental.sbi.ModelComparisonNetwork;
import io.github.zzuegg.brainiac.experimental.stats.ModelComparison;
import io.github.zzuegg.brainiac.tensor.Matrices;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Model checking and comparison: PSIS-LOO, WAIC and stacking on conjugate Bayesian linear
 * regressions (13.5.12), where exact LOO and evidences are closed-form, and amortised model
 * comparison with classifiers and evidence networks against exact Bayes factors, nested
 * sampling and ABC model choice (13.5.16).
 */
public final class ModelChecking {

    private ModelChecking() {
    }

    private static final double NOISE = 0.5;
    private static final double PRIOR_SD = 3;

    /** A conjugate Gaussian linear regression with known noise. */
    private record Fit(double[] mean, double[][] cov) {
    }

    private static Fit fit(double[][] x, double[] y, int skip) {
        int p = x[0].length;
        double[][] prec = new double[p][p];
        double[] b = new double[p];
        for (int i = 0; i < x.length; i++) {
            if (i == skip) {
                continue;
            }
            for (int a = 0; a < p; a++) {
                b[a] += x[i][a] * y[i] / (NOISE * NOISE);
                for (int c = 0; c < p; c++) {
                    prec[a][c] += x[i][a] * x[i][c] / (NOISE * NOISE);
                }
            }
        }
        for (int a = 0; a < p; a++) {
            prec[a][a] += 1 / (PRIOR_SD * PRIOR_SD);
        }
        double[][] cov = Matrices.inverse(prec);
        double[] m = new double[p];
        for (int a = 0; a < p; a++) {
            for (int c = 0; c < p; c++) {
                m[a] += cov[a][c] * b[c];
            }
        }
        return new Fit(m, cov);
    }

    private static double predictive(Fit f, double[] x, double y) {
        double mu = 0;
        double v = NOISE * NOISE;
        for (int a = 0; a < x.length; a++) {
            mu += f.mean()[a] * x[a];
            for (int c = 0; c < x.length; c++) {
                v += x[a] * f.cov()[a][c] * x[c];
            }
        }
        return -0.5 * (y - mu) * (y - mu) / v - 0.5 * Math.log(2 * Math.PI * v);
    }

    private static double[][] draws(Fit f, int s, SplittableRandom r) {
        double[][] l = Matrices.cholesky(f.cov());
        int p = f.mean().length;
        double[][] out = new double[s][p];
        for (int d = 0; d < s; d++) {
            double[] z = new double[p];
            for (int a = 0; a < p; a++) {
                z[a] = r.nextGaussian();
            }
            for (int a = 0; a < p; a++) {
                out[d][a] = f.mean()[a];
                for (int c = 0; c <= a; c++) {
                    out[d][a] += l[a][c] * z[c];
                }
            }
        }
        return out;
    }

    private static double[][] logLik(double[][] beta, double[][] x, double[] y) {
        double[][] out = new double[beta.length][x.length];
        for (int d = 0; d < beta.length; d++) {
            for (int i = 0; i < x.length; i++) {
                double mu = 0;
                for (int a = 0; a < x[i].length; a++) {
                    mu += beta[d][a] * x[i][a];
                }
                double z = (y[i] - mu) / NOISE;
                out[d][i] = -0.5 * z * z - Math.log(NOISE) - 0.5 * Math.log(2 * Math.PI);
            }
        }
        return out;
    }

    private static double logEvidence(double[][] x, double[] y) {
        int n = x.length;
        double[][] c = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dot = 0;
                for (int a = 0; a < x[i].length; a++) {
                    dot += x[i][a] * x[j][a];
                }
                c[i][j] = PRIOR_SD * PRIOR_SD * dot + (i == j ? NOISE * NOISE : 0);
            }
        }
        double[][] l = Matrices.cholesky(c);
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            double s = y[i];
            for (int k = 0; k < i; k++) {
                s -= l[i][k] * z[k];
            }
            z[i] = s / l[i][i];
        }
        double q = 0;
        double logDet = 0;
        for (int i = 0; i < n; i++) {
            q += z[i] * z[i];
            logDet += 2 * Math.log(l[i][i]);
        }
        return -0.5 * q - 0.5 * logDet - 0.5 * n * Math.log(2 * Math.PI);
    }

    /**
     * PSIS-LOO against exact LOO (13.5.12) on a linear regression with planted outliers: the
     * absolute error of elpd for PSIS-LOO, plain IS-LOO and WAIC, and the outlier detection of
     * {@code k-hat > 0.7}.
     *
     * @param n points
     * @param outliers planted outliers
     * @param draws posterior draws
     * @param seed the seed
     * @return rows: {@code PSIS-LOO, IS-LOO, WAIC -> {|elpd error|, elpd}}, {@code exact LOO ->
     *     {0, elpd}}, {@code k-hat -> {recall of outliers, false flags among clean points}}
     */
    public static Map<String, double[]> loo(int n, int outliers, int draws, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            // outliers sit at high leverage, far from the other inputs and off the line
            double v = i < outliers ? 6 + r.nextDouble() : -2 + 4 * r.nextDouble();
            x[i] = new double[] {1, v};
            y[i] = 1 + 2 * v + NOISE * r.nextGaussian() + (i < outliers ? -6 - r.nextDouble() : 0);
        }
        Fit full = fit(x, y, -1);
        double[][] ll = logLik(draws(full, draws, r), x, y);
        double exact = 0;
        for (int i = 0; i < n; i++) {
            exact += predictive(fit(x, y, i), x[i], y[i]);
        }
        ModelComparison.Loo psis = ModelComparison.psisLoo(ll);
        ModelComparison.Loo is = ModelComparison.isLoo(ll);
        double[] waic = ModelComparison.waic(ll);
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("exact LOO", new double[] {0, exact});
        out.put("PSIS-LOO", new double[] {Math.abs(psis.elpd() - exact), psis.elpd()});
        out.put("IS-LOO", new double[] {Math.abs(is.elpd() - exact), is.elpd()});
        out.put("WAIC", new double[] {Math.abs(waic[0] - exact), waic[0]});
        int hit = 0;
        int false0 = 0;
        for (int i = 0; i < n; i++) {
            boolean flag = psis.khat()[i] > 0.7;
            if (i < outliers && flag) {
                hit++;
            } else if (i >= outliers && flag) {
                false0++;
            }
        }
        out.put("k-hat>0.7 {recall, false flags}", new double[] {hit / (double) Math.max(1, outliers), false0});
        return out;
    }

    /**
     * Stacking in an M-open setting (13.5.12): each point follows a linear or a quadratic
     * regression at random; the candidates are the linear and the quadratic model. Weights from
     * stacking on PSIS-LOO, pseudo-BMA+, BMA from closed-form evidences, and the single best
     * model are scored by the held-out log predictive density.
     *
     * @param n training points
     * @param test held-out points
     * @param draws posterior draws
     * @param seed the seed
     * @return rows {@code {held-out mean log score, weight of the linear model}}
     */
    public static Map<String, double[]> stacking(int n, int test, int draws, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] xs = new double[n + test];
        double[] y = new double[n + test];
        for (int i = 0; i < n + test; i++) {
            double v = -2 + 4 * r.nextDouble();
            xs[i] = v;
            y[i] = (r.nextBoolean() ? 1 + 1.5 * v : -1 + 1.2 * v * v) + NOISE * r.nextGaussian();
        }
        double[][][] feats = new double[2][n + test][];
        for (int i = 0; i < n + test; i++) {
            feats[0][i] = new double[] {1, xs[i]};
            feats[1][i] = new double[] {1, xs[i] * xs[i]};
        }
        double[][] loo = new double[2][];
        double[] logZ = new double[2];
        double[][] heldOut = new double[2][test];
        double[] trainY = java.util.Arrays.copyOf(y, n);
        for (int k = 0; k < 2; k++) {
            double[][] tx = java.util.Arrays.copyOf(feats[k], n);
            Fit f = fit(tx, trainY, -1);
            loo[k] = ModelComparison.psisLoo(logLik(draws(f, draws, r), tx, trainY)).pointwise();
            logZ[k] = logEvidence(tx, trainY);
            for (int i = 0; i < test; i++) {
                heldOut[k][i] = predictive(f, feats[k][n + i], y[n + i]);
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        double[] stack = ModelComparison.stackingWeights(loo);
        double[] pbma = ModelComparison.pseudoBmaPlus(loo, 200, seed);
        double[] bma = ModelComparison.bmaWeights(logZ);
        double elpd0 = java.util.Arrays.stream(loo[0]).sum();
        double elpd1 = java.util.Arrays.stream(loo[1]).sum();
        double[] best = elpd0 >= elpd1 ? new double[] {1, 0} : new double[] {0, 1};
        out.put("stacking", new double[] {ModelComparison.mixtureLogScore(heldOut, stack) / test, stack[0]});
        out.put("pseudo-BMA+", new double[] {ModelComparison.mixtureLogScore(heldOut, pbma) / test, pbma[0]});
        out.put("BMA", new double[] {ModelComparison.mixtureLogScore(heldOut, bma) / test, bma[0]});
        out.put("best single (LOO)", new double[] {ModelComparison.mixtureLogScore(heldOut, best) / test, best[0]});
        return out;
    }

    // ---------------------------------------------------------------- amortised comparison

    private static final int TRIALS = 20;

    /** Exact {@code log p(x | M1) - log p(x | M0)} for the Gaussian-mean pair given the summary. */
    static double exactLogBf(double mean) {
        double tau2 = 1;
        return -0.5 * Math.log(1 + TRIALS * tau2) + TRIALS * TRIALS * mean * mean * tau2 / (2 * (1 + TRIALS * tau2));
    }

    /**
     * Amortised model comparison (13.5.16) on a conjugate pair: {@code M0: y ~ N(0, 1)} against
     * {@code M1: y ~ N(mu, 1), mu ~ N(0, 1)}, 20 trials summarised by their mean and standard
     * deviation, with exact Bayes factors. The classifier and the evidence network are scored by
     * the absolute log-Bayes-factor error in bins of the exact {@code |log BF|}, model-choice
     * accuracy and calibration error; rejection-ABC model choice (acceptance shares among the
     * nearest 2% of the simulations) and the per-observation cost against nested sampling are
     * the baselines. A last row gives classifier accuracy and calibration on SIR against SEIR.
     *
     * @param simulations training simulations
     * @param tests test observations
     * @param seed the seed
     * @return rows {@code {error |lbf| < 5, error 5-20, error > 20, accuracy, ECE}} and a cost row
     *     {@code {network microseconds per observation, nested-sampling microseconds, ratio, 0,
     *     0}}
     */
    public static Map<String, double[]> amortised(int simulations, int tests, long seed) {
        Prior none = new Prior() {
            @Override
            public double[] sample(SplittableRandom random) {
                return new double[] {0};
            }

            @Override
            public double logDensity(double[] theta) {
                return 0;
            }
        };
        Prior mu = SimulatorSuite.gaussian(new double[1], 1);
        Simulator sim = (t, r) -> {
            double s = 0;
            double s2 = 0;
            for (int i = 0; i < TRIALS; i++) {
                double v = t[0] + r.nextGaussian();
                s += v;
                s2 += v * v;
            }
            double m = s / TRIALS;
            return new double[] {m, Math.sqrt(Math.max(s2 / TRIALS - m * m, 1e-12))};
        };
        List<Prior> priors = List.of(none, mu);
        List<Simulator> sims = List.of(sim, sim);
        SplittableRandom r = new SplittableRandom(seed + 1);
        double[][] xs = new double[tests][];
        int[] labels = new int[tests];
        double[] exact = new double[tests];
        for (int i = 0; i < tests; i++) {
            labels[i] = r.nextInt(2);
            xs[i] = sim.simulate(priors.get(labels[i]).sample(r), r);
            exact[i] = exactLogBf(xs[i][0]);
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        for (ModelComparisonNetwork.Loss loss : ModelComparisonNetwork.Loss.values()) {
            ModelComparisonNetwork net = ModelComparisonNetwork.train(loss, priors, sims, simulations, 32, 1500, seed);
            long t0 = System.nanoTime();
            double[] lbf = net.logBayesFactors(xs);
            double us = (System.nanoTime() - t0) / 1e3 / tests;
            double[] row = binned(lbf, exact, labels);
            row[4] = ModelComparisonNetwork.calibrationError(net.probabilities(xs), labels);
            out.put(loss.name(), row);
            if (loss == ModelComparisonNetwork.Loss.CLASSIFIER) {
                // cost against nested sampling of M1's evidence for one observation
                double[] x0 = xs[0];
                long t1 = System.nanoTime();
                NestedSampling.of(NestedSampling.Replacement.SLICE, 100, 3, 0.01, 10000).run(1,
                        u -> new double[] {io.github.zzuegg.brainiac.experimental.ml.NormalTails
                                .quantile(Math.min(1 - 1e-12, Math.max(1e-12, u[0])))},
                        t -> -0.5 * TRIALS * ((x0[0] - t[0]) * (x0[0] - t[0]) + x0[1] * x0[1])
                                - 0.5 * TRIALS * Math.log(2 * Math.PI),
                        new SplittableRandom(seed));
                double nsUs = (System.nanoTime() - t1) / 1e3;
                out.put("cost {net us/obs, nested sampling us/obs, ratio}", new double[] {us, nsUs, nsUs / us, 0, 0});
            }
        }
        // rejection-ABC model choice on a pool of the same size
        double[][] pool = new double[simulations][];
        int[] poolM = new int[simulations];
        for (int i = 0; i < simulations; i++) {
            poolM[i] = r.nextInt(2);
            pool[i] = sim.simulate(priors.get(poolM[i]).sample(r), r);
        }
        double[] abc = new double[tests];
        int keep = Math.max(20, simulations / 50);
        for (int i = 0; i < tests; i++) {
            double[] d = new double[simulations];
            Integer[] order = new Integer[simulations];
            for (int j = 0; j < simulations; j++) {
                d[j] = Math.hypot(pool[j][0] - xs[i][0], (pool[j][1] - xs[i][1]) * 3);
                order[j] = j;
            }
            java.util.Arrays.sort(order, (a, b) -> Double.compare(d[a], d[b]));
            double c1 = 0.5;
            for (int k = 0; k < keep; k++) {
                c1 += poolM[order[k]];
            }
            abc[i] = Math.log(c1 / (keep + 1 - c1));
        }
        out.put("ABC model choice", binned(abc, exact, labels));
        // SIR against SEIR, classifier only
        var sir = SimulatorSuite.sir(0);
        var seir = SimulatorSuite.seir();
        ModelComparisonNetwork epi = ModelComparisonNetwork.train(ModelComparisonNetwork.Loss.CLASSIFIER,
                List.of(sir.prior(), seir.prior()), List.of(sir.simulator(), seir.simulator()), 1500, 32, 800, seed);
        double[][] ex = new double[300][];
        int[] el = new int[300];
        for (int i = 0; i < ex.length; i++) {
            el[i] = r.nextInt(2);
            var p = el[i] == 0 ? sir : seir;
            ex[i] = p.simulator().simulate(p.prior().sample(r), r);
        }
        double[][] pr = epi.probabilities(ex);
        double acc = 0;
        for (int i = 0; i < ex.length; i++) {
            acc += ((pr[i][1] > 0.5 ? 1 : 0) == el[i] ? 1.0 : 0) / ex.length;
        }
        out.put("SIR vs SEIR classifier {-, -, -, accuracy, ECE}", new double[] {0, 0, 0, acc,
            ModelComparisonNetwork.calibrationError(pr, el)});
        return out;
    }

    private static double[] binned(double[] lbf, double[] exact, int[] labels) {
        double[] err = new double[3];
        int[] cnt = new int[3];
        double acc = 0;
        for (int i = 0; i < lbf.length; i++) {
            double a = Math.abs(exact[i]);
            int bin = a < 5 ? 0 : a < 20 ? 1 : 2;
            err[bin] += Math.abs(lbf[i] - exact[i]);
            cnt[bin]++;
            acc += ((lbf[i] > 0 ? 1 : 0) == labels[i] ? 1.0 : 0) / lbf.length;
        }
        return new double[] {err[0] / Math.max(1, cnt[0]), err[1] / Math.max(1, cnt[1]),
            err[2] / Math.max(1, cnt[2]), acc, 0};
    }
}
