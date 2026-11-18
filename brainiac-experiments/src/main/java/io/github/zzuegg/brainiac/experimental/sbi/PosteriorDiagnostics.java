package io.github.zzuegg.brainiac.experimental.sbi;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Prior;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Simulator;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Distributions;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Diagnostics for amortised posteriors that need no reference posterior.
 * <ul>
 *   <li>{@link #sbc}: simulation-based calibration (Talts et al. 2018,
 *       <a href="https://arxiv.org/abs/1804.06788">arXiv:1804.06788</a>): the rank of the
 *       true parameter among posterior samples is uniform for a calibrated posterior; a
 *       chi-squared test (core {@link Distributions#chiSquaredSurvival}) on the rank histogram
 *       per coordinate, with a Bonferroni minimum;</li>
 *   <li>{@link #expectedCoverage}: the share of true parameters inside the {@code alpha}
 *       credible region, here the ellipsoid of posterior Mahalanobis distance; below the
 *       diagonal means overconfident;</li>
 *   <li>{@link #tarp}: tests of accuracy with random points (Lemos et al., ICML 2023,
 *       <a href="https://arxiv.org/abs/2302.03026">arXiv:2302.03026</a>): for a random reference
 *       point {@code theta_r}, the share of posterior samples closer to it than the true
 *       parameter is uniform if and only if the posterior is correct; reported as the coverage
 *       curve, its largest deviation and a Kolmogorov-Smirnov p-value;</li>
 *   <li>{@link #localC2st}: local classifier two-sample tests (Linhart et al., NeurIPS 2023,
 *       <a href="https://arxiv.org/abs/2306.03580">arXiv:2306.03580</a>): one classifier trained
 *       on calibration pairs to tell {@code (theta, x)} from {@code (theta ~ q(.|x), x)} is
 *       evaluated at a single observation; its mean squared deviation from 1/2 over
 *       {@code q(.|x_o)} samples is the statistic, with a null distribution from classifiers
 *       trained where both classes come from {@code q}.</li>
 * </ul>
 */
public final class PosteriorDiagnostics {

    private PosteriorDiagnostics() {
    }

    /** An amortised posterior to diagnose. */
    @FunctionalInterface
    public interface PosteriorSampler {
        /**
         * Draws posterior samples.
         *
         * @param x the observation
         * @param n the number of samples
         * @param random the random source
         * @return {@code [n][d]}
         */
        double[][] sample(double[] x, int n, SplittableRandom random);
    }

    /**
     * Simulation-based calibration ranks and their uniformity tests.
     *
     * @param ranks ranks {@code [draws][d]} in {@code 0..samples}
     * @param pValues the chi-squared p-value per coordinate
     * @param minPValue the Bonferroni-corrected smallest p-value
     */
    public record SbcResult(int[][] ranks, double[] pValues, double minPValue) {
    }

    /**
     * A coverage curve.
     *
     * @param levels credibility levels
     * @param coverage empirical coverage at each level
     * @param maxDeviation the largest {@code |coverage - level|}
     * @param pValue a Kolmogorov-Smirnov p-value for uniformity of the underlying statistic
     */
    public record Coverage(double[] levels, double[] coverage, double maxDeviation, double pValue) {
    }

    /**
     * A local C2ST.
     *
     * @param statistic the mean squared deviation of the classifier from 1/2
     * @param pValue the share of null statistics at least as large
     */
    public record LocalTest(double statistic, double pValue) {
    }

    /** The classifier for local C2ST. */
    public enum Classifier {
        /** L2-regularised logistic regression on standardised linear and quadratic features. */
        LOGISTIC_QUADRATIC,
        /** A small tanh MLP. */
        MLP
    }

    /**
     * Draws {@code draws} (theta, x) pairs from the prior predictive, with posterior samples for
     * each, in the layout every diagnostic takes.
     *
     * @param prior the prior
     * @param sim the simulator
     * @param q the posterior
     * @param draws the number of pairs
     * @param samples posterior samples per pair
     * @param seed the seed
     * @return {@code {thetas [draws][d], xs [draws][dx], posteriors flattened [draws*samples][d]}}
     */
    public static double[][][] simulateChecks(Prior prior, Simulator sim, PosteriorSampler q, int draws, int samples,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] t = new double[draws][];
        double[][] x = new double[draws][];
        double[][] post = new double[draws * samples][];
        for (int i = 0; i < draws; i++) {
            t[i] = prior.sample(r);
            x[i] = sim.simulate(t[i], r);
            double[][] s = q.sample(x[i], samples, r);
            System.arraycopy(s, 0, post, i * samples, samples);
        }
        return new double[][][] {t, x, post};
    }

    private static double[][] block(double[][] flat, int i, int samples) {
        return Arrays.copyOfRange(flat, i * samples, (i + 1) * samples);
    }

    /**
     * SBC from checks made by {@link #simulateChecks}.
     *
     * @param thetas true parameters {@code [draws][d]}
     * @param posteriors flattened posterior samples {@code [draws*samples][d]}
     * @param samples samples per draw {@code L}
     * @param bins histogram bins
     * @return ranks and tests
     */
    public static SbcResult sbc(double[][] thetas, double[][] posteriors, int samples, int bins) {
        int draws = thetas.length;
        int d = thetas[0].length;
        int[][] ranks = new int[draws][d];
        for (int i = 0; i < draws; i++) {
            for (int s = 0; s < samples; s++) {
                double[] v = posteriors[i * samples + s];
                for (int j = 0; j < d; j++) {
                    if (v[j] < thetas[i][j]) {
                        ranks[i][j]++;
                    }
                }
            }
        }
        double[] p = new double[d];
        double min = 1;
        for (int j = 0; j < d; j++) {
            double[] counts = new double[bins];
            for (int[] rk : ranks) {
                counts[Math.min(bins - 1, rk[j] * bins / (samples + 1))]++;
            }
            double e = draws / (double) bins;
            double chi = 0;
            for (double c : counts) {
                chi += (c - e) * (c - e) / e;
            }
            p[j] = Distributions.chiSquaredSurvival(chi, bins - 1);
            min = Math.min(min, p[j] * d);
        }
        return new SbcResult(ranks, p, Math.min(1, min));
    }

    private static double[] levels() {
        double[] l = new double[19];
        for (int i = 0; i < l.length; i++) {
            l[i] = 0.05 * (i + 1);
        }
        return l;
    }

    private static Coverage curve(double[] stat) {
        double[] lv = levels();
        double[] cov = new double[lv.length];
        double max = 0;
        for (int k = 0; k < lv.length; k++) {
            int c = 0;
            for (double s : stat) {
                if (s < lv[k]) {
                    c++;
                }
            }
            cov[k] = c / (double) stat.length;
            max = Math.max(max, Math.abs(cov[k] - lv[k]));
        }
        return new Coverage(lv, cov, max, ksUniformPValue(stat));
    }

    /**
     * The Kolmogorov-Smirnov p-value (asymptotic) for a sample against {@code U(0, 1)}.
     *
     * @param u the sample
     * @return the p-value
     */
    public static double ksUniformPValue(double[] u) {
        double[] s = u.clone();
        Arrays.sort(s);
        int n = s.length;
        double dmax = 0;
        for (int i = 0; i < n; i++) {
            dmax = Math.max(dmax, Math.max((i + 1.0) / n - s[i], s[i] - i / (double) n));
        }
        double lambda = (Math.sqrt(n) + 0.12 + 0.11 / Math.sqrt(n)) * dmax;
        double p = 0;
        for (int k = 1; k <= 100; k++) {
            p += 2 * Math.pow(-1, k - 1) * Math.exp(-2 * k * k * lambda * lambda);
        }
        return Math.max(0, Math.min(1, p));
    }

    /**
     * Expected coverage of Mahalanobis-ellipsoid credible regions: the credibility of each true
     * parameter is the share of posterior samples nearer the posterior mean, in the posterior's
     * own covariance metric.
     *
     * @param thetas true parameters
     * @param posteriors flattened posterior samples
     * @param samples samples per draw
     * @return the coverage curve
     */
    public static Coverage expectedCoverage(double[][] thetas, double[][] posteriors, int samples) {
        double[] cred = new double[thetas.length];
        for (int i = 0; i < thetas.length; i++) {
            double[][] s = block(posteriors, i, samples);
            double[] m = SbiArrays.mean(s);
            double[] sd = SbiArrays.sd(s, m);
            double dt = 0;
            for (int j = 0; j < m.length; j++) {
                double z = (thetas[i][j] - m[j]) / sd[j];
                dt += z * z;
            }
            int closer = 0;
            for (double[] v : s) {
                double dv = 0;
                for (int j = 0; j < m.length; j++) {
                    double z = (v[j] - m[j]) / sd[j];
                    dv += z * z;
                }
                if (dv < dt) {
                    closer++;
                }
            }
            cred[i] = (closer + 0.5) / (samples + 1);
        }
        return curve(cred);
    }

    /**
     * TARP with reference points drawn from the prior, distances standardised by the prior's
     * spread.
     *
     * @param thetas true parameters
     * @param posteriors flattened posterior samples
     * @param samples samples per draw
     * @param prior the prior, for reference points and scale
     * @param seed the seed
     * @return the coverage curve of the TARP statistic
     */
    public static Coverage tarp(double[][] thetas, double[][] posteriors, int samples, Prior prior, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] pool = new double[500][];
        for (int i = 0; i < pool.length; i++) {
            pool[i] = prior.sample(r);
        }
        double[] m = SbiArrays.mean(pool);
        double[] sd = SbiArrays.sd(pool, m);
        double[] f = new double[thetas.length];
        for (int i = 0; i < thetas.length; i++) {
            double[] ref = prior.sample(r);
            double dt = dist(thetas[i], ref, sd);
            int closer = 0;
            for (int s = 0; s < samples; s++) {
                if (dist(posteriors[i * samples + s], ref, sd) < dt) {
                    closer++;
                }
            }
            f[i] = (closer + r.nextDouble()) / (samples + 1);
        }
        return curve(f);
    }

    private static double dist(double[] a, double[] b, double[] sd) {
        double s = 0;
        for (int j = 0; j < a.length; j++) {
            double z = (a[j] - b[j]) / sd[j];
            s += z * z;
        }
        return s;
    }

    /** A trained probabilistic classifier on joint {@code (theta, x)} features. */
    private interface Probability {
        double[] predict(double[][] z);
    }

    private static double[][] joint(double[][] t, double[][] x) {
        double[][] z = new double[t.length][];
        for (int i = 0; i < t.length; i++) {
            z[i] = SbiArrays.concat(t[i], x[i]);
        }
        return z;
    }

    private static Probability train(Classifier kind, double[][] pos, double[][] neg, long seed) {
        double[][] all = SbiArrays.stack(pos, neg);
        double[] m = SbiArrays.mean(all);
        double[] sd = SbiArrays.sd(all, m);
        int n = all.length;
        float[] y = new float[n];
        for (int i = 0; i < pos.length; i++) {
            y[i] = 1;
        }
        if (kind == Classifier.LOGISTIC_QUADRATIC) {
            double[][] f = new double[n][];
            for (int i = 0; i < n; i++) {
                f[i] = quadratic(all[i], m, sd);
            }
            int p = f[0].length;
            double[] w = new double[p + 1];
            for (int it = 0; it < 25; it++) {   // Newton-Raphson (IRLS) with a small ridge
                double[] g = new double[p + 1];
                double[][] h = new double[p + 1][p + 1];
                for (int i = 0; i < n; i++) {
                    double[] fi = Arrays.copyOf(f[i], p + 1);
                    fi[p] = 1;
                    double eta = 0;
                    for (int j = 0; j <= p; j++) {
                        eta += w[j] * fi[j];
                    }
                    double mu = 1 / (1 + Math.exp(-eta));
                    double v = Math.max(mu * (1 - mu), 1e-9);
                    for (int a = 0; a <= p; a++) {
                        g[a] += (mu - y[i]) * fi[a];
                        for (int b = 0; b <= p; b++) {
                            h[a][b] += v * fi[a] * fi[b];
                        }
                    }
                }
                for (int a = 0; a < p; a++) {
                    g[a] += 1e-2 * w[a];
                    h[a][a] += 1e-2;
                }
                h[p][p] += 1e-9;
                double[] step = io.github.zzuegg.brainiac.tensor.Matrices.solve(h, g);
                double norm = 0;
                for (int j = 0; j <= p; j++) {
                    w[j] -= step[j];
                    norm += step[j] * step[j];
                }
                if (norm < 1e-12) {
                    break;
                }
            }
            return z -> {
                double[] out = new double[z.length];
                for (int i = 0; i < z.length; i++) {
                    double[] fi = quadratic(z[i], m, sd);
                    double eta = w[p];
                    for (int j = 0; j < p; j++) {
                        eta += w[j] * fi[j];
                    }
                    out[i] = 1 / (1 + Math.exp(-eta));
                }
                return out;
            };
        }
        SimpleMlp net = SimpleMlp.tanh(seed, m.length, 32, 32, 1);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int b = Math.min(128, n);
        for (int it = 0; it < 300; it++) {
            int[] idx = SbiArrays.batch(n, b, r);
            float[] yb = new float[b];
            for (int i = 0; i < b; i++) {
                yb[i] = y[idx[i]];
            }
            Var logit = net.forward(Var.of(SbiArrays.standardisedRows(all, idx, m, sd))).reshape(b);
            Tensor yt = Tensor.of(Shape.of(b), yb);
            // binary cross-entropy: softplus(logit) - y * logit
            Var loss = logit.softplus().sub(logit.mul(yt)).mean();
            loss.backward();
            opt.step(net.parameters());
        }
        return z -> Var.noGrad(() -> net.forward(Var.of(SbiArrays.standardised(z, m, sd))).sigmoid()).value()
                .toDoubleArray();
    }

    private static double[] quadratic(double[] v, double[] m, double[] sd) {
        int d = v.length;
        double[] z = new double[d];
        for (int j = 0; j < d; j++) {
            z[j] = (v[j] - m[j]) / sd[j];
        }
        double[] f = new double[d + d * (d + 1) / 2];
        int k = 0;
        for (int j = 0; j < d; j++) {
            f[k++] = z[j];
        }
        for (int a = 0; a < d; a++) {
            for (int b = a; b < d; b++) {
                f[k++] = z[a] * z[b] / 2;
            }
        }
        return f;
    }

    /**
     * Local C2ST at several observations, with a shared null from {@code nulls} classifiers
     * trained on data where both classes come from {@code q}.
     *
     * @param kind the classifier
     * @param calibrationThetas calibration parameters from the prior
     * @param calibrationXs their simulations
     * @param q the posterior under test
     * @param observations the observations to test
     * @param evaluation {@code q(.|x_o)} samples per evaluation
     * @param nulls null classifiers
     * @param seed the seed
     * @return one test per observation
     */
    public static LocalTest[] localC2st(Classifier kind, double[][] calibrationThetas, double[][] calibrationXs,
            PosteriorSampler q, double[][] observations, int evaluation, int nulls, long seed) {
        Objects.requireNonNull(kind, "kind");
        SplittableRandom r = new SplittableRandom(seed);
        int n = calibrationThetas.length;
        double[][] fromQ = new double[n][];
        for (int i = 0; i < n; i++) {
            fromQ[i] = q.sample(calibrationXs[i], 1, r)[0];
        }
        Probability clf = train(kind, joint(calibrationThetas, calibrationXs), joint(fromQ, calibrationXs), seed);
        Probability[] nullClf = new Probability[nulls];
        for (int h = 0; h < nulls; h++) {
            double[][] a = new double[n][];
            double[][] b = new double[n][];
            for (int i = 0; i < n; i++) {
                a[i] = q.sample(calibrationXs[i], 1, r)[0];
                b[i] = q.sample(calibrationXs[i], 1, r)[0];
            }
            nullClf[h] = train(kind, joint(a, calibrationXs), joint(b, calibrationXs), seed + 7 + h);
        }
        LocalTest[] out = new LocalTest[observations.length];
        for (int o = 0; o < observations.length; o++) {
            double[][] s = q.sample(observations[o], evaluation, r);
            double[][] xo = new double[evaluation][];
            Arrays.fill(xo, observations[o]);
            double[][] z = joint(s, xo);
            double t = msd(clf.predict(z));
            int ge = 0;
            for (Probability h : nullClf) {
                if (msd(h.predict(z)) >= t) {
                    ge++;
                }
            }
            out[o] = new LocalTest(t, (ge + 1.0) / (nulls + 1));
        }
        return out;
    }

    private static double msd(double[] p) {
        double s = 0;
        for (double v : p) {
            s += (v - 0.5) * (v - 0.5) / p.length;
        }
        return s;
    }
}
