package io.github.zzuegg.brainiac.experiments.estimation;

import io.github.zzuegg.brainiac.experimental.estimators.ClassifierTwoSample;
import io.github.zzuegg.brainiac.experimental.estimators.ExpectationPropagation;
import io.github.zzuegg.brainiac.experimental.estimators.MutualInformation;
import io.github.zzuegg.brainiac.experimental.estimators.NestedSampling;
import io.github.zzuegg.brainiac.experimental.estimators.SteinTests;
import io.github.zzuegg.brainiac.experimental.ml.AnnealedImportanceSampler;
import io.github.zzuegg.brainiac.experimental.ml.GaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.KernelTests;
import io.github.zzuegg.brainiac.experimental.ml.LogDensity;
import io.github.zzuegg.brainiac.experimental.ml.McmcDiagnostics;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.NormalTails;
import io.github.zzuegg.brainiac.experimental.ml.StationaryKernel;
import io.github.zzuegg.brainiac.experimental.train.HamiltonianMonteCarlo;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;
import java.util.function.UnaryOperator;

/**
 * Approximate inference, evidence, information and testing (wave 12, items 12.5.10 to 12.5.13).
 *
 * <ul>
 *   <li>{@link #gpClassification}: probit GP classification by sequential and parallel EP, the
 *       Laplace approximation (probit, and the wave-6 logistic classifier), against long HMC on the
 *       whitened latents as ground truth for predictions and marginals, and annealed importance
 *       sampling for the evidence;</li>
 *   <li>{@link #evidence}: nested sampling with three replacement samplers against annealed
 *       importance sampling, an SMC sampler and the harmonic-mean estimator, on a well-separated
 *       Gaussian mixture and the Gaussian-shells problem, both with known {@code log Z};</li>
 *   <li>{@link #mutualInformation}: binned, KSG and four variational estimators on correlated
 *       Gaussians, with and without a cubic transform of {@code y};</li>
 *   <li>{@link #goodnessOfFit}: rejection rates of KSD, FSSD, MMD, HSIC and classifier two-sample
 *       tests against a standard normal model under the null and three alternatives.</li>
 * </ul>
 */
public final class InferenceAndTesting {

    private InferenceAndTesting() {
    }

    // ------------------------------------------------------------------ 12.5.10

    private record Data(double[][] x, int[] y) {
    }

    private static Data dataset(String name, int n, SplittableRandom r) {
        double[][] x = new double[n][2];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            switch (name) {
                case "moons" -> {
                    y[i] = r.nextInt(2);
                    double t = Math.PI * r.nextDouble();
                    x[i][0] = (y[i] == 0 ? Math.cos(t) : 1 - Math.cos(t)) + 0.2 * gauss(r);
                    x[i][1] = (y[i] == 0 ? Math.sin(t) : 0.5 - Math.sin(t)) + 0.2 * gauss(r);
                }
                case "noisy xor" -> {
                    x[i][0] = 2 * r.nextDouble() - 1;
                    x[i][1] = 2 * r.nextDouble() - 1;
                    y[i] = x[i][0] * x[i][1] > 0 ? 1 : 0;
                    if (r.nextDouble() < 0.1) {
                        y[i] = 1 - y[i];
                    }
                }
                default -> {
                    y[i] = r.nextDouble() < 0.15 ? 1 : 0;
                    double c = y[i] == 1 ? 1.5 : 0;
                    x[i][0] = c + 0.8 * gauss(r);
                    x[i][1] = c + 0.8 * gauss(r);
                }
            }
        }
        return new Data(x, y);
    }

    private static double nll(double[] p, int[] y) {
        double s = 0;
        for (int i = 0; i < p.length; i++) {
            double q = Math.min(Math.max(y[i] == 1 ? p[i] : 1 - p[i], 1e-12), 1);
            s -= Math.log(q);
        }
        return s / p.length;
    }

    /**
     * The expected calibration error with ten equal-width confidence bins.
     *
     * @param p predicted {@code p(y = 1)}
     * @param y the labels
     * @return the ECE
     */
    static double ece(double[] p, int[] y) {
        int bins = 10;
        double[] conf = new double[bins];
        double[] acc = new double[bins];
        int[] count = new int[bins];
        for (int i = 0; i < p.length; i++) {
            double c = Math.max(p[i], 1 - p[i]);
            int b = Math.min(bins - 1, (int) ((c - 0.5) * 2 * bins));
            conf[b] += c;
            acc[b] += (p[i] >= 0.5 ? 1 : 0) == y[i] ? 1 : 0;
            count[b]++;
        }
        double e = 0;
        for (int b = 0; b < bins; b++) {
            if (count[b] > 0) {
                e += Math.abs(conf[b] - acc[b]) / p.length;
            }
        }
        return e;
    }

    private static double marginalKl(double[] mTrue, double[] vTrue, double[] m, double[] v) {
        double s = 0;
        for (int i = 0; i < m.length; i++) {
            s += 0.5 * (Math.log(v[i] / vTrue[i]) + (vTrue[i] + Math.pow(mTrue[i] - m[i], 2)) / v[i] - 1);
        }
        return s / m.length;
    }

    /**
     * EP against Laplace for probit GP classification, with HMC as ground truth.
     *
     * @param trainSize the training points per dataset, for example 60
     * @param hmcSamples HMC samples after warm-up, for example 1500
     * @param seed the seed
     * @return {@code "dataset / method"} to {test NLL, ECE, mean marginal KL(truth || approx),
     *     log Z, sweeps}; the {@code truth} row carries HMC's predictions, the evidence from annealed
     *     importance sampling and, in place of sweeps, AIS's relative ESS
     */
    public static Map<String, double[]> gpClassification(int trainSize, int hmcSamples, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        StationaryKernel kernel = StationaryKernel.rbf(0.6, 9.0);
        for (String name : new String[] {"moons", "noisy xor", "imbalanced blobs"}) {
            SplittableRandom r = new SplittableRandom(seed);
            Data train = dataset(name, trainSize, r);
            Data test = dataset(name, 400, r);
            int n = trainSize;
            double[][] k = kernel.gram(train.x());
            for (int i = 0; i < n; i++) {
                k[i][i] += 1e-6;
            }
            double[][] lk = MlMatrices.cholesky(k);
            double[] ys = new double[n];
            for (int i = 0; i < n; i++) {
                ys[i] = train.y()[i] == 1 ? 1 : -1;
            }
            double logNorm = -0.5 * n * Math.log(2 * Math.PI);
            LogDensity whitened = (u, grad) -> {
                double[] f = lower(lk, u);
                double lp = logNorm;
                double[] g = new double[n];
                for (int i = 0; i < n; i++) {
                    double z = ys[i] * f[i];
                    double lc = NormalTails.logSf(-z);
                    lp += lc - 0.5 * u[i] * u[i];
                    g[i] = ys[i] * Math.exp(NormalTails.logPdf(z) - lc);
                }
                if (grad != null) {
                    for (int j = 0; j < n; j++) {
                        double s = -u[j];
                        for (int i = j; i < n; i++) {
                            s += lk[i][j] * g[i];
                        }
                        grad[j] = s;
                    }
                }
                return lp;
            };
            HamiltonianMonteCarlo.Result chain = HamiltonianMonteCarlo.hmc(whitened, new double[n], 0, 15,
                    hmcSamples / 3, hmcSamples, 0.8, new SplittableRandom(seed + 7));
            double[] mTrue = new double[n];
            double[] vTrue = new double[n];
            double[][] fs = new double[hmcSamples][];
            for (int s = 0; s < hmcSamples; s++) {
                fs[s] = lower(lk, chain.samples()[s]);
                for (int i = 0; i < n; i++) {
                    mTrue[i] += fs[s][i] / hmcSamples;
                    vTrue[i] += fs[s][i] * fs[s][i] / hmcSamples;
                }
            }
            for (int i = 0; i < n; i++) {
                vTrue[i] -= mTrue[i] * mTrue[i];
            }
            // HMC predictive: f* | f ~ N(k*^T K^-1 f, k** - k*^T K^-1 k*)
            int m = test.x().length;
            double[][] a = new double[m][];
            double[] cond = new double[m];
            for (int t = 0; t < m; t++) {
                double[] ks = new double[n];
                for (int i = 0; i < n; i++) {
                    ks[i] = kernel.value(train.x()[i], test.x()[t]);
                }
                a[t] = MlMatrices.choleskySolve(lk, ks);
                cond[t] = Math.max(kernel.value(test.x()[t], test.x()[t]) - MlMatrices.dot(ks, a[t]), 1e-12);
            }
            double[] pTrue = new double[m];
            for (double[] f : fs) {
                for (int t = 0; t < m; t++) {
                    pTrue[t] += NormalTails.cdf(MlMatrices.dot(a[t], f) / Math.sqrt(1 + cond[t])) / hmcSamples;
                }
            }
            ExpectationPropagation seq = ExpectationPropagation.probit(train.x(), train.y(), kernel,
                    ExpectationPropagation.Schedule.SEQUENTIAL, 1.0, 50, 1e-6);
            ExpectationPropagation par = ExpectationPropagation.probit(train.x(), train.y(), kernel,
                    ExpectationPropagation.Schedule.PARALLEL, 0.5, 200, 1e-6);
            ExpectationPropagation.Laplace lap = ExpectationPropagation.laplace(train.x(), train.y(), kernel, 50);
            GaussianProcess.Classifier logistic = GaussianProcess.classifier(train.x(), train.y(), kernel);
            LogDensity prior = (u, grad) -> {
                double s = logNorm;
                for (int i = 0; i < n; i++) {
                    s -= 0.5 * u[i] * u[i];
                    if (grad != null) {
                        grad[i] = -u[i];
                    }
                }
                return s;
            };
            AnnealedImportanceSampler.Estimate ais = AnnealedImportanceSampler.anneal(prior, rr -> {
                double[] u = new double[n];
                for (int i = 0; i < n; i++) {
                    u[i] = gauss(rr);
                }
                return u;
            }, whitened, 32, 300, 10, 0.1, false, new SplittableRandom(seed + 11));
            out.put(name + " / truth (HMC, AIS evidence)",
                    new double[] {nll(pTrue, test.y()), ece(pTrue, test.y()), 0, ais.logZ(), ais.ess() / 32});
            for (ExpectationPropagation ep : new ExpectationPropagation[] {seq, par}) {
                double[] p = ep.predictProbability(test.x());
                out.put(name + " / EP " + (ep == seq ? "sequential" : "parallel (damping 0.5)"),
                        new double[] {nll(p, test.y()), ece(p, test.y()),
                            marginalKl(mTrue, vTrue, ep.posteriorMean(), ep.posteriorVariance()),
                            ep.logMarginalLikelihood(), ep.sweeps()});
            }
            double[] pl = lap.predictProbability(test.x());
            out.put(name + " / Laplace probit", new double[] {nll(pl, test.y()), ece(pl, test.y()),
                marginalKl(mTrue, vTrue, lap.posteriorMean(), lap.posteriorVariance()), lap.logMarginalLikelihood(),
                0});
            double[] pg = logistic.predictProbability(test.x());
            out.put(name + " / Laplace logistic (wave 6)", new double[] {nll(pg, test.y()), ece(pg, test.y()),
                Double.NaN, logistic.logMarginalLikelihood(), 0});
        }
        return out;
    }

    private static double[] lower(double[][] l, double[] u) {
        double[] f = new double[u.length];
        for (int i = 0; i < u.length; i++) {
            double s = 0;
            for (int j = 0; j <= i; j++) {
                s += l[i][j] * u[j];
            }
            f[i] = s;
        }
        return f;
    }

    // ------------------------------------------------------------------ 12.5.11

    private record Problem(String name, int dimension, double low, double high, double logZ,
            LogDensity logLikelihood) {
    }

    private static Problem mixture() {
        int d = 5;
        double sigma = 1;
        return new Problem("gaussian mixture d=5", d, -10, 10, -d * Math.log(20), (x, grad) -> {
            double[] lc = new double[2];
            for (int c = 0; c < 2; c++) {
                double mean = c == 0 ? -4 : 4;
                double s = 0;
                for (double v : x) {
                    s += (v - mean) * (v - mean);
                }
                lc[c] = Math.log(0.5) - s / (2 * sigma * sigma) - d / 2.0 * Math.log(2 * Math.PI * sigma * sigma);
            }
            double mx = Math.max(lc[0], lc[1]);
            double lse = mx + Math.log(Math.exp(lc[0] - mx) + Math.exp(lc[1] - mx));
            if (grad != null) {
                double w0 = Math.exp(lc[0] - lse);
                for (int j = 0; j < d; j++) {
                    grad[j] = -(w0 * (x[j] + 4) + (1 - w0) * (x[j] - 4)) / (sigma * sigma);
                }
            }
            return lse;
        });
    }

    private static Problem shells() {
        double radius = 2;
        double width = 0.1;
        double[][] centres = {{-3.5, 0}, {3.5, 0}};
        // each shell integrates to about 2 pi r over the prior box of area 144
        return new Problem("gaussian shells d=2", 2, -6, 6, Math.log(2 * 2 * Math.PI * radius / 144), (x, grad) -> {
            double[] lc = new double[2];
            double[][] g = new double[2][2];
            for (int c = 0; c < 2; c++) {
                double dx = x[0] - centres[c][0];
                double dy = x[1] - centres[c][1];
                double rho = Math.sqrt(dx * dx + dy * dy) + 1e-12;
                double e = rho - radius;
                lc[c] = -e * e / (2 * width * width) - 0.5 * Math.log(2 * Math.PI * width * width);
                g[c][0] = -e / (width * width) * dx / rho;
                g[c][1] = -e / (width * width) * dy / rho;
            }
            double mx = Math.max(lc[0], lc[1]);
            double lse = mx + Math.log(Math.exp(lc[0] - mx) + Math.exp(lc[1] - mx));
            if (grad != null) {
                double w0 = Math.exp(lc[0] - lse);
                grad[0] = w0 * g[0][0] + (1 - w0) * g[1][0];
                grad[1] = w0 * g[0][1] + (1 - w0) * g[1][1];
            }
            return lse;
        });
    }

    /**
     * Nested sampling against AIS, SMC and the harmonic mean on problems with known evidence.
     *
     * @param livePoints nested-sampling live points, for example 300
     * @param seed the seed
     * @return {@code "problem / method"} to {log Z estimate, reported error (NaN if none),
     *     |error|, likelihood or gradient evaluations in thousands}
     */
    public static Map<String, double[]> evidence(int livePoints, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Problem pr : new Problem[] {mixture(), shells()}) {
            int d = pr.dimension();
            double width = pr.high() - pr.low();
            UnaryOperator<double[]> transform = u -> {
                double[] t = new double[d];
                for (int j = 0; j < d; j++) {
                    t[j] = pr.low() + width * u[j];
                }
                return t;
            };
            ToDoubleFunction<double[]> logL = pr.logLikelihood()::value;
            NestedSampling.Result reference = null;
            for (NestedSampling.Replacement rep : NestedSampling.Replacement.values()) {
                int walk = rep == NestedSampling.Replacement.SLICE ? 3 : 25;
                NestedSampling.Result res = NestedSampling.of(rep, livePoints, walk, 0.01, 200_000).run(d, transform,
                        logL, new SplittableRandom(seed));
                out.put(pr.name() + " / NS " + rep, new double[] {res.logZ(), res.error(),
                    Math.abs(res.logZ() - pr.logZ()), res.likelihoodEvaluations() / 1000.0});
                if (rep == NestedSampling.Replacement.SLICE) {
                    reference = res;
                }
            }
            double logPrior = -d * Math.log(width);
            LogDensity prior = (x, grad) -> {
                if (grad != null) {
                    java.util.Arrays.fill(grad, 0);
                }
                for (double v : x) {
                    if (v < pr.low() || v > pr.high()) {
                        return Double.NEGATIVE_INFINITY;
                    }
                }
                return logPrior;
            };
            LogDensity target = (x, grad) -> {
                double lp = prior.evaluate(x, null);
                if (lp == Double.NEGATIVE_INFINITY) {
                    if (grad != null) {
                        java.util.Arrays.fill(grad, 0);
                    }
                    return lp;
                }
                return lp + pr.logLikelihood().evaluate(x, grad);
            };
            java.util.function.Function<SplittableRandom, double[]> sampler = r -> {
                double[] x = new double[d];
                for (int j = 0; j < d; j++) {
                    x[j] = pr.low() + width * r.nextDouble();
                }
                return x;
            };
            for (boolean resample : new boolean[] {false, true}) {
                AnnealedImportanceSampler.Estimate e = AnnealedImportanceSampler.anneal(prior, sampler, target, 100,
                        200, 10, 0.1, resample, new SplittableRandom(seed));
                out.put(pr.name() + " / " + (resample ? "SMC sampler" : "AIS"), new double[] {e.logZ(), Double.NaN,
                    Math.abs(e.logZ() - pr.logZ()), e.gradientEvaluations() / 1000.0});
            }
            double[][] post = reference.resample(2000, new SplittableRandom(seed + 1));
            double[] neg = new double[post.length];
            for (int i = 0; i < post.length; i++) {
                neg[i] = -logL.applyAsDouble(post[i]);
            }
            double hm = -McmcDiagnostics.logMeanExp(neg);
            out.put(pr.name() + " / harmonic mean", new double[] {hm, Double.NaN, Math.abs(hm - pr.logZ()),
                post.length / 1000.0});
            out.put(pr.name() + " / analytic", new double[] {pr.logZ(), 0, 0, 0});
        }
        return out;
    }

    // ------------------------------------------------------------------ 12.5.12

    /**
     * Mutual-information estimators on correlated Gaussians ({@code d = 5} coordinate pairs).
     *
     * @param samples the sample size for the neural bounds, for example 4000 (KSG and binning use
     *     the first 800)
     * @param steps critic training steps
     * @param seed the seed
     * @return {@code "setting / estimator"} to the estimate in nats; {@code "setting / true"} holds
     *     the truth
     */
    public static Map<String, Double> mutualInformation(int samples, int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        int d = 5;
        String[] settings = {"gaussian I=2", "gaussian I=6", "cubic I=2"};
        double[] mis = {2, 6, 2};
        for (int s = 0; s < settings.length; s++) {
            SplittableRandom r = new SplittableRandom(seed + s);
            double rho = MutualInformation.correlationFor(mis[s], d);
            double[][] x = new double[samples][d];
            double[][] y = new double[samples][d];
            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < d; j++) {
                    x[i][j] = gauss(r);
                    y[i][j] = rho * x[i][j] + Math.sqrt(1 - rho * rho) * gauss(r);
                    if (s == 2) {
                        y[i][j] = y[i][j] * y[i][j] * y[i][j];
                    }
                }
            }
            String key = settings[s] + " / ";
            out.put(key + "true", MutualInformation.gaussian(rho, d));
            int small = Math.min(800, samples);
            double[][] xs = java.util.Arrays.copyOf(x, small);
            double[][] ysm = java.util.Arrays.copyOf(y, small);
            out.put(key + "binned (4 per dim)", MutualInformation.binned(xs, ysm, 4));
            out.put(key + "KSG k=3", MutualInformation.ksg(xs, ysm, 3));
            for (MutualInformation.Bound b : MutualInformation.Bound.values()) {
                out.put(key + b, MutualInformation.variational(b, x, y, 64, 16, 64, steps, 5e-4, 5, seed));
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ 12.5.13

    private static double[][] sample(String setting, int n, int d, SplittableRandom r) {
        double[][] x = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                x[i][j] = gauss(r);
            }
            switch (setting) {
                case "laplace" -> {
                    for (int j = 0; j < d; j++) {
                        double u = r.nextDouble() - 0.5;
                        x[i][j] = -Math.signum(u) * Math.log(1 - 2 * Math.abs(u)) / Math.sqrt(2);
                    }
                }
                case "variance 2" -> x[i][0] *= Math.sqrt(2);
                case "mean shift 0.5" -> x[i][0] += 0.5;
                default -> {
                }
            }
        }
        return x;
    }

    /**
     * Rejection rates at level 0.05 of goodness-of-fit and two-sample tests against
     * {@code N(0, I_5)}: KSD and FSSD use only the model's score; MMD, HSIC (sample-label
     * dependence) and C2ST compare with an equally large model sample.
     *
     * @param n the sample size
     * @param trials repetitions per setting
     * @param seed the seed
     * @return {@code "test / setting"} to the rejection rate (type-I error under {@code null})
     */
    public static Map<String, Double> goodnessOfFit(int n, int trials, long seed) {
        int d = 5;
        UnaryOperator<double[]> score = x -> {
            double[] s = new double[x.length];
            for (int j = 0; j < x.length; j++) {
                s[j] = -x[j];
            }
            return s;
        };
        String[] tests = {"KSD (IMQ)", "FSSD (opt, J=5)", "MMD (median)", "HSIC (sample-label)",
            "C2ST logistic", "C2ST kNN"};
        String[] settings = {"null", "laplace", "variance 2", "mean shift 0.5"};
        Map<String, Double> out = new LinkedHashMap<>();
        for (String setting : settings) {
            int[] rejections = new int[tests.length];
            for (int t = 0; t < trials; t++) {
                SplittableRandom r = new SplittableRandom(seed * 1000 + t);
                double[][] x = sample(setting, n, d, r);
                double[][] p = sample("null", n, d, r);
                long sd = seed + t;
                double[][] pooled = new double[2 * n][];
                double[][] lab = new double[2 * n][1];
                for (int i = 0; i < n; i++) {
                    pooled[i] = x[i];
                    pooled[n + i] = p[i];
                    lab[n + i][0] = 1;
                }
                KernelTests.Result[] res = {
                    SteinTests.ksd(x, score, 1, -0.5, 200, sd),
                    SteinTests.fssd(x, score, 5, 0.2, 15, 1000, sd),
                    KernelTests.mmdTest(x, p, 0, 100, sd),
                    KernelTests.hsicTest(pooled, lab, 100, sd),
                    ClassifierTwoSample.test(x, p, ClassifierTwoSample.Classifier.LOGISTIC_QUADRATIC, 0.5, sd),
                    ClassifierTwoSample.test(x, p, ClassifierTwoSample.Classifier.KNN, 0.5, sd)};
                for (int k = 0; k < tests.length; k++) {
                    if (res[k].rejects(0.05)) {
                        rejections[k]++;
                    }
                }
            }
            for (int k = 0; k < tests.length; k++) {
                out.put(tests[k] + " / " + setting, (double) rejections[k] / trials);
            }
        }
        return out;
    }

    private static double gauss(SplittableRandom r) {
        double u = r.nextDouble();
        double v = r.nextDouble();
        return Math.sqrt(-2 * Math.log(1 - u)) * Math.cos(2 * Math.PI * v);
    }
}
