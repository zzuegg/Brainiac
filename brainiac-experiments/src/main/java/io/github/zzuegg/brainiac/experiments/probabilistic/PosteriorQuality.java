package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.experimental.ml.AnnealedImportanceSampler;
import io.github.zzuegg.brainiac.experimental.ml.FlowVariationalInference;
import io.github.zzuegg.brainiac.experimental.ml.FlowVariationalInference.Family;
import io.github.zzuegg.brainiac.experimental.ml.LogDensity;
import io.github.zzuegg.brainiac.experimental.ml.McmcDiagnostics;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.train.HamiltonianMonteCarlo;
import io.github.zzuegg.brainiac.experimental.train.SteinVariationalGradientDescent;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * How well samplers and variational families recover known posteriors.
 *
 * <ul>
 *   <li>{@link #samplers}: random-walk Metropolis, HMC and NUTS on the banana and on Neal's
 *       funnel, four chains each: ESS per 1000 gradient (or density) evaluations, split
 *       R-hat, and the sliced Wasserstein distance to exact samples.</li>
 *   <li>{@link #steinMixture}: SVGD against an ensemble of independent gradient-ascent
 *       particles of the same size and a single NUTS chain on a four-mode 2-D mixture: MMD
 *       to exact samples and the number of modes covered.</li>
 *   <li>{@link #variational}: mean-field and full-rank ADVI, planar, radial and IAF flows on
 *       the banana, the funnel and the two-moons energy: ELBO, {@code KL(q || p) = -ELBO}
 *       for the normalised targets, and wall time.</li>
 *   <li>{@link #evidence}: AIS and an SMC sampler against plain importance sampling from the
 *       prior for a 5-D Gaussian mixture with known {@code log Z}, at equal target
 *       evaluations: bias and spread over repeats.</li>
 *   <li>{@link #linearGaussianVae}: the test log-likelihood of a linear-Gaussian decoder (a
 *       VAE whose {@code log p(x)} is exact) estimated by the ELBO, IWAE-k and AIS.</li>
 * </ul>
 */
public final class PosteriorQuality {

    private PosteriorQuality() {
    }

    private static LogDensity target(String name) {
        return switch (name) {
            case "banana" -> LogDensity.banana(2, 0.25);
            case "funnel" -> LogDensity.funnel(5);
            case "funnel2" -> LogDensity.funnel(2);
            case "two moons" -> twoMoons();
            default -> throw new IllegalArgumentException(name);
        };
    }

    private static double[][] exact(String name, int n, SplittableRandom r) {
        return switch (name) {
            case "banana" -> LogDensity.bananaSamples(2, 0.25, n, r);
            case "funnel" -> LogDensity.funnelSamples(5, n, r);
            default -> throw new IllegalArgumentException(name);
        };
    }

    /** The two-moons energy U1 of Rezende and Mohamed 2015, figure 3 (unnormalised). */
    private static LogDensity twoMoons() {
        return LogDensity.numerical(z -> {
            double r = Math.hypot(z[0], z[1]);
            double a = Math.exp(-0.5 * Math.pow((z[0] - 2) / 0.6, 2));
            double b = Math.exp(-0.5 * Math.pow((z[0] + 2) / 0.6, 2));
            return -0.5 * Math.pow((r - 2) / 0.4, 2) + Math.log(a + b + 1e-300);
        }, 1e-5);
    }

    /**
     * Sampler diagnostics on one target.
     *
     * @param name {@code "banana"} or {@code "funnel"} (5-D)
     * @param seed the seed
     * @return method to {@code {ESS per 1000 evaluations, max split R-hat, sliced W to exact}}
     */
    public static Map<String, double[]> samplers(String name, long seed) {
        LogDensity p = target(name);
        int d = name.equals("banana") ? 2 : 5;
        SplittableRandom r = new SplittableRandom(seed);
        double[][] reference = exact(name, 1000, new SplittableRandom(seed + 1));
        Map<String, double[]> out = new LinkedHashMap<>();
        for (String method : new String[] {"random-walk MH", "HMC (L=10)", "NUTS"}) {
            double[][][] chains = new double[4][][];
            long evals = 0;
            double ess = 0;
            for (int c = 0; c < 4; c++) {
                double[] start = new double[d];
                for (int i = 0; i < d; i++) {
                    start[i] = r.nextGaussian();
                }
                HamiltonianMonteCarlo.Result res = switch (method) {
                    case "random-walk MH" -> HamiltonianMonteCarlo.randomWalkMetropolis(p, start, 0.5, 2000, 10000, r);
                    case "HMC (L=10)" -> HamiltonianMonteCarlo.hmc(p, start, 0, 10, 300, 1000, 0.8, r);
                    default -> HamiltonianMonteCarlo.nuts(p, start, 300, 1000, 0.8, 8, r);
                };
                chains[c] = res.samples();
                evals += res.gradientEvaluations();
                ess += McmcDiagnostics.minEffectiveSampleSize(res.samples());
            }
            double[][] pooled = thin(chains, 1000);
            out.put(method, new double[] {1000 * ess / evals, McmcDiagnostics.maxSplitRHat(chains),
                McmcDiagnostics.slicedWasserstein(pooled, reference, 50, seed)});
        }
        return out;
    }

    private static double[][] thin(double[][][] chains, int n) {
        double[][] out = new double[n][];
        int per = n / chains.length;
        for (int c = 0; c < chains.length; c++) {
            int len = chains[c].length;
            for (int i = 0; i < per; i++) {
                out[c * per + i] = chains[c][(int) ((long) i * len / per)];
            }
        }
        return out;
    }

    /**
     * Mode coverage on a mixture of four well-separated Gaussians at {@code (+-3, +-3)}.
     *
     * @param seed the seed
     * @return method to {@code {MMD^2 to exact samples, modes covered}}
     */
    public static Map<String, double[]> steinMixture(long seed) {
        double[][] means = {{3, 3}, {-3, 3}, {3, -3}, {-3, -3}};
        double std = 0.7;
        LogDensity p = LogDensity.mixture(means, new double[] {0.25, 0.25, 0.25, 0.25}, std);
        SplittableRandom r = new SplittableRandom(seed);
        int n = 64;
        double[][] exact = new double[400][2];
        for (int i = 0; i < exact.length; i++) {
            double[] m = means[i % 4];
            exact[i][0] = m[0] + std * r.nextGaussian();
            exact[i][1] = m[1] + std * r.nextGaussian();
        }
        double[][] init = new double[n][2];
        for (int i = 0; i < n; i++) {
            init[i][0] = 3 * r.nextGaussian();
            init[i][1] = 3 * r.nextGaussian();
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        double[][] svgd = SteinVariationalGradientDescent.create(p, init, 0.3).run(500);
        out.put("SVGD", new double[] {McmcDiagnostics.mmd2(svgd, exact, 1), covered(svgd, means)});
        double[][] ens = new double[n][];
        for (int i = 0; i < n; i++) {
            double[] x = init[i].clone();
            double[] g = new double[2];
            for (int s = 0; s < 500; s++) {
                p.evaluate(x, g);
                x[0] += 0.1 * g[0];
                x[1] += 0.1 * g[1];
            }
            ens[i] = x;
        }
        out.put("gradient-ascent ensemble", new double[] {McmcDiagnostics.mmd2(ens, exact, 1), covered(ens, means)});
        double[][] chain = HamiltonianMonteCarlo.nuts(p, init[0], 200, 640, 0.8, 8, r).samples();
        double[][] thinned = new double[n][];
        for (int i = 0; i < n; i++) {
            thinned[i] = chain[i * 10];
        }
        out.put("NUTS (one chain)", new double[] {McmcDiagnostics.mmd2(thinned, exact, 1), covered(thinned, means)});
        return out;
    }

    private static double covered(double[][] x, double[][] means) {
        boolean[] hit = new boolean[means.length];
        for (double[] p : x) {
            int best = 0;
            for (int j = 1; j < means.length; j++) {
                if (MlMatrices.squaredDistance(p, means[j]) < MlMatrices.squaredDistance(p, means[best])) {
                    best = j;
                }
            }
            hit[best] = true;
        }
        int c = 0;
        for (boolean h : hit) {
            c += h ? 1 : 0;
        }
        return c;
    }

    /**
     * Fits every variational family to one 2-D target.
     *
     * @param name {@code "banana"}, {@code "funnel2"} (2-D funnel) or {@code "two moons"}
     * @param iterations Adam steps per family
     * @param seed the seed
     * @return family to {@code {ELBO, KL(q||p) (NaN when unnormalised), seconds}}
     */
    public static Map<String, double[]> variational(String name, int iterations, long seed) {
        LogDensity p = target(name);
        boolean normalised = !name.equals("two moons");
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Family f : Family.values()) {
            long t = System.nanoTime();
            FlowVariationalInference q = FlowVariationalInference.create(f, 2, f == Family.IAF ? 2 : 8, seed);
            q.fit(p, iterations, 32, 0.02f, seed);
            double elbo = q.elbo(p, 4000, seed + 1);
            out.put(f.name(), new double[] {elbo, normalised ? -elbo : Double.NaN, (System.nanoTime() - t) / 1e9});
        }
        return out;
    }

    /**
     * Evidence estimation for {@code f(x) = Z * mixture(x)} in 5-D, {@code log Z = 3}, from the
     * prior {@code N(0, 2^2 I)}, all methods at about 30k target evaluations, five repeats.
     *
     * @param seed the seed
     * @return method to {@code {mean error of log Z, std over repeats, evaluations}}
     */
    public static Map<String, double[]> evidence(long seed) {
        int d = 5;
        double[][] means = new double[2][d];
        means[0][0] = 1.5;
        means[1][0] = -1.5;
        means[1][1] = 1;
        LogDensity mix = LogDensity.mixture(means, new double[] {0.3, 0.7}, 0.3);
        LogDensity f = (x, g) -> mix.evaluate(x, g) + 3;
        LogDensity prior = LogDensity.gaussian(d, 0, 2);
        java.util.function.Function<SplittableRandom, double[]> draw = rr -> {
            double[] x = new double[d];
            for (int i = 0; i < d; i++) {
                x[i] = 2 * rr.nextGaussian();
            }
            return x;
        };
        Map<String, double[]> out = new LinkedHashMap<>();
        for (String method : new String[] {"importance (prior)", "AIS", "SMC sampler"}) {
            double[] est = new double[5];
            long evals = 0;
            for (int rep = 0; rep < 5; rep++) {
                SplittableRandom r = new SplittableRandom(seed * 31 + rep);
                AnnealedImportanceSampler.Estimate e = switch (method) {
                    case "importance (prior)" -> AnnealedImportanceSampler.importance(prior, draw, f, 30000, r);
                    case "AIS" -> AnnealedImportanceSampler.anneal(prior, draw, f, 100, 60, 4, 0.2, false, r);
                    default -> AnnealedImportanceSampler.anneal(prior, draw, f, 100, 60, 4, 0.2, true, r);
                };
                est[rep] = e.logZ();
                evals = e.gradientEvaluations() == 0 ? 30000 : e.gradientEvaluations();
            }
            double mean = 0;
            for (double v : est) {
                mean += v / est.length;
            }
            double var = 0;
            for (double v : est) {
                var += (v - mean) * (v - mean) / (est.length - 1);
            }
            out.put(method, new double[] {Math.abs(mean - 3), Math.sqrt(var), evals});
        }
        return out;
    }

    /**
     * Log-likelihood evaluation of a linear-Gaussian decoder {@code x = W z + N(0, 0.3^2)},
     * {@code z ~ N(0, I_8)}, {@code x in R^20}, with the mean-field Gaussian encoder that is
     * optimal for the ELBO; averaged over 5 test points. AIS runs 50 particles through 100
     * stages (about 30k gradients), annealing either from the prior or from the encoder.
     *
     * @param seed the seed
     * @return estimator to {@code {mean log p(x) estimate, gap to the exact value}}
     */
    public static Map<String, double[]> linearGaussianVae(long seed) {
        int dz = 8;
        int dx = 20;
        double s = 0.3;
        SplittableRandom r = new SplittableRandom(seed);
        double[][] w = new double[dx][dz];
        for (double[] row : w) {
            for (int j = 0; j < dz; j++) {
                row[j] = r.nextGaussian() / Math.sqrt(dz);
            }
        }
        // posterior precision I + W^T W / s^2 (the same for every x)
        double[][] prec = MlMatrices.identity(dz);
        for (int a = 0; a < dz; a++) {
            for (int b = 0; b < dz; b++) {
                for (double[] row : w) {
                    prec[a][b] += row[a] * row[b] / (s * s);
                }
            }
        }
        double[][] cov = MlMatrices.choleskyInverse(MlMatrices.cholesky(prec));
        // marginal covariance W W^T + s^2 I
        double[][] marg = MlMatrices.multiply(w, MlMatrices.transpose(w));
        for (int i = 0; i < dx; i++) {
            marg[i][i] += s * s;
        }
        double[][] margChol = MlMatrices.cholesky(marg);
        String[] names = {"ELBO (k=1)", "IWAE k=50", "IWAE k=5000", "AIS from prior", "AIS from encoder"};
        double[] sums = new double[names.length];
        double exactSum = 0;
        int points = 5;
        for (int pt = 0; pt < points; pt++) {
            double[] z = new double[dz];
            for (int j = 0; j < dz; j++) {
                z[j] = r.nextGaussian();
            }
            double[] x = new double[dx];
            for (int i = 0; i < dx; i++) {
                x[i] = MlMatrices.dot(w[i], z) + s * r.nextGaussian();
            }
            double[] y = MlMatrices.forwardSolve(margChol, x);
            double exact = -0.5 * MlMatrices.dot(y, y) - 0.5 * MlMatrices.logDet(margChol) - 0.5 * dx * Math.log(2 * Math.PI);
            exactSum += exact;
            // posterior mean = cov W^T x / s^2
            double[] wx = new double[dz];
            for (int j = 0; j < dz; j++) {
                for (int i = 0; i < dx; i++) {
                    wx[j] += w[i][j] * x[i] / (s * s);
                }
            }
            double[] mu = new double[dz];
            double[] sd = new double[dz];
            for (int j = 0; j < dz; j++) {
                mu[j] = MlMatrices.dot(cov[j], wx);
                sd[j] = Math.sqrt(1 / prec[j][j]);
            }
            LogDensity joint = (zz, g) -> {
                double lp = 0;
                if (g != null) {
                    java.util.Arrays.fill(g, 0);
                }
                for (int i = 0; i < dx; i++) {
                    double res = x[i] - MlMatrices.dot(w[i], zz);
                    lp += -0.5 * res * res / (s * s);
                    if (g != null) {
                        for (int j = 0; j < dz; j++) {
                            g[j] += res * w[i][j] / (s * s);
                        }
                    }
                }
                for (int j = 0; j < dz; j++) {
                    lp += -0.5 * zz[j] * zz[j];
                    if (g != null) {
                        g[j] -= zz[j];
                    }
                }
                return lp - 0.5 * (dx + dz) * Math.log(2 * Math.PI) - dx * Math.log(s);
            };
            int[] ks = {1, 50, 5000};
            for (int m = 0; m < 3; m++) {
                int k = ks[m];
                int reps = m == 0 ? 200 : 1;
                double acc = 0;
                for (int rep = 0; rep < reps; rep++) {
                    double[] lw = new double[k];
                    for (int i = 0; i < k; i++) {
                        double[] zz = new double[dz];
                        double lq = 0;
                        for (int j = 0; j < dz; j++) {
                            double e = r.nextGaussian();
                            zz[j] = mu[j] + sd[j] * e;
                            lq += -0.5 * e * e - Math.log(sd[j]) - 0.5 * Math.log(2 * Math.PI);
                        }
                        lw[i] = joint.value(zz) - lq;
                    }
                    acc += McmcDiagnostics.logMeanExp(lw) / reps;
                }
                sums[m] += acc;
            }
            LogDensity prior = LogDensity.gaussian(dz, 0, 1);
            sums[3] += AnnealedImportanceSampler.anneal(prior, rr -> {
                double[] zz = new double[dz];
                for (int j = 0; j < dz; j++) {
                    zz[j] = rr.nextGaussian();
                }
                return zz;
            }, joint, 50, 100, 5, 0.05, false, r).logZ();
            LogDensity encoder = (zz, g) -> {
                double lq = 0;
                for (int j = 0; j < dz; j++) {
                    double e = (zz[j] - mu[j]) / sd[j];
                    lq += -0.5 * e * e - Math.log(sd[j]) - 0.5 * Math.log(2 * Math.PI);
                    if (g != null) {
                        g[j] = -e / sd[j];
                    }
                }
                return lq;
            };
            sums[4] += AnnealedImportanceSampler.anneal(encoder, rr -> {
                double[] zz = new double[dz];
                for (int j = 0; j < dz; j++) {
                    zz[j] = mu[j] + sd[j] * rr.nextGaussian();
                }
                return zz;
            }, joint, 50, 100, 5, 0.05, false, r).logZ();
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("exact", new double[] {exactSum / points, 0});
        for (int m = 0; m < names.length; m++) {
            out.put(names[m], new double[] {sums[m] / points, (exactSum - sums[m]) / points});
        }
        return out;
    }
}
