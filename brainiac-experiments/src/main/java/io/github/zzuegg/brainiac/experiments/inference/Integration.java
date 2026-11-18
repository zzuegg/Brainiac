package io.github.zzuegg.brainiac.experiments.inference;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.estimators.NestedSampling;
import io.github.zzuegg.brainiac.experimental.ml.FlowVariationalInference;
import io.github.zzuegg.brainiac.experimental.ml.LogDensity;
import io.github.zzuegg.brainiac.experimental.quadrature.BayesianQuadrature;
import io.github.zzuegg.brainiac.experimental.quadrature.Cubature;
import io.github.zzuegg.brainiac.experimental.quadrature.KernelHerding;
import io.github.zzuegg.brainiac.experimental.quadrature.LowDiscrepancy;
import io.github.zzuegg.brainiac.experimental.quadrature.RqmcEstimator;
import io.github.zzuegg.brainiac.experimental.quadrature.RqmcEstimator.Method;
import io.github.zzuegg.brainiac.tensor.Matrices;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Special;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Numerical integration: randomised QMC against Monte Carlo, antithetic and Latin-hypercube
 * sampling on Genz integrands, and RQMC for reparameterised ELBO gradients (13.5.10); Bayesian
 * quadrature, kernel herding and Gauss-Hermite/Smolyak cubature on Gaussian-measure integrals
 * and model evidences (13.5.11).
 */
public final class Integration {

    private Integration() {
    }

    /** A test integrand on the unit cube with its exact integral. */
    private record Genz(String name, ToDoubleFunction<double[]> f, double exact) {
    }

    private static double erf(double x) {
        return 1 - Special.erfc(x);
    }

    private static Genz genz(String kind, int d) {
        double[] a = new double[d];
        double[] w = new double[d];
        for (int j = 0; j < d; j++) {
            a[j] = 4.0 / ((j + 1.0) * (j + 1.0));
            w[j] = 0.3 + 0.4 * ((j * 0.618) % 1);
        }
        return switch (kind) {
            case "gaussian" -> {
                double exact = 1;
                for (int j = 0; j < d; j++) {
                    exact *= Math.sqrt(Math.PI) / (2 * a[j]) * (erf(a[j] * (1 - w[j])) + erf(a[j] * w[j]));
                }
                yield new Genz(kind, u -> {
                    double s = 0;
                    for (int j = 0; j < d; j++) {
                        s += a[j] * a[j] * (u[j] - w[j]) * (u[j] - w[j]);
                    }
                    return Math.exp(-s);
                }, exact);
            }
            case "oscillatory" -> {
                // Re(exp(i 2 pi w0) prod_j (exp(i a_j) - 1) / (i a_j))
                double re = Math.cos(2 * Math.PI * w[0]);
                double im = Math.sin(2 * Math.PI * w[0]);
                for (int j = 0; j < d; j++) {
                    double fr = Math.sin(a[j]) / a[j];
                    double fi = (1 - Math.cos(a[j])) / a[j];
                    double nr = re * fr - im * fi;
                    im = re * fi + im * fr;
                    re = nr;
                }
                yield new Genz(kind, u -> {
                    double s = 2 * Math.PI * w[0];
                    for (int j = 0; j < d; j++) {
                        s += a[j] * u[j];
                    }
                    return Math.cos(s);
                }, re);
            }
            case "discontinuous" -> {
                double exact = 1;
                for (int j = 0; j < d; j++) {
                    double upper = j < 2 ? w[j] : 1;
                    exact *= (Math.exp(a[j] * upper) - 1) / a[j];
                }
                yield new Genz(kind, u -> {
                    if (u[0] > w[0] || (d > 1 && u[1] > w[1])) {
                        return 0;
                    }
                    double s = 0;
                    for (int j = 0; j < d; j++) {
                        s += a[j] * u[j];
                    }
                    return Math.exp(s);
                }, exact);
            }
            default -> new Genz("additive", u -> {
                double s = 0;
                for (double v : u) {
                    s += v * v;
                }
                return s;
            }, d / 3.0);
        };
    }

    /**
     * The QMC table: for each integrand, dimension and point set, the log-log slope of RMSE
     * against {@code n = 2^7 .. 2^maxLog2} and the relative RMSE at the largest {@code n}.
     *
     * @param dims the dimensions
     * @param maxLog2 the largest {@code log2 n}
     * @param replicates replicates per {@code n}
     * @param seed the seed
     * @return rows "integrand d=.. METHOD" with {@code {slope, relative RMSE}}
     */
    public static Map<String, double[]> qmc(int[] dims, int maxLog2, int replicates, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        Method[] methods = {Method.MC, Method.ANTITHETIC, Method.LATIN_HYPERCUBE, Method.HALTON, Method.SOBOL_OWEN,
            Method.SOBOL_LINEAR, Method.LATTICE};
        for (String kind : new String[] {"gaussian", "oscillatory", "discontinuous", "additive"}) {
            for (int d : dims) {
                Genz g = genz(kind, d);
                for (Method m : methods) {
                    int points = maxLog2 - 6;
                    double[] lx = new double[points];
                    double[] ly = new double[points];
                    for (int p = 0; p < points; p++) {
                        int n = 1 << (7 + p);
                        RqmcEstimator.Estimate e = RqmcEstimator.estimate(g.f(), d, m, n, replicates, seed + p);
                        double mse = 0;
                        for (double v : e.replicates()) {
                            mse += (v - g.exact()) * (v - g.exact()) / replicates;
                        }
                        lx[p] = Math.log(n);
                        ly[p] = Math.log(Math.sqrt(mse) + 1e-300);
                    }
                    out.put(kind + " d=" + d + " " + m.name(), new double[] {slope(lx, ly),
                        Math.exp(ly[points - 1]) / Math.abs(g.exact())});
                }
            }
        }
        return out;
    }

    private static double slope(double[] x, double[] y) {
        double mx = 0;
        double my = 0;
        for (int i = 0; i < x.length; i++) {
            mx += x[i] / x.length;
            my += y[i] / x.length;
        }
        double sxy = 0;
        double sxx = 0;
        for (int i = 0; i < x.length; i++) {
            sxy += (x[i] - mx) * (y[i] - my);
            sxx += (x[i] - mx) * (x[i] - mx);
        }
        return sxy / sxx;
    }

    /**
     * RQMC for variational inference: the total variance of the reparameterised ELBO gradient
     * of a mean-field Gaussian on the banana at its initialisation, and the ELBO after a short
     * fit, with i.i.d. against scrambled-Sobol Gaussian noise.
     *
     * @param batch samples per gradient
     * @param replicates gradient replicates
     * @param steps fitting steps
     * @param seed the seed
     * @return rows {@code {gradient variance, final ELBO, ELBO after 50 steps, mean training ELBO
     *     of the last 20 steps}}
     */
    public static Map<String, double[]> variational(int batch, int replicates, int steps, long seed) {
        LogDensity banana = LogDensity.banana(1, 0.5);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int qmc = 0; qmc < 2; qmc++) {
            FlowVariationalInference q = FlowVariationalInference.create(FlowVariationalInference.Family.MEAN_FIELD, 2,
                    0, seed);
            List<Var> params = q.parameters();
            // move away from the optimum so gradients are informative
            params.get(0).assign(Tensor.of(params.get(0).shape(), 3f, -3f));
            SplittableRandom r = new SplittableRandom(seed + 1);
            double[] sum = null;
            double[] sum2 = null;
            for (int rep = 0; rep < replicates; rep++) {
                Var[] s = q.sample(noise(qmc == 1, batch, r));
                Var elbo = FlowVariationalInference.logDensity(banana, s[0]).sub(s[1]).mean();
                elbo.backward();
                double[] g = flatGrad(params);
                for (Var p : params) {
                    p.zeroGrad();
                }
                if (sum == null) {
                    sum = new double[g.length];
                    sum2 = new double[g.length];
                }
                for (int i = 0; i < g.length; i++) {
                    sum[i] += g[i];
                    sum2[i] += g[i] * g[i];
                }
            }
            double var = 0;
            for (int i = 0; i < sum.length; i++) {
                double m = sum[i] / replicates;
                var += sum2[i] / replicates - m * m;
            }
            FlowVariationalInference fit = FlowVariationalInference.create(FlowVariationalInference.Family.MEAN_FIELD,
                    2, 0, seed);
            List<Var> fp = fit.parameters();
            fp.get(0).assign(Tensor.of(fp.get(0).shape(), 3f, -3f));
            Optimizer opt = Optimizer.adam(0.02f);
            double tail = 0;
            double early = Double.NaN;
            for (int it = 0; it < steps; it++) {
                Var[] s = fit.sample(noise(qmc == 1, batch, r));
                Var elbo = FlowVariationalInference.logDensity(banana, s[0]).sub(s[1]).mean();
                double v = elbo.value().item();
                if (it >= steps - 20) {
                    tail += v / 20;
                }
                elbo.neg().backward();
                opt.step(fp);
                if (it == 49) {
                    early = fit.elbo(banana, 20000, seed);
                }
            }
            out.put(qmc == 1 ? "RQMC (Owen Sobol)" : "MC", new double[] {var, fit.elbo(banana, 20000, seed), early,
                tail});
        }
        return out;
    }

    private static Tensor noise(boolean qmc, int batch, SplittableRandom r) {
        double[][] z = qmc ? LowDiscrepancy.toGaussian(LowDiscrepancy.sobol(batch, 2, LowDiscrepancy.Scramble.OWEN,
                r.nextLong())) : LowDiscrepancy.toGaussian(RqmcEstimator.points(Method.MC, batch, 2, r.nextLong()));
        float[] f = new float[batch * 2];
        for (int i = 0; i < batch; i++) {
            f[2 * i] = (float) z[i][0];
            f[2 * i + 1] = (float) z[i][1];
        }
        return Tensor.of(Shape.of(batch, 2), f);
    }

    private static double[] flatGrad(List<Var> params) {
        int n = 0;
        for (Var p : params) {
            n += p.grad().toDoubleArray().length;
        }
        double[] g = new double[n];
        int k = 0;
        for (Var p : params) {
            for (double v : p.grad().toDoubleArray()) {
                g[k++] = v;
            }
        }
        return g;
    }

    /** A Gaussian-measure integrand with its exact value. */
    private record Target(String name, ToDoubleFunction<double[]> f, double exact) {
    }

    private static Target target(String kind, int d) {
        if (kind.equals("smooth")) {
            double b2 = 1.5 * 1.5;
            double c2 = 0.09 * d;
            double exact = Math.pow(b2 / (b2 + 1), d / 2.0) * Math.exp(-c2 / (2 * (b2 + 1)));
            return new Target(kind, x -> {
                double s = 0;
                for (double v : x) {
                    s += (v - 0.3) * (v - 0.3);
                }
                return Math.exp(-s / (2 * b2));
            }, exact);
        }
        double exact = Math.pow(1 + Math.sqrt(2 / Math.PI), d) / Math.pow(2, d);
        return new Target(kind, x -> {
            double p = 1;
            for (double v : x) {
                p *= (1 + Math.abs(v)) / 2;
            }
            return p;
        }, exact);
    }

    /**
     * The Bayesian-quadrature table: relative absolute error at {@code n} evaluations on
     * smooth and rough Gaussian-measure integrals, the coverage of BQ's 95% interval, and kernel
     * herding's MMD.
     *
     * @param dims the dimensions
     * @param n evaluations
     * @param replicates replicates of the randomised methods
     * @param seed the seed
     * @return rows "kind d=.. METHOD" with {@code {relative error, coverage (BQ) or evaluations}}
     */
    public static Map<String, double[]> quadrature(int[] dims, int n, int replicates, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (String kind : new String[] {"smooth", "rough"}) {
            for (int d : dims) {
                Target t = target(kind, d);
                double mc = 0;
                double sob = 0;
                double bq = 0;
                double cover = 0;
                for (int rep = 0; rep < replicates; rep++) {
                    double[][] xm = LowDiscrepancy.toGaussian(RqmcEstimator.points(Method.MC, n, d, seed + rep));
                    double[][] xs = LowDiscrepancy.toGaussian(LowDiscrepancy.sobol(n, d, LowDiscrepancy.Scramble.OWEN,
                            seed + rep));
                    mc += Math.abs(mean(t.f(), xm) - t.exact()) / replicates;
                    sob += Math.abs(mean(t.f(), xs) - t.exact()) / replicates;
                    double[] fy = new double[n];
                    for (int i = 0; i < n; i++) {
                        fy[i] = t.f().applyAsDouble(xs[i]);
                    }
                    BayesianQuadrature.Estimate e = BayesianQuadrature.fit(xs, fy);
                    bq += Math.abs(e.mean() - t.exact()) / replicates;
                    cover += Math.abs(e.mean() - t.exact()) <= 1.96 * e.sd() ? 1.0 / replicates : 0;
                }
                String p = kind + " d=" + d + " ";
                double ex = Math.abs(t.exact());
                out.put(p + "MC", new double[] {mc / ex, n});
                out.put(p + "Sobol", new double[] {sob / ex, n});
                out.put(p + "BQ (Sobol points)", new double[] {bq / ex, cover});
                BayesianQuadrature.Estimate seq = BayesianQuadrature.sequential(t.f(), d, n, Math.min(n, 2 * d + 2),
                        100, seed);
                out.put(p + "sequential BQ", new double[] {Math.abs(seq.mean() - t.exact()) / ex,
                    Math.abs(seq.mean() - t.exact()) <= 1.96 * seq.sd() ? 1 : 0});
                double[][] herd = KernelHerding.herd(n, d, 1, 300, seed);
                out.put(p + "herding (uniform weights)",
                        new double[] {Math.abs(mean(t.f(), herd) - t.exact()) / ex, n});
                int per = (int) Math.max(2, Math.floor(Math.pow(n, 1.0 / d)));
                Cubature.Rule gh = Cubature.product(per, d);
                out.put(p + "Gauss-Hermite product", new double[] {Math.abs(gh.integrate(t.f()) - t.exact()) / ex,
                    gh.size()});
                Cubature.Rule sm = smolyakWithin(n, d);
                out.put(p + "Smolyak", new double[] {Math.abs(sm.integrate(t.f()) - t.exact()) / ex, sm.size()});
            }
        }
        return out;
    }

    private static Cubature.Rule smolyakWithin(int n, int d) {
        Cubature.Rule best = Cubature.smolyak(1, d);
        for (int level = 2; level < 8; level++) {
            Cubature.Rule r = Cubature.smolyak(level, d);
            if (r.size() > n) {
                break;
            }
            best = r;
        }
        return best;
    }

    private static double mean(ToDoubleFunction<double[]> f, double[][] x) {
        double s = 0;
        for (double[] row : x) {
            s += f.applyAsDouble(row);
        }
        return s / x.length;
    }

    /**
     * Herding's MMD to {@code N(0, I)} against i.i.d. and scrambled-Sobol Gaussian points.
     *
     * @param d the dimension
     * @param sizes the point counts
     * @param seed the seed
     * @return rows per method with the MMD at each size, then the log-log slope
     */
    public static Map<String, double[]> herding(int d, int[] sizes, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        double[][] rows = new double[3][sizes.length + 1];
        double[] lx = new double[sizes.length];
        for (int s = 0; s < sizes.length; s++) {
            int n = sizes[s];
            lx[s] = Math.log(n);
            rows[0][s] = KernelHerding.mmd(KernelHerding.herd(n, d, 1, 300, seed), 1);
            double iid = 0;
            double sob = 0;
            for (int rep = 0; rep < 5; rep++) {
                iid += KernelHerding.mmd(LowDiscrepancy.toGaussian(RqmcEstimator.points(Method.MC, n, d, seed + rep)),
                        1) / 5;
                sob += KernelHerding.mmd(LowDiscrepancy.toGaussian(LowDiscrepancy.sobol(n, d,
                        LowDiscrepancy.Scramble.OWEN, seed + rep)), 1) / 5;
            }
            rows[1][s] = iid;
            rows[2][s] = sob;
        }
        String[] names = {"herding", "i.i.d.", "Owen Sobol"};
        for (int m = 0; m < 3; m++) {
            double[] ly = new double[sizes.length];
            for (int s = 0; s < sizes.length; s++) {
                ly[s] = Math.log(rows[m][s]);
            }
            rows[m][sizes.length] = slope(lx, ly);
            out.put(names[m], rows[m]);
        }
        return out;
    }

    /**
     * Log evidence of a Bayesian linear regression with prior {@code N(0, I)}: BQ and WSABI on
     * Sobol points, MC and Sobol averages of the likelihood, and nested sampling, against the
     * closed form.
     *
     * @param d the number of weights
     * @param n integrand evaluations for the quadrature methods
     * @param seed the seed
     * @return rows {@code {|log Z error|, likelihood evaluations}}
     */
    public static Map<String, double[]> evidence(int d, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int m = 10;
        double sigma = 1.0;
        double[][] x = new double[m][d];
        double[] y = new double[m];
        for (int i = 0; i < m; i++) {
            double s = 0;
            for (int j = 0; j < d; j++) {
                x[i][j] = r.nextGaussian();
                s += x[i][j] * 0.5;
            }
            y[i] = s + sigma * r.nextGaussian();
        }
        // exact: y ~ N(0, X X^T + sigma^2 I)
        double[][] cov = new double[m][m];
        for (int a = 0; a < m; a++) {
            for (int b = 0; b < m; b++) {
                double s = a == b ? sigma * sigma : 0;
                for (int j = 0; j < d; j++) {
                    s += x[a][j] * x[b][j];
                }
                cov[a][b] = s;
            }
        }
        double[] alpha = Matrices.solve(cov, y);
        double quad = 0;
        for (int i = 0; i < m; i++) {
            quad += y[i] * alpha[i];
        }
        double exact = -0.5 * quad - 0.5 * Math.log(Matrices.determinant(cov)) - 0.5 * m * Math.log(2 * Math.PI);
        ToDoubleFunction<double[]> logLik = th -> {
            double s = 0;
            for (int i = 0; i < m; i++) {
                double mu = 0;
                for (int j = 0; j < d; j++) {
                    mu += x[i][j] * th[j];
                }
                s += -0.5 * (y[i] - mu) * (y[i] - mu) / (sigma * sigma);
            }
            return s - m * Math.log(sigma) - 0.5 * m * Math.log(2 * Math.PI);
        };
        // scale the likelihood by exp(-offset) so values are O(1)
        double offset = exact;
        ToDoubleFunction<double[]> lik = th -> Math.exp(logLik.applyAsDouble(th) - offset);
        Map<String, double[]> out = new LinkedHashMap<>();
        double[][] xs = LowDiscrepancy.toGaussian(LowDiscrepancy.sobol(n, d, LowDiscrepancy.Scramble.OWEN, seed));
        double[][] xm = LowDiscrepancy.toGaussian(RqmcEstimator.points(Method.MC, n, d, seed));
        double[] fy = new double[n];
        for (int i = 0; i < n; i++) {
            fy[i] = lik.applyAsDouble(xs[i]);
        }
        out.put("MC", new double[] {Math.abs(Math.log(Math.max(mean(lik, xm), 1e-300))), n});
        out.put("Owen Sobol", new double[] {Math.abs(Math.log(Math.max(mean(lik, xs), 1e-300))), n});
        out.put("BQ (Sobol points)", new double[] {Math.abs(Math.log(Math.max(BayesianQuadrature.fit(xs, fy).mean(),
                1e-300))), n});
        out.put("WSABI-L (Sobol points)", new double[] {Math.abs(Math.log(Math.max(BayesianQuadrature.wsabi(xs, fy)
                .mean(), 1e-300))), n});
        NestedSampling.Result ns = NestedSampling.of(NestedSampling.Replacement.SLICE, 100, 3, 0.01, 20000).run(d,
                u -> {
                    double[] t = new double[d];
                    for (int j = 0; j < d; j++) {
                        t[j] = Special.normalQuantile(Math.min(1 - 1e-12, Math.max(1e-12, u[j])));
                    }
                    return t;
                }, logLik, new SplittableRandom(seed));
        out.put("nested sampling", new double[] {Math.abs(ns.logZ() - exact), ns.likelihoodEvaluations()});
        return out;
    }
}
