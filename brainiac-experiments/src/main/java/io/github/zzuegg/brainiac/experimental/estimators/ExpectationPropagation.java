package io.github.zzuegg.brainiac.experimental.estimators;

import io.github.zzuegg.brainiac.experimental.ml.KernelFunction;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.NormalTails;

import java.util.Objects;

/**
 * Expectation propagation for binary Gaussian-process classification with the probit likelihood
 * {@code p(y | f) = Phi(y f)}, {@code y in {-1, +1}} (Minka, UAI 2001; Rasmussen and Williams,
 * <i>Gaussian Processes for Machine Learning</i>, 2006, Algorithms 3.5 and 3.6).
 *
 * <p>Each likelihood term is replaced by an unnormalised Gaussian site
 * {@code t_i(f_i) = Z_i N(f_i; nu_i / tau_i, 1 / tau_i)}. A site update removes the site from the
 * current posterior marginal (the cavity), matches the first two moments of the tilted
 * distribution {@code cavity x Phi(y f)} in closed form, and sets the site so that the new
 * marginal has those moments. {@link Schedule#SEQUENTIAL} updates one site at a time with a
 * rank-one update of the posterior covariance; {@link Schedule#PARALLEL} computes every site from
 * the same posterior and then refactorises, which needs damping to converge. The EP
 * approximation of the log marginal likelihood follows GPML's {@code infEP}.
 *
 * <p>{@link #laplace} gives the Laplace approximation for the same probit model (GPML
 * Algorithm 3.1/3.2), the baseline EP is usually compared against (Kuss and Rasmussen, JMLR 2005).
 */
public final class ExpectationPropagation {

    /** The site-update schedule. */
    public enum Schedule {
        /** One site at a time with rank-one covariance updates. */
        SEQUENTIAL,
        /** All sites from the same posterior, then one refactorisation. */
        PARALLEL
    }

    private final double[][] x;
    private final double[] y;
    private final KernelFunction kernel;
    private final double[][] k;
    private final double[] tau;
    private final double[] nu;
    private double[][] sigma;
    private double[] mu;
    private double[][] l;
    private double[] sw;
    private int sweeps;
    private double logZ;

    private ExpectationPropagation(double[][] x, int[] labels, KernelFunction kernel) {
        this.x = x;
        this.kernel = kernel;
        int n = x.length;
        this.y = new double[n];
        for (int i = 0; i < n; i++) {
            y[i] = labels[i] > 0 ? 1 : -1;
        }
        this.k = kernel.gram(x);
        this.tau = new double[n];
        this.nu = new double[n];
        this.sigma = new double[n][];
        for (int i = 0; i < n; i++) {
            sigma[i] = k[i].clone();
        }
        this.mu = new double[n];
    }

    /**
     * Runs expectation propagation to convergence.
     *
     * @param x the training inputs {@code [n][d]}
     * @param labels the labels, {@code 0/1} or {@code -1/+1}
     * @param kernel the prior covariance
     * @param schedule the update schedule
     * @param damping the damping factor in {@code (0, 1]}, 1 for undamped updates
     * @param maxSweeps the maximum number of sweeps over all sites
     * @param tolerance stop when the largest change of a site precision falls below this
     * @return the fitted approximation
     */
    public static ExpectationPropagation probit(double[][] x, int[] labels, KernelFunction kernel, Schedule schedule,
            double damping, int maxSweeps, double tolerance) {
        Objects.requireNonNull(schedule, "schedule");
        if (!(damping > 0 && damping <= 1)) {
            throw new IllegalArgumentException("damping must lie in (0, 1]: " + damping);
        }
        ExpectationPropagation ep = new ExpectationPropagation(x, labels, kernel);
        ep.run(schedule, damping, maxSweeps, tolerance);
        return ep;
    }

    private void run(Schedule schedule, double damping, int maxSweeps, double tolerance) {
        int n = y.length;
        for (int sweep = 0; sweep < maxSweeps; sweep++) {
            double change = 0;
            if (schedule == Schedule.SEQUENTIAL) {
                for (int i = 0; i < n; i++) {
                    double[] site = siteUpdate(i, damping);
                    double dTau = site[0] - tau[i];
                    change = Math.max(change, Math.abs(dTau));
                    tau[i] = site[0];
                    nu[i] = site[1];
                    double c = dTau / (1 + dTau * sigma[i][i]);
                    double[] si = sigma[i].clone();
                    for (int a = 0; a < n; a++) {
                        double f = c * si[a];
                        for (int b = 0; b < n; b++) {
                            sigma[a][b] -= f * si[b];
                        }
                    }
                    for (int a = 0; a < n; a++) {
                        double s = 0;
                        for (int b = 0; b < n; b++) {
                            s += sigma[a][b] * nu[b];
                        }
                        mu[a] = s;
                    }
                }
            } else {
                double[][] next = new double[n][];
                for (int i = 0; i < n; i++) {
                    next[i] = siteUpdate(i, damping);
                }
                for (int i = 0; i < n; i++) {
                    change = Math.max(change, Math.abs(next[i][0] - tau[i]));
                    tau[i] = next[i][0];
                    nu[i] = next[i][1];
                }
            }
            refactor();
            sweeps = sweep + 1;
            if (change < tolerance) {
                break;
            }
        }
        logZ = computeLogZ();
    }

    /** The damped new site {@code (tau_i, nu_i)} from the current marginal of {@code f_i}. */
    private double[] siteUpdate(int i, double damping) {
        double tauCav = 1 / sigma[i][i] - tau[i];
        double nuCav = mu[i] / sigma[i][i] - nu[i];
        if (tauCav <= 0) {
            return new double[] {tau[i], nu[i]};
        }
        double[] m = tiltedMoments(y[i], nuCav / tauCav, 1 / tauCav);
        double tauNew = Math.max(1 / m[2] - tauCav, 0);
        double nuNew = m[1] / m[2] - nuCav;
        return new double[] {(1 - damping) * tau[i] + damping * tauNew, (1 - damping) * nu[i] + damping * nuNew};
    }

    /**
     * Moments of {@code N(f; m, v) Phi(y f)}.
     *
     * @return {@code {log Z, mean, variance}}
     */
    static double[] tiltedMoments(double y, double m, double v) {
        double s = Math.sqrt(1 + v);
        double z = y * m / s;
        double logPhi = logCdf(z);
        double ratio = Math.exp(NormalTails.logPdf(z) - logPhi);
        double mean = m + y * v * ratio / s;
        double var = v - v * v * ratio * (z + ratio) / (1 + v);
        return new double[] {logPhi, mean, Math.max(var, 1e-12)};
    }

    static double logCdf(double z) {
        return NormalTails.logSf(-z);
    }

    private void refactor() {
        int n = y.length;
        sw = new double[n];
        for (int i = 0; i < n; i++) {
            sw[i] = Math.sqrt(Math.max(tau[i], 0));
        }
        double[][] b = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i][j] = sw[i] * k[i][j] * sw[j] + (i == j ? 1 : 0);
            }
        }
        l = MlMatrices.cholesky(b);
        // V = L^-1 (sW K): column j of V solves L v = sW o K[:, j]
        double[][] swk = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                swk[i][j] = sw[i] * k[i][j];
            }
        }
        double[][] vt = new double[n][];
        for (int j = 0; j < n; j++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) {
                col[i] = swk[i][j];
            }
            vt[j] = MlMatrices.forwardSolve(l, col);
        }
        sigma = new double[n][n];
        for (int a = 0; a < n; a++) {
            for (int c = a; c < n; c++) {
                double s = k[a][c] - MlMatrices.dot(vt[a], vt[c]);
                sigma[a][c] = s;
                sigma[c][a] = s;
            }
        }
        for (int a = 0; a < n; a++) {
            double s = 0;
            for (int c = 0; c < n; c++) {
                s += sigma[a][c] * nu[c];
            }
            mu[a] = s;
        }
    }

    /**
     * GPML equation 3.65: {@code log Z_EP = -1/2 log|K + S^-1| - 1/2 mu~^T (K + S^-1)^-1 mu~
     * + sum_i [log Z_i + 1/2 log(s2_-i + 1/tau_i) + (mu_-i - mu~_i)^2 / (2 (s2_-i + 1/tau_i))]}
     * with site means {@code mu~ = nu / tau} and cavity moments {@code mu_-i, s2_-i}.
     */
    private double computeLogZ() {
        int n = y.length;
        double lz = 0;
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            // log|K + S^-1| = log|B| - log|S|
            lz -= Math.log(l[i][i]);
            lz += 0.5 * Math.log(Math.max(tau[i], 1e-300));
            b[i] = sw[i] > 0 ? nu[i] / sw[i] : 0;
        }
        double[] c = MlMatrices.forwardSolve(l, b);
        lz -= 0.5 * MlMatrices.dot(c, c);
        for (int i = 0; i < n; i++) {
            double tauCav = 1 / sigma[i][i] - tau[i];
            double muCav = (mu[i] / sigma[i][i] - nu[i]) / tauCav;
            double s2Cav = 1 / tauCav;
            lz += tiltedMoments(y[i], muCav, s2Cav)[0];
            if (tau[i] > 0) {
                double total = s2Cav + 1 / tau[i];
                double siteMean = nu[i] / tau[i];
                lz += 0.5 * Math.log(total) + (muCav - siteMean) * (muCav - siteMean) / (2 * total);
            }
        }
        return lz;
    }

    /**
     * The EP approximation of {@code log p(y | X)}.
     *
     * @return the log marginal likelihood
     */
    public double logMarginalLikelihood() {
        return logZ;
    }

    /**
     * The sweeps until convergence.
     *
     * @return the number of sweeps run
     */
    public int sweeps() {
        return sweeps;
    }

    /**
     * The approximate posterior mean of the training latents.
     *
     * @return {@code mu} {@code [n]}
     */
    public double[] posteriorMean() {
        return mu.clone();
    }

    /**
     * The approximate posterior marginal variances of the training latents.
     *
     * @return {@code diag(Sigma)} {@code [n]}
     */
    public double[] posteriorVariance() {
        double[] v = new double[mu.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = sigma[i][i];
        }
        return v;
    }

    /**
     * The approximate posterior covariance of the training latents.
     *
     * @return a copy of {@code Sigma}
     */
    public double[][] posteriorCovariance() {
        double[][] c = new double[mu.length][];
        for (int i = 0; i < c.length; i++) {
            c[i] = sigma[i].clone();
        }
        return c;
    }

    /**
     * Predictive latent moments at test inputs (GPML Algorithm 3.6).
     *
     * @param xs the test inputs {@code [m][d]}
     * @return {@code {mean[m], variance[m]}}
     */
    public double[][] predictLatent(double[][] xs) {
        int n = y.length;
        // z = sW L^T \ (L \ (sW K nu))
        double[] kn = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) {
                s += k[i][j] * nu[j];
            }
            kn[i] = sw[i] * s;
        }
        double[] z = backSolve(l, MlMatrices.forwardSolve(l, kn));
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = nu[i] - sw[i] * z[i];
        }
        double[] mean = new double[xs.length];
        double[] var = new double[xs.length];
        for (int t = 0; t < xs.length; t++) {
            double[] ks = new double[n];
            double[] sks = new double[n];
            for (int i = 0; i < n; i++) {
                ks[i] = kernel.value(x[i], xs[t]);
                sks[i] = sw[i] * ks[i];
            }
            mean[t] = MlMatrices.dot(ks, w);
            double[] v = MlMatrices.forwardSolve(l, sks);
            var[t] = Math.max(kernel.value(xs[t], xs[t]) - MlMatrices.dot(v, v), 1e-12);
        }
        return new double[][] {mean, var};
    }

    /**
     * Predictive class probabilities {@code Phi(m / sqrt(1 + v))}.
     *
     * @param xs the test inputs
     * @return {@code p(y = 1 | x)} {@code [m]}
     */
    public double[] predictProbability(double[][] xs) {
        return probit(predictLatent(xs));
    }

    static double[] probit(double[][] mv) {
        double[] p = new double[mv[0].length];
        for (int t = 0; t < p.length; t++) {
            p[t] = NormalTails.cdf(mv[0][t] / Math.sqrt(1 + mv[1][t]));
        }
        return p;
    }

    static double[] backSolve(double[][] l, double[] b) {
        int n = b.length;
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double s = b[i];
            for (int c = i + 1; c < n; c++) {
                s -= l[c][i] * x[c];
            }
            x[i] = s / l[i][i];
        }
        return x;
    }

    /**
     * The Laplace approximation to the same probit GP classifier: Newton iterations to the
     * posterior mode, a Gaussian with the negative inverse Hessian there (GPML Algorithms 3.1
     * and 3.2).
     *
     * @param x the training inputs
     * @param labels the labels, {@code 0/1} or {@code -1/+1}
     * @param kernel the prior covariance
     * @param iterations the maximum Newton iterations
     * @return the approximation
     */
    public static Laplace laplace(double[][] x, int[] labels, KernelFunction kernel, int iterations) {
        return new Laplace(x, labels, kernel, iterations);
    }

    /** The Laplace approximation of the probit GP classifier. */
    public static final class Laplace {

        private final double[][] x;
        private final KernelFunction kernel;
        private final double[] f;
        private final double[] grad;
        private final double[] sw;
        private final double[][] l;
        private final double[][] sigma;
        private final double logZ;

        private Laplace(double[][] x, int[] labels, KernelFunction kernel, int iterations) {
            this.x = x;
            this.kernel = kernel;
            int n = x.length;
            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                y[i] = labels[i] > 0 ? 1 : -1;
            }
            double[][] k = kernel.gram(x);
            double[] fv = new double[n];
            double[] g = new double[n];
            double[] w = new double[n];
            double[][] lv = null;
            double[] s = new double[n];
            double[] a = new double[n];
            double previous = Double.NEGATIVE_INFINITY;
            for (int it = 0; it < iterations; it++) {
                derivatives(y, fv, g, w);
                for (int i = 0; i < n; i++) {
                    s[i] = Math.sqrt(w[i]);
                }
                lv = MlMatrices.cholesky(scaledB(k, s));
                double[] b = new double[n];
                for (int i = 0; i < n; i++) {
                    b[i] = w[i] * fv[i] + g[i];
                }
                double[] kb = new double[n];
                for (int i = 0; i < n; i++) {
                    double t = 0;
                    for (int j = 0; j < n; j++) {
                        t += k[i][j] * b[j];
                    }
                    kb[i] = s[i] * t;
                }
                double[] c = backSolve(lv, MlMatrices.forwardSolve(lv, kb));
                for (int i = 0; i < n; i++) {
                    a[i] = b[i] - s[i] * c[i];
                }
                for (int i = 0; i < n; i++) {
                    double t = 0;
                    for (int j = 0; j < n; j++) {
                        t += k[i][j] * a[j];
                    }
                    fv[i] = t;
                }
                double obj = -MlMatrices.dot(a, fv) / 2;
                for (int i = 0; i < n; i++) {
                    obj += logCdf(y[i] * fv[i]);
                }
                if (Math.abs(obj - previous) < 1e-10) {
                    break;
                }
                previous = obj;
            }
            derivatives(y, fv, g, w);
            for (int i = 0; i < n; i++) {
                s[i] = Math.sqrt(w[i]);
            }
            lv = MlMatrices.cholesky(scaledB(k, s));
            double lz = -MlMatrices.dot(a, fv) / 2;
            for (int i = 0; i < n; i++) {
                lz += logCdf(y[i] * fv[i]) - Math.log(lv[i][i]);
            }
            double[][] vt = new double[n][];
            for (int j = 0; j < n; j++) {
                double[] col = new double[n];
                for (int i = 0; i < n; i++) {
                    col[i] = s[i] * k[i][j];
                }
                vt[j] = MlMatrices.forwardSolve(lv, col);
            }
            double[][] sg = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    sg[i][j] = k[i][j] - MlMatrices.dot(vt[i], vt[j]);
                }
            }
            this.f = fv;
            this.grad = g;
            this.sw = s;
            this.l = lv;
            this.sigma = sg;
            this.logZ = lz;
        }

        private static double[][] scaledB(double[][] k, double[] s) {
            int n = s.length;
            double[][] b = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    b[i][j] = s[i] * k[i][j] * s[j] + (i == j ? 1 : 0);
                }
            }
            return b;
        }

        private static void derivatives(double[] y, double[] f, double[] g, double[] w) {
            for (int i = 0; i < y.length; i++) {
                double z = y[i] * f[i];
                double r = Math.exp(NormalTails.logPdf(z) - logCdf(z));
                g[i] = y[i] * r;
                w[i] = r * r + z * r;
            }
        }

        /**
         * The Laplace approximation of {@code log p(y | X)}.
         *
         * @return the log marginal likelihood
         */
        public double logMarginalLikelihood() {
            return logZ;
        }

        /**
         * The posterior mode of the training latents.
         *
         * @return {@code f_hat}
         */
        public double[] posteriorMean() {
            return f.clone();
        }

        /**
         * The approximate posterior marginal variances.
         *
         * @return {@code diag((K^-1 + W)^-1)}
         */
        public double[] posteriorVariance() {
            double[] v = new double[f.length];
            for (int i = 0; i < v.length; i++) {
                v[i] = sigma[i][i];
            }
            return v;
        }

        /**
         * Predictive latent moments (GPML Algorithm 3.2).
         *
         * @param xs the test inputs
         * @return {@code {mean[m], variance[m]}}
         */
        public double[][] predictLatent(double[][] xs) {
            int n = f.length;
            double[] mean = new double[xs.length];
            double[] var = new double[xs.length];
            for (int t = 0; t < xs.length; t++) {
                double[] ks = new double[n];
                double[] sks = new double[n];
                for (int i = 0; i < n; i++) {
                    ks[i] = kernel.value(x[i], xs[t]);
                    sks[i] = sw[i] * ks[i];
                }
                mean[t] = MlMatrices.dot(ks, grad);
                double[] v = MlMatrices.forwardSolve(l, sks);
                var[t] = Math.max(kernel.value(xs[t], xs[t]) - MlMatrices.dot(v, v), 1e-12);
            }
            return new double[][] {mean, var};
        }

        /**
         * Predictive class probabilities.
         *
         * @param xs the test inputs
         * @return {@code p(y = 1 | x)}
         */
        public double[] predictProbability(double[][] xs) {
            return probit(predictLatent(xs));
        }
    }
}
