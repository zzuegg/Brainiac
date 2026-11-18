package io.github.zzuegg.brainiac.experimental.ml;

import java.util.Arrays;

/**
 * A variational Bayesian Gaussian mixture (Corduneanu and Bishop, AISTATS 2001,
 * <a href="https://proceedings.mlr.press/r3/corduneanu01a.html">r3/corduneanu01a</a>; Bishop,
 * <i>Pattern Recognition and Machine Learning</i>, 2006, section 10.2) fitted by coordinate
 * ascent (CAVI). Mixing weights get a symmetric Dirichlet prior ({@link Prior#FINITE_DIRICHLET})
 * or a truncated stick-breaking (Dirichlet-process) prior ({@link Prior#STICK_BREAKING}); each
 * component's mean and precision a Normal-Wishart prior. With a small concentration, components
 * that are not needed are driven to zero weight, so one fit with a generous truncation chooses
 * the number of clusters without a BIC sweep, and the evidence lower bound (ELBO) scores the
 * model:
 * <pre>
 *   ELBO = sum_n log sum_k exp(E[log pi_k] + E[log N(x_n | mu_k, Lambda_k)])
 *          - KL(q(pi) || p(pi)) - sum_k KL(q(mu_k, Lambda_k) || p(mu_k, Lambda_k)),
 * </pre>
 * the bound with the assignment posterior already optimal. Initialised from k-means++.
 */
public final class VariationalGaussianMixture {

    /** The prior on the mixing weights. */
    public enum Prior {
        /** A symmetric Dirichlet with concentration {@code alpha0} per component. */
        FINITE_DIRICHLET,
        /** Truncated stick-breaking with {@code v_k ~ Beta(1, alpha0)}. */
        STICK_BREAKING
    }

    private final double[][] resp;
    private final double[] counts;
    private final double[] expectedWeights;
    private final double elbo;
    private final int iterations;
    private final double[][] means;
    private final double[][][] precisions;

    private VariationalGaussianMixture(double[][] resp, double[] counts, double[] expectedWeights, double elbo,
            int iterations, double[][] means, double[][][] precisions) {
        this.resp = resp;
        this.counts = counts;
        this.expectedWeights = expectedWeights;
        this.elbo = elbo;
        this.iterations = iterations;
        this.means = means;
        this.precisions = precisions;
    }

    private static double digamma(double x) {
        return DirichletProcessMixture.digamma(x);
    }

    private static double logGamma(double x) {
        return io.github.zzuegg.brainiac.experimental.discovery.Distributions.logGamma(x);
    }

    /**
     * Fits the mixture.
     *
     * @param x points {@code [n][d]}
     * @param truncation the number of components {@code K}
     * @param prior the weight prior
     * @param alpha0 the concentration
     * @param clusterVariance the prior's expected per-dimension cluster variance (the Wishart
     *     scale is {@code W0^-1 = nu0 clusterVariance I}, {@code nu0 = d})
     * @param iterations the largest number of CAVI sweeps (stops when the ELBO stalls)
     * @param seed the k-means++ seed
     * @return the fitted mixture
     */
    public static VariationalGaussianMixture fit(double[][] x, int truncation, Prior prior, double alpha0,
            double clusterVariance, int iterations, long seed) {
        int n = x.length;
        int d = x[0].length;
        int k = truncation;
        double[] m0 = MlMatrices.columnMeans(x);
        double beta0 = 1;
        double nu0 = d;
        double[][] w0inv = new double[d][d];
        for (int a = 0; a < d; a++) {
            w0inv[a][a] = nu0 * clusterVariance;
        }
        double logDetW0 = -MlMatrices.logDet(MlMatrices.cholesky(w0inv));
        KMeans start = KMeans.fit(x, k, KMeans.Init.PLUS_PLUS, 1, 20, seed);
        double[][] r = new double[n][k];
        for (int i = 0; i < n; i++) {
            r[i][start.labels()[i]] = 1;
        }
        double[] nk = new double[k];
        double[] eLogPi = new double[k];
        double previous = Double.NEGATIVE_INFINITY;
        double bound = Double.NEGATIVE_INFINITY;
        double[] alpha = new double[k];
        double[][] gamma = new double[k][2];
        int it = 0;
        double[][] lastM = new double[k][d];
        double[][][] lastPrecision = new double[k][d][d];
        for (; it < iterations; it++) {
            // M-step (variational parameters)
            Arrays.fill(nk, 0);
            double[][] xbar = new double[k][d];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < k; j++) {
                    nk[j] += r[i][j];
                    for (int a = 0; a < d; a++) {
                        xbar[j][a] += r[i][j] * x[i][a];
                    }
                }
            }
            double[] beta = new double[k];
            double[] nu = new double[k];
            double[][] m = new double[k][d];
            double[][][] w = new double[k][][];
            double[] logDetW = new double[k];
            double klParams = 0;
            for (int j = 0; j < k; j++) {
                double cnt = nk[j] + 1e-10;
                for (int a = 0; a < d; a++) {
                    xbar[j][a] /= cnt;
                }
                double[][] s = new double[d][d];
                for (int i = 0; i < n; i++) {
                    if (r[i][j] < 1e-12) {
                        continue;
                    }
                    for (int a = 0; a < d; a++) {
                        double da = x[i][a] - xbar[j][a];
                        for (int b = 0; b < d; b++) {
                            s[a][b] += r[i][j] * da * (x[i][b] - xbar[j][b]);
                        }
                    }
                }
                beta[j] = beta0 + nk[j];
                nu[j] = nu0 + nk[j];
                double[][] winv = new double[d][d];
                for (int a = 0; a < d; a++) {
                    m[j][a] = (beta0 * m0[a] + nk[j] * xbar[j][a]) / beta[j];
                }
                double f = beta0 * nk[j] / (beta0 + nk[j]);
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        winv[a][b] = w0inv[a][b] + s[a][b] + f * (xbar[j][a] - m0[a]) * (xbar[j][b] - m0[b]);
                    }
                }
                double[][] lw = MlMatrices.cholesky(winv);
                w[j] = MlMatrices.choleskyInverse(lw);
                logDetW[j] = -MlMatrices.logDet(lw);
                klParams += klNormalWishart(m[j], beta[j], w[j], logDetW[j], nu[j], m0, beta0, w0inv, logDetW0,
                        nu0);
            }
            for (int j = 0; j < k; j++) {
                lastM[j] = m[j];
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        lastPrecision[j][a][b] = nu[j] * w[j][a][b];
                    }
                }
            }
            double klWeights;
            if (prior == Prior.FINITE_DIRICHLET) {
                double sum = 0;
                for (int j = 0; j < k; j++) {
                    alpha[j] = alpha0 + nk[j];
                    sum += alpha[j];
                }
                for (int j = 0; j < k; j++) {
                    eLogPi[j] = digamma(alpha[j]) - digamma(sum);
                }
                double[] a0 = new double[k];
                Arrays.fill(a0, alpha0);
                klWeights = klDirichlet(alpha, a0);
            } else {
                double tail = 0;
                for (int j = k - 1; j >= 0; j--) {
                    gamma[j][0] = 1 + nk[j];
                    gamma[j][1] = alpha0 + tail;
                    tail += nk[j];
                }
                double acc = 0;
                klWeights = 0;
                for (int j = 0; j < k; j++) {
                    double dg = digamma(gamma[j][0] + gamma[j][1]);
                    double eLogV = digamma(gamma[j][0]) - dg;
                    double eLog1mV = digamma(gamma[j][1]) - dg;
                    eLogPi[j] = eLogV + acc;
                    acc += eLog1mV;
                    if (j < k - 1) {
                        klWeights += klDirichlet(gamma[j], new double[] {1, alpha0});
                    }
                }
            }
            // E-step
            double[] eLogDet = new double[k];
            for (int j = 0; j < k; j++) {
                double s = d * Math.log(2) + logDetW[j];
                for (int a = 1; a <= d; a++) {
                    s += digamma((nu[j] + 1 - a) / 2);
                }
                eLogDet[j] = s;
            }
            double data = 0;
            double[] diff = new double[d];
            for (int i = 0; i < n; i++) {
                double max = Double.NEGATIVE_INFINITY;
                double[] lr = new double[k];
                for (int j = 0; j < k; j++) {
                    for (int a = 0; a < d; a++) {
                        diff[a] = x[i][a] - m[j][a];
                    }
                    double quad = 0;
                    for (int a = 0; a < d; a++) {
                        double row = 0;
                        for (int b = 0; b < d; b++) {
                            row += w[j][a][b] * diff[b];
                        }
                        quad += diff[a] * row;
                    }
                    lr[j] = eLogPi[j] + 0.5 * eLogDet[j] - 0.5 * d * Math.log(2 * Math.PI)
                            - 0.5 * (d / beta[j] + nu[j] * quad);
                    max = Math.max(max, lr[j]);
                }
                double s = 0;
                for (int j = 0; j < k; j++) {
                    r[i][j] = Math.exp(lr[j] - max);
                    s += r[i][j];
                }
                for (int j = 0; j < k; j++) {
                    r[i][j] /= s;
                }
                data += max + Math.log(s);
            }
            bound = data - klWeights - klParams;
            if (Math.abs(bound - previous) < 1e-6 * n) {
                it++;
                break;
            }
            previous = bound;
        }
        Arrays.fill(nk, 0);
        for (double[] row : r) {
            for (int j = 0; j < k; j++) {
                nk[j] += row[j];
            }
        }
        double[] ew = new double[k];
        for (int j = 0; j < k; j++) {
            ew[j] = Math.exp(eLogPi[j]);
        }
        double s = Arrays.stream(ew).sum();
        for (int j = 0; j < k; j++) {
            ew[j] /= s;
        }
        return new VariationalGaussianMixture(r, nk.clone(), ew, bound, it, lastM, lastPrecision);
    }

    private static double klDirichlet(double[] a, double[] a0) {
        double sa = Arrays.stream(a).sum();
        double s0 = Arrays.stream(a0).sum();
        double kl = logGamma(sa) - logGamma(s0);
        double dsa = digamma(sa);
        for (int j = 0; j < a.length; j++) {
            kl += logGamma(a0[j]) - logGamma(a[j]) + (a[j] - a0[j]) * (digamma(a[j]) - dsa);
        }
        return kl;
    }

    private static double klNormalWishart(double[] m, double beta, double[][] w, double logDetW, double nu,
            double[] m0, double beta0, double[][] w0inv, double logDetW0, double nu0) {
        int d = m.length;
        double tr = 0;
        for (int a = 0; a < d; a++) {
            for (int b = 0; b < d; b++) {
                tr += w0inv[a][b] * w[b][a];
            }
        }
        // ln |W0^-1 W| = ln |W| - ln |W0|
        double logRatio = logDetW - logDetW0;
        double kl = -nu0 / 2 * logRatio + nu / 2 * (tr - d);
        for (int i = 1; i <= d; i++) {
            kl += logGamma((nu0 + 1 - i) / 2) - logGamma((nu + 1 - i) / 2) + (nu - nu0) / 2 * digamma((nu + 1 - i) / 2);
        }
        double quad = 0;
        for (int a = 0; a < d; a++) {
            for (int b = 0; b < d; b++) {
                quad += (m[a] - m0[a]) * w[a][b] * (m[b] - m0[b]);
            }
        }
        kl += 0.5 * (d * beta0 / beta - d + beta0 * nu * quad + d * Math.log(beta / beta0));
        return kl;
    }

    /**
     * The plug-in density: the mixture with weights {@link #expectedWeights}, posterior mean
     * locations and expected precisions {@code nu_k W_k}.
     *
     * @param x a point
     * @return {@code log p(x)}
     */
    public double logDensity(double[] x) {
        int d = x.length;
        double max = Double.NEGATIVE_INFINITY;
        double[] lp = new double[means.length];
        for (int j = 0; j < means.length; j++) {
            double[] diff = new double[d];
            for (int a = 0; a < d; a++) {
                diff[a] = x[a] - means[j][a];
            }
            double quad = 0;
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    quad += diff[a] * precisions[j][a][b] * diff[b];
                }
            }
            double logDet = MlMatrices.logDet(MlMatrices.cholesky(precisions[j]));
            lp[j] = Math.log(Math.max(expectedWeights[j], 1e-300)) + 0.5 * logDet - 0.5 * d * Math.log(2 * Math.PI)
                    - 0.5 * quad;
            max = Math.max(max, lp[j]);
        }
        double s = 0;
        for (double v : lp) {
            s += Math.exp(v - max);
        }
        return max + Math.log(s);
    }

    /** @return the most responsible component of every training point */
    public int[] labels() {
        int[] out = new int[resp.length];
        for (int i = 0; i < resp.length; i++) {
            int b = 0;
            for (int j = 1; j < resp[i].length; j++) {
                b = resp[i][j] > resp[i][b] ? j : b;
            }
            out[i] = b;
        }
        return out;
    }

    /**
     * @param minCount the smallest expected count of a kept component
     * @return the number of components with at least that many expected points
     */
    public int activeComponents(double minCount) {
        int c = 0;
        for (double v : counts) {
            c += v >= minCount ? 1 : 0;
        }
        return c;
    }

    /** @return the expected mixing weights {@code exp(E[log pi])}, normalised */
    public double[] expectedWeights() {
        return expectedWeights.clone();
    }

    /** @return the evidence lower bound at convergence */
    public double elbo() {
        return elbo;
    }

    /** @return the CAVI sweeps run */
    public int iterations() {
        return iterations;
    }
}
