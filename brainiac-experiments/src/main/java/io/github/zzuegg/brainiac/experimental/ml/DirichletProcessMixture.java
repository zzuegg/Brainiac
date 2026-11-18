package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Dirichlet-process Gaussian mixtures: clustering where the number of clusters is inferred.
 *
 * <ul>
 *   <li>{@link #gibbs}: the collapsed Gibbs sampler of Neal (2000, algorithm 3). Each point is
 *       reassigned with probability proportional to the Chinese-restaurant prior times the
 *       Normal-inverse-Wishart posterior predictive (a multivariate Student-t) of each cluster,
 *       or of a new cluster with weight {@code alpha}.</li>
 *   <li>{@link #variational}: truncated stick-breaking mean-field variational inference
 *       (Blei and Jordan 2006) with Gaussian-Wishart components.</li>
 *   <li>{@link #dpMeans}: the small-variance limit (Kulis and Jordan 2012,
 *       <a href="https://arxiv.org/abs/1111.0352">arXiv:1111.0352</a>): k-means where a point
 *       further than {@code sqrt(lambda)} from every centre opens a new cluster.</li>
 * </ul>
 *
 * <p>The sufficient statistics are kept locally ({@link GaussianMixture} keeps its M step
 * private).
 */
public final class DirichletProcessMixture {

    private DirichletProcessMixture() {
    }

    /**
     * A Normal-inverse-Wishart prior on each cluster's mean and covariance.
     *
     * @param mean the prior mean {@code mu0}
     * @param kappa the prior mean strength {@code kappa0}
     * @param dof the inverse-Wishart degrees of freedom {@code nu0 > d - 1}
     * @param scale the inverse-Wishart scale {@code Psi0}; the prior mean covariance is
     *     {@code Psi0 / (nu0 - d - 1)}
     */
    public record Prior(double[] mean, double kappa, double dof, double[][] scale) {

        /**
         * A weak prior centred on the data with an isotropic expected cluster covariance.
         *
         * @param x the data
         * @param clusterVariance the expected per-coordinate variance of one cluster
         * @return the prior ({@code kappa0 = 0.01}, {@code nu0 = d + 2})
         */
        public static Prior of(double[][] x, double clusterVariance) {
            int d = x[0].length;
            double[][] s = new double[d][d];
            for (int k = 0; k < d; k++) {
                s[k][k] = clusterVariance;
            }
            return new Prior(MlMatrices.columnMeans(x), 0.01, d + 2, s);
        }
    }

    /**
     * A clustering.
     *
     * @param labels each point's cluster, {@code 0 .. clusters - 1}
     * @param clusters the number of non-empty clusters
     * @param weights the predictive weight of each cluster (and, for Gibbs, a last entry for a
     *     new cluster)
     * @param means the component means
     * @param covariances the component predictive covariances
     */
    public record Result(int[] labels, int clusters, double[] weights, double[][] means, double[][][] covariances) {

        /**
         * The mean held-out log-likelihood under the fitted mixture (Gaussian components with
         * the predictive covariances).
         *
         * @param x the points
         * @return the average log density
         */
        public double meanLogLikelihood(double[][] x) {
            double total = 0;
            int k = means.length;
            double[][][] chol = new double[k][][];
            double[] logDet = new double[k];
            for (int j = 0; j < k; j++) {
                chol[j] = MlMatrices.cholesky(covariances[j]);
                logDet[j] = MlMatrices.logDet(chol[j]);
            }
            for (double[] p : x) {
                double max = Double.NEGATIVE_INFINITY;
                double[] lp = new double[k];
                for (int j = 0; j < k; j++) {
                    lp[j] = Math.log(weights[j]) + gaussianLogDensity(p, means[j], chol[j], logDet[j]);
                    max = Math.max(max, lp[j]);
                }
                double s = 0;
                for (double v : lp) {
                    s += Math.exp(v - max);
                }
                total += max + Math.log(s);
            }
            return total / x.length;
        }
    }

    private static double gaussianLogDensity(double[] x, double[] mean, double[][] chol, double logDet) {
        int d = x.length;
        double[] diff = new double[d];
        for (int k = 0; k < d; k++) {
            diff[k] = x[k] - mean[k];
        }
        double[] z = MlMatrices.forwardSolve(chol, diff);
        return -0.5 * (MlMatrices.dot(z, z) + logDet + d * Math.log(2 * Math.PI));
    }

    /** Running sufficient statistics of one cluster. */
    private static final class Stats {
        int n;
        final double[] sum;
        final double[][] outer;
        boolean dirty = true;
        double[] loc;
        double[][] chol;
        double logDet;
        double dof;

        Stats(int d) {
            sum = new double[d];
            outer = new double[d][d];
        }

        void add(double[] x, int sign) {
            n += sign;
            for (int a = 0; a < x.length; a++) {
                sum[a] += sign * x[a];
                for (int b = 0; b < x.length; b++) {
                    outer[a][b] += sign * x[a] * x[b];
                }
            }
            dirty = true;
        }

        /** Posterior NIW parameters, then the Student-t predictive's location and scale. */
        void refresh(Prior p) {
            if (!dirty) {
                return;
            }
            int d = sum.length;
            double kn = p.kappa() + n;
            double vn = p.dof() + n;
            loc = new double[d];
            for (int a = 0; a < d; a++) {
                loc[a] = (p.kappa() * p.mean()[a] + sum[a]) / kn;
            }
            double[][] psi = new double[d][d];
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    // Psi_n = Psi0 + sum x x^T + kappa0 mu0 mu0^T - kappa_n mu_n mu_n^T
                    psi[a][b] = p.scale()[a][b] + outer[a][b] + p.kappa() * p.mean()[a] * p.mean()[b]
                            - kn * loc[a] * loc[b];
                }
            }
            dof = vn - d + 1;
            double f = (kn + 1) / (kn * dof);
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    psi[a][b] *= f;
                }
                psi[a][a] += 1e-9;
            }
            chol = MlMatrices.cholesky(psi);
            logDet = MlMatrices.logDet(chol);
            dirty = false;
        }

        double[][] scale() {
            int d = chol.length;
            double[][] s = new double[d][d];
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    for (int k = 0; k < d; k++) {
                        s[a][b] += chol[a][k] * chol[b][k];
                    }
                }
            }
            return s;
        }

        double logPredictive(double[] x) {
            int d = x.length;
            double[] diff = new double[d];
            for (int a = 0; a < d; a++) {
                diff[a] = x[a] - loc[a];
            }
            double[] z = MlMatrices.forwardSolve(chol, diff);
            double m = MlMatrices.dot(z, z);
            return ProbabilisticProgram.logGamma((dof + d) / 2) - ProbabilisticProgram.logGamma(dof / 2)
                    - 0.5 * d * Math.log(dof * Math.PI) - 0.5 * logDet - 0.5 * (dof + d) * Math.log1p(m / dof);
        }
    }

    /**
     * Collapsed Gibbs sampling, starting from one cluster or from given labels.
     *
     * @param x the points {@code [n][d]}
     * @param alpha the concentration
     * @param prior the cluster prior
     * @param sweeps the passes over the data
     * @param start initial labels (any non-negative ints), or {@code null} for a single cluster
     * @param seed the seed
     * @return the final state; covariances are the predictive Student-t scales
     */
    public static Result gibbs(double[][] x, double alpha, Prior prior, int sweeps, int[] start, long seed) {
        int n = x.length;
        int d = x[0].length;
        SplittableRandom r = new SplittableRandom(seed);
        int[] z = new int[n];
        List<Stats> clusters = new ArrayList<>();
        if (start == null) {
            clusters.add(new Stats(d));
        } else {
            int max = Arrays.stream(start).max().orElse(0);
            for (int k = 0; k <= max; k++) {
                clusters.add(new Stats(d));
            }
        }
        for (int i = 0; i < n; i++) {
            z[i] = start == null ? 0 : start[i];
            clusters.get(z[i]).add(x[i], 1);
        }
        Stats empty = new Stats(d);
        empty.refresh(prior);
        for (int sweep = 0; sweep < sweeps; sweep++) {
            for (int i = 0; i < n; i++) {
                Stats own = clusters.get(z[i]);
                own.add(x[i], -1);
                if (own.n == 0) {
                    int last = clusters.size() - 1;
                    if (z[i] != last) {
                        clusters.set(z[i], clusters.get(last));
                        for (int j = 0; j < n; j++) {
                            if (z[j] == last) {
                                z[j] = z[i];
                            }
                        }
                    }
                    clusters.remove(last);
                }
                int k = clusters.size();
                double[] lp = new double[k + 1];
                double max = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < k; c++) {
                    Stats s = clusters.get(c);
                    s.refresh(prior);
                    lp[c] = Math.log(s.n) + s.logPredictive(x[i]);
                    max = Math.max(max, lp[c]);
                }
                lp[k] = Math.log(alpha) + empty.logPredictive(x[i]);
                max = Math.max(max, lp[k]);
                double total = 0;
                for (int c = 0; c <= k; c++) {
                    lp[c] = Math.exp(lp[c] - max);
                    total += lp[c];
                }
                double u = r.nextDouble() * total;
                int pick = 0;
                while (pick < k && (u -= lp[pick]) > 0) {
                    pick++;
                }
                if (pick == k) {
                    clusters.add(new Stats(d));
                }
                z[i] = pick;
                clusters.get(pick).add(x[i], 1);
            }
        }
        int k = clusters.size();
        double[] w = new double[k + 1];
        double[][] means = new double[k + 1][];
        double[][][] cov = new double[k + 1][][];
        for (int c = 0; c < k; c++) {
            Stats s = clusters.get(c);
            s.refresh(prior);
            w[c] = s.n / (n + alpha);
            means[c] = s.loc;
            cov[c] = s.scale();
        }
        w[k] = alpha / (n + alpha);
        means[k] = empty.loc;
        cov[k] = empty.scale();
        return new Result(z, k, w, means, cov);
    }

    /**
     * The digamma function {@code d/dx log Gamma(x)}.
     *
     * @param x the argument, {@code > 0}
     * @return {@code psi(x)}
     */
    public static double digamma(double x) {
        double result = 0;
        while (x < 6) {
            result -= 1 / x;
            x += 1;
        }
        double f = 1 / (x * x);
        return result + Math.log(x) - 0.5 / x
                - f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f / 132))));
    }

    /**
     * Truncated stick-breaking variational inference with Gaussian-Wishart components.
     *
     * @param x the points {@code [n][d]}
     * @param alpha the concentration
     * @param prior the cluster prior (its inverse-Wishart scale sets the Wishart one)
     * @param truncation the number of sticks
     * @param iterations the coordinate-ascent rounds
     * @param seed seeds the k-means++ initialisation of the responsibilities
     * @return clusters with at least one hard-assigned point, relabelled {@code 0 ..}
     */
    public static Result variational(double[][] x, double alpha, Prior prior, int truncation, int iterations,
            long seed) {
        int n = x.length;
        int d = x[0].length;
        int t = Math.min(truncation, n);
        double[][] resp = new double[n][t];
        int[] init = KMeans.fit(x, t, KMeans.Init.PLUS_PLUS, 1, 10, seed).labels();
        for (int i = 0; i < n; i++) {
            resp[i][init[i]] = 1;
        }
        double[][] w0Inv = prior.scale();
        double beta0 = prior.kappa();
        double nu0 = prior.dof();
        double[] counts = new double[t];
        double[][] m = new double[t][d];
        double[] beta = new double[t];
        double[] nu = new double[t];
        double[][][] wChol = new double[t][][];
        double[][][] wInv = new double[t][][];
        double[] eLogPi = new double[t];
        double[] restLog = new double[t];
        double[] eLogDet = new double[t];
        for (int it = 0; it <= iterations; it++) {
            // M step: sufficient statistics, then the variational posteriors
            for (int k = 0; k < t; k++) {
                double nk = 1e-10;
                double[] mean = new double[d];
                for (int i = 0; i < n; i++) {
                    nk += resp[i][k];
                    for (int a = 0; a < d; a++) {
                        mean[a] += resp[i][k] * x[i][a];
                    }
                }
                for (int a = 0; a < d; a++) {
                    mean[a] /= nk;
                }
                double[][] s = new double[d][d];
                for (int i = 0; i < n; i++) {
                    for (int a = 0; a < d; a++) {
                        for (int b = 0; b < d; b++) {
                            s[a][b] += resp[i][k] * (x[i][a] - mean[a]) * (x[i][b] - mean[b]);
                        }
                    }
                }
                counts[k] = nk;
                beta[k] = beta0 + nk;
                nu[k] = nu0 + nk;
                double[][] wi = new double[d][d];
                for (int a = 0; a < d; a++) {
                    m[k][a] = (beta0 * prior.mean()[a] + nk * mean[a]) / beta[k];
                }
                double c = beta0 * nk / (beta0 + nk);
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        wi[a][b] = w0Inv[a][b] + s[a][b]
                                + c * (mean[a] - prior.mean()[a]) * (mean[b] - prior.mean()[b]);
                    }
                }
                wInv[k] = wi;
                wChol[k] = MlMatrices.cholesky(wi);
                // E log |Lambda| = sum psi((nu + 1 - i) / 2) + d log 2 - log |W^-1|
                double ld = d * Math.log(2) - MlMatrices.logDet(wChol[k]);
                for (int a = 1; a <= d; a++) {
                    ld += digamma((nu[k] + 1 - a) / 2);
                }
                eLogDet[k] = ld;
            }
            double tail = 0;
            for (int k = t - 1; k >= 0; k--) {
                double g1 = 1 + counts[k];
                double g2 = alpha + tail;
                tail += counts[k];
                double dg = digamma(g1 + g2);
                eLogPi[k] = digamma(g1) - dg;
                restLog[k] = digamma(g2) - dg;
            }
            if (it == iterations) {
                break;
            }
            // E step
            for (int i = 0; i < n; i++) {
                double max = Double.NEGATIVE_INFINITY;
                double before = 0;
                for (int k = 0; k < t; k++) {
                    double[] diff = new double[d];
                    for (int a = 0; a < d; a++) {
                        diff[a] = x[i][a] - m[k][a];
                    }
                    double[] zz = MlMatrices.forwardSolve(wChol[k], diff);
                    double maha = nu[k] * MlMatrices.dot(zz, zz);
                    // E log pi_k = E log v_k + sum_{j<k} E log (1 - v_j)
                    resp[i][k] = eLogPi[k] + before + 0.5 * eLogDet[k] - d / (2 * beta[k]) - 0.5 * maha;
                    before += restLog[k];
                    max = Math.max(max, resp[i][k]);
                }
                double total = 0;
                for (int k = 0; k < t; k++) {
                    resp[i][k] = Math.exp(resp[i][k] - max);
                    total += resp[i][k];
                }
                for (int k = 0; k < t; k++) {
                    resp[i][k] /= total;
                }
            }
        }
        int[] raw = new int[n];
        int[] used = new int[t];
        for (int i = 0; i < n; i++) {
            int best = 0;
            for (int k = 1; k < t; k++) {
                best = resp[i][k] > resp[i][best] ? k : best;
            }
            raw[i] = best;
            used[best]++;
        }
        int[] relabel = new int[t];
        int count = 0;
        for (int k = 0; k < t; k++) {
            relabel[k] = used[k] > 0 ? count++ : -1;
        }
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            labels[i] = relabel[raw[i]];
        }
        double[] w = new double[count];
        double[][] means = new double[count][];
        double[][][] cov = new double[count][][];
        double totalW = 0;
        for (int k = 0; k < t; k++) {
            if (relabel[k] >= 0) {
                int j = relabel[k];
                w[j] = counts[k];
                totalW += counts[k];
                means[j] = m[k];
                double f = 1 / Math.max(nu[k] - d - 1, 1);
                cov[j] = new double[d][d];
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        cov[j][a][b] = wInv[k][a][b] * f;
                    }
                }
            }
        }
        for (int j = 0; j < count; j++) {
            w[j] /= totalW;
        }
        return new Result(labels, count, w, means, cov);
    }

    /**
     * DP-means: Lloyd iterations where a point whose squared distance to every centre exceeds
     * {@code lambda} opens a new cluster.
     *
     * @param x the points {@code [n][d]}
     * @param lambda the squared-distance penalty per cluster
     * @param iterations the maximum passes
     * @return the clustering (covariances are each cluster's empirical one)
     */
    public static Result dpMeans(double[][] x, double lambda, int iterations) {
        int n = x.length;
        int d = x[0].length;
        List<double[]> centres = new ArrayList<>();
        centres.add(MlMatrices.columnMeans(x));
        int[] z = new int[n];
        for (int it = 0; it < iterations; it++) {
            boolean changed = false;
            for (int i = 0; i < n; i++) {
                int best = 0;
                double bd = Double.POSITIVE_INFINITY;
                for (int c = 0; c < centres.size(); c++) {
                    double dist = MlMatrices.squaredDistance(x[i], centres.get(c));
                    if (dist < bd) {
                        bd = dist;
                        best = c;
                    }
                }
                if (bd > lambda) {
                    centres.add(x[i].clone());
                    best = centres.size() - 1;
                }
                changed |= best != z[i];
                z[i] = best;
            }
            double[][] sums = new double[centres.size()][d];
            int[] counts = new int[centres.size()];
            for (int i = 0; i < n; i++) {
                counts[z[i]]++;
                for (int a = 0; a < d; a++) {
                    sums[z[i]][a] += x[i][a];
                }
            }
            List<double[]> next = new ArrayList<>();
            int[] relabel = new int[centres.size()];
            for (int c = 0; c < centres.size(); c++) {
                if (counts[c] > 0) {
                    relabel[c] = next.size();
                    for (int a = 0; a < d; a++) {
                        sums[c][a] /= counts[c];
                    }
                    next.add(sums[c]);
                }
            }
            for (int i = 0; i < n; i++) {
                z[i] = relabel[z[i]];
            }
            centres = next;
            if (!changed && it > 0) {
                break;
            }
        }
        return empirical(x, z, centres.size());
    }

    /**
     * A {@link Result} from hard labels with empirical weights, means and covariances.
     *
     * @param x the points
     * @param z the labels {@code 0 .. k - 1}
     * @param k the number of clusters
     * @return the result
     */
    public static Result empirical(double[][] x, int[] z, int k) {
        int n = x.length;
        int d = x[0].length;
        double[] w = new double[k];
        double[][] means = new double[k][d];
        double[][][] cov = new double[k][d][d];
        for (int i = 0; i < n; i++) {
            w[z[i]]++;
            for (int a = 0; a < d; a++) {
                means[z[i]][a] += x[i][a];
            }
        }
        for (int c = 0; c < k; c++) {
            for (int a = 0; a < d; a++) {
                means[c][a] /= Math.max(w[c], 1);
            }
        }
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    cov[z[i]][a][b] += (x[i][a] - means[z[i]][a]) * (x[i][b] - means[z[i]][b]);
                }
            }
        }
        for (int c = 0; c < k; c++) {
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    cov[c][a][b] /= Math.max(w[c], 1);
                }
                cov[c][a][a] += 1e-3;
            }
            w[c] /= n;
        }
        return new Result(z, k, w, means, cov);
    }
}
