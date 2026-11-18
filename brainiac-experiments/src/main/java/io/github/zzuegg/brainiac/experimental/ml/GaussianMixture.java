package io.github.zzuegg.brainiac.experimental.ml;

import java.util.SplittableRandom;

/**
 * A full-covariance Gaussian mixture fitted by expectation maximisation (Dempster, Laird and
 * Rubin 1977, <a href="https://www.jstor.org/stable/2984875">Maximum likelihood from incomplete
 * data via the EM algorithm</a>), seeded by {@link KMeans k-means++}, with the Bayesian
 * information criterion for choosing the number of components.
 *
 * <p>Unlike k-means it models elongated (anisotropic) clusters and gives soft assignments and a
 * normalised density, so it is both a clusterer and the classic density-estimation baseline for
 * normalising flows.
 */
public final class GaussianMixture {

    private final double[] weights;
    private final double[][] means;
    private final double[][][] covariances;
    private final double[][][] chols;
    private final double[] logDets;

    private GaussianMixture(double[] weights, double[][] means, double[][][] covariances) {
        this.weights = weights;
        this.means = means;
        this.covariances = covariances;
        this.chols = new double[weights.length][][];
        this.logDets = new double[weights.length];
        for (int j = 0; j < weights.length; j++) {
            chols[j] = MlMatrices.cholesky(covariances[j]);
            logDets[j] = MlMatrices.logDet(chols[j]);
        }
    }

    /**
     * Fits a mixture by EM.
     *
     * @param x the points {@code [n][d]}
     * @param k the number of components
     * @param iterations EM iterations (stops early when the likelihood stalls)
     * @param regularization added to every covariance diagonal, for example {@code 1e-6}
     * @param seed seeds the k-means++ start
     * @return the mixture
     */
    public static GaussianMixture fit(double[][] x, int k, int iterations, double regularization, long seed) {
        int n = x.length;
        KMeans start = KMeans.fit(x, k, KMeans.Init.PLUS_PLUS, 1, 20, seed);
        double[][] resp = new double[n][k];
        for (int i = 0; i < n; i++) {
            resp[i][start.labels()[i]] = 1;
        }
        GaussianMixture model = mStep(x, resp, regularization);
        double previous = Double.NEGATIVE_INFINITY;
        for (int it = 0; it < iterations; it++) {
            double ll = 0;
            for (int i = 0; i < n; i++) {
                double[] lp = model.componentLogDensities(x[i]);
                double max = Double.NEGATIVE_INFINITY;
                for (double v : lp) {
                    max = Math.max(max, v);
                }
                double s = 0;
                for (int j = 0; j < k; j++) {
                    resp[i][j] = Math.exp(lp[j] - max);
                    s += resp[i][j];
                }
                for (int j = 0; j < k; j++) {
                    resp[i][j] /= s;
                }
                ll += max + Math.log(s);
            }
            model = mStep(x, resp, regularization);
            if (Math.abs(ll - previous) < 1e-7 * n) {
                break;
            }
            previous = ll;
        }
        return model;
    }

    private static GaussianMixture mStep(double[][] x, double[][] resp, double regularization) {
        int n = x.length;
        int d = x[0].length;
        int k = resp[0].length;
        double[] w = new double[k];
        double[][] mu = new double[k][d];
        double[][][] cov = new double[k][d][d];
        for (int j = 0; j < k; j++) {
            double nk = 1e-10;
            for (int i = 0; i < n; i++) {
                nk += resp[i][j];
                for (int t = 0; t < d; t++) {
                    mu[j][t] += resp[i][j] * x[i][t];
                }
            }
            for (int t = 0; t < d; t++) {
                mu[j][t] /= nk;
            }
            for (int i = 0; i < n; i++) {
                double r = resp[i][j];
                if (r == 0) {
                    continue;
                }
                for (int a = 0; a < d; a++) {
                    double da = x[i][a] - mu[j][a];
                    for (int b = a; b < d; b++) {
                        cov[j][a][b] += r * da * (x[i][b] - mu[j][b]);
                    }
                }
            }
            for (int a = 0; a < d; a++) {
                for (int b = a; b < d; b++) {
                    cov[j][a][b] /= nk;
                    cov[j][b][a] = cov[j][a][b];
                }
                cov[j][a][a] += regularization;
            }
            w[j] = nk / n;
        }
        return new GaussianMixture(w, mu, cov);
    }

    /** {@code log w_j + log N(x | mu_j, Sigma_j)} for every component. */
    private double[] componentLogDensities(double[] x) {
        int d = x.length;
        double[] out = new double[weights.length];
        double[] diff = new double[d];
        for (int j = 0; j < weights.length; j++) {
            for (int t = 0; t < d; t++) {
                diff[t] = x[t] - means[j][t];
            }
            double[] z = MlMatrices.forwardSolve(chols[j], diff);
            out[j] = Math.log(Math.max(weights[j], 1e-300))
                    - 0.5 * (d * Math.log(2 * Math.PI) + logDets[j] + MlMatrices.dot(z, z));
        }
        return out;
    }

    /**
     * The log density of a point under the mixture.
     *
     * @param x a point
     * @return {@code log p(x)}
     */
    public double logDensity(double[] x) {
        double[] lp = componentLogDensities(x);
        double max = Double.NEGATIVE_INFINITY;
        for (double v : lp) {
            max = Math.max(max, v);
        }
        double s = 0;
        for (double v : lp) {
            s += Math.exp(v - max);
        }
        return max + Math.log(s);
    }

    /**
     * The mean log-likelihood per point.
     *
     * @param x points
     * @return the average {@code log p(x)}
     */
    public double meanLogLikelihood(double[][] x) {
        double s = 0;
        for (double[] p : x) {
            s += logDensity(p);
        }
        return s / x.length;
    }

    /**
     * The Bayesian information criterion {@code -2 log L + p log n}, lower is better.
     *
     * @param x the data
     * @return the BIC
     */
    public double bic(double[][] x) {
        int d = means[0].length;
        int k = weights.length;
        double params = (k - 1) + k * d + k * d * (d + 1) / 2.0;
        return -2 * meanLogLikelihood(x) * x.length + params * Math.log(x.length);
    }

    /**
     * Fits mixtures with 1 to {@code maxComponents} components and keeps the lowest BIC.
     *
     * @param x the data
     * @param maxComponents the largest number of components tried
     * @param iterations EM iterations per fit
     * @param seed the seed
     * @return the selected mixture
     */
    public static GaussianMixture selectByBic(double[][] x, int maxComponents, int iterations, long seed) {
        GaussianMixture best = null;
        double bestBic = Double.POSITIVE_INFINITY;
        for (int k = 1; k <= maxComponents; k++) {
            GaussianMixture m = fit(x, k, iterations, 1e-6, seed + k);
            double b = m.bic(x);
            if (b < bestBic) {
                bestBic = b;
                best = m;
            }
        }
        return best;
    }

    /**
     * The most responsible component of each point.
     *
     * @param x points
     * @return component indices
     */
    public int[] predict(double[][] x) {
        int[] out = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            double[] lp = componentLogDensities(x[i]);
            int best = 0;
            for (int j = 1; j < lp.length; j++) {
                if (lp[j] > lp[best]) {
                    best = j;
                }
            }
            out[i] = best;
        }
        return out;
    }

    /**
     * Draws a point from the mixture.
     *
     * @param random the source
     * @return a sample
     */
    public double[] sample(SplittableRandom random) {
        double u = random.nextDouble();
        int j = 0;
        while (j < weights.length - 1 && (u -= weights[j]) > 0) {
            j++;
        }
        int d = means[j].length;
        double[] z = new double[d];
        for (int t = 0; t < d; t++) {
            z[t] = random.nextGaussian();
        }
        double[] out = means[j].clone();
        for (int a = 0; a < d; a++) {
            for (int b = 0; b <= a; b++) {
                out[a] += chols[j][a][b] * z[b];
            }
        }
        return out;
    }

    /** @return the number of components */
    public int components() {
        return weights.length;
    }

    /** @return the mixing weights (a copy) */
    public double[] weights() {
        return weights.clone();
    }

    /**
     * Returns one component's covariance.
     *
     * @param j the component
     * @return its covariance (a copy)
     */
    public double[][] covariance(int j) {
        double[][] c = new double[covariances[j].length][];
        for (int i = 0; i < c.length; i++) {
            c[i] = covariances[j][i].clone();
        }
        return c;
    }

    /**
     * Returns one component's mean.
     *
     * @param j the component
     * @return its mean (a copy)
     */
    public double[] mean(int j) {
        return means[j].clone();
    }
}
