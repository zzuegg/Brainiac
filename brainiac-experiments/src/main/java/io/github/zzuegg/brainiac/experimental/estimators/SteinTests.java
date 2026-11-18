package io.github.zzuegg.brainiac.experimental.estimators;

import io.github.zzuegg.brainiac.experimental.ml.KernelTests;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Goodness-of-fit tests of a sample against a model known only through its score
 * {@code s_p(x) = grad log p(x)}, so the normaliser is never needed.
 *
 * <ul>
 *   <li>{@link #ksd}: the kernel Stein discrepancy (Liu, Lee and Jordan, ICML 2016; Chwialkowski,
 *       Strathmann and Gretton, ICML 2016) with the inverse multiquadric kernel
 *       {@code (c^2 + |x - y|^2)^beta}, {@code beta in (-1, 0)}, which detects non-convergence
 *       (Gorham and Mackey, ICML 2017). The V-statistic {@code n KSD^2} is calibrated with a wild
 *       bootstrap using Rademacher weights.</li>
 *   <li>{@link #fssd}: the finite-set Stein discrepancy (Jitkrittum et al., NeurIPS 2017), a
 *       linear-time statistic from the Stein witness at {@code J} test locations. Locations and the
 *       Gaussian bandwidth are optimised on a held-out split to maximise the test-power proxy
 *       {@code FSSD^2 / sigma_H1}; the null is the weighted chi-square sum
 *       {@code sum_j w_j (Z_j^2 - 1)} with {@code w} the eigenvalues of the feature covariance.</li>
 * </ul>
 * Results use {@link KernelTests.Result}, as the MMD and HSIC tests do.
 */
public final class SteinTests {

    private SteinTests() {
    }

    /**
     * The kernel Stein discrepancy test.
     *
     * @param x the sample {@code [n][d]}
     * @param score the model score {@code grad log p}
     * @param c the IMQ scale, for example 1
     * @param beta the IMQ exponent, for example -0.5
     * @param bootstraps wild-bootstrap replicates
     * @param seed the bootstrap seed
     * @return {@code n KSD_V^2} and its p-value
     */
    public static KernelTests.Result ksd(double[][] x, UnaryOperator<double[]> score, double c, double beta,
            int bootstraps, long seed) {
        int n = x.length;
        int d = x[0].length;
        double[][] s = new double[n][];
        for (int i = 0; i < n; i++) {
            s[i] = score.apply(x[i]);
        }
        double[][] u = new double[n][n];
        double stat = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double r2 = 0;
                double sxr = 0;
                double syr = 0;
                double ss = 0;
                for (int a = 0; a < d; a++) {
                    double r = x[i][a] - x[j][a];
                    r2 += r * r;
                    sxr += s[i][a] * r;
                    syr += s[j][a] * r;
                    ss += s[i][a] * s[j][a];
                }
                double q = c * c + r2;
                double k = Math.pow(q, beta);
                double kq1 = Math.pow(q, beta - 1);
                double kq2 = Math.pow(q, beta - 2);
                // grad_x k = 2 beta q^(beta-1) r, grad_y k = -grad_x k
                double v = ss * k - sxr * 2 * beta * kq1 + syr * 2 * beta * kq1
                        - 4 * beta * (beta - 1) * kq2 * r2 - 2 * beta * d * kq1;
                u[i][j] = v;
                u[j][i] = v;
                stat += i == j ? v : 2 * v;
            }
        }
        stat /= n;
        SplittableRandom r = new SplittableRandom(seed);
        int exceed = 0;
        double[] w = new double[n];
        for (int b = 0; b < bootstraps; b++) {
            for (int i = 0; i < n; i++) {
                w[i] = r.nextBoolean() ? 1 : -1;
            }
            double t = 0;
            for (int i = 0; i < n; i++) {
                double row = 0;
                for (int j = 0; j < n; j++) {
                    row += u[i][j] * w[j];
                }
                t += w[i] * row;
            }
            if (t / n >= stat) {
                exceed++;
            }
        }
        return new KernelTests.Result(stat, (exceed + 1.0) / (bootstraps + 1));
    }

    /**
     * The finite-set Stein discrepancy test with optimised locations and bandwidth.
     *
     * @param x the sample {@code [n][d]}
     * @param score the model score
     * @param locations the number of test locations {@code J}
     * @param trainFraction the share of the sample used to optimise locations, for example 0.2
     * @param optimiseSteps gradient-ascent steps on the power proxy
     * @param nullSamples draws from the weighted chi-square null
     * @param seed the seed
     * @return {@code n FSSD_u^2} on the test split and its p-value
     */
    public static KernelTests.Result fssd(double[][] x, UnaryOperator<double[]> score, int locations,
            double trainFraction, int optimiseSteps, int nullSamples, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = x.length;
        int d = x[0].length;
        int[] perm = permutation(n, r);
        int nTrain = (int) Math.round(n * trainFraction);
        double[][] train = new double[nTrain][];
        double[][] test = new double[n - nTrain][];
        for (int i = 0; i < n; i++) {
            if (i < nTrain) {
                train[i] = x[perm[i]];
            } else {
                test[i - nTrain] = x[perm[i]];
            }
        }
        double[][] sTrain = scores(train, score);
        double[][] sTest = scores(test, score);
        // parameters: J locations then log bandwidth
        double[] theta = new double[locations * d + 1];
        double[] mean = MlMatrices.columnMeans(train);
        double[] sd = new double[d];
        for (double[] p : train) {
            for (int a = 0; a < d; a++) {
                sd[a] += (p[a] - mean[a]) * (p[a] - mean[a]) / train.length;
            }
        }
        for (int j = 0; j < locations; j++) {
            for (int a = 0; a < d; a++) {
                theta[j * d + a] = mean[a] + Math.sqrt(sd[a]) * gaussian(r);
            }
        }
        theta[locations * d] = Math.log(Math.max(KernelTests.medianDistance(train), 1e-3));
        double step = 0.1;
        double h = 1e-4;
        for (int it = 0; it < optimiseSteps; it++) {
            double[] g = new double[theta.length];
            double norm = 0;
            for (int p = 0; p < theta.length; p++) {
                double keep = theta[p];
                theta[p] = keep + h;
                double up = powerProxy(train, sTrain, theta, locations);
                theta[p] = keep - h;
                double down = powerProxy(train, sTrain, theta, locations);
                theta[p] = keep;
                g[p] = (up - down) / (2 * h);
                norm += g[p] * g[p];
            }
            norm = Math.sqrt(norm);
            if (!(norm > 0) || !Double.isFinite(norm)) {
                break;
            }
            for (int p = 0; p < theta.length; p++) {
                theta[p] += step * g[p] / norm;
            }
        }
        double[][] tau = features(test, sTest, theta, locations);
        int m = tau.length;
        int f = tau[0].length;
        double[] mu = new double[f];
        double sumSq = 0;
        for (double[] t : tau) {
            for (int a = 0; a < f; a++) {
                mu[a] += t[a];
                sumSq += t[a] * t[a];
            }
        }
        double tot = 0;
        for (int a = 0; a < f; a++) {
            tot += mu[a] * mu[a];
        }
        double fssd2 = (tot - sumSq) / (m * (m - 1.0));
        double stat = m * fssd2;
        for (int a = 0; a < f; a++) {
            mu[a] /= m;
        }
        double[][] cov = new double[f][f];
        for (double[] t : tau) {
            for (int a = 0; a < f; a++) {
                for (int b = 0; b < f; b++) {
                    cov[a][b] += (t[a] - mu[a]) * (t[b] - mu[b]) / m;
                }
            }
        }
        double[] eig = MlMatrices.eigh(cov).values();
        int exceed = 0;
        for (int s = 0; s < nullSamples; s++) {
            double v = 0;
            for (double e : eig) {
                double z = gaussian(r);
                v += Math.max(e, 0) * (z * z - 1);
            }
            if (v >= stat) {
                exceed++;
            }
        }
        return new KernelTests.Result(stat, (exceed + 1.0) / (nullSamples + 1));
    }

    private static double powerProxy(double[][] x, double[][] s, double[] theta, int locations) {
        double[][] tau = features(x, s, theta, locations);
        int m = tau.length;
        int f = tau[0].length;
        double[] mu = new double[f];
        double sumSq = 0;
        for (double[] t : tau) {
            for (int a = 0; a < f; a++) {
                mu[a] += t[a];
                sumSq += t[a] * t[a];
            }
        }
        double tot = 0;
        for (int a = 0; a < f; a++) {
            tot += mu[a] * mu[a];
        }
        double fssd2 = (tot - sumSq) / (m * (m - 1.0));
        for (int a = 0; a < f; a++) {
            mu[a] /= m;
        }
        // sigma_H1^2 = 4 mu^T Sigma mu = 4 var(tau . mu)
        double mean = 0;
        double sq = 0;
        for (double[] t : tau) {
            double v = MlMatrices.dot(t, mu);
            mean += v / m;
            sq += v * v / m;
        }
        double var = 4 * (sq - mean * mean);
        return fssd2 / Math.sqrt(var + 1e-4);
    }

    /** {@code tau(x) in R^{dJ}}: {@code (s(x) k(x, v_j) + grad_x k(x, v_j)) / sqrt(dJ)}. */
    private static double[][] features(double[][] x, double[][] s, double[] theta, int locations) {
        int n = x.length;
        int d = x[0].length;
        double sigma = Math.exp(theta[locations * d]);
        double inv = 1 / (sigma * sigma);
        double norm = 1 / Math.sqrt(d * locations);
        double[][] tau = new double[n][d * locations];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < locations; j++) {
                double r2 = 0;
                for (int a = 0; a < d; a++) {
                    double r = x[i][a] - theta[j * d + a];
                    r2 += r * r;
                }
                double k = Math.exp(-r2 * inv / 2);
                for (int a = 0; a < d; a++) {
                    double r = x[i][a] - theta[j * d + a];
                    tau[i][j * d + a] = (s[i][a] * k - k * r * inv) * norm;
                }
            }
        }
        return tau;
    }

    private static double[][] scores(double[][] x, UnaryOperator<double[]> score) {
        double[][] s = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            s[i] = score.apply(x[i]);
        }
        return s;
    }

    static int[] permutation(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    static double gaussian(SplittableRandom r) {
        double u = r.nextDouble();
        double v = r.nextDouble();
        return Math.sqrt(-2 * Math.log(1 - u)) * Math.cos(2 * Math.PI * v);
    }
}
