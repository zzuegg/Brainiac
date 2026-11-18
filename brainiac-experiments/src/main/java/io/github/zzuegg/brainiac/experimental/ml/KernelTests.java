package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Kernel two-sample and independence tests.
 *
 * <p>{@link #mmdTest}: the unbiased squared maximum mean discrepancy between two samples
 * (Gretton et al. 2012, <a href="https://jmlr.org/papers/v13/gretton12a.html">JMLR</a>), the
 * distance between their kernel mean embeddings, which is zero only when the distributions are
 * equal for a characteristic kernel; the p-value comes from permuting the pooled sample (the
 * pooled Gram matrix is computed once). The bandwidth is the median pairwise distance.
 * {@link #deepKernelMmdTest} learns the kernel on one half of the data by maximising the test
 * power proxy {@code MMD^2 / sigma_H1} over a small network feature map (Liu et al. 2020,
 * <a href="https://arxiv.org/abs/2002.09116">arXiv:2002.09116</a>) and tests on the other half.
 *
 * <p>{@link #hsicTest}: the Hilbert-Schmidt independence criterion (Gretton et al. 2008,
 * <a href="https://papers.nips.cc/paper/3201-a-kernel-statistical-test-of-independence">NeurIPS
 * </a>), the squared norm of the cross-covariance operator between RKHS embeddings of
 * {@code x} and {@code y}, zero only under independence, so it detects dependence of any form;
 * the p-value is by the gamma approximation of the null or by permutation.
 *
 * <p>The baselines are here too: Welch's t-test and the Kolmogorov-Smirnov test per coordinate
 * with a Bonferroni correction, the Pearson correlation test and the distance-correlation
 * permutation test.
 */
public final class KernelTests {

    /**
     * A test result.
     *
     * @param statistic the test statistic
     * @param pValue the p-value
     */
    public record Result(double statistic, double pValue) {
        /**
         * Whether the test rejects at a level.
         *
         * @param alpha the level, for example 0.05
         * @return {@code pValue < alpha}
         */
        public boolean rejects(double alpha) {
            return pValue < alpha;
        }
    }

    private KernelTests() {
    }

    /**
     * The median pairwise Euclidean distance of a sample (subsampled to 300 points).
     *
     * @param x the points
     * @return the median distance
     */
    public static double medianDistance(double[][] x) {
        int n = Math.min(x.length, 300);
        double[] d = new double[n * (n - 1) / 2];
        int c = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                d[c++] = Math.sqrt(MlMatrices.squaredDistance(x[i], x[j]));
            }
        }
        Arrays.sort(d);
        return Math.max(d[d.length / 2], 1e-12);
    }

    private static double[][] concat(double[][] a, double[][] b) {
        double[][] z = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, z, a.length, b.length);
        return z;
    }

    private static double[][] gaussianGram(double[][] z, double sigma) {
        int n = z.length;
        double g = 1 / (2 * sigma * sigma);
        double[][] k = new double[n][n];
        for (int i = 0; i < n; i++) {
            k[i][i] = 1;
            for (int j = i + 1; j < n; j++) {
                k[i][j] = Math.exp(-g * MlMatrices.squaredDistance(z[i], z[j]));
                k[j][i] = k[i][j];
            }
        }
        return k;
    }

    /** Unbiased MMD^2 from a pooled Gram matrix and a split of its indices. */
    private static double mmd2(double[][] k, int[] idx, int n) {
        int m = idx.length - n;
        double xx = 0;
        double yy = 0;
        double xy = 0;
        for (int a = 0; a < idx.length; a++) {
            double[] row = k[idx[a]];
            for (int b = 0; b < idx.length; b++) {
                if (a == b) {
                    continue;
                }
                double v = row[idx[b]];
                if (a < n && b < n) {
                    xx += v;
                } else if (a >= n && b >= n) {
                    yy += v;
                } else if (a < n) {
                    xy += v;
                }
            }
        }
        return xx / (n * (n - 1.0)) + yy / (m * (m - 1.0)) - 2 * xy / ((double) n * m);
    }

    private static Result permutationMmd(double[][] k, int n, int permutations, SplittableRandom r) {
        int total = k.length;
        int[] idx = new int[total];
        for (int i = 0; i < total; i++) {
            idx[i] = i;
        }
        double stat = mmd2(k, idx, n);
        int exceed = 0;
        for (int p = 0; p < permutations; p++) {
            for (int i = total - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = idx[i];
                idx[i] = idx[j];
                idx[j] = t;
            }
            if (mmd2(k, idx, n) >= stat) {
                exceed++;
            }
        }
        return new Result(stat, (exceed + 1.0) / (permutations + 1.0));
    }

    /**
     * The MMD permutation test with a Gaussian kernel.
     *
     * @param x one sample {@code [n][d]}
     * @param y another {@code [m][d]}
     * @param sigma the bandwidth, or {@code <= 0} for the median heuristic on the pooled sample
     * @param permutations the number of permutations, for example 200
     * @param seed the permutation seed
     * @return {@code MMD^2_u} and its p-value
     */
    public static Result mmdTest(double[][] x, double[][] y, double sigma, int permutations, long seed) {
        double[][] z = concat(x, y);
        double s = sigma > 0 ? sigma : medianDistance(z);
        return permutationMmd(gaussianGram(z, s), x.length, permutations, new SplittableRandom(seed));
    }

    /**
     * The deep-kernel MMD test: a feature map {@code phi} (per-coordinate scales times a
     * residual tanh layer that starts at zero, so training begins at an ARD kernel) and bandwidth
     * are trained on the first half of both samples to maximise {@code MMD^2 / sigma_H1}; the
     * kernel {@code (1 - eps) k(phi(a), phi(b)) q(a, b) + eps q(a, b)}, with {@code q} a Gaussian
     * kernel on the inputs, is then tested by permutation on the second half.
     *
     * @param x one sample
     * @param y another, of the same size
     * @param hidden the feature map's hidden width, for example 16
     * @param steps Adam steps, for example 100
     * @param permutations the number of permutations
     * @param seed the seed
     * @return the held-out statistic and p-value
     */
    public static Result deepKernelMmdTest(double[][] x, double[][] y, int hidden, int steps, int permutations,
            long seed) {
        int half = x.length / 2;
        int d = x[0].length;
        SplittableRandom r = new SplittableRandom(seed);
        double[][] xTrain = Arrays.copyOfRange(x, 0, half);
        double[][] yTrain = Arrays.copyOfRange(y, 0, half);
        double q = medianDistance(concat(xTrain, yTrain));
        Var w1 = Var.param(Tensor.normal(Shape.of(d, hidden), 0, (float) (1 / Math.sqrt(d)), r));
        Var b1 = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        Var w2 = Var.param(Tensor.zeros(Shape.of(hidden, d)));
        Var ard = Var.param(Tensor.zeros(Shape.of(1, d)));
        Var logSigma = Var.param(Tensor.full(Shape.of(1, 1), (float) Math.log(q)));
        List<Var> ps = new ArrayList<>(List.of(w1, b1, w2, ard, logSigma));
        Optimizer opt = Optimizer.adam(0.01f);
        Tensor tx = tensor(xTrain);
        Tensor ty = tensor(yTrain);
        Tensor qx = Tensor.of(Shape.of(half, half), toFloat(gaussianGram(xTrain, q)));
        Tensor qy = Tensor.of(Shape.of(half, half), toFloat(gaussianGram(yTrain, q)));
        Tensor qxy = Tensor.of(Shape.of(half, half), toFloat(crossGram(xTrain, yTrain, q)));
        float eps = 0.1f;
        float[] mask = new float[half * half];
        java.util.Arrays.fill(mask, 1f);
        for (int i = 0; i < half; i++) {
            mask[i * half + i] = 0;
        }
        Tensor offDiagonal = Tensor.of(Shape.of(half, half), mask);
        for (int s = 0; s < steps; s++) {
            Var fx = features(Var.of(tx), w1, b1, w2, ard);
            Var fy = features(Var.of(ty), w1, b1, w2, ard);
            Var sig2 = logSigma.mul(2f).exp().mul(2f);
            Var kxx = gram(fx, fx, sig2).mul(1 - eps).add(eps).mul(qx);
            Var kyy = gram(fy, fy, sig2).mul(1 - eps).add(eps).mul(qy);
            Var kxy = gram(fx, fy, sig2).mul(1 - eps).add(eps).mul(qxy);
            // the unbiased statistic: the diagonal of H would reward a vanishing bandwidth
            Var h = kxx.add(kyy).sub(kxy).sub(kxy.transpose()).mul(offDiagonal);
            Var mmd = h.sum().div((float) (half * (half - 1.0)));
            Var rows = h.sum(1, false).div(half - 1f);
            Var var = rows.square().mean().sub(mmd.square()).mul(4f).add(1e-6f);
            mmd.div(var.sqrt()).neg().backward();
            opt.step(ps);
        }
        double[][] xTest = Arrays.copyOfRange(x, half, x.length);
        double[][] yTest = Arrays.copyOfRange(y, half, y.length);
        double[][] z = concat(xTest, yTest);
        double[][] fz = rowsOf(Var.noGrad(() -> features(Var.of(tensor(z)), w1, b1, w2, ard)).value(), d);
        double sig = Math.exp(logSigma.value().item());
        double[][] k = new double[z.length][z.length];
        for (int i = 0; i < z.length; i++) {
            for (int j = 0; j < z.length; j++) {
                double deep = Math.exp(-MlMatrices.squaredDistance(fz[i], fz[j]) / (2 * sig * sig));
                double qq = Math.exp(-MlMatrices.squaredDistance(z[i], z[j]) / (2 * q * q));
                k[i][j] = ((1 - eps) * deep + eps) * qq;
            }
        }
        return permutationMmd(k, xTest.length, permutations, r);
    }

    private static Var features(Var x, Var w1, Var b1, Var w2, Var ard) {
        return x.matmul(w1).add(b1).tanh().matmul(w2).add(x).mul(ard.exp());
    }

    private static Var gram(Var a, Var b, Var twoSigma2) {
        Var sq = a.square().sum(1, true).add(b.square().sum(1, true).transpose()).sub(a.matmul(b.transpose()).mul(2f));
        return sq.maximum(0f).div(twoSigma2).neg().exp();
    }

    private static double[][] crossGram(double[][] a, double[][] b, double sigma) {
        double[][] k = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                k[i][j] = Math.exp(-MlMatrices.squaredDistance(a[i], b[j]) / (2 * sigma * sigma));
            }
        }
        return k;
    }

    private static Tensor tensor(double[][] x) {
        return Tensor.of(Shape.of(x.length, x[0].length), toFloat(x));
    }

    private static float[] toFloat(double[][] x) {
        int d = x[0].length;
        float[] v = new float[x.length * d];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) x[i][j];
            }
        }
        return v;
    }

    private static double[][] rowsOf(Tensor t, int d) {
        double[] v = t.toDoubleArray();
        double[][] out = new double[v.length / d][];
        for (int i = 0; i < out.length; i++) {
            out[i] = Arrays.copyOfRange(v, i * d, (i + 1) * d);
        }
        return out;
    }

    private static double[][] centredGram(double[][] x) {
        double[][] k = gaussianGram(x, medianDistance(x));
        int n = k.length;
        double[] rowMean = new double[n];
        double all = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                rowMean[i] += k[i][j] / n;
            }
            all += rowMean[i] / n;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                k[i][j] = k[i][j] - rowMean[i] - rowMean[j] + all;
            }
        }
        return k;
    }

    /**
     * The HSIC independence test with Gaussian kernels (median bandwidths).
     *
     * @param x one variable {@code [n][dx]}
     * @param y the other {@code [n][dy]}
     * @param permutations the number of permutations, or 0 for the gamma approximation
     * @param seed the permutation seed
     * @return the biased statistic {@code tr(K H L H) / n^2} and its p-value
     */
    public static Result hsicTest(double[][] x, double[][] y, int permutations, long seed) {
        int n = x.length;
        double[][] kc = centredGram(x);
        double[][] lc = centredGram(y);
        double stat = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                stat += kc[i][j] * lc[i][j];
            }
        }
        stat /= (double) n * n;
        if (permutations > 0) {
            SplittableRandom r = new SplittableRandom(seed);
            int[] p = new int[n];
            for (int i = 0; i < n; i++) {
                p[i] = i;
            }
            int exceed = 0;
            for (int t = 0; t < permutations; t++) {
                for (int i = n - 1; i > 0; i--) {
                    int j = r.nextInt(i + 1);
                    int tmp = p[i];
                    p[i] = p[j];
                    p[j] = tmp;
                }
                double s = 0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        s += kc[i][j] * lc[p[i]][p[j]];
                    }
                }
                if (s / ((double) n * n) >= stat) {
                    exceed++;
                }
            }
            return new Result(stat, (exceed + 1.0) / (permutations + 1.0));
        }
        // gamma approximation of the null (Gretton et al. 2008, section 3)
        double[][] k = gaussianGram(x, medianDistance(x));
        double[][] l = gaussianGram(y, medianDistance(y));
        double muX = 0;
        double muY = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    muX += k[i][j];
                    muY += l[i][j];
                }
            }
        }
        muX /= n * (n - 1.0);
        muY /= n * (n - 1.0);
        double mean = (1 + muX * muY - muX - muY) / n;
        double var = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    double v = kc[i][j] * lc[i][j] / 6;
                    var += v * v;
                }
            }
        }
        var = var / (n * (n - 1.0)) * 72.0 * (n - 4) * (n - 5) / ((double) n * (n - 1) * (n - 2) * (n - 3));
        double shape = mean * mean / var;
        double scale = var * n / mean;
        return new Result(stat, 1 - regularizedGammaP(shape, stat * n / scale));
    }

    /**
     * The regularised lower incomplete gamma function {@code P(a, x)}.
     *
     * @param a the shape
     * @param x the argument
     * @return {@code P(a, x)}
     */
    public static double regularizedGammaP(double a, double x) {
        if (x <= 0) {
            return 0;
        }
        double lg = ProbabilisticProgram.logGamma(a);
        if (x < a + 1) {
            double sum = 1 / a;
            double term = sum;
            for (int k = 1; k < 500; k++) {
                term *= x / (a + k);
                sum += term;
                if (term < sum * 1e-14) {
                    break;
                }
            }
            return sum * Math.exp(-x + a * Math.log(x) - lg);
        }
        // continued fraction for Q
        double b = x + 1 - a;
        double c = 1 / 1e-300;
        double d = 1 / b;
        double h = d;
        for (int i = 1; i < 500; i++) {
            double an = -i * (i - a);
            b += 2;
            d = an * d + b;
            d = Math.abs(d) < 1e-300 ? 1e-300 : d;
            c = b + an / c;
            c = Math.abs(c) < 1e-300 ? 1e-300 : c;
            d = 1 / d;
            double del = d * c;
            h *= del;
            if (Math.abs(del - 1) < 1e-14) {
                break;
            }
        }
        return 1 - Math.exp(-x + a * Math.log(x) - lg) * h;
    }

    // ------------------------------------------------------------ baselines

    /**
     * Welch's t-test on every coordinate (normal approximation), Bonferroni-corrected.
     *
     * @param x one sample
     * @param y another
     * @return the largest |t| and the corrected p-value
     */
    public static Result tTest(double[][] x, double[][] y) {
        int d = x[0].length;
        double best = 0;
        for (int j = 0; j < d; j++) {
            double[] a = McmcDiagnostics.column(x, j);
            double[] b = McmcDiagnostics.column(y, j);
            double ma = Arrays.stream(a).average().orElse(0);
            double mb = Arrays.stream(b).average().orElse(0);
            double va = 0;
            double vb = 0;
            for (double v : a) {
                va += (v - ma) * (v - ma) / (a.length - 1);
            }
            for (double v : b) {
                vb += (v - mb) * (v - mb) / (b.length - 1);
            }
            best = Math.max(best, Math.abs(ma - mb) / Math.sqrt(va / a.length + vb / b.length));
        }
        double p = 2 * (1 - MlMetrics.normalCdf(best));
        return new Result(best, Math.min(1, p * d));
    }

    /**
     * The two-sample Kolmogorov-Smirnov test on every coordinate, Bonferroni-corrected.
     *
     * @param x one sample
     * @param y another
     * @return the largest KS distance and the corrected p-value
     */
    public static Result ksTest(double[][] x, double[][] y) {
        int d = x[0].length;
        double best = 0;
        for (int j = 0; j < d; j++) {
            double[] a = McmcDiagnostics.column(x, j);
            double[] b = McmcDiagnostics.column(y, j);
            Arrays.sort(a);
            Arrays.sort(b);
            int i = 0;
            int k = 0;
            double dmax = 0;
            while (i < a.length && k < b.length) {
                double v = Math.min(a[i], b[k]);
                while (i < a.length && a[i] <= v) {
                    i++;
                }
                while (k < b.length && b[k] <= v) {
                    k++;
                }
                dmax = Math.max(dmax, Math.abs((double) i / a.length - (double) k / b.length));
            }
            best = Math.max(best, dmax);
        }
        double ne = (double) x.length * y.length / (x.length + y.length);
        double lambda = (Math.sqrt(ne) + 0.12 + 0.11 / Math.sqrt(ne)) * best;
        double q = 0;
        for (int k = 1; k <= 100; k++) {
            q += 2 * (k % 2 == 1 ? 1 : -1) * Math.exp(-2 * k * k * lambda * lambda);
        }
        return new Result(best, Math.min(1, Math.max(0, q) * d));
    }

    /**
     * The Pearson correlation test between two scalars (Fisher z, normal approximation).
     *
     * @param x one variable
     * @param y the other
     * @return the correlation and its p-value
     */
    public static Result pearsonTest(double[] x, double[] y) {
        int n = x.length;
        double mx = Arrays.stream(x).average().orElse(0);
        double my = Arrays.stream(y).average().orElse(0);
        double sxy = 0;
        double sxx = 0;
        double syy = 0;
        for (int i = 0; i < n; i++) {
            sxy += (x[i] - mx) * (y[i] - my);
            sxx += (x[i] - mx) * (x[i] - mx);
            syy += (y[i] - my) * (y[i] - my);
        }
        double rho = sxy / Math.sqrt(sxx * syy);
        double z = 0.5 * Math.log((1 + rho) / (1 - rho)) * Math.sqrt(n - 3.0);
        return new Result(rho, 2 * (1 - MlMetrics.normalCdf(Math.abs(z))));
    }

    /**
     * The distance-correlation permutation test (Szekely et al. 2007).
     *
     * @param x one variable {@code [n][dx]}
     * @param y the other {@code [n][dy]}
     * @param permutations the number of permutations
     * @param seed the seed
     * @return the squared distance covariance and its p-value
     */
    public static Result distanceCorrelationTest(double[][] x, double[][] y, int permutations, long seed) {
        double[][] a = centredDistances(x);
        double[][] b = centredDistances(y);
        int n = x.length;
        double stat = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                stat += a[i][j] * b[i][j];
            }
        }
        SplittableRandom r = new SplittableRandom(seed);
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        int exceed = 0;
        for (int t = 0; t < permutations; t++) {
            for (int i = n - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int tmp = p[i];
                p[i] = p[j];
                p[j] = tmp;
            }
            double s = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    s += a[i][j] * b[p[i]][p[j]];
                }
            }
            if (s >= stat) {
                exceed++;
            }
        }
        return new Result(stat / ((double) n * n), (exceed + 1.0) / (permutations + 1.0));
    }

    private static double[][] centredDistances(double[][] x) {
        int n = x.length;
        double[][] d = new double[n][n];
        double[] row = new double[n];
        double all = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                d[i][j] = Math.sqrt(MlMatrices.squaredDistance(x[i], x[j]));
                row[i] += d[i][j] / n;
            }
            all += row[i] / n;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                d[i][j] = d[i][j] - row[i] - row[j] + all;
            }
        }
        return d;
    }
}
