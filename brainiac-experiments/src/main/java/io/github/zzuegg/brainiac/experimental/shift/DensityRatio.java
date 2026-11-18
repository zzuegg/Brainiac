package io.github.zzuegg.brainiac.experimental.shift;

import io.github.zzuegg.brainiac.experimental.ml.KernelTests;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Density-ratio estimation {@code w(x) = q(x) / p(x)} from source samples {@code p} and
 * target samples {@code q}, for covariate-shift correction: under covariate shift
 * {@code p(y | x)} is fixed, so weighting the training loss by {@code w} is unbiased for the
 * target risk.
 *
 * <ul>
 *   <li>{@link #kliep}: a non-negative Gaussian-kernel expansion around target points,
 *       maximising the target log-likelihood {@code sum log w(x_t)} subject to
 *       {@code mean_s w = 1} by projected gradient ascent (Sugiyama et al., AISM 2008,
 *       <a href="https://doi.org/10.1007/s10463-008-0197-x">doi:10.1007/s10463-008-0197-x</a>);
 *       the bandwidth by held-out likelihood.</li>
 *   <li>{@link #ulsif}: the same model fitted by least squares,
 *       {@code alpha = (H + lambda I)^-1 h}, in closed form, with bandwidth and ridge by
 *       cross-validated squared loss (Kanamori et al., JMLR 2009,
 *       <a href="https://jmlr.org/papers/v10/kanamori09a.html">JMLR 10</a>).</li>
 *   <li>{@link #classifierOdds}: a logistic domain classifier on {@code (x, x^2)} features;
 *       {@code w = (n_s / n_t) p(target | x) / p(source | x)}.</li>
 * </ul>
 *
 * <p>Weights can be clipped ({@link #clip}); {@link #effectiveSampleSize} shows what weighting
 * costs. {@link WeightedConformal} keeps split-conformal coverage under covariate shift with the
 * same weights (Tibshirani et al., NeurIPS 2019, <a href="https://arxiv.org/abs/1904.06019">arXiv:1904.06019</a>).
 */
public final class DensityRatio {

    private final double[][] centres;
    private final double sigma;
    private final double[] alpha;
    private final double[] logistic;

    private DensityRatio(double[][] centres, double sigma, double[] alpha, double[] logistic) {
        this.centres = centres;
        this.sigma = sigma;
        this.alpha = alpha;
        this.logistic = logistic;
    }

    /**
     * @param x inputs
     * @return the estimated ratio {@code q(x) / p(x)} per row
     */
    public double[] weights(double[][] x) {
        double[] w = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            if (logistic != null) {
                double[] f = quadratic(x[i]);
                double s = 0;
                for (int j = 0; j < f.length; j++) {
                    s += f[j] * logistic[j];
                }
                w[i] = Math.exp(Math.min(30, s));
            } else {
                double[] phi = basis(x[i], centres, sigma);
                double s = 0;
                for (int j = 0; j < phi.length; j++) {
                    s += phi[j] * alpha[j];
                }
                w[i] = Math.max(0, s);
            }
        }
        return w;
    }

    /** @return the kernel bandwidth (NaN for the classifier) */
    public double sigma() {
        return logistic == null ? sigma : Double.NaN;
    }

    /**
     * uLSIF with cross-validated bandwidth and ridge.
     *
     * @param source source inputs
     * @param target target inputs
     * @param centres the number of kernel centres (target points), such as 100
     * @param seed the seed
     * @return the estimator
     */
    public static DensityRatio ulsif(double[][] source, double[][] target, int centres, long seed) {
        double[][] c = pickCentres(target, centres, seed);
        double med = KernelTests.medianDistance(concat(source, target));
        double bestScore = Double.POSITIVE_INFINITY;
        double bestSigma = med;
        double bestLambda = 1e-3;
        for (double sm : new double[] {0.25, 0.5, 1, 2}) {
            for (double lambda : new double[] {1e-3, 1e-2, 1e-1, 1}) {
                double score = 0;
                for (int fold = 0; fold < 5; fold++) {
                    double[][] st = fold(source, fold, false);
                    double[][] tt = fold(target, fold, false);
                    double[] a = ulsifFit(st, tt, c, med * sm, lambda);
                    DensityRatio r = new DensityRatio(c, med * sm, a, null);
                    double[] ws = r.weights(fold(source, fold, true));
                    double[] wt = r.weights(fold(target, fold, true));
                    score += 0.5 * mean(square(ws)) - mean(wt);
                }
                if (score < bestScore) {
                    bestScore = score;
                    bestSigma = med * sm;
                    bestLambda = lambda;
                }
            }
        }
        return new DensityRatio(c, bestSigma, ulsifFit(source, target, c, bestSigma, bestLambda), null);
    }

    private static double[] ulsifFit(double[][] source, double[][] target, double[][] c, double sigma,
            double lambda) {
        int b = c.length;
        float[] h = new float[b * b];
        float[] hv = new float[b];
        for (double[] x : source) {
            double[] phi = basis(x, c, sigma);
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < b; j++) {
                    h[i * b + j] += (float) (phi[i] * phi[j] / source.length);
                }
            }
        }
        for (double[] x : target) {
            double[] phi = basis(x, c, sigma);
            for (int i = 0; i < b; i++) {
                hv[i] += (float) (phi[i] / target.length);
            }
        }
        for (int i = 0; i < b; i++) {
            h[i * b + i] += (float) lambda;
        }
        double[] a = Linalg.solve(Tensor.of(Shape.of(b, b), h), Tensor.of(Shape.of(b, 1), hv)).toDoubleArray();
        for (int i = 0; i < b; i++) {
            a[i] = Math.max(0, a[i]);
        }
        return a;
    }

    /**
     * KLIEP with the bandwidth chosen by held-out target log-likelihood.
     *
     * @param source source inputs
     * @param target target inputs
     * @param centres the number of kernel centres, such as 100
     * @param seed the seed
     * @return the estimator
     */
    public static DensityRatio kliep(double[][] source, double[][] target, int centres, long seed) {
        double[][] c = pickCentres(target, centres, seed);
        double med = KernelTests.medianDistance(concat(source, target));
        double best = Double.NEGATIVE_INFINITY;
        double bestSigma = med;
        for (double sm : new double[] {0.25, 0.5, 1, 2}) {
            double score = 0;
            for (int fold = 0; fold < 5; fold++) {
                double[] a = kliepFit(source, fold(target, fold, false), c, med * sm);
                double[] wt = new DensityRatio(c, med * sm, a, null).weights(fold(target, fold, true));
                for (double w : wt) {
                    score += Math.log(Math.max(w, 1e-12));
                }
            }
            if (score > best) {
                best = score;
                bestSigma = med * sm;
            }
        }
        return new DensityRatio(c, bestSigma, kliepFit(source, target, c, bestSigma), null);
    }

    private static double[] kliepFit(double[][] source, double[][] target, double[][] c, double sigma) {
        int b = c.length;
        double[][] a = new double[target.length][];
        for (int i = 0; i < target.length; i++) {
            a[i] = basis(target[i], c, sigma);
        }
        double[] bv = new double[b];
        for (double[] x : source) {
            double[] phi = basis(x, c, sigma);
            for (int j = 0; j < b; j++) {
                bv[j] += phi[j] / source.length;
            }
        }
        double bb = 0;
        for (double v : bv) {
            bb += v * v;
        }
        double[] alpha = new double[b];
        Arrays.fill(alpha, 1);
        normalise(alpha, bv);
        for (double eps : new double[] {1e-1, 1e-2, 1e-3}) {
            for (int it = 0; it < 100; it++) {
                double[] grad = new double[b];
                for (double[] row : a) {
                    double s = 1e-12;
                    for (int j = 0; j < b; j++) {
                        s += row[j] * alpha[j];
                    }
                    for (int j = 0; j < b; j++) {
                        grad[j] += row[j] / s;
                    }
                }
                double dot = 0;
                for (int j = 0; j < b; j++) {
                    alpha[j] += eps * grad[j] / target.length;
                    dot += bv[j] * alpha[j];
                }
                for (int j = 0; j < b; j++) {
                    alpha[j] = Math.max(0, alpha[j] + (1 - dot) * bv[j] / bb);
                }
                normalise(alpha, bv);
            }
        }
        return alpha;
    }

    private static void normalise(double[] alpha, double[] bv) {
        double dot = 0;
        for (int j = 0; j < alpha.length; j++) {
            dot += bv[j] * alpha[j];
        }
        for (int j = 0; j < alpha.length; j++) {
            alpha[j] /= Math.max(1e-12, dot);
        }
    }

    /**
     * The probabilistic-classification estimator.
     *
     * @param source source inputs
     * @param target target inputs
     * @param l2 the ridge on the logistic weights, such as 1e-3
     * @return the estimator
     */
    public static DensityRatio classifierOdds(double[][] source, double[][] target, double l2) {
        int dim = quadratic(source[0]).length;
        double[] w = new double[dim];
        int n = source.length + target.length;
        for (int it = 0; it < 2000; it++) {
            double[] g = new double[dim];
            for (int i = 0; i < n; i++) {
                boolean isTarget = i >= source.length;
                double[] f = quadratic(isTarget ? target[i - source.length] : source[i]);
                double s = 0;
                for (int j = 0; j < dim; j++) {
                    s += f[j] * w[j];
                }
                double p = 1 / (1 + Math.exp(-s));
                double err = p - (isTarget ? 1 : 0);
                for (int j = 0; j < dim; j++) {
                    g[j] += err * f[j] / n;
                }
            }
            for (int j = 0; j < dim; j++) {
                w[j] -= 0.5 * (g[j] + l2 * w[j]);
            }
        }
        w[0] += Math.log(source.length / (double) target.length);   // prior odds correction
        return new DensityRatio(null, Double.NaN, null, w);
    }

    /**
     * @param w weights
     * @param max the largest weight kept
     * @return the clipped weights
     */
    public static double[] clip(double[] w, double max) {
        double[] out = w.clone();
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.min(out[i], max);
        }
        return out;
    }

    /**
     * @param w importance weights
     * @return {@code (sum w)^2 / sum w^2}, the sample size an unweighted set of equal variance would have
     */
    public static double effectiveSampleSize(double[] w) {
        double s = 0;
        double s2 = 0;
        for (double v : w) {
            s += v;
            s2 += v * v;
        }
        return s2 > 0 ? s * s / s2 : 0;
    }

    /** Weighted split-conformal classification with the score {@code 1 - p_y}. */
    public static final class WeightedConformal {

        private final double[] scores;
        private final double[] weights;
        private final double alpha;

        /**
         * Calibrates on held-out source data.
         *
         * @param probs source calibration probabilities {@code [n][classes]}
         * @param labels their labels
         * @param weights their density-ratio weights (all 1 for plain split conformal)
         * @param alpha the miscoverage, such as 0.1
         */
        public WeightedConformal(double[][] probs, int[] labels, double[] weights, double alpha) {
            this.scores = new double[labels.length];
            for (int i = 0; i < labels.length; i++) {
                scores[i] = 1 - probs[i][labels[i]];
            }
            this.weights = weights.clone();
            this.alpha = alpha;
        }

        /**
         * The score threshold for a test point of weight {@code w}: the {@code 1 - alpha}
         * quantile of the calibration scores weighted by {@code w_i / (sum w + w)}, with the
         * test point's mass at infinity.
         *
         * @param testWeight the test point's weight
         * @return the threshold (infinite if the set must be all classes)
         */
        public double threshold(double testWeight) {
            Integer[] order = new Integer[scores.length];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> Double.compare(scores[a], scores[b]));
            double total = testWeight;
            for (double w : weights) {
                total += w;
            }
            double cum = 0;
            for (int i : order) {
                cum += weights[i] / total;
                if (cum >= 1 - alpha) {
                    return scores[i];
                }
            }
            return Double.POSITIVE_INFINITY;
        }

        /**
         * @param p a test point's probabilities
         * @param testWeight its weight
         * @return the prediction set
         */
        public int[] predictSet(double[] p, double testWeight) {
            double t = threshold(testWeight);
            return java.util.stream.IntStream.range(0, p.length).filter(c -> 1 - p[c] <= t).toArray();
        }
    }

    // ------------------------------------------------------------------ helpers

    private static double[] quadratic(double[] x) {
        double[] f = new double[1 + 2 * x.length];
        f[0] = 1;
        for (int j = 0; j < x.length; j++) {
            f[1 + j] = x[j];
            f[1 + x.length + j] = x[j] * x[j];
        }
        return f;
    }

    private static double[] basis(double[] x, double[][] c, double sigma) {
        double[] phi = new double[c.length];
        for (int j = 0; j < c.length; j++) {
            double d2 = 0;
            for (int k = 0; k < x.length; k++) {
                double v = x[k] - c[j][k];
                d2 += v * v;
            }
            phi[j] = Math.exp(-d2 / (2 * sigma * sigma));
        }
        return phi;
    }

    private static double[][] pickCentres(double[][] target, int count, long seed) {
        int b = Math.min(count, target.length);
        int[] idx = new SplittableRandom(seed).ints(0, target.length).distinct().limit(b).toArray();
        double[][] c = new double[b][];
        for (int i = 0; i < b; i++) {
            c[i] = target[idx[i]];
        }
        return c;
    }

    private static double[][] fold(double[][] x, int fold, boolean heldOut) {
        return java.util.stream.IntStream.range(0, x.length).filter(i -> (i % 5 == fold) == heldOut)
                .mapToObj(i -> x[i]).toArray(double[][]::new);
    }

    private static double[][] concat(double[][] a, double[][] b) {
        double[][] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private static double[] square(double[] v) {
        double[] o = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            o[i] = v[i] * v[i];
        }
        return o;
    }

    private static double mean(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        return s / v.length;
    }
}
