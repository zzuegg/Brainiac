package io.github.zzuegg.brainiac.experimental.shift;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.KernelTests;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Discrepancy-based domain adaptation: align source and target feature statistics directly,
 * without an adversary.
 *
 * <ul>
 *   <li>{@link #coral}: Deep CORAL, the squared Frobenius distance between the source and
 *       target feature covariances, divided by {@code 4 d^2} (Sun and Saenko, ECCV-W 2016,
 *       <a href="https://arxiv.org/abs/1607.01719">arXiv:1607.01719</a>).</li>
 *   <li>{@link #mmdLinear}: DAN's multi-kernel MMD with the unbiased linear-time estimator over
 *       disjoint pairs, a sum of Gaussian kernels at bandwidths {@code sigma * 2^j},
 *       {@code j = -2..2} (Long et al., ICML 2015, <a href="https://arxiv.org/abs/1502.02791">arXiv:1502.02791</a>);
 *       {@link #mmd} is the quadratic-time unbiased estimator, less noisy on small batches.
 *       {@link #medianBandwidth} sets {@code sigma} by {@link KernelTests#medianDistance}'s
 *       median heuristic.</li>
 * </ul>
 */
public final class DiscrepancyAdaptation {

    private static final double[] SCALES = {0.25, 0.5, 1, 2, 4};

    private DiscrepancyAdaptation() {
    }

    /**
     * @param source source features {@code [n, d]}
     * @param target target features {@code [m, d]}
     * @return the CORAL loss
     */
    public static Var coral(Var source, Var target) {
        int d = source.shape().dim(1);
        Var cs = covariance(source);
        Var ct = covariance(target);
        return cs.sub(ct).square().sum().div(4f * d * d);
    }

    private static Var covariance(Var x) {
        int n = x.shape().dim(0);
        Var c = x.sub(x.mean(0, true));
        return c.transpose().matmul(c).div(Math.max(1, n - 1));
    }

    /**
     * The median-heuristic bandwidth of the pooled (detached) features.
     *
     * @param source source features
     * @param target target features
     * @return the median pairwise distance
     */
    public static double medianBandwidth(Tensor source, Tensor target) {
        Tensor all = Tensor.concat(0, source, target);
        int d = all.shape().dim(1);
        double[] v = all.toDoubleArray();
        double[][] rows = new double[v.length / d][d];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(v, i * d, rows[i], 0, d);
        }
        return Math.max(1e-3, KernelTests.medianDistance(rows));
    }

    /**
     * Linear-time multi-kernel MMD^2: the mean over {@code i} of
     * {@code k(x_2i, x_2i+1) + k(y_2i, y_2i+1) - k(x_2i, y_2i+1) - k(x_2i+1, y_2i)}.
     *
     * @param source source features {@code [n, d]}
     * @param target target features {@code [n, d]}, the same count
     * @param sigma the base bandwidth
     * @return the estimate (unbiased, can be negative)
     */
    public static Var mmdLinear(Var source, Var target, double sigma) {
        int n = Math.min(source.shape().dim(0), target.shape().dim(0)) / 2 * 2;
        Var x1 = everyOther(source, n, 0);
        Var x2 = everyOther(source, n, 1);
        Var y1 = everyOther(target, n, 0);
        Var y2 = everyOther(target, n, 1);
        return pairKernel(x1, x2, sigma).add(pairKernel(y1, y2, sigma)).sub(pairKernel(x1, y2, sigma))
                .sub(pairKernel(x2, y1, sigma)).mean();
    }

    /**
     * Quadratic-time unbiased multi-kernel MMD^2.
     *
     * @param source source features {@code [n, d]}
     * @param target target features {@code [m, d]}
     * @param sigma the base bandwidth
     * @return the estimate
     */
    public static Var mmd(Var source, Var target, double sigma) {
        int n = source.shape().dim(0);
        int m = target.shape().dim(0);
        Var kxx = gram(source, source, sigma);
        Var kyy = gram(target, target, sigma);
        Var kxy = gram(source, target, sigma);
        float diag = SCALES.length;   // k(x, x) = 1 per kernel
        Var xx = kxx.sum().sub(diag * n).div((float) n * (n - 1));
        Var yy = kyy.sum().sub(diag * m).div((float) m * (m - 1));
        return xx.add(yy).sub(kxy.sum().mul(2f / ((float) n * m)));
    }

    private static Var everyOther(Var x, int n, int offset) {
        int d = x.shape().dim(1);
        return x.slice(0, 0, n).reshape(n / 2, 2, d).slice(1, offset, offset + 1).reshape(n / 2, d);
    }

    private static Var pairKernel(Var a, Var b, double sigma) {
        Var d2 = a.sub(b).square().sum(1, false);
        Var k = null;
        for (double s : SCALES) {
            double bw = sigma * s;
            Var term = d2.mul((float) (-1 / (2 * bw * bw))).exp();
            k = k == null ? term : k.add(term);
        }
        return k;
    }

    private static Var gram(Var a, Var b, double sigma) {
        Var aa = a.square().sum(1, true);
        Var bb = b.square().sum(1, true).transpose();
        Var d2 = aa.add(bb).sub(a.matmul(b.transpose()).mul(2f)).relu();
        Var k = null;
        for (double s : SCALES) {
            double bw = sigma * s;
            Var term = d2.mul((float) (-1 / (2 * bw * bw))).exp();
            k = k == null ? term : k.add(term);
        }
        return k;
    }
}
