package io.github.zzuegg.brainiac.experimental.quadrature;

import java.util.SplittableRandom;

/**
 * Kernel herding (Chen, Welling and Smola, UAI 2010) for the standard normal with the Gaussian
 * kernel {@code exp(-|x - x'|^2 / (2 l^2))}: each new point maximises
 * {@code mu_p(x) - 1/(t+1) sum_i k(x, x_i)} over random candidates, where the kernel mean
 * {@code mu_p(x) = (l^2/(l^2+1))^(d/2) exp(-|x|^2 / (2(l^2+1)))} is closed-form. Herding is
 * Bayesian quadrature with uniform weights (Huszar and Duvenaud, UAI 2012); {@link #mmd} gives
 * the exact maximum mean discrepancy of any point set to the target.
 */
public final class KernelHerding {

    private KernelHerding() {
    }

    private static double kernelMean(double[] x, double l) {
        double s = 0;
        for (double v : x) {
            s += v * v;
        }
        return Math.pow(l * l / (l * l + 1), x.length / 2.0) * Math.exp(-s / (2 * (l * l + 1)));
    }

    private static double k(double[] a, double[] b, double l) {
        double s = 0;
        for (int j = 0; j < a.length; j++) {
            s += (a[j] - b[j]) * (a[j] - b[j]);
        }
        return Math.exp(-s / (2 * l * l));
    }

    /**
     * Herds points.
     *
     * @param n the number of points
     * @param d the dimension
     * @param lengthScale the kernel length scale
     * @param candidates random candidates per step
     * @param seed the seed
     * @return points {@code [n][d]}
     */
    public static double[][] herd(int n, int d, double lengthScale, int candidates, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][];
        for (int t = 0; t < n; t++) {
            double best = Double.NEGATIVE_INFINITY;
            double[] arg = null;
            for (int c = 0; c < candidates; c++) {
                double[] cand = new double[d];
                for (int j = 0; j < d; j++) {
                    cand[j] = 1.3 * r.nextGaussian();
                }
                double s = 0;
                for (int i = 0; i < t; i++) {
                    s += k(cand, x[i], lengthScale);
                }
                double v = kernelMean(cand, lengthScale) - s / (t + 1);
                if (v > best) {
                    best = v;
                    arg = cand;
                }
            }
            x[t] = arg;
        }
        return x;
    }

    /**
     * The exact MMD between the uniform empirical measure on {@code x} and {@code N(0, I)}.
     *
     * @param x points
     * @param lengthScale the kernel length scale
     * @return the MMD
     */
    public static double mmd(double[][] x, double lengthScale) {
        int n = x.length;
        int d = x[0].length;
        double l2 = lengthScale * lengthScale;
        double pp = Math.pow(l2 / (l2 + 2), d / 2.0);
        double px = 0;
        double xx = 0;
        for (int i = 0; i < n; i++) {
            px += kernelMean(x[i], lengthScale) / n;
            for (int j = 0; j < n; j++) {
                xx += k(x[i], x[j], lengthScale) / ((double) n * n);
            }
        }
        return Math.sqrt(Math.max(pp - 2 * px + xx, 0));
    }
}
